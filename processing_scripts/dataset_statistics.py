"""Report per-language, per-split, and total statistics for the Multilingual USAS Wikipedia dataset."""

import csv
import dataclasses
import os
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from datasets import Dataset, get_dataset_config_names, load_dataset
from dotenv import load_dotenv
from rich import print as rprint
from rich.table import Table

from wikipedia_processing.utils import (
    get_valid_usas_language_processing_wikipedia_codes,
    language_display_name,
)

DATASET_SPLITS = ("train", "validation")


class DatasetSplit(str, Enum):
    """Which split(s) of the Hub dataset to compute statistics over.

    `all` reports `train`, `validation`, and their combined total as
    separate rows. `combined` loads both splits but reports only their
    combined total, without the separate `train`/`validation` rows.
    """

    train = "train"
    validation = "validation"
    all = "all"
    combined = "combined"


COLUMN_LABELS = {
    "language": "Language",
    "split": "Split",
    "number_of_articles": "Articles",
    "number_of_sentences": "Sentences (M)",
    "number_of_tokens": "Tokens (M)",
    "number_of_labelled_tokens": "Labelled Tokens (M)",
    "labels_per_token": "Labels per Token",
    "multi_tag_membership_percentage": "Multi Tag Membership (%)",
    "number_of_unique_tags": "Unique Tags",
    "number_of_mwes": "MWEs (M)",
    "mwe_token_percentage": "MWE Tokens (%)",
}
COLUMNS = tuple(COLUMN_LABELS)

# Columns whose value is expressed in millions (rounded to 3 decimal places)
# rather than as a raw count.
MILLION_SCALED_COLUMNS = frozenset({"number_of_sentences", "number_of_tokens", "number_of_labelled_tokens", "number_of_mwes"})

WikipediaLanguageCode = Enum("WikipediaLanguageCode", [(value, value) for value in get_valid_usas_language_processing_wikipedia_codes()], type=str)
ColumnName = Enum("ColumnName", [(value, value) for value in COLUMNS], type=str)


@dataclasses.dataclass
class DatasetStatistics:
    """Aggregate statistics for a set of articles.

    Attributes:
        number_of_articles: Number of articles the statistics were computed over.
        number_of_sentences: Total number of sentences across all articles.
        number_of_tokens: Total number of tokens across all articles.
        number_of_labelled_tokens: Number of tokens with at least one USAS tag
            (a token's `tags` entry, e.g. `tags[0][0]`, is non-empty).
        number_of_multi_tag_tokens: Total number of individual USAS tag
            labels that belong to a "multi tag membership" group -- any tag
            group (a labelled token's `tags` entry, or an individual group
            within its `other_tags` entry; both are positive labels when
            training) that itself contains more than one USAS tag. Every
            tag within such a group counts individually, e.g. a token whose
            `tags` entry is `["A3", "M6"]` contributes 2, and a further
            `other_tags` group of `["Z2", "Z9"]` on that same token
            contributes another 2. Since every tag counted here is also
            counted in `number_of_tag_labels`, this is always <=
            `number_of_tag_labels`.
        number_of_tag_labels: Total number of individual USAS tag labels
            across all tokens, counting both the `tags` and `other_tags`
            columns (both are positive labels when training) -- a token with
            one `tags` entry and two `other_tags` groups of one tag each
            contributes 3.
        unique_tags: The set of distinct USAS tag strings seen across all
            articles, from both the `tags` and `other_tags` columns.
        number_of_mwes: Total number of Multi-Word Expressions (MWEs) across all articles.
        number_of_mwe_tokens: Number of tokens that are part of at least one
            Multi-Word Expression, i.e. whose `mwes` entry is non-empty.
    """

    number_of_articles: int = 0
    number_of_sentences: int = 0
    number_of_tokens: int = 0
    number_of_labelled_tokens: int = 0
    number_of_multi_tag_tokens: int = 0
    number_of_tag_labels: int = 0
    unique_tags: set[str] = dataclasses.field(default_factory=set)
    number_of_mwes: int = 0
    number_of_mwe_tokens: int = 0

    @property
    def labels_per_token(self) -> float:
        """Average number of USAS tag labels per token, across `tags` and `other_tags`.

        Examples:
            >>> DatasetStatistics(number_of_tokens=4, number_of_tag_labels=6).labels_per_token
            1.5
            >>> DatasetStatistics().labels_per_token
            0.0
        """
        if self.number_of_tokens == 0:
            return 0.0
        return self.number_of_tag_labels / self.number_of_tokens

    @property
    def multi_tag_membership_percentage(self) -> float:
        """Percentage of USAS tag labels that belong to a "multi tag membership" group.

        A "multi tag membership" group is any tag group -- a labelled
        token's `tags` entry, or an individual group within its
        `other_tags` entry -- that itself contains more than one USAS tag
        (e.g. `tags[0][0]` is `["A3", "M6"]`, or one of `other_tags[0][0]`'s
        groups is `["A3", "M6"]`). Every tag within such a group counts
        towards both the numerator (`number_of_multi_tag_tokens`) and the
        denominator (`number_of_tag_labels`, the total count of individual
        tag labels across `tags` and `other_tags`), so this always falls
        between 0% and 100%.

        Examples:
            >>> DatasetStatistics(number_of_tag_labels=4, number_of_multi_tag_tokens=1).multi_tag_membership_percentage
            25.0
            >>> DatasetStatistics().multi_tag_membership_percentage
            0.0
        """
        if self.number_of_tag_labels == 0:
            return 0.0
        return self.number_of_multi_tag_tokens / self.number_of_tag_labels * 100

    @property
    def number_of_unique_tags(self) -> int:
        """Number of distinct USAS tag strings seen across all articles."""
        return len(self.unique_tags)

    @property
    def mwe_token_percentage(self) -> float:
        """Percentage of tokens that are part of at least one Multi-Word Expression (MWE).

        Examples:
            >>> DatasetStatistics(number_of_tokens=4, number_of_mwe_tokens=1).mwe_token_percentage
            25.0
            >>> DatasetStatistics().mwe_token_percentage
            0.0
        """
        if self.number_of_tokens == 0:
            return 0.0
        return self.number_of_mwe_tokens / self.number_of_tokens * 100

    def merged_with(self, other: "DatasetStatistics") -> "DatasetStatistics":
        """Combine these statistics with another set of statistics.

        Args:
            other: The statistics to merge in.

        Returns:
            A new `DatasetStatistics` with counts summed and `unique_tags` unioned.

        Examples:
            >>> a = DatasetStatistics(number_of_articles=1, number_of_tokens=3, unique_tags={"Z2"})
            >>> b = DatasetStatistics(number_of_articles=1, number_of_tokens=5, unique_tags={"A3"})
            >>> merged = a.merged_with(b)
            >>> merged.number_of_articles, merged.number_of_tokens
            (2, 8)
            >>> sorted(merged.unique_tags)
            ['A3', 'Z2']
        """
        return DatasetStatistics(
            number_of_articles=self.number_of_articles + other.number_of_articles,
            number_of_sentences=self.number_of_sentences + other.number_of_sentences,
            number_of_tokens=self.number_of_tokens + other.number_of_tokens,
            number_of_labelled_tokens=self.number_of_labelled_tokens + other.number_of_labelled_tokens,
            number_of_multi_tag_tokens=self.number_of_multi_tag_tokens + other.number_of_multi_tag_tokens,
            number_of_tag_labels=self.number_of_tag_labels + other.number_of_tag_labels,
            unique_tags=self.unique_tags | other.unique_tags,
            number_of_mwes=self.number_of_mwes + other.number_of_mwes,
            number_of_mwe_tokens=self.number_of_mwe_tokens + other.number_of_mwe_tokens,
        )


def compute_article_statistics(
    tokens: list[list[str]],
    tags: list[list[list[str]]],
    other_tags: list[list[list[list[str]]]],
    mwes: list[list[list[int]]],
) -> DatasetStatistics:
    """Compute statistics for a single article.

    Args:
        tokens: Per-sentence lists of token strings, as stored in the
            dataset's `tokens` column.
        tags: Per-sentence, per-token lists of USAS tag strings, as stored
            in the dataset's `tags` column. A token is "labelled" if its
            list of tags is non-empty.
        other_tags: Per-sentence, per-token lists of other valid USAS tag
            groups, one level deeper than `tags`, as stored in the dataset's
            `other_tags` column.
        mwes: Per-sentence, per-token lists of Multi-Word Expression (MWE)
            labels, as stored in the dataset's `mwes` column. Labels are
            unique per sentence and reset at each sentence boundary, so MWEs
            are counted per sentence before being summed.

    Returns:
        A `DatasetStatistics` for this single article (`number_of_articles`
        is always 1).

    Examples:
        >>> tokens = [["A", "cat", "sat"]]
        >>> tags = [[["Z2"], [], ["A3"]]]
        >>> other_tags = [[[["M6", "Z9"]], [], []]]
        >>> mwes = [[[], [1], [1]]]
        >>> stats = compute_article_statistics(tokens, tags, other_tags, mwes)
        >>> stats.number_of_articles, stats.number_of_sentences, stats.number_of_tokens
        (1, 1, 3)
        >>> stats.number_of_labelled_tokens, stats.number_of_multi_tag_tokens
        (2, 2)
        >>> stats.number_of_tag_labels
        4
        >>> sorted(stats.unique_tags)
        ['A3', 'M6', 'Z2', 'Z9']
        >>> stats.number_of_mwes, stats.number_of_mwe_tokens
        (1, 2)
    """
    number_of_sentences = len(tokens)
    number_of_tokens = sum(len(sentence_tokens) for sentence_tokens in tokens)

    number_of_labelled_tokens = 0
    number_of_multi_tag_tokens = 0
    number_of_tag_labels = 0
    unique_tags: set[str] = set()
    for sentence_tags, sentence_other_tags in zip(tags, other_tags):
        for token_tags, token_other_tag_groups in zip(sentence_tags, sentence_other_tags):
            number_of_tag_labels += len(token_tags) + sum(len(group) for group in token_other_tag_groups)
            unique_tags.update(token_tags)
            for group in token_other_tag_groups:
                unique_tags.update(group)
            if token_tags:
                number_of_labelled_tokens += 1
                if len(token_tags) > 1:
                    number_of_multi_tag_tokens += len(token_tags)
                for group in token_other_tag_groups:
                    if len(group) > 1:
                        number_of_multi_tag_tokens += len(group)

    number_of_mwes = 0
    number_of_mwe_tokens = 0
    for sentence_mwes in mwes:
        sentence_mwe_labels: set[int] = set()
        for token_mwe_labels in sentence_mwes:
            if token_mwe_labels:
                number_of_mwe_tokens += 1
            sentence_mwe_labels.update(token_mwe_labels)
        number_of_mwes += len(sentence_mwe_labels)

    return DatasetStatistics(
        number_of_articles=1,
        number_of_sentences=number_of_sentences,
        number_of_tokens=number_of_tokens,
        number_of_labelled_tokens=number_of_labelled_tokens,
        number_of_multi_tag_tokens=number_of_multi_tag_tokens,
        number_of_tag_labels=number_of_tag_labels,
        unique_tags=unique_tags,
        number_of_mwes=number_of_mwes,
        number_of_mwe_tokens=number_of_mwe_tokens,
    )


def compute_split_statistics(dataset: Dataset) -> DatasetStatistics:
    """Compute aggregate statistics across every article in a dataset split.

    Args:
        dataset: A dataset split (e.g. one language's `train` or
            `validation` split) with `tokens`, `tags`, `other_tags`, and
            `mwes` columns.

    Returns:
        A `DatasetStatistics` aggregated over every article in `dataset`.
    """
    statistics = DatasetStatistics()
    for example in dataset.select_columns(["tokens", "tags", "other_tags", "mwes"]):
        statistics = statistics.merged_with(compute_article_statistics(example["tokens"], example["tags"], example["other_tags"], example["mwes"]))
    return statistics


def statistics_row(language: str, split: str, statistics: DatasetStatistics) -> dict[str, str | int | float]:
    """Format one `DatasetStatistics` as a flat row for display/export.

    Args:
        language: Language label for the row, e.g. a Wikipedia language
            code, or `"Total"` for the all-languages rows.
        split: Split label for the row, e.g. `"train"`, `"validation"`, or
            `"total"` (the `train` + `validation` combination).
        statistics: The statistics to format.

    Returns:
        A dict of column name (matching `COLUMNS`) to formatted value,
        suitable for a `rich.table.Table` row or a CSV row. The columns in
        `MILLION_SCALED_COLUMNS` (sentences, tokens, labelled tokens, MWEs)
        are expressed in millions, rounded to 3 decimal places, rather than
        as raw counts.

    Examples:
        >>> stats = DatasetStatistics(number_of_sentences=2_000_000, number_of_tokens=5_000_000, number_of_labelled_tokens=4_500_000, number_of_mwes=1_500_000)
        >>> row = statistics_row("English", "train", stats)
        >>> row["number_of_sentences"], row["number_of_tokens"], row["number_of_labelled_tokens"], row["number_of_mwes"]
        (2.0, 5.0, 4.5, 1.5)
    """
    return {
        "language": language,
        "split": split,
        "number_of_articles": statistics.number_of_articles,
        "number_of_sentences": round(statistics.number_of_sentences / 1_000_000, 3),
        "number_of_tokens": round(statistics.number_of_tokens / 1_000_000, 3),
        "number_of_labelled_tokens": round(statistics.number_of_labelled_tokens / 1_000_000, 3),
        "labels_per_token": round(statistics.labels_per_token, 2),
        "multi_tag_membership_percentage": round(statistics.multi_tag_membership_percentage, 2),
        "number_of_unique_tags": statistics.number_of_unique_tags,
        "number_of_mwes": round(statistics.number_of_mwes / 1_000_000, 3),
        "mwe_token_percentage": round(statistics.mwe_token_percentage, 2),
    }


def format_row_value(value: str | int | float, decimal_places: int = 2) -> str:
    """Format a single row value for table display, adding `,` thousands separators to numbers.

    Args:
        value: The value to format, as produced by `statistics_row`.
        decimal_places: Number of decimal places to use when `value` is a
            float. Defaults to 2.

    Returns:
        `value` unchanged if it is a string, otherwise formatted with `,`
        thousands separators (and, for floats, `decimal_places` decimal
        places).

    Examples:
        >>> format_row_value("da")
        'da'
        >>> format_row_value(1369932)
        '1,369,932'
        >>> format_row_value(7739.7288)
        '7,739.73'
        >>> format_row_value(7739.7288, decimal_places=3)
        '7,739.729'
    """
    match value:
        case int():
            return f"{value:,}"
        case float():
            return f"{value:,.{decimal_places}f}"
        case _:
            return str(value)


def decimal_places_for_column(column: str) -> int:
    """Return how many decimal places a column's float values should be shown with.

    Args:
        column: A column name, matching `COLUMNS`.

    Returns:
        3 for a column in `MILLION_SCALED_COLUMNS`, otherwise 2.

    Examples:
        >>> decimal_places_for_column("number_of_tokens")
        3
        >>> decimal_places_for_column("labels_per_token")
        2
    """
    return 3 if column in MILLION_SCALED_COLUMNS else 2


def escape_latex(text: str) -> str:
    r"""Escape LaTeX special characters in a string.

    Args:
        text: The text to escape.

    Returns:
        `text` with LaTeX special characters replaced by their escaped equivalents.

    Examples:
        >>> escape_latex("50%")
        '50\\%'
        >>> escape_latex("a_b & c")
        'a\\_b \\& c'
    """
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def rows_to_latex(rows: list[dict[str, str | int | float]], columns: tuple[str, ...], column_labels: dict[str, str]) -> str:
    r"""Render statistics rows as a LaTeX `tabular` environment.

    Args:
        rows: Rows to render, as produced by `statistics_row`.
        columns: Column names to include, and their order, e.g. `COLUMNS`.
        column_labels: Human-readable header text for each entry in `columns`.

    Returns:
        A LaTeX `tabular` environment (using `booktabs` rules), ready to be
        embedded within a `table` environment in a LaTeX document.

    Examples:
        >>> rows = [{"language": "da", "number_of_articles": 187}]
        >>> print(rows_to_latex(rows, ("language", "number_of_articles"), {"language": "Language", "number_of_articles": "Articles"}))
        \begin{tabular}{ll}
        \toprule
        Language & Articles \\
        \midrule
        da & 187 \\
        \bottomrule
        \end{tabular}
    """
    lines = [
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        r"\toprule",
        " & ".join(escape_latex(column_labels[column]) for column in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(escape_latex(format_row_value(row[column], decimal_places_for_column(column))) for column in columns) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main(
    languages: Annotated[list[WikipediaLanguageCode] | None, typer.Option("-l", "--language", help="Language config(s) to compute statistics for. Repeatable. Defaults to every config found in --hf-dataset-repo-id.")] = None,
    hf_dataset_repo_id: Annotated[str, typer.Option("--hf-dataset-repo-id", help="HuggingFace Hub dataset repository (`namespace/name`) to read from.")] = "ucrelnlp/Multilingual-USAS-Labelled-Silver-Wikipedia",
    hf_dataset_revision: Annotated[str | None, typer.Option("--hf-dataset-revision", help="Branch (or other revision) of the Hub dataset repo to read. Defaults to the repo's default branch.")] = None,
    output_csv: Annotated[Path | None, typer.Option("--output-csv", help="Optional path to also write the statistics table to as a CSV file.")] = None,
    output_latex: Annotated[Path | None, typer.Option("--output-latex", help="Optional path to also write the statistics table to as a LaTeX tabular environment.")] = None,
    exclude_columns: Annotated[list[ColumnName] | None, typer.Option("-x", "--exclude-column", help="Column(s) to omit from the output table, CSV, and LaTeX. Repeatable.")] = None,
    split: Annotated[DatasetSplit, typer.Option("-s", "--split", help="Dataset split to report statistics for. `all` reports `train`, `validation`, and their combined total; `combined` reports only the combined total.")] = DatasetSplit.all,
) -> None:
    """Report per-language, per-split, and total statistics for the Multilingual USAS Wikipedia dataset.

    For every language config in `hf_dataset_repo_id` (or those given via
    `--language`), loads the split(s) selected by `--split` and reports, for
    each: the number of articles, number of sentences, number of tokens,
    number of labelled tokens (tokens with at least one USAS tag), and
    number of Multi-Word Expressions (MWEs) -- sentences, tokens, labelled
    tokens, and MWEs are all expressed in millions, rounded to 3 decimal
    places -- labels per token (the average number of USAS tag labels per
    token, across both the `tags` and `other_tags` columns), Multi Tag
    Membership (%) (the percentage of USAS tag labels that belong to a
    "multi tag membership" group -- a labelled token's `tags` entry, or an
    individual `other_tags` group, that itself contains more than one USAS
    tag), number of unique USAS tags (from both `tags` and `other_tags`),
    and MWE Tokens (%) (the percentage of tokens that are part of at least
    one MWE). Each language is shown by its full display name (e.g.
    "Danish", via `language_display_name`) rather than its Wikipedia code,
    and rows are sorted by that name. A final `"Total"` language aggregates
    every language together the same way. With `--split all` (the default),
    both `train` and `validation` rows are shown per language/`"Total"`,
    plus a `"total"` row combining them; `--split combined` also loads both
    splits but shows only the combined `"total"` row.

    Reads `HF_TOKEN` from the environment (e.g. via a `.env` file, loaded
    with `python-dotenv`) to authenticate with the Hub, which is required if
    `hf_dataset_repo_id` is private.

    Examples:
        Report statistics for every language in the default dataset:

        $ uv run processing_scripts/dataset_statistics.py

        Report statistics for a single language's `train` split only,
        omitting the MWE columns, and also save to CSV and LaTeX:

        $ uv run processing_scripts/dataset_statistics.py -l da --split train \\
              -x number_of_mwes -x mwe_token_percentage \\
              --output-csv ./stats.csv --output-latex ./stats.tex
    """
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")

    wikipedia_language_codes = [language.value for language in languages] if languages else get_dataset_config_names(hf_dataset_repo_id, revision=hf_dataset_revision, token=hf_token)
    wikipedia_language_codes = sorted(wikipedia_language_codes, key=language_display_name)

    excluded_columns = {column.value for column in exclude_columns} if exclude_columns else set()
    columns_to_include = tuple(column for column in COLUMNS if column not in excluded_columns)

    match split:
        case DatasetSplit.all:
            splits_to_load = DATASET_SPLITS
            emit_individual_splits = True
        case DatasetSplit.combined:
            splits_to_load = DATASET_SPLITS
            emit_individual_splits = False
        case _:
            splits_to_load = (split.value,)
            emit_individual_splits = True

    rows: list[dict[str, str | int | float]] = []
    overall_by_split: dict[str, DatasetStatistics] = {split_name: DatasetStatistics() for split_name in splits_to_load}

    for wikipedia_language_code in wikipedia_language_codes:
        per_language_by_split: dict[str, DatasetStatistics] = {}
        for split_name in splits_to_load:
            dataset = load_dataset(hf_dataset_repo_id, wikipedia_language_code, split=split_name, revision=hf_dataset_revision, token=hf_token)
            statistics = compute_split_statistics(dataset)
            per_language_by_split[split_name] = statistics
            overall_by_split[split_name] = overall_by_split[split_name].merged_with(statistics)
            if emit_individual_splits:
                rows.append(statistics_row(language_display_name(wikipedia_language_code), split_name, statistics))
        if len(splits_to_load) > 1:
            combined_statistics = DatasetStatistics()
            for split_name in splits_to_load:
                combined_statistics = combined_statistics.merged_with(per_language_by_split[split_name])
            rows.append(statistics_row(language_display_name(wikipedia_language_code), "total", combined_statistics))

    if emit_individual_splits:
        for split_name in splits_to_load:
            rows.append(statistics_row("Total", split_name, overall_by_split[split_name]))
    if len(splits_to_load) > 1:
        overall_total = DatasetStatistics()
        for split_name in splits_to_load:
            overall_total = overall_total.merged_with(overall_by_split[split_name])
        rows.append(statistics_row("Total", "total", overall_total))

    table = Table(title="Multilingual USAS Wikipedia dataset statistics")
    for column in columns_to_include:
        table.add_column(COLUMN_LABELS[column])
    for row in rows:
        table.add_row(*(format_row_value(row[column], decimal_places_for_column(column)) for column in columns_to_include))
    rprint(table)

    if output_csv is not None:
        with output_csv.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(columns_to_include), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        rprint(f"Wrote statistics to {output_csv!r}")

    if output_latex is not None:
        output_latex.write_text(rows_to_latex(rows, columns_to_include, COLUMN_LABELS) + "\n", encoding="utf-8")
        rprint(f"Wrote statistics to {output_latex!r}")


if __name__ == "__main__":
    typer.run(main)
