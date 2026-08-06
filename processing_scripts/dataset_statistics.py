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
)

DATASET_SPLITS = ("train", "validation")
COLUMN_LABELS = {
    "language": "Language",
    "split": "Split",
    "number_of_articles": "Articles",
    "average_article_size_tokens": "Avg. Tokens / Article",
    "average_number_of_sentences": "Avg. Sentences / Article",
    "number_of_tokens": "Tokens",
    "number_of_labelled_tokens": "Labelled Tokens",
    "average_tags_per_labelled_token": "Avg. Tags / Labelled Token",
    "number_of_unique_tags": "Unique Tags",
    "number_of_mwes": "MWEs",
}
COLUMNS = tuple(COLUMN_LABELS)

WikipediaLanguageCode = Enum("WikipediaLanguageCode", [(value, value) for value in get_valid_usas_language_processing_wikipedia_codes()], type=str)


@dataclasses.dataclass
class DatasetStatistics:
    """Aggregate statistics for a set of articles.

    Attributes:
        number_of_articles: Number of articles the statistics were computed over.
        number_of_sentences: Total number of sentences across all articles.
        number_of_tokens: Total number of tokens across all articles.
        number_of_labelled_tokens: Number of tokens with at least one USAS tag
            (a token's `tags` entry, e.g. `tags[0][0]`, is non-empty).
        number_of_tag_assignments: Total number of USAS tags assigned across
            all tokens (a labelled token can have more than one tag, e.g.
            `tags[0][0]` can be `["A3", "M6"]`).
        unique_tags: The set of distinct USAS tag strings seen across all articles.
        number_of_mwes: Total number of Multi-Word Expressions (MWEs) across all articles.
    """

    number_of_articles: int = 0
    number_of_sentences: int = 0
    number_of_tokens: int = 0
    number_of_labelled_tokens: int = 0
    number_of_tag_assignments: int = 0
    unique_tags: set[str] = dataclasses.field(default_factory=set)
    number_of_mwes: int = 0

    @property
    def average_article_size_tokens(self) -> float:
        """Average number of tokens per article.

        Examples:
            >>> DatasetStatistics(number_of_articles=2, number_of_tokens=10).average_article_size_tokens
            5.0
            >>> DatasetStatistics().average_article_size_tokens
            0.0
        """
        if self.number_of_articles == 0:
            return 0.0
        return self.number_of_tokens / self.number_of_articles

    @property
    def average_number_of_sentences(self) -> float:
        """Average number of sentences per article.

        Examples:
            >>> DatasetStatistics(number_of_articles=2, number_of_sentences=5).average_number_of_sentences
            2.5
            >>> DatasetStatistics().average_number_of_sentences
            0.0
        """
        if self.number_of_articles == 0:
            return 0.0
        return self.number_of_sentences / self.number_of_articles

    @property
    def average_tags_per_labelled_token(self) -> float:
        """Average number of USAS tags per labelled token, ignoring unlabelled tokens.

        Examples:
            >>> DatasetStatistics(number_of_labelled_tokens=2, number_of_tag_assignments=3).average_tags_per_labelled_token
            1.5
            >>> DatasetStatistics().average_tags_per_labelled_token
            0.0
        """
        if self.number_of_labelled_tokens == 0:
            return 0.0
        return self.number_of_tag_assignments / self.number_of_labelled_tokens

    @property
    def number_of_unique_tags(self) -> int:
        """Number of distinct USAS tag strings seen across all articles."""
        return len(self.unique_tags)

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
            number_of_tag_assignments=self.number_of_tag_assignments + other.number_of_tag_assignments,
            unique_tags=self.unique_tags | other.unique_tags,
            number_of_mwes=self.number_of_mwes + other.number_of_mwes,
        )


def compute_article_statistics(
    tokens: list[list[str]],
    tags: list[list[list[str]]],
    mwes: list[list[list[int]]],
) -> DatasetStatistics:
    """Compute statistics for a single article.

    Args:
        tokens: Per-sentence lists of token strings, as stored in the
            dataset's `tokens` column.
        tags: Per-sentence, per-token lists of USAS tag strings, as stored
            in the dataset's `tags` column. A token is "labelled" if its
            list of tags is non-empty.
        mwes: Per-sentence, per-token lists of Multi-Word Expression (MWE)
            labels, as stored in the dataset's `mwes` column. Labels are
            unique per sentence and reset at each sentence boundary, so MWEs
            are counted per sentence before being summed.

    Returns:
        A `DatasetStatistics` for this single article (`number_of_articles`
        is always 1).

    Examples:
        >>> tokens = [["A", "cat", "sat"]]
        >>> tags = [[["Z2"], [], ["A3", "M6"]]]
        >>> mwes = [[[], [1], [1]]]
        >>> stats = compute_article_statistics(tokens, tags, mwes)
        >>> stats.number_of_articles, stats.number_of_sentences, stats.number_of_tokens
        (1, 1, 3)
        >>> stats.number_of_labelled_tokens, stats.number_of_tag_assignments
        (2, 3)
        >>> sorted(stats.unique_tags)
        ['A3', 'M6', 'Z2']
        >>> stats.number_of_mwes
        1
    """
    number_of_sentences = len(tokens)
    number_of_tokens = sum(len(sentence_tokens) for sentence_tokens in tokens)

    number_of_labelled_tokens = 0
    number_of_tag_assignments = 0
    unique_tags: set[str] = set()
    for sentence_tags in tags:
        for token_tags in sentence_tags:
            if token_tags:
                number_of_labelled_tokens += 1
                number_of_tag_assignments += len(token_tags)
            unique_tags.update(token_tags)

    number_of_mwes = 0
    for sentence_mwes in mwes:
        sentence_mwe_labels: set[int] = set()
        for token_mwe_labels in sentence_mwes:
            sentence_mwe_labels.update(token_mwe_labels)
        number_of_mwes += len(sentence_mwe_labels)

    return DatasetStatistics(
        number_of_articles=1,
        number_of_sentences=number_of_sentences,
        number_of_tokens=number_of_tokens,
        number_of_labelled_tokens=number_of_labelled_tokens,
        number_of_tag_assignments=number_of_tag_assignments,
        unique_tags=unique_tags,
        number_of_mwes=number_of_mwes,
    )


def compute_split_statistics(dataset: Dataset) -> DatasetStatistics:
    """Compute aggregate statistics across every article in a dataset split.

    Args:
        dataset: A dataset split (e.g. one language's `train` or
            `validation` split) with `tokens`, `tags`, and `mwes` columns.

    Returns:
        A `DatasetStatistics` aggregated over every article in `dataset`.
    """
    statistics = DatasetStatistics()
    for example in dataset.select_columns(["tokens", "tags", "mwes"]):
        statistics = statistics.merged_with(compute_article_statistics(example["tokens"], example["tags"], example["mwes"]))
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
        suitable for a `rich.table.Table` row or a CSV row.
    """
    return {
        "language": language,
        "split": split,
        "number_of_articles": statistics.number_of_articles,
        "average_article_size_tokens": round(statistics.average_article_size_tokens, 2),
        "average_number_of_sentences": round(statistics.average_number_of_sentences, 2),
        "number_of_tokens": statistics.number_of_tokens,
        "number_of_labelled_tokens": statistics.number_of_labelled_tokens,
        "average_tags_per_labelled_token": round(statistics.average_tags_per_labelled_token, 2),
        "number_of_unique_tags": statistics.number_of_unique_tags,
        "number_of_mwes": statistics.number_of_mwes,
    }


def format_row_value(value: str | int | float) -> str:
    """Format a single row value for table display, adding `,` thousands separators to numbers.

    Args:
        value: The value to format, as produced by `statistics_row`.

    Returns:
        `value` unchanged if it is a string, otherwise formatted with `,`
        thousands separators (and, for floats, two decimal places).

    Examples:
        >>> format_row_value("da")
        'da'
        >>> format_row_value(1369932)
        '1,369,932'
        >>> format_row_value(7739.7288)
        '7,739.73'
    """
    match value:
        case int():
            return f"{value:,}"
        case float():
            return f"{value:,.2f}"
        case _:
            return str(value)


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
        lines.append(" & ".join(escape_latex(format_row_value(row[column])) for column in columns) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main(
    languages: Annotated[list[WikipediaLanguageCode] | None, typer.Option("-l", "--language", help="Language config(s) to compute statistics for. Repeatable. Defaults to every config found in --hf-dataset-repo-id.")] = None,
    hf_dataset_repo_id: Annotated[str, typer.Option("--hf-dataset-repo-id", help="HuggingFace Hub dataset repository (`namespace/name`) to read from.")] = "ucrelnlp/Multilingual-USAS-Labelled-Silver-Wikipedia",
    hf_dataset_revision: Annotated[str | None, typer.Option("--hf-dataset-revision", help="Branch (or other revision) of the Hub dataset repo to read. Defaults to the repo's default branch.")] = None,
    output_csv: Annotated[Path | None, typer.Option("--output-csv", help="Optional path to also write the statistics table to as a CSV file.")] = None,
    output_latex: Annotated[Path | None, typer.Option("--output-latex", help="Optional path to also write the statistics table to as a LaTeX tabular environment.")] = None,
) -> None:
    """Report per-language, per-split, and total statistics for the Multilingual USAS Wikipedia dataset.

    For every language config in `hf_dataset_repo_id` (or those given via
    `--language`), loads the `train` and `validation` splits and reports,
    for each split: the number of articles, average article size in tokens,
    average number of sentences per article, number of tokens, number of
    labelled tokens (tokens with at least one USAS tag), average number of
    USAS tags per labelled token, number of unique USAS tags, and number of
    Multi-Word Expressions (MWEs). A final `"Total"` language aggregates
    every language together, broken down into `train`, `validation`, and the
    overall total.

    Reads `HF_TOKEN` from the environment (e.g. via a `.env` file, loaded
    with `python-dotenv`) to authenticate with the Hub, which is required if
    `hf_dataset_repo_id` is private.

    Examples:
        Report statistics for every language in the default dataset:

        $ uv run processing_scripts/dataset_statistics.py

        Report statistics for a single language and also save to CSV and LaTeX:

        $ uv run processing_scripts/dataset_statistics.py -l da \\
              --output-csv ./stats.csv --output-latex ./stats.tex
    """
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")

    wikipedia_language_codes = [language.value for language in languages] if languages else get_dataset_config_names(hf_dataset_repo_id, revision=hf_dataset_revision, token=hf_token)

    rows: list[dict[str, str | int | float]] = []
    overall_by_split: dict[str, DatasetStatistics] = {split: DatasetStatistics() for split in DATASET_SPLITS}

    for wikipedia_language_code in wikipedia_language_codes:
        for split in DATASET_SPLITS:
            dataset = load_dataset(hf_dataset_repo_id, wikipedia_language_code, split=split, revision=hf_dataset_revision, token=hf_token)
            statistics = compute_split_statistics(dataset)
            rows.append(statistics_row(wikipedia_language_code, split, statistics))
            overall_by_split[split] = overall_by_split[split].merged_with(statistics)

    for split in DATASET_SPLITS:
        rows.append(statistics_row("Total", split, overall_by_split[split]))
    overall_total = overall_by_split["train"].merged_with(overall_by_split["validation"])
    rows.append(statistics_row("Total", "total", overall_total))

    table = Table(title="Multilingual USAS Wikipedia dataset statistics")
    for column in COLUMNS:
        table.add_column(COLUMN_LABELS[column])
    for row in rows:
        table.add_row(*(format_row_value(row[column]) for column in COLUMNS))
    rprint(table)

    if output_csv is not None:
        with output_csv.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
        rprint(f"Wrote statistics to {output_csv!r}")

    if output_latex is not None:
        output_latex.write_text(rows_to_latex(rows, COLUMNS, COLUMN_LABELS) + "\n", encoding="utf-8")
        rprint(f"Wrote statistics to {output_latex!r}")


if __name__ == "__main__":
    typer.run(main)
