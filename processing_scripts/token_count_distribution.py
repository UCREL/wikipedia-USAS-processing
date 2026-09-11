"""Plot and tabulate token-count distributions for the Multilingual USAS Wikipedia dataset.

For every language config in a Hub dataset repository (default
``ucrelnlp/Multilingual-USAS-Labelled-Silver-Wikipedia``), collects the number
of tokens per sentence and the number of tokens per article, then:

* Renders one histogram per granularity (tokens-per-sentence,
  tokens-per-article), with every language overlaid as its own colored,
  density-normalized step curve on a shared log-scaled x-axis, so
  differently-sized corpora remain comparable by shape.
* Renders one quantile table per granularity (25/50/75/90/95/99%, plus the
  maximum), one row per language, plus a "Macro Avg" row -- the unweighted
  mean of each language's own values (equal weight per language, regardless
  of corpus size), as either a Markdown or LaTeX table.
"""

import dataclasses
import os
from enum import Enum
from pathlib import Path
from typing import Annotated

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import typer
from datasets import Dataset, get_dataset_config_names, load_dataset
from dotenv import load_dotenv
from rich import print as rprint

from wikipedia_processing.utils import (
    get_valid_usas_language_processing_wikipedia_codes,
    language_display_name,
)

QUANTILES: tuple[float, ...] = (0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
QUANTILE_COLUMN_LABELS: tuple[str, ...] = ("P25", "P50", "P75", "P90", "P95", "P99")
SUMMARY_COLUMN_LABELS: tuple[str, ...] = (*QUANTILE_COLUMN_LABELS, "Max")

# Fixed categorical slots from the project's validated data-viz palette,
# assigned by language identity (not by selection order) so a color never
# repaints when --language narrows the set. Order matches the 8 languages
# this dataset is restricted to `./wikipedia_processing/data/usas_wikipedia_processing.yaml`,
# i.e. every "training: true" entry in usas_wikipedia_processing.yaml.
LANGUAGE_COLORS: dict[str, str] = {
    "en": "#2a78d6",
    "nl": "#eb6834",
    "es": "#1baf7a",
    "da": "#eda100",
    "it": "#e87ba4",
    "pt": "#008300",
    "zh": "#4a3aa7",
    "fi": "#e34948",
}
FALLBACK_LANGUAGE_COLOR = "#898781"

WikipediaLanguageCode = Enum("WikipediaLanguageCode", [(value, value) for value in get_valid_usas_language_processing_wikipedia_codes()], type=str)


class DatasetSplit(str, Enum):
    """Which split(s) of the Hub dataset to compute statistics over."""

    train = "train"
    validation = "validation"
    all = "all"


class TableFormat(str, Enum):
    """Supported rendering formats for the quantile table."""

    MARKDOWN = "markdown"
    LATEX = "latex"


@dataclasses.dataclass
class TokenCounts:
    """Per-sentence and per-article token counts collected for a dataset.

    Attributes:
        sentence_token_counts: Number of tokens in each sentence, one entry
            per sentence across every article.
        article_token_counts: Total number of tokens in each article, one
            entry per article.
    """

    sentence_token_counts: list[int] = dataclasses.field(default_factory=list)
    article_token_counts: list[int] = dataclasses.field(default_factory=list)

    def extend_with(self, other: "TokenCounts") -> None:
        """Append another `TokenCounts`'s values onto this one, in place.

        Args:
            other: The counts to append.

        Examples:
            >>> counts = TokenCounts([1, 2], [3])
            >>> counts.extend_with(TokenCounts([4], [5, 6]))
            >>> counts.sentence_token_counts
            [1, 2, 4]
            >>> counts.article_token_counts
            [3, 5, 6]
        """
        self.sentence_token_counts.extend(other.sentence_token_counts)
        self.article_token_counts.extend(other.article_token_counts)


def compute_token_counts(dataset: Dataset) -> TokenCounts:
    """Compute per-sentence and per-article token counts for a dataset split.

    Args:
        dataset: A dataset split with a `tokens` column of per-sentence
            lists of token strings, as stored in the Multilingual USAS
            Wikipedia dataset.

    Returns:
        The `TokenCounts` for every article in `dataset`.

    Examples:
        >>> ds = Dataset.from_dict({"tokens": [[["a", "b"], ["c"]], [["d", "e", "f"]]]})
        >>> counts = compute_token_counts(ds)
        >>> counts.sentence_token_counts
        [2, 1, 3]
        >>> counts.article_token_counts
        [3, 3]
    """
    sentence_token_counts: list[int] = []
    article_token_counts: list[int] = []
    for example in dataset.select_columns(["tokens"]):
        sentence_lengths = [len(sentence_tokens) for sentence_tokens in example["tokens"]]
        sentence_token_counts.extend(sentence_lengths)
        article_token_counts.append(sum(sentence_lengths))
    return TokenCounts(sentence_token_counts, article_token_counts)


def summary_row(counts: list[int]) -> list[float]:
    """Compute the `QUANTILES` plus the maximum of a list of counts, rounded to one decimal place.

    Args:
        counts: The counts to summarize.

    Returns:
        One value per entry in `SUMMARY_COLUMN_LABELS` (each of `QUANTILES`,
        then the maximum), in that order. All zeros if `counts` is empty.

    Examples:
        >>> summary_row(list(range(101)))
        [25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 100.0]
        >>> summary_row([])
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    """
    if not counts:
        return [0.0] * len(SUMMARY_COLUMN_LABELS)
    quantile_values = np.quantile(np.asarray(counts), QUANTILES)
    return [round(float(value), 1) for value in quantile_values] + [round(float(max(counts)), 1)]


def build_quantile_table_rows(counts_by_language: dict[str, list[int]]) -> tuple[list[str], list[list[str]]]:
    """Build headers and string rows for a per-language quantile table.

    Languages are sorted by display name, with a trailing "Max" column
    holding each language's largest observed count. A final "Macro Avg" row
    holds the unweighted, column-wise mean of each language's own values --
    equal weight per language, regardless of how much data it contributed
    (as opposed to a "micro" average, which would pool every language's raw
    counts before computing quantiles).

    Args:
        counts_by_language: Mapping of Wikipedia language code to its list
            of counts (e.g. `sentence_token_counts` or `article_token_counts`).

    Returns:
        A `(headers, rows)` pair, each row a list of formatted string cells
        aligned with `headers`.

    Examples:
        >>> headers, rows = build_quantile_table_rows({"en": list(range(101)), "nl": list(range(101))})
        >>> headers
        ['Language', 'P25', 'P50', 'P75', 'P90', 'P95', 'P99', 'Max']
        >>> rows[0]
        ['Dutch', '25.0', '50.0', '75.0', '90.0', '95.0', '99.0', '100.0']
        >>> rows[-1]
        ['Macro Avg', '25.0', '50.0', '75.0', '90.0', '95.0', '99.0', '100.0']
    """
    headers = ["Language", *SUMMARY_COLUMN_LABELS]
    rows: list[list[str]] = []
    per_language_summaries: list[list[float]] = []

    for code in sorted(counts_by_language, key=language_display_name):
        summary = summary_row(counts_by_language[code])
        per_language_summaries.append(summary)
        rows.append([language_display_name(code), *(f"{value:,.1f}" for value in summary)])

    if per_language_summaries:
        macro_average = [sum(column) / len(column) for column in zip(*per_language_summaries)]
        rows.append(["Macro Avg", *(f"{value:,.1f}" for value in macro_average)])

    return headers, rows


def _pad(value: str, width: int, alignment: str) -> str:
    """Pad `value` to `width` using left or right alignment."""
    return value.rjust(width) if alignment == "right" else value.ljust(width)


def _column_widths(headers: list[str], rows: list[list[str]]) -> list[int]:
    """Return the display width of each column, header included."""
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    return widths


def render_markdown_table(headers: list[str], rows: list[list[str]], alignments: list[str]) -> str:
    """Render a padded GitHub-flavoured Markdown table.

    Args:
        headers: Column headers.
        rows: String cells, one list per row, aligned with `headers`.
        alignments: Per-column `"left"` or `"right"`.

    Returns:
        The table as a single string, without a trailing newline.

    Examples:
        >>> print(render_markdown_table(["Language", "P50"], [["Danish", "22.0"]], ["left", "right"]))
        | Language | P50  |
        | :------- | ---: |
        | Danish   | 22.0 |
    """
    widths = _column_widths(headers, rows)

    def render_row(cells: list[str]) -> str:
        padded = (_pad(cell, width, alignment) for cell, width, alignment in zip(cells, widths, alignments))
        return "| " + " | ".join(padded) + " |"

    separator_cells = []
    for width, alignment in zip(widths, alignments):
        dashes = "-" * (max(width, 3) - 1)
        separator_cells.append(dashes + ":" if alignment == "right" else ":" + dashes)

    lines = [render_row(headers), "| " + " | ".join(separator_cells) + " |"]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


_LATEX_SPECIAL_CHARACTERS = {"%": r"\%", "&": r"\&", "_": r"\_", "#": r"\#"}


def _latex_escape(cell: str) -> str:
    """Escape the LaTeX special characters that can occur in a table cell."""
    return "".join(_LATEX_SPECIAL_CHARACTERS.get(character, character) for character in cell)


def render_latex_table(headers: list[str], rows: list[list[str]], alignments: list[str]) -> str:
    r"""Render a LaTeX `booktabs` `tabular`.

    The output requires `\usepackage{booktabs}` in the document preamble.

    Args:
        headers: Column headers.
        rows: String cells, one list per row, aligned with `headers`.
        alignments: Per-column `"left"` (`l`) or `"right"` (`r`).

    Returns:
        The `tabular` environment as a single string, without a trailing
        newline.

    Examples:
        >>> print(render_latex_table(["Language", "P50"], [["Danish", "22.0"]], ["left", "right"]))
        \begin{tabular}{lr}
        \toprule
        Language & P50 \\
        \midrule
        Danish & 22.0 \\
        \bottomrule
        \end{tabular}
    """
    column_spec = "".join("r" if alignment == "right" else "l" for alignment in alignments)
    lines = [
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(_latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(_latex_escape(cell) for cell in row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def render_table(headers: list[str], rows: list[list[str]], alignments: list[str], table_format: TableFormat) -> str:
    """Render a table in the requested `TableFormat`.

    Args:
        headers: Column headers.
        rows: String cells, one list per row.
        alignments: Per-column `"left"` or `"right"`.
        table_format: Which renderer to use.

    Returns:
        The rendered table, without a trailing newline.

    Examples:
        >>> print(render_table(["Language"], [["Danish"]], ["left"], TableFormat.MARKDOWN))
        | Language |
        | :------- |
        | Danish   |
    """
    match table_format:
        case TableFormat.MARKDOWN:
            return render_markdown_table(headers, rows, alignments)
        case TableFormat.LATEX:
            return render_latex_table(headers, rows, alignments)


def write_or_print_table(rendered_table: str, output_path: Path | None, description: str) -> None:
    """Write a rendered table to a file, or print it to the console.

    Args:
        rendered_table: The table text, as returned by `render_table`.
        output_path: Path to write to, or `None` to print to stdout instead.
        description: Short label for the console confirmation message, e.g.
            `"tokens-per-sentence"`.
    """
    if output_path is None:
        rprint(rendered_table)
        return
    output_path.write_text(rendered_table + "\n", encoding="utf-8")
    rprint(f"Wrote {description} quantile table to {output_path!r}")


def plot_token_count_histogram(counts_by_language: dict[str, list[int]], title: str, xlabel: str, output: Path) -> None:
    """Render an overlaid, density-normalized histogram of token counts by language.

    Every language is drawn as its own colored step curve (rather than
    filled bars) on a shared set of log-spaced bins, so up to eight
    overlapping distributions stay legible. Density normalization keeps
    differently-sized corpora comparable by shape rather than raw count.

    Args:
        counts_by_language: Mapping of Wikipedia language code to its list
            of counts (e.g. `sentence_token_counts` or `article_token_counts`).
        title: Figure title.
        xlabel: X-axis label (the y-axis is always labelled "Density").
        output: File path the figure is saved to.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    max_count = max(count for counts in counts_by_language.values() for count in counts)
    bins = np.geomspace(1, max_count, num=41).tolist()

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    for code in sorted(counts_by_language, key=language_display_name):
        ax.hist(
            counts_by_language[code],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=LANGUAGE_COLORS.get(code, FALLBACK_LANGUAGE_COLOR),
            label=language_display_name(code),
        )

    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(True, which="both", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main(
    languages: Annotated[list[WikipediaLanguageCode] | None, typer.Option("-l", "--language", help="Language config(s) to compute statistics for. Repeatable. Defaults to every config found in --hf-dataset-repo-id.")] = None,
    hf_dataset_repo_id: Annotated[str, typer.Option("--hf-dataset-repo-id", help="HuggingFace Hub dataset repository (`namespace/name`) to read from.")] = "ucrelnlp/Multilingual-USAS-Labelled-Silver-Wikipedia",
    hf_dataset_revision: Annotated[str, typer.Option("--hf-dataset-revision", help="Branch (or other revision) of the Hub dataset repo to read.")] = "main",
    split: Annotated[DatasetSplit, typer.Option("-s", "--split", help="Dataset split to compute statistics over. `all` combines `train` and `validation`.")] = DatasetSplit.train,
    table_format: Annotated[TableFormat, typer.Option("-f", "--format", help="Quantile table output format.")] = TableFormat.MARKDOWN,
    output_histogram_sentences: Annotated[Path, typer.Option(help="Path to write the tokens-per-sentence histogram to.")] = Path("data/plots/token_count_per_sentence_histogram.png"),
    output_histogram_articles: Annotated[Path, typer.Option(help="Path to write the tokens-per-article histogram to.")] = Path("data/plots/token_count_per_article_histogram.png"),
    output_table_sentences: Annotated[Path | None, typer.Option(help="Optional path to write the tokens-per-sentence quantile table to. Defaults to printing to the console.")] = None,
    output_table_articles: Annotated[Path | None, typer.Option(help="Optional path to write the tokens-per-article quantile table to. Defaults to printing to the console.")] = None,
) -> None:
    """Plot and tabulate tokens-per-sentence and tokens-per-article distributions.

    For every language config in `hf_dataset_repo_id` (or those given via
    `--language`), collects token counts from `--split` and renders two
    histograms (tokens-per-sentence, tokens-per-article -- every language
    overlaid as its own colored curve) plus two quantile tables (25/50/75/
    90/95/99% and the maximum, one row per language, plus a "Macro Avg" row).

    Reads `HF_TOKEN` from the environment (e.g. via a `.env` file, loaded
    with `python-dotenv`) to authenticate with the Hub, which is required if
    `hf_dataset_repo_id` is private.

    Examples:
        Report on every language in the default dataset's `train` split:

        $ uv run processing_scripts/token_count_distribution.py

        Report on two languages across both splits, and save both tables as LaTeX:

        $ uv run processing_scripts/token_count_distribution.py -l da -l en \\
              --split all --format latex \\
              --output-table-sentences data/tables/sentence_quantiles.tex \\
              --output-table-articles data/tables/article_quantiles.tex
    """
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")

    wikipedia_language_codes = [language.value for language in languages] if languages else get_dataset_config_names(hf_dataset_repo_id, revision=hf_dataset_revision, token=hf_token)

    match split:
        case DatasetSplit.all:
            splits_to_load = ("train", "validation")
        case _:
            splits_to_load = (split.value,)

    sentence_counts_by_language: dict[str, list[int]] = {}
    article_counts_by_language: dict[str, list[int]] = {}

    for wikipedia_language_code in wikipedia_language_codes:
        combined_counts = TokenCounts()
        for split_name in splits_to_load:
            dataset = load_dataset(hf_dataset_repo_id, wikipedia_language_code, split=split_name, revision=hf_dataset_revision, token=hf_token)
            combined_counts.extend_with(compute_token_counts(dataset))
        sentence_counts_by_language[wikipedia_language_code] = combined_counts.sentence_token_counts
        article_counts_by_language[wikipedia_language_code] = combined_counts.article_token_counts
        rprint(f"{language_display_name(wikipedia_language_code)}: {len(combined_counts.article_token_counts):,} articles, {len(combined_counts.sentence_token_counts):,} sentences")

    plot_token_count_histogram(sentence_counts_by_language, "Tokens per sentence by language", "Tokens per sentence (log scale)", output_histogram_sentences)
    plot_token_count_histogram(article_counts_by_language, "Tokens per article by language", "Tokens per article (log scale)", output_histogram_articles)
    rprint(f"Wrote histograms to {output_histogram_sentences!r} and {output_histogram_articles!r}")

    alignments = ["left"] + ["right"] * len(SUMMARY_COLUMN_LABELS)

    sentence_headers, sentence_rows = build_quantile_table_rows(sentence_counts_by_language)
    write_or_print_table(render_table(sentence_headers, sentence_rows, alignments, table_format), output_table_sentences, "tokens-per-sentence")

    article_headers, article_rows = build_quantile_table_rows(article_counts_by_language)
    write_or_print_table(render_table(article_headers, article_rows, alignments, table_format), output_table_articles, "tokens-per-article")


if __name__ == "__main__":
    typer.run(main)
