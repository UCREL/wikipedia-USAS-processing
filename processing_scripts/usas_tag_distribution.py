"""Report USAS tag distributions for the Multilingual USAS Wikipedia dataset.

For every language config in a Hub dataset repository (default
``ucrelnlp/Multilingual-USAS-Labelled-Silver-Wikipedia``), counts individual
USAS tag occurrences from both the `tags` and `other_tags` columns (both are
positive labels when training -- `other_tags` holds every other valid tag
group PyMUSAS considered besides the most likely one in `tags`), then
reports, per language plus a "Macro Avg" column (the unweighted mean of each
language's own percentages, equal weight per language regardless of corpus
size):

* The full major tag (first character of a USAS tag) distribution.
* The top N and bottom N individual tags in the distribution, N configurable
  via the CLI.
* A five-number summary (min, 25th/50th/75th percentile, max) of how
  individual tags' percentages and raw counts are spread out.
"""

import dataclasses
import os
from collections import Counter
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
from matplotlib.colors import LinearSegmentedColormap
from rich import print as rprint

from wikipedia_processing.utils import (
    ImageFormat,
    get_valid_usas_language_processing_wikipedia_codes,
    language_display_name,
    resolve_image_path,
)

# Light-to-dark sequential ramp from the project's validated data-viz
# palette (single hue: blue), used to color the major tag heatmap by
# magnitude. Matches the hues `LANGUAGE_COLORS` draws from in
# `token_count_distribution.py`.
SEQUENTIAL_BLUE_RAMP: tuple[str, ...] = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)

WikipediaLanguageCode = Enum("WikipediaLanguageCode", [(value, value) for value in get_valid_usas_language_processing_wikipedia_codes()], type=str)


class DatasetSplit(str, Enum):
    """Which split(s) of the Hub dataset to compute statistics over."""

    train = "train"
    validation = "validation"
    all = "all"


class TableFormat(str, Enum):
    """Supported rendering formats for the distribution tables."""

    MARKDOWN = "markdown"
    LATEX = "latex"


@dataclasses.dataclass
class TagCounts:
    """Individual USAS tag occurrence counts collected for a dataset.

    Attributes:
        tag_counts: Number of times each individual USAS tag occurs, counting
            tags from both the `tags` and `other_tags` columns, across every
            article.
    """

    tag_counts: Counter[str] = dataclasses.field(default_factory=Counter)

    @property
    def total(self) -> int:
        """Total number of tag occurrences across every tag.

        Examples:
            >>> TagCounts(Counter({"A3": 2, "Z2": 1})).total
            3
        """
        return sum(self.tag_counts.values())

    def extend_with(self, other: "TagCounts") -> None:
        """Add another `TagCounts`'s occurrences onto this one, in place.

        Args:
            other: The counts to add.

        Examples:
            >>> counts = TagCounts(Counter({"A3": 2}))
            >>> counts.extend_with(TagCounts(Counter({"A3": 1, "Z2": 1})))
            >>> sorted(counts.tag_counts.items())
            [('A3', 3), ('Z2', 1)]
        """
        self.tag_counts.update(other.tag_counts)


def compute_tag_counts(dataset: Dataset) -> TagCounts:
    """Compute individual USAS tag occurrence counts for a dataset split.

    Counts tags from both the `tags` and `other_tags` columns, since both
    are positive labels when training -- `other_tags` holds every other
    valid tag group PyMUSAS considered besides the most likely one in `tags`.

    Args:
        dataset: A dataset split with `tags` and `other_tags` columns, as
            stored in the Multilingual USAS Wikipedia dataset.

    Returns:
        The `TagCounts` aggregated over every article in `dataset`.

    Examples:
        >>> ds = Dataset.from_dict({
        ...     "tags": [[[["A3"], []]]],
        ...     "other_tags": [[[[], [["Z2", "Z9"]]]]],
        ... })
        >>> sorted(compute_tag_counts(ds).tag_counts.items())
        [('A3', 1), ('Z2', 1), ('Z9', 1)]
    """
    tag_counts: Counter[str] = Counter()
    for example in dataset.select_columns(["tags", "other_tags"]):
        for sentence_tags in example["tags"]:
            for token_tags in sentence_tags:
                tag_counts.update(token_tags)
        for sentence_other_tags in example["other_tags"]:
            for token_other_tag_groups in sentence_other_tags:
                for tag_group in token_other_tag_groups:
                    tag_counts.update(tag_group)
    return TagCounts(tag_counts)


def tag_percentages(counts: TagCounts) -> dict[str, float]:
    """Convert a `TagCounts`'s raw counts into percentages of its total occurrences.

    Args:
        counts: The counts to convert.

    Returns:
        Mapping of tag to its percentage of `counts.total`. Empty if
        `counts.total` is 0.

    Examples:
        >>> tag_percentages(TagCounts(Counter({"A3": 3, "Z2": 1})))
        {'A3': 75.0, 'Z2': 25.0}
        >>> tag_percentages(TagCounts())
        {}
    """
    total = counts.total
    if total == 0:
        return {}
    return {tag: count / total * 100 for tag, count in counts.tag_counts.items()}


def major_tag_percentages(counts: TagCounts) -> dict[str, float]:
    """Convert a `TagCounts`'s raw counts into major-tag percentages of its total occurrences.

    A major tag is the first character of a USAS tag, e.g. `"A3"` and
    `"A1"` both belong to major tag `"A"`.

    Args:
        counts: The counts to convert.

    Returns:
        Mapping of major tag to its percentage of `counts.total`. Empty if
        `counts.total` is 0.

    Examples:
        >>> major_tag_percentages(TagCounts(Counter({"A3": 2, "A1": 1, "Z2": 1})))
        {'A': 75.0, 'Z': 25.0}
    """
    total = counts.total
    if total == 0:
        return {}
    major_counts: Counter[str] = Counter()
    for tag, count in counts.tag_counts.items():
        major_counts[tag[0]] += count
    return {major_tag: count / total * 100 for major_tag, count in major_counts.items()}


def compute_macro_average_percentages(percentages_by_language: dict[str, dict[str, float]]) -> dict[str, float]:
    """Compute each tag's macro-average percentage of occurrences across languages.

    Every language contributes 0.0 for a tag it never used, so this is an
    unweighted mean across all languages (equal weight per language,
    regardless of corpus size) -- not a pooled/micro-average computed on the
    languages that happen to use a given tag.

    Args:
        percentages_by_language: Mapping of Wikipedia language code to that
            language's tag -> percentage-of-occurrences mapping (as returned
            by `tag_percentages` or `major_tag_percentages`).

    Returns:
        Mapping of tag to its macro-average percentage.

    Examples:
        >>> percentages = {"en": {"A3": 40.0}, "nl": {"A3": 20.0, "Z2": 10.0}}
        >>> sorted(compute_macro_average_percentages(percentages).items())
        [('A3', 30.0), ('Z2', 5.0)]
    """
    all_tags = sorted({tag for language_percentages in percentages_by_language.values() for tag in language_percentages})
    number_of_languages = len(percentages_by_language)
    return {
        tag: sum(language_percentages.get(tag, 0.0) for language_percentages in percentages_by_language.values()) / number_of_languages
        for tag in all_tags
    }


def rank_tags(macro_averages: dict[str, float], descending: bool = True) -> list[str]:
    """Order tags by macro-average percentage, breaking ties alphabetically.

    Args:
        macro_averages: Mapping of tag to its macro-average percentage, as
            returned by `compute_macro_average_percentages`.
        descending: Most common tag first if True, least common first if False.

    Returns:
        Every tag in `macro_averages`, ordered as described above.

    Examples:
        >>> rank_tags({"A3": 10.0, "Z2": 30.0, "B1": 30.0})
        ['B1', 'Z2', 'A3']
        >>> rank_tags({"A3": 10.0, "Z2": 30.0}, descending=False)
        ['A3', 'Z2']
    """
    return sorted(macro_averages, key=lambda tag: (-macro_averages[tag] if descending else macro_averages[tag], tag))


def percentile(values: list[float], percentile_rank: float) -> float:
    """Compute a percentile of a list of values by linear interpolation.

    Args:
        values: The values to summarize. Order does not matter.
        percentile_rank: The percentile to compute, in the closed range
            `[0, 100]`.

    Returns:
        The interpolated value at `percentile_rank`, or 0.0 if `values` is
        empty.

    Examples:
        >>> percentile([10.0, 20.0, 30.0, 40.0], 50)
        25.0
        >>> percentile([10.0, 20.0, 30.0, 40.0], 25)
        17.5
        >>> percentile([], 50)
        0.0
    """
    if not values:
        return 0.0
    sorted_values = sorted(values)
    number_of_values = len(sorted_values)
    if number_of_values == 1:
        return sorted_values[0]
    rank = percentile_rank / 100 * (number_of_values - 1)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, number_of_values - 1)
    fraction = rank - lower_index
    return sorted_values[lower_index] + (sorted_values[upper_index] - sorted_values[lower_index]) * fraction


TAG_SUMMARY_STATISTICS = ("Min", "P25", "P50", "P75", "Max")


def five_number_summary(values: list[float]) -> dict[str, float]:
    """Compute the min, 25th/50th/75th percentile, and max of a list of values.

    Args:
        values: The values to summarize. Order does not matter.

    Returns:
        Mapping with keys `"Min"`, `"P25"`, `"P50"`, `"P75"`, `"Max"`,
        summarizing how `values` is spread out. All 0.0 if `values` is empty.

    Examples:
        >>> five_number_summary([10.0, 30.0, 20.0, 40.0])
        {'Min': 10.0, 'P25': 17.5, 'P50': 25.0, 'P75': 32.5, 'Max': 40.0}
        >>> five_number_summary([])
        {'Min': 0.0, 'P25': 0.0, 'P50': 0.0, 'P75': 0.0, 'Max': 0.0}
    """
    if not values:
        return dict.fromkeys(TAG_SUMMARY_STATISTICS, 0.0)
    return {
        "Min": min(values),
        "P25": percentile(values, 25),
        "P50": percentile(values, 50),
        "P75": percentile(values, 75),
        "Max": max(values),
    }


def build_tag_summary_rows(
    percentage_summary_by_language: dict[str, dict[str, float]],
    count_summary_by_language: dict[str, dict[str, float]],
) -> tuple[list[str], list[list[str]]]:
    """Build headers and string rows for the tag-frequency summary table.

    Each cell combines a language's raw count value and its percentage
    value for that statistic, e.g. `"120 (12.0%)"`.

    Args:
        percentage_summary_by_language: Mapping of Wikipedia language code
            to that language's five-number tag-percentage summary, as
            returned by `five_number_summary` over a language's
            `tag_percentages` values.
        count_summary_by_language: Mapping of Wikipedia language code to
            that language's five-number tag-count summary, as returned by
            `five_number_summary` over a language's raw `TagCounts.tag_counts`
            values. Must have the same keys as `percentage_summary_by_language`.

    Returns:
        A `(headers, rows)` pair: one column per language (sorted by display
        name), plus a trailing "Macro Avg" column (the unweighted mean of
        each language's own count and percentage for that statistic); one
        row per entry in `TAG_SUMMARY_STATISTICS`.

    Examples:
        >>> percentages = {"en": {"Min": 10.0, "P25": 20.0, "P50": 30.0, "P75": 40.0, "Max": 50.0}}
        >>> counts = {"en": {"Min": 100.0, "P25": 200.0, "P50": 300.0, "P75": 400.0, "Max": 500.0}}
        >>> headers, rows = build_tag_summary_rows(percentages, counts)
        >>> headers
        ['Statistic', 'English', 'Macro Avg']
        >>> rows[0]
        ['Min', '100 (10.0%)', '100 (10.0%)']
    """
    language_codes = sorted(percentage_summary_by_language, key=language_display_name)
    headers = ["Statistic", *(language_display_name(code) for code in language_codes), "Macro Avg"]
    rows: list[list[str]] = []
    for statistic in TAG_SUMMARY_STATISTICS:
        per_language_percentages = [percentage_summary_by_language[code][statistic] for code in language_codes]
        per_language_counts = [count_summary_by_language[code][statistic] for code in language_codes]
        macro_average_percentage = sum(per_language_percentages) / len(per_language_percentages) if per_language_percentages else 0.0
        macro_average_count = sum(per_language_counts) / len(per_language_counts) if per_language_counts else 0.0
        cells = [f"{count:,.0f} ({percentage:,.1f}%)" for count, percentage in zip(per_language_counts, per_language_percentages)]
        cells.append(f"{macro_average_count:,.0f} ({macro_average_percentage:,.1f}%)")
        rows.append([statistic, *cells])
    return headers, rows


def build_tag_distribution_rows(percentages_by_language: dict[str, dict[str, float]], tags: list[str]) -> tuple[list[str], list[list[str]]]:
    """Build headers and string rows for a tag distribution table.

    Args:
        percentages_by_language: Mapping of Wikipedia language code to that
            language's tag -> percentage-of-occurrences mapping.
        tags: Which tags to include as rows, and their order.

    Returns:
        A `(headers, rows)` pair: one column per language (sorted by display
        name), plus a trailing "Macro Avg" column (the unweighted mean of
        each language's own percentage for that tag); one row per entry in
        `tags`.

    Examples:
        >>> percentages = {"en": {"A3": 40.0, "Z2": 60.0}, "nl": {"A3": 20.0}}
        >>> headers, rows = build_tag_distribution_rows(percentages, ["A3", "Z2"])
        >>> headers
        ['Tag', 'Dutch', 'English', 'Macro Avg']
        >>> rows
        [['A3', '20.0', '40.0', '30.0'], ['Z2', '0.0', '60.0', '30.0']]
    """
    language_codes = sorted(percentages_by_language, key=language_display_name)
    headers = ["Tag", *(language_display_name(code) for code in language_codes), "Macro Avg"]
    rows: list[list[str]] = []
    for tag in tags:
        per_language_values = [percentages_by_language[code].get(tag, 0.0) for code in language_codes]
        macro_average = sum(per_language_values) / len(per_language_values) if per_language_values else 0.0
        rows.append([tag, *(f"{value:,.1f}" for value in per_language_values), f"{macro_average:,.1f}"])
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
        >>> print(render_markdown_table(["Tag", "P50"], [["A3", "22.0"]], ["left", "right"]))
        | Tag | P50  |
        | :-- | ---: |
        | A3  | 22.0 |
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
        >>> print(render_latex_table(["Tag", "P50"], [["A3", "22.0"]], ["left", "right"]))
        \begin{tabular}{lr}
        \toprule
        Tag & P50 \\
        \midrule
        A3 & 22.0 \\
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
        >>> print(render_table(["Tag"], [["A3"]], ["left"], TableFormat.MARKDOWN))
        | Tag |
        | :-- |
        | A3  |
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
            `"major tag"`.
    """
    if output_path is None:
        rprint(rendered_table)
        return
    output_path.write_text(rendered_table + "\n", encoding="utf-8")
    rprint(f"Wrote {description} distribution table to {output_path!r}")


def plot_major_tag_heatmap(percentages_by_language: dict[str, dict[str, float]], output: Path) -> None:
    """Render a heatmap of major tag percentages, one row per tag, one column per language.

    Tags are ordered alphabetically (rather than by frequency, as in the
    Markdown/LaTeX major tag distribution table) so the heatmap's row axis
    stays fixed and scannable. Languages are columns, sorted by display
    name, plus a trailing "Macro Avg" column -- the unweighted mean of each
    language's own percentage for that tag. Cells are colored on a single
    light-to-dark blue scale, from 0 to the matrix-wide maximum percentage.

    Args:
        percentages_by_language: Mapping of Wikipedia language code to that
            language's major tag -> percentage-of-occurrences mapping.
        output: File path the figure is saved to.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    language_codes = sorted(percentages_by_language, key=language_display_name)
    macro_averages = compute_macro_average_percentages(percentages_by_language)
    tags = sorted(macro_averages)
    column_labels = [*(language_display_name(code) for code in language_codes), "Macro Avg"]
    matrix = np.array([[*(percentages_by_language[code].get(tag, 0.0) for code in language_codes), macro_averages[tag]] for tag in tags])

    color_map = LinearSegmentedColormap.from_list("usas_sequential_blue", SEQUENTIAL_BLUE_RAMP)

    fig, ax = plt.subplots(figsize=(0.9 * len(column_labels) + 2, 0.35 * len(tags) + 2), dpi=150)
    image = ax.imshow(matrix, cmap=color_map, aspect="auto", vmin=0)

    ax.set_xticks(range(len(column_labels)))
    ax.set_xticklabels(column_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels(tags)
    ax.set_title("Major Tag Distribution by Language (%)")

    text_color_threshold = matrix.max() / 2 if matrix.size else 0.0
    for row_index, row_values in enumerate(matrix):
        for column_index, value in enumerate(row_values):
            text_color = "white" if value > text_color_threshold else "#0b0b0b"
            ax.text(column_index, row_index, f"{value:.1f}", ha="center", va="center", color=text_color, fontsize=8)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(column_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(tags), 1), minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.colorbar(image, ax=ax, label="Percentage of tag occurrences", fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main(
    languages: Annotated[list[WikipediaLanguageCode] | None, typer.Option("-l", "--language", help="Language config(s) to compute statistics for. Repeatable. Defaults to every config found in --hf-dataset-repo-id.")] = None,
    hf_dataset_repo_id: Annotated[str, typer.Option("--hf-dataset-repo-id", help="HuggingFace Hub dataset repository (`namespace/name`) to read from.")] = "ucrelnlp/Multilingual-USAS-Labelled-Silver-Wikipedia",
    hf_dataset_revision: Annotated[str, typer.Option("--hf-dataset-revision", help="Branch (or other revision) of the Hub dataset repo to read.")] = "main",
    split: Annotated[DatasetSplit, typer.Option("-s", "--split", help="Dataset split to compute statistics over. `all` combines `train` and `validation`.")] = DatasetSplit.train,
    top_bottom_count: Annotated[int, typer.Option("-n", "--top-bottom-count", min=1, help="Number of most-common (top) and least-common (bottom) individual tags to report.")] = 10,
    table_format: Annotated[TableFormat, typer.Option("-f", "--format", help="Distribution table output format.")] = TableFormat.MARKDOWN,
    output_table_major: Annotated[Path | None, typer.Option(help="Optional path to write the major tag distribution table to. Defaults to printing to the console.")] = None,
    output_table_top: Annotated[Path | None, typer.Option(help="Optional path to write the top-tags distribution table to. Defaults to printing to the console.")] = None,
    output_table_bottom: Annotated[Path | None, typer.Option(help="Optional path to write the bottom-tags distribution table to. Defaults to printing to the console.")] = None,
    output_table_summary: Annotated[Path | None, typer.Option(help="Optional path to write the tag-frequency summary table to. Defaults to printing to the console.")] = None,
    output_heatmap_major: Annotated[Path, typer.Option(help="Path to write the major tag distribution heatmap to.")] = Path("data/plots/major_tag_distribution_heatmap.png"),
    image_format: Annotated[ImageFormat, typer.Option("-i", "--image-format", help="File format for the saved heatmap figure. Overrides the file extension of --output-heatmap-major.")] = ImageFormat.PNG,
) -> None:
    """Report per-language and macro-average USAS tag distributions.

    For every language config in `hf_dataset_repo_id` (or those given via
    `--language`), collects individual USAS tag occurrences from `--split`,
    counting both the `tags` and `other_tags` columns (both are positive
    labels when training), then renders four distribution tables -- one row
    per tag (or per summary statistic), one column per language plus a
    "Macro Avg" column (the unweighted mean of each language's own value,
    equal weight per language regardless of corpus size):

    * The full major tag (first character of a USAS tag) distribution.
    * The top `--top-bottom-count` most common individual tags.
    * The bottom `--top-bottom-count` least common individual tags.
    * A five-number summary (min, 25th/50th/75th percentile, max) of how
      individual tags' percentages and raw counts are spread out.

    Also renders the major tag distribution as a PNG heatmap (tags sorted
    alphabetically, rather than ranked by macro-average percentage as in
    the table).

    Reads `HF_TOKEN` from the environment (e.g. via a `.env` file, loaded
    with `python-dotenv`) to authenticate with the Hub, which is required if
    `hf_dataset_repo_id` is private.

    Examples:
        Report on every language in the default dataset's `train` split:

        $ uv run processing_scripts/usas_tag_distribution.py

        Report on two languages across both splits, top/bottom 5 tags, as LaTeX:

        $ uv run processing_scripts/usas_tag_distribution.py -l da -l en \\
              --split all --top-bottom-count 5 --format latex \\
              --output-table-major data/tables/major_tags.tex \\
              --output-table-top data/tables/top_tags.tex \\
              --output-table-bottom data/tables/bottom_tags.tex \\
              --output-table-summary data/tables/tag_summary.tex

        Save the major tag heatmap as PDF instead of PNG:

        $ uv run processing_scripts/usas_tag_distribution.py --image-format pdf
    """
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")

    wikipedia_language_codes = [language.value for language in languages] if languages else get_dataset_config_names(hf_dataset_repo_id, revision=hf_dataset_revision, token=hf_token)

    match split:
        case DatasetSplit.all:
            splits_to_load = ("train", "validation")
        case _:
            splits_to_load = (split.value,)

    tag_counts_by_language: dict[str, TagCounts] = {}

    for wikipedia_language_code in wikipedia_language_codes:
        combined_counts = TagCounts()
        for split_name in splits_to_load:
            dataset = load_dataset(hf_dataset_repo_id, wikipedia_language_code, split=split_name, revision=hf_dataset_revision, token=hf_token)
            combined_counts.extend_with(compute_tag_counts(dataset))
        tag_counts_by_language[wikipedia_language_code] = combined_counts
        rprint(f"{language_display_name(wikipedia_language_code)}: {combined_counts.total:,} tag occurrences, {len(combined_counts.tag_counts):,} unique tags")

    tag_percentages_by_language = {code: tag_percentages(counts) for code, counts in tag_counts_by_language.items()}
    major_tag_percentages_by_language = {code: major_tag_percentages(counts) for code, counts in tag_counts_by_language.items()}

    output_heatmap_major = resolve_image_path(output_heatmap_major, image_format)
    plot_major_tag_heatmap(major_tag_percentages_by_language, output_heatmap_major)
    rprint(f"Wrote major tag distribution heatmap to {output_heatmap_major!r}")

    alignments = ["left"] + ["right"] * (len(tag_percentages_by_language) + 1)

    major_tag_macro_averages = compute_macro_average_percentages(major_tag_percentages_by_language)
    major_headers, major_rows = build_tag_distribution_rows(major_tag_percentages_by_language, rank_tags(major_tag_macro_averages))
    write_or_print_table(render_table(major_headers, major_rows, alignments, table_format), output_table_major, "major tag")

    tag_macro_averages = compute_macro_average_percentages(tag_percentages_by_language)
    top_tags = rank_tags(tag_macro_averages)[:top_bottom_count]
    top_headers, top_rows = build_tag_distribution_rows(tag_percentages_by_language, top_tags)
    write_or_print_table(render_table(top_headers, top_rows, alignments, table_format), output_table_top, "top tags")

    bottom_tags = rank_tags(tag_macro_averages, descending=False)[:top_bottom_count]
    bottom_headers, bottom_rows = build_tag_distribution_rows(tag_percentages_by_language, bottom_tags)
    write_or_print_table(render_table(bottom_headers, bottom_rows, alignments, table_format), output_table_bottom, "bottom tags")

    percentage_summary_by_language = {code: five_number_summary(list(percentages.values())) for code, percentages in tag_percentages_by_language.items()}
    count_summary_by_language = {code: five_number_summary([float(count) for count in counts.tag_counts.values()]) for code, counts in tag_counts_by_language.items()}
    summary_headers, summary_rows = build_tag_summary_rows(percentage_summary_by_language, count_summary_by_language)
    write_or_print_table(render_table(summary_headers, summary_rows, alignments, table_format), output_table_summary, "tag-frequency summary")


if __name__ == "__main__":
    typer.run(main)
