"""Estimate per-language document loss from Wikipedia-page-ID deduplication.

`deduplicate_wikipedia_dataset.py` groups a language's `train` + `validation`
rows by `id` and drops every row except the highest-`version` copy, but it
never records anywhere how many documents that step removed from a given
build. This script estimates that count after the fact, by diffing two
independently generated LaTeX tables:

* The `"Kept"` column of `report_pipeline_document_funnel.py --view dropped`
  (default path: `data/tables/pipeline_funnel.tex`) -- documents surviving
  the pipeline's own GA/FA, test-URL, min-words, exact-dedup, and
  MinHash-dedup filters, per language.
* The `"Articles"` column of `dataset_statistics.py --output-latex` (default
  path: `data/tables/overall_dataset_statistics.tex`) -- the final published
  article count per language, summed across whichever splits are present.

`documents_after_filtering - final_articles` is then the number of documents
removed afterwards -- exactly what `deduplicate_wikipedia_dataset.py`'s
`id`-based dedup does to a freshly-built dataset.

Caveat: this is only a valid measurement when the funnel table's `log_data`
run is the *same, complete* run that produced the final dataset. If the
dataset was instead built from several separate pipeline runs merged
together (see "Commands used to create the original dataset" in
`README.md`), the funnel table only reflects one of those runs, and this
diff will not isolate dedup-by-id loss -- it will also pick up every
document contributed by the other runs.
"""

import csv
import dataclasses
from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint
from rich.table import Table

COLUMN_LABELS = {
    "language": "Language",
    "documents_after_filtering": "Documents After Filtering",
    "final_articles": "Final Articles",
    "dropped": "Dropped",
    "dropped_percentage": "Dropped (%)",
}
COLUMNS = tuple(COLUMN_LABELS)

TOTAL_ROW_LABEL = "Total (matched languages)"

_LATEX_SPECIAL_CHARACTERS = {"%": r"\%", "&": r"\&", "_": r"\_", "#": r"\#"}
_LATEX_UNESCAPE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"\textbackslash{}", "\\"),
    (r"\textasciitilde{}", "~"),
    (r"\textasciicircum{}", "^"),
    (r"\&", "&"),
    (r"\%", "%"),
    (r"\$", "$"),
    (r"\#", "#"),
    (r"\_", "_"),
    (r"\{", "{"),
    (r"\}", "}"),
)


def unescape_latex(text: str) -> str:
    r"""Reverse this repo's LaTeX table renderers' character escaping.

    Args:
        text: Text possibly containing LaTeX-escaped special characters.

    Returns:
        `text` with every escape in `_LATEX_UNESCAPE_REPLACEMENTS` replaced
        by its original character.

    Examples:
        >>> unescape_latex(r"50\%")
        '50%'
        >>> unescape_latex(r"a\_b \& c")
        'a_b & c'
    """
    for escaped, original in _LATEX_UNESCAPE_REPLACEMENTS:
        text = text.replace(escaped, original)
    return text


def parse_row_cells(line: str) -> list[str]:
    r"""Split one rendered LaTeX table row into its unescaped cell values.

    Args:
        line: A row line ending in `` \\`` (as rendered by this repo's LaTeX
            table renderers), e.g. `"Danish & 187 (0.064\%) \\\\"`.

    Returns:
        The row's cells, in order, with LaTeX escaping reversed.

    Examples:
        >>> parse_row_cells(r"Danish & 187 (0.064\%) \\")
        ['Danish', '187 (0.064%)']
    """
    stripped = line.rstrip()
    if stripped.endswith(r"\\"):
        stripped = stripped[: -len(r"\\")].rstrip()
    return [unescape_latex(cell.strip()) for cell in stripped.split(" & ")]


def parse_latex_tabular(text: str) -> tuple[list[str], list[list[str]]]:
    r"""Parse the header and rows out of a single LaTeX `tabular` environment.

    Expects exactly the structure produced by this repo's own LaTeX table
    renderers (see e.g. `dataset_statistics.rows_to_latex`): one `\toprule`,
    a single header row, one `\midrule`, the body rows, then `\bottomrule`
    and `\end{tabular}` -- no per-row `\midrule`s or a wrapping `table`
    environment.

    Args:
        text: The full contents of a `.tex` file holding one `tabular`
            environment.

    Returns:
        A `(headers, rows)` pair. Every cell has any LaTeX escaping (e.g.
        `\%`) reversed via `unescape_latex`.

    Raises:
        ValueError: If `text` does not have the expected `\toprule`/
            `\midrule`/`\bottomrule` structure.

    Examples:
        >>> text = r'''\begin{tabular}{ll}
        ... \toprule
        ... Language & Kept \\
        ... \midrule
        ... Danish & 187 (0.064\%) \\
        ... \bottomrule
        ... \end{tabular}
        ... '''
        >>> headers, rows = parse_latex_tabular(text)
        >>> headers
        ['Language', 'Kept']
        >>> rows
        [['Danish', '187 (0.064%)']]
    """
    lines = [line.strip() for line in text.strip().splitlines()]
    try:
        top_rule_index = lines.index(r"\toprule")
        mid_rule_index = lines.index(r"\midrule")
        bottom_rule_index = lines.index(r"\bottomrule")
    except ValueError as error:
        raise ValueError(r"Expected a \toprule/\midrule/\bottomrule tabular structure") from error

    header_line = lines[top_rule_index + 1]
    row_lines = lines[mid_rule_index + 1 : bottom_rule_index]

    headers = parse_row_cells(header_line)
    rows = [parse_row_cells(line) for line in row_lines]
    return headers, rows


def load_latex_table(path: Path) -> tuple[list[str], list[list[str]]]:
    """Read a `.tex` file and parse its single `tabular` environment.

    Args:
        path: Path to the `.tex` file.

    Returns:
        The `(headers, rows)` pair, as returned by `parse_latex_tabular`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If `path`'s contents are not a single well-formed
            `tabular` environment (see `parse_latex_tabular`).
    """
    return parse_latex_tabular(path.read_text(encoding="utf-8"))


def column_index(headers: list[str], name: str) -> int:
    """Return the index of a required column, or raise a clear error.

    Args:
        headers: Column headers to search.
        name: The column name to find.

    Returns:
        The index of `name` within `headers`.

    Raises:
        ValueError: If `name` is not in `headers`.

    Examples:
        >>> column_index(["Language", "Kept"], "Kept")
        1
        >>> column_index(["Language"], "Kept")
        Traceback (most recent call last):
            ...
        ValueError: Column 'Kept' not found. Available columns: ['Language']
    """
    try:
        return headers.index(name)
    except ValueError as error:
        raise ValueError(f"Column {name!r} not found. Available columns: {headers!r}") from error


def parse_int(text: str) -> int:
    """Parse an integer that may contain `,` thousands separators.

    Args:
        text: The number to parse, e.g. `"1,369,932"`.

    Returns:
        The parsed integer.

    Examples:
        >>> parse_int("1,369,932")
        1369932
        >>> parse_int("187")
        187
    """
    return int(text.replace(",", ""))


def parse_kept_cell(cell: str) -> int | None:
    """Extract the leading document count from a "Kept" cell, e.g. `"187 (0.064%)"`.

    Args:
        cell: A "Kept" column cell, as rendered by
            `report_pipeline_document_funnel.py`'s `format_kept_cell`.

    Returns:
        The leading document count, or `None` for a `"n/a"` cell.

    Examples:
        >>> parse_kept_cell("187 (0.064%)")
        187
        >>> parse_kept_cell("n/a")
    """
    if cell.strip().lower() == "n/a":
        return None
    return parse_int(cell.split("(")[0].strip())


def extract_funnel_kept_counts(headers: list[str], rows: list[list[str]]) -> dict[str, int | None]:
    """Extract each language's "Kept" document count from parsed funnel-table rows.

    Args:
        headers: Column headers, as returned by `parse_latex_tabular`. Must
            include `"Language"` and `"Kept"` (as produced by
            `report_pipeline_document_funnel.py --view dropped`).
        rows: Parsed row cells, as returned by `parse_latex_tabular`.

    Returns:
        Mapping of language display name to its "Kept" document count, or
        `None` for a `"n/a"` cell.

    Raises:
        ValueError: If `headers` is missing a `"Language"` or `"Kept"` column.

    Examples:
        >>> headers = ["Language", "Kept"]
        >>> rows = [["Danish", "187 (0.064%)"], ["Dutch", "n/a"]]
        >>> extract_funnel_kept_counts(headers, rows)
        {'Danish': 187, 'Dutch': None}
    """
    language_index = column_index(headers, "Language")
    kept_index = column_index(headers, "Kept")
    return {row[language_index]: parse_kept_cell(row[kept_index]) for row in rows}


def extract_final_article_counts(headers: list[str], rows: list[list[str]]) -> dict[str, int]:
    """Compute each language's total article count from parsed statistics-table rows.

    Skips the aggregate `"Total"` language row. For a language with an
    explicit combined row (`Split` of `"total"`, e.g. from `--split
    all`/`--split combined` in `dataset_statistics.py`), uses only that
    row's `Articles` value, to avoid double-counting; otherwise sums
    `Articles` across every row for that language (e.g. separate
    `train`/`validation` rows with no combined row).

    Args:
        headers: Column headers, as returned by `parse_latex_tabular`. Must
            include `"Language"` and `"Articles"`.
        rows: Parsed row cells, as returned by `parse_latex_tabular`.

    Returns:
        Mapping of language display name to its total article count.

    Raises:
        ValueError: If `headers` is missing a `"Language"` or `"Articles"` column.

    Examples:
        No combined row: sums `train` + `validation`, skipping the "Total"
        aggregate language row:

        >>> headers = ["Language", "Split", "Articles"]
        >>> rows = [["Danish", "train", "168"], ["Danish", "validation", "19"], ["Total", "total", "187"]]
        >>> extract_final_article_counts(headers, rows)
        {'Danish': 187}

        A combined row is preferred over summing (190, not 168 + 19 = 187):

        >>> headers = ["Language", "Split", "Articles"]
        >>> rows = [["Danish", "train", "168"], ["Danish", "validation", "19"], ["Danish", "total", "190"]]
        >>> extract_final_article_counts(headers, rows)
        {'Danish': 190}
    """
    language_index = column_index(headers, "Language")
    articles_index = column_index(headers, "Articles")
    split_index = headers.index("Split") if "Split" in headers else None

    rows_by_language: dict[str, list[list[str]]] = {}
    for row in rows:
        language = row[language_index]
        if language == "Total":
            continue
        rows_by_language.setdefault(language, []).append(row)

    counts: dict[str, int] = {}
    for language, language_rows in rows_by_language.items():
        if split_index is not None:
            combined_rows = [row for row in language_rows if row[split_index] == "total"]
            if combined_rows:
                counts[language] = parse_int(combined_rows[0][articles_index])
                continue
        counts[language] = sum(parse_int(row[articles_index]) for row in language_rows)
    return counts


@dataclasses.dataclass(frozen=True)
class DeduplicationLoss:
    """Estimated document loss between two stages of one language's dataset build.

    Attributes:
        language: Language display name, or a summary label like
            `TOTAL_ROW_LABEL`.
        documents_after_filtering: Documents surviving the pipeline's own
            filters (a funnel table's "Kept" count).
        final_articles: The final published article count.
    """

    language: str
    documents_after_filtering: int
    final_articles: int

    @property
    def dropped(self) -> int:
        """Documents removed after filtering, presumably by id-based dedup.

        Examples:
            >>> DeduplicationLoss("English", 49242, 49218).dropped
            24
        """
        return self.documents_after_filtering - self.final_articles

    @property
    def dropped_percentage(self) -> float:
        """Percentage of post-filtering documents subsequently dropped.

        Examples:
            >>> DeduplicationLoss("English", 49242, 49218).dropped_percentage
            0.05
            >>> DeduplicationLoss("English", 0, 0).dropped_percentage
            0.0
        """
        if self.documents_after_filtering == 0:
            return 0.0
        return round(self.dropped / self.documents_after_filtering * 100, 2)


def compute_deduplication_losses(funnel_kept: dict[str, int | None], final_articles: dict[str, int]) -> list[DeduplicationLoss]:
    """Pair up matching languages from both tables into `DeduplicationLoss` rows.

    Only languages present, with a known "Kept" count, in both `funnel_kept`
    and `final_articles` are included -- a language missing from either
    table, or with an unknown (`"n/a"`) "Kept" count, is silently skipped
    (see `main` for a report of what got skipped).

    Args:
        funnel_kept: Mapping of language to its funnel-table "Kept" count
            (or `None` for an `"n/a"` cell), as returned by
            `extract_funnel_kept_counts`.
        final_articles: Mapping of language to its final article count, as
            returned by `extract_final_article_counts`.

    Returns:
        One `DeduplicationLoss` per matching language, sorted by language name.

    Examples:
        >>> losses = compute_deduplication_losses(
        ...     {"Danish": 187, "English": 49242, "Chinese": None},
        ...     {"Danish": 187, "English": 49218},
        ... )
        >>> [(loss.language, loss.dropped) for loss in losses]
        [('Danish', 0), ('English', 24)]
    """
    known_kept: dict[str, int] = {language: kept for language, kept in funnel_kept.items() if kept is not None}
    languages = sorted(language for language in known_kept if language in final_articles)
    return [DeduplicationLoss(language, known_kept[language], final_articles[language]) for language in languages]


def loss_to_row(loss: DeduplicationLoss) -> dict[str, str | int | float]:
    """Format one `DeduplicationLoss` as a flat row for display/export.

    Args:
        loss: The loss figures to format.

    Returns:
        A dict of column name (matching `COLUMNS`) to value.

    Examples:
        >>> loss_to_row(DeduplicationLoss("English", 49242, 49218))
        {'language': 'English', 'documents_after_filtering': 49242, 'final_articles': 49218, 'dropped': 24, 'dropped_percentage': 0.05}
    """
    return {
        "language": loss.language,
        "documents_after_filtering": loss.documents_after_filtering,
        "final_articles": loss.final_articles,
        "dropped": loss.dropped,
        "dropped_percentage": loss.dropped_percentage,
    }


def format_row_value(value: str | int | float) -> str:
    """Format a row value for console display, adding `,` thousands separators to numbers.

    Args:
        value: The value to format, as produced by `loss_to_row`.

    Returns:
        `value` unchanged if it is a string, otherwise formatted with `,`
        thousands separators (and, for floats, two decimal places).

    Examples:
        >>> format_row_value("English")
        'English'
        >>> format_row_value(49242)
        '49,242'
        >>> format_row_value(0.05)
        '0.05'
    """
    match value:
        case int():
            return f"{value:,}"
        case float():
            return f"{value:,.2f}"
        case _:
            return str(value)


def _latex_escape(cell: str) -> str:
    """Escape the LaTeX special characters that can occur in a table cell."""
    return "".join(_LATEX_SPECIAL_CHARACTERS.get(character, character) for character in cell)


def rows_to_latex(rows: list[dict[str, str | int | float]]) -> str:
    r"""Render loss rows as a LaTeX `tabular` environment.

    Args:
        rows: Rows to render, as produced by `loss_to_row`.

    Returns:
        A LaTeX `tabular` environment (using `booktabs` rules), ready to be
        embedded within a `table` environment in a LaTeX document.

    Examples:
        >>> rows = [{"language": "English", "documents_after_filtering": 49242, "final_articles": 49218, "dropped": 24, "dropped_percentage": 0.05}]
        >>> print(rows_to_latex(rows))
        \begin{tabular}{lrrrr}
        \toprule
        Language & Documents After Filtering & Final Articles & Dropped & Dropped (\%) \\
        \midrule
        English & 49,242 & 49,218 & 24 & 0.05 \\
        \bottomrule
        \end{tabular}
    """
    column_spec = "l" + "r" * (len(COLUMNS) - 1)
    lines = [
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(_latex_escape(COLUMN_LABELS[column]) for column in COLUMNS) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(format_row_value(row[column])) for column in COLUMNS) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main(
    funnel_table: Annotated[
        Path,
        typer.Argument(help='Path to the LaTeX table from `report_pipeline_document_funnel.py --view dropped` (must have "Language" and "Kept" columns).'),
    ] = Path("data/tables/pipeline_funnel.tex"),
    statistics_table: Annotated[
        Path,
        typer.Argument(help='Path to the LaTeX table from `dataset_statistics.py --output-latex` (must have "Language" and "Articles" columns).'),
    ] = Path("data/tables/overall_dataset_statistics.tex"),
    output_csv: Annotated[Path | None, typer.Option("--output-csv", help="Optional path to also write the report to as a CSV file.")] = None,
    output_latex: Annotated[Path | None, typer.Option("--output-latex", help="Optional path to also write the report as a LaTeX tabular environment.")] = None,
) -> None:
    """Estimate per-language document loss from Wikipedia-page-ID deduplication.

    Reads `funnel_table` and `statistics_table` and, for every language
    present in both, reports how many documents were dropped between the
    two: `documents_after_filtering` (the funnel table's "Kept" count) minus
    `final_articles` (the statistics table's "Articles" count, summed across
    whichever splits are present). This is exactly what
    `deduplicate_wikipedia_dataset.py`'s `id`-based dedup does to a
    freshly-built dataset -- see the module docstring for the important
    caveat about when this diff is (and is not) a valid measurement.

    A language present in only one table, or with an unavailable ("n/a")
    "Kept" count, is skipped and reported separately. A final `"Total
    (matched languages)"` row sums the figures across every included
    language.

    Examples:
        Report using the default table paths:

        $ uv run processing_scripts/report_deduplication_loss.py

        Report using explicit paths, and also save to CSV:

        $ uv run processing_scripts/report_deduplication_loss.py \\
              data/tables/pipeline_funnel.tex data/tables/overall_dataset_statistics.tex \\
              --output-csv data/tables/dedup_loss.csv
    """
    funnel_headers, funnel_rows = load_latex_table(funnel_table)
    funnel_kept = extract_funnel_kept_counts(funnel_headers, funnel_rows)

    statistics_headers, statistics_rows = load_latex_table(statistics_table)
    final_articles = extract_final_article_counts(statistics_headers, statistics_rows)

    losses = compute_deduplication_losses(funnel_kept, final_articles)
    matched_languages = {loss.language for loss in losses}

    missing_from_statistics = sorted(language for language, kept in funnel_kept.items() if kept is not None and language not in final_articles)
    missing_from_funnel = sorted(language for language in final_articles if language not in funnel_kept)
    unknown_kept = sorted(language for language, kept in funnel_kept.items() if kept is None)

    if missing_from_statistics:
        rprint(f"[yellow]No matching row in {statistics_table} for: {', '.join(missing_from_statistics)}[/yellow]")
    if missing_from_funnel:
        rprint(f"[yellow]No matching row in {funnel_table} for: {', '.join(missing_from_funnel)}[/yellow]")
    if unknown_kept:
        rprint(f"[yellow]'Kept' count unavailable (n/a) in {funnel_table} for: {', '.join(unknown_kept)}[/yellow]")

    total_after_filtering = sum(loss.documents_after_filtering for loss in losses)
    total_final_articles = sum(loss.final_articles for loss in losses)
    rows = [loss_to_row(loss) for loss in losses]
    rows.append(loss_to_row(DeduplicationLoss(TOTAL_ROW_LABEL, total_after_filtering, total_final_articles)))

    table = Table(title="Estimated document loss from id-based deduplication")
    for column in COLUMNS:
        table.add_column(COLUMN_LABELS[column])
    for row in rows:
        table.add_row(*(format_row_value(row[column]) for column in COLUMNS))
    rprint(table)

    if not matched_languages:
        rprint("[yellow]No languages matched between the two tables -- nothing to report.[/yellow]")

    if output_csv is not None:
        with output_csv.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
        rprint(f"Wrote report to {output_csv!r}")

    if output_latex is not None:
        output_latex.write_text(rows_to_latex(rows) + "\n", encoding="utf-8")
        rprint(f"Wrote report to {output_latex!r}")


if __name__ == "__main__":
    typer.run(main)
