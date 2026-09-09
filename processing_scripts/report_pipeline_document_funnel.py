"""Summarise per-language document attrition through the pipeline's filter stages from a ``log_data`` folder.

The DataTrove pipeline writes one sub-directory per language under ``log_data``
(plus a ``driver`` folder). Each language folder holds one sub-directory per
pipeline stage, and every stage writes a ``stats.json`` recording each pipeline
block's document counts, already summed across that stage's array tasks.

Only six blocks actually discard documents. In pipeline order:

* ``reading`` / ``Lambda`` -- pages that are not Wikipedia "Good"/"Featured".
* ``reading`` / ``Simple URL Filter`` -- held-out test-set URLs.
* ``initial_process`` / ``Empty text filter`` -- documents empty after the
  markdown -> plain-text conversion.
* ``initial_process`` / ``Minimum Words Document Filter`` -- documents below the
  min-word threshold.
* ``exact_dedup_filter`` / ``exact-deduplication stage 3`` -- exact-duplicate
  documents.
* ``minhash_dedup_filter`` / ``MinHash stage 4`` -- near-duplicate documents.

Every one of those blocks records ``total`` documents in and ``forwarded``
documents out, so ``total - forwarded`` is the number it dropped. This script
turns that into a per-language table in one of two views:

* ``dropped`` (default) -- documents removed at each stage, the total removed,
  and the number (with percentage) of documents kept.
* ``survived`` -- the FineWiki input count and the number of documents still
  alive after each stage.

A filter stage that dropped nothing for every language is left out of the table
(the empty-text filter is normally dormant); pass ``--all-stages`` to keep every
column.

The table is rendered as Markdown (default) or a LaTeX ``booktabs`` table.

Example:
    Print the Markdown "dropped" table for the checked-out logs::

        $ uv run processing_scripts/report_pipeline_document_funnel.py ./log_data

    Write the "survived" funnel as a LaTeX table, largest corpus first::

        $ uv run processing_scripts/report_pipeline_document_funnel.py ./log_data \\
            --view survived --format latex --output-file funnel.tex
"""

import dataclasses
import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import typer

from wikipedia_processing.utils import (
    discover_log_data_language_directories,
    language_display_name,
)


@dataclasses.dataclass(frozen=True)
class FilterStageSpec:
    """Locates one document-dropping pipeline block inside a language's logs.

    Attributes:
        column_label: Short header used for this stage's table column.
        stage_directory_name: ``log_data/<language>`` sub-directory holding the
            stage's ``stats.json``.
        block_name_substring: Substring identifying the block within that
            ``stats.json`` (block names carry emoji, so a substring is matched).
    """

    column_label: str
    stage_directory_name: str
    block_name_substring: str


# The document-dropping blocks, in pipeline order.
FILTER_STAGE_SPECS: tuple[FilterStageSpec, ...] = (
    FilterStageSpec("Good/Featured", "reading", "Lambda"),
    FilterStageSpec("Test URL", "reading", "URL Filter"),
    FilterStageSpec("Empty text", "initial_process", "Empty text filter"),
    FilterStageSpec("Min words", "initial_process", "Minimum Words"),
    FilterStageSpec("Exact dedup", "exact_dedup_filter", "exact-deduplication"),
    FilterStageSpec("MinHash dedup", "minhash_dedup_filter", "MinHash stage 4"),
)

STAGE_COLUMN_LABELS: tuple[str, ...] = tuple(
    spec.column_label for spec in FILTER_STAGE_SPECS
)


class TableFormat(str, Enum):
    """Supported rendering formats for the funnel table."""

    MARKDOWN = "markdown"
    LATEX = "latex"


class FunnelView(str, Enum):
    """Which per-language figures the table shows."""

    DROPPED = "dropped"
    SURVIVED = "survived"


class SortKey(str, Enum):
    """Supported row orderings for the funnel table."""

    INPUT = "input"
    KEPT = "kept"
    LANGUAGE = "language"


@dataclasses.dataclass(frozen=True)
class FilterStageResult:
    """Document counts flowing through one filter block for one language.

    Attributes:
        column_label: The stage's table-column header.
        documents_in: Documents the block received (its ``total`` stat), or
            ``None`` when the block was not found.
        documents_out: Documents the block forwarded (its ``forwarded`` stat),
            or ``None`` when the block was not found.
    """

    column_label: str
    documents_in: int | None
    documents_out: int | None

    @property
    def dropped(self) -> int | None:
        """Documents discarded by this block, or ``None`` if counts are missing.

        Examples:
            >>> FilterStageResult("Exact dedup", 197, 190).dropped
            7
            >>> FilterStageResult("Exact dedup", None, 190).dropped is None
            True
        """
        if self.documents_in is None or self.documents_out is None:
            return None
        return self.documents_in - self.documents_out


@dataclasses.dataclass(frozen=True)
class LanguageDocumentFunnel:
    """The full filter-stage document funnel for a single language.

    Attributes:
        language_code: Wikipedia language code (the ``log_data`` sub-directory
            name), e.g. ``"da"``.
        language_name: Human-readable language name, e.g. ``"Danish"``.
        stage_results: One :class:`FilterStageResult` per entry in
            :data:`FILTER_STAGE_SPECS`, in pipeline order.
    """

    language_code: str
    language_name: str
    stage_results: tuple[FilterStageResult, ...]

    @property
    def input_document_count(self) -> int | None:
        """Documents entering the first filter stage (the FineWiki page count).

        Examples:
            >>> funnel = LanguageDocumentFunnel("da", "Danish", (
            ...     FilterStageResult("Good/Featured", 291961, 197),
            ...     FilterStageResult("MinHash dedup", 190, 187),
            ... ))
            >>> funnel.input_document_count
            291961
        """
        return self.stage_results[0].documents_in if self.stage_results else None

    @property
    def final_document_count(self) -> int | None:
        """Documents forwarded by the last stage that recorded a count.

        Examples:
            >>> funnel = LanguageDocumentFunnel("da", "Danish", (
            ...     FilterStageResult("Good/Featured", 291961, 197),
            ...     FilterStageResult("MinHash dedup", 190, 187),
            ... ))
            >>> funnel.final_document_count
            187
        """
        for result in reversed(self.stage_results):
            if result.documents_out is not None:
                return result.documents_out
        return None

    @property
    def total_dropped(self) -> int | None:
        """Documents removed across the whole pipeline, or ``None`` if unknown.

        Examples:
            >>> funnel = LanguageDocumentFunnel("da", "Danish", (
            ...     FilterStageResult("Good/Featured", 291961, 197),
            ...     FilterStageResult("MinHash dedup", 190, 187),
            ... ))
            >>> funnel.total_dropped
            291774
        """
        if self.input_document_count is None or self.final_document_count is None:
            return None
        return self.input_document_count - self.final_document_count


def format_document_count(count: int | None) -> str:
    """Format a document count with thousands separators, or ``"n/a"``.

    Args:
        count: A document count, or ``None`` when the figure is unavailable.

    Returns:
        The grouped count, e.g. ``"6,614,655"``, or ``"n/a"``.

    Examples:
        >>> format_document_count(6614655)
        '6,614,655'
        >>> format_document_count(0)
        '0'
        >>> format_document_count(None)
        'n/a'
    """
    return "n/a" if count is None else f"{count:,}"


def format_kept_cell(
    final_document_count: int | None, input_document_count: int | None
) -> str:
    """Format the "documents kept" cell as ``"<count> (<percentage>%)"``.

    Args:
        final_document_count: Documents surviving the whole pipeline.
        input_document_count: Documents that entered the first filter stage.

    Returns:
        The kept count with its share of the input to three significant
        figures, e.g. ``"49,242 (0.744%)"``. ``"n/a"`` when either figure is
        missing or the input count is zero.

    Examples:
        >>> format_kept_cell(49242, 6614655)
        '49,242 (0.744%)'
        >>> format_kept_cell(378, 2072865)
        '378 (0.0182%)'
        >>> format_kept_cell(None, 100)
        'n/a'
    """
    if final_document_count is None or not input_document_count:
        return "n/a"
    percentage = 100 * final_document_count / input_document_count
    return f"{final_document_count:,} ({percentage:.3g}%)"


def load_stats_blocks(stats_file: Path) -> list[dict]:
    """Parse a DataTrove ``stats.json`` file into its list of stat blocks.

    Args:
        stats_file: Path to a ``<stage>/stats.json`` file.

    Returns:
        The parsed list of block objects, or ``[]`` when the file is absent or
        not a JSON list.

    Examples:
        >>> load_stats_blocks(Path("log_data/da/reading/stats.json"))  # doctest: +SKIP
        [{'name': '📖 - READER: 🤗 HuggingFace', ...}, ...]
    """
    if not stats_file.is_file():
        return []
    try:
        parsed = json.loads(stats_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def find_stats_block(blocks: list[dict], name_substring: str) -> dict | None:
    """Return the first stat block whose ``name`` contains ``name_substring``.

    Args:
        blocks: Stat blocks parsed from a ``stats.json`` file.
        name_substring: Substring to search for in each block's ``name``.

    Returns:
        The matching block, or ``None`` when nothing matches.

    Examples:
        >>> blocks = [{"name": "\U0001f53b - FILTER: \U0001f464 Lambda", "stats": {}}]
        >>> find_stats_block(blocks, "Lambda")["name"].endswith("Lambda")
        True
        >>> find_stats_block(blocks, "MinHash") is None
        True
    """
    for block in blocks:
        if name_substring in block.get("name", ""):
            return block
    return None


def _stat_total(entry: Any) -> int | None:
    """Return the document count from a DataTrove stat entry.

    A stat is stored either as a bare number (stages merged from a single task)
    or as an aggregate ``{"total": <number>, "mean": ..., ...}`` mapping (stages
    merged across several tasks); both are handled.
    """
    if isinstance(entry, dict):
        entry = entry.get("total")
    if isinstance(entry, bool):
        return None
    if isinstance(entry, int | float):
        return int(entry)
    return None


def block_document_flow(block: dict | None) -> tuple[int | None, int | None]:
    """Return ``(documents_in, documents_out)`` for a DataTrove filter block.

    Args:
        block: A stat block from :func:`find_stats_block`, or ``None``.

    Returns:
        The block's ``total`` and ``forwarded`` document counts. Either element
        is ``None`` when the block is missing or does not record that figure.

    Examples:
        >>> block_document_flow(
        ...     {"stats": {"total": {"total": 197}, "forwarded": {"total": 190}}}
        ... )
        (197, 190)
        >>> block_document_flow({"stats": {"total": 291961, "forwarded": 197}})
        (291961, 197)
        >>> block_document_flow({"stats": {}})
        (None, None)
        >>> block_document_flow(None)
        (None, None)
    """
    if block is None:
        return None, None
    stats = block.get("stats", {})
    return _stat_total(stats.get("total")), _stat_total(stats.get("forwarded"))


def build_language_funnel(language_directory: Path) -> LanguageDocumentFunnel:
    """Build the :class:`LanguageDocumentFunnel` for one language folder.

    Args:
        language_directory: A ``log_data/<language>`` folder.

    Returns:
        The language's filter-stage document funnel.

    Examples:
        >>> build_language_funnel(Path("log_data/da"))  # doctest: +SKIP
        LanguageDocumentFunnel(language_code='da', language_name='Danish', ...)
    """
    blocks_by_stage: dict[str, list[dict]] = {}
    stage_results: list[FilterStageResult] = []
    for spec in FILTER_STAGE_SPECS:
        if spec.stage_directory_name not in blocks_by_stage:
            blocks_by_stage[spec.stage_directory_name] = load_stats_blocks(
                language_directory / spec.stage_directory_name / "stats.json"
            )
        block = find_stats_block(
            blocks_by_stage[spec.stage_directory_name], spec.block_name_substring
        )
        documents_in, documents_out = block_document_flow(block)
        stage_results.append(
            FilterStageResult(spec.column_label, documents_in, documents_out)
        )

    language_code = language_directory.name
    return LanguageDocumentFunnel(
        language_code=language_code,
        language_name=language_display_name(language_code),
        stage_results=tuple(stage_results),
    )


def sort_funnels(
    funnels: list[LanguageDocumentFunnel], sort_by: SortKey
) -> list[LanguageDocumentFunnel]:
    """Return ``funnels`` ordered according to ``sort_by``.

    Args:
        funnels: The per-language funnels to order.
        sort_by: ``INPUT`` and ``KEPT`` sort descending (missing counts last);
            ``LANGUAGE`` sorts ascending by language name.

    Returns:
        A new, ordered list. The input list is left unchanged.

    Examples:
        >>> da = LanguageDocumentFunnel("da", "Danish", (
        ...     FilterStageResult("Good/Featured", 291961, 187),))
        >>> en = LanguageDocumentFunnel("en", "English", (
        ...     FilterStageResult("Good/Featured", 6614655, 49242),))
        >>> [f.language_code for f in sort_funnels([da, en], SortKey.INPUT)]
        ['en', 'da']
        >>> [f.language_code for f in sort_funnels([en, da], SortKey.LANGUAGE)]
        ['da', 'en']
    """
    ordered = list(funnels)
    match sort_by:
        case SortKey.LANGUAGE:
            ordered.sort(key=lambda funnel: funnel.language_name)
        case SortKey.KEPT:
            ordered.sort(
                key=lambda funnel: funnel.final_document_count or 0, reverse=True
            )
        case SortKey.INPUT:
            ordered.sort(
                key=lambda funnel: funnel.input_document_count or 0, reverse=True
            )
    return ordered


def informative_stage_indices(funnels: list[LanguageDocumentFunnel]) -> list[int]:
    """Return the :data:`FILTER_STAGE_SPECS` indices worth showing as columns.

    A stage is kept when at least one language has an unknown drop count for it
    (so the gap stays visible) or any language dropped a non-zero number of
    documents there. A stage that provably dropped nothing everywhere is
    omitted.

    Args:
        funnels: The per-language funnels.

    Returns:
        Ascending stage indices to render, always a subset of
        ``range(len(FILTER_STAGE_SPECS))``.

    Examples:
        >>> a = LanguageDocumentFunnel("da", "Danish", (
        ...     FilterStageResult("Good/Featured", 100, 10),
        ...     FilterStageResult("Empty text", 10, 10),
        ... ))
        >>> b = LanguageDocumentFunnel("en", "English", (
        ...     FilterStageResult("Good/Featured", 200, 20),
        ...     FilterStageResult("Empty text", 20, 20),
        ... ))
        >>> informative_stage_indices([a, b])
        [0]
    """
    stage_count = len(funnels[0].stage_results) if funnels else 0
    kept: list[int] = []
    for index in range(stage_count):
        known_drops = [
            funnel.stage_results[index].dropped
            for funnel in funnels
            if funnel.stage_results[index].dropped is not None
        ]
        if len(known_drops) < len(funnels) or any(drop != 0 for drop in known_drops):
            kept.append(index)
    return kept


def build_table(
    funnels: list[LanguageDocumentFunnel],
    view: FunnelView,
    stage_indices: list[int],
) -> tuple[list[str], list[list[str]], list[str]]:
    """Return ``(headers, rows, alignments)`` for the requested funnel view.

    Args:
        funnels: The per-language funnels, already in display order.
        view: ``DROPPED`` shows documents removed at each stage plus the total
            removed and the kept count/percentage; ``SURVIVED`` shows the
            FineWiki input and documents alive after each stage.
        stage_indices: Which :data:`FILTER_STAGE_SPECS` stages to include as
            columns (see :func:`informative_stage_indices`).

    Returns:
        The column headers, the string cells for each row, and the per-column
        alignment (``"left"`` or ``"right"``).

    Examples:
        >>> funnel = LanguageDocumentFunnel("da", "Danish", (
        ...     FilterStageResult("Good/Featured", 291961, 197),
        ...     FilterStageResult("Test URL", 197, 197),
        ...     FilterStageResult("Empty text", 197, 197),
        ...     FilterStageResult("Min words", 197, 197),
        ...     FilterStageResult("Exact dedup", 197, 190),
        ...     FilterStageResult("MinHash dedup", 190, 187),
        ... ))
        >>> headers, rows, alignments = build_table(
        ...     [funnel], FunnelView.DROPPED, [0, 1, 3, 4, 5]
        ... )
        >>> headers
        ['Language', 'Good/Featured', 'Test URL', 'Min words', 'Exact dedup', 'MinHash dedup', 'Total removed', 'Kept']
        >>> rows[0]
        ['Danish', '291,764', '0', '0', '7', '3', '291,774', '187 (0.064%)']
        >>> build_table([funnel], FunnelView.SURVIVED, [0, 4, 5])[1][0]
        ['Danish', '291,961', '197', '190', '187']
    """
    stage_labels = [STAGE_COLUMN_LABELS[index] for index in stage_indices]

    match view:
        case FunnelView.DROPPED:
            headers = ["Language", *stage_labels, "Total removed", "Kept"]
            rows = [
                [
                    funnel.language_name,
                    *(
                        format_document_count(funnel.stage_results[index].dropped)
                        for index in stage_indices
                    ),
                    format_document_count(funnel.total_dropped),
                    format_kept_cell(
                        funnel.final_document_count, funnel.input_document_count
                    ),
                ]
                for funnel in funnels
            ]
        case FunnelView.SURVIVED:
            headers = ["Language", "FineWiki input", *stage_labels]
            rows = [
                [
                    funnel.language_name,
                    format_document_count(funnel.input_document_count),
                    *(
                        format_document_count(funnel.stage_results[index].documents_out)
                        for index in stage_indices
                    ),
                ]
                for funnel in funnels
            ]

    alignments = ["left", *["right"] * (len(headers) - 1)]
    return headers, rows, alignments


def _pad(value: str, width: int, alignment: str) -> str:
    """Pad ``value`` to ``width`` using left or right alignment."""
    return value.rjust(width) if alignment == "right" else value.ljust(width)


def _column_widths(headers: list[str], rows: list[list[str]]) -> list[int]:
    """Return the display width of each column, header included."""
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    return widths


def render_markdown_table(
    headers: list[str], rows: list[list[str]], alignments: list[str]
) -> str:
    """Render a padded GitHub-flavoured Markdown table.

    Args:
        headers: Column headers.
        rows: String cells, one list per row, aligned with ``headers``.
        alignments: Per-column ``"left"`` or ``"right"``.

    Returns:
        The table as a single string, without a trailing newline.

    Examples:
        >>> print(render_markdown_table(
        ...     ["Language", "Exact dedup", "Kept"],
        ...     [["Danish", "7", "187 (0.0641%)"]],
        ...     ["left", "right", "right"],
        ... ))
        | Language | Exact dedup |          Kept |
        | :------- | ----------: | ------------: |
        | Danish   |           7 | 187 (0.0641%) |
    """
    widths = _column_widths(headers, rows)

    def render_row(cells: list[str]) -> str:
        padded = (
            _pad(cell, width, alignment)
            for cell, width, alignment in zip(cells, widths, alignments)
        )
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


def render_latex_table(
    headers: list[str], rows: list[list[str]], alignments: list[str]
) -> str:
    r"""Render a LaTeX ``booktabs`` ``tabular``.

    The output requires ``\usepackage{booktabs}`` in the document preamble.

    Args:
        headers: Column headers.
        rows: String cells, one list per row, aligned with ``headers``.
        alignments: Per-column ``"left"`` (``l``) or ``"right"`` (``r``).

    Returns:
        The ``tabular`` environment as a single string, without a trailing
        newline.

    Examples:
        >>> print(render_latex_table(
        ...     ["Language", "Kept"],
        ...     [["Danish", "187 (0.0641%)"]],
        ...     ["left", "right"],
        ... ))
        \begin{tabular}{lr}
        \toprule
        Language & Kept \\
        \midrule
        Danish & 187 (0.0641\%) \\
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
    lines.extend(
        " & ".join(_latex_escape(cell) for cell in row) + r" \\" for row in rows
    )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def render_table(
    headers: list[str],
    rows: list[list[str]],
    alignments: list[str],
    table_format: TableFormat,
) -> str:
    """Render a table in the requested :class:`TableFormat`.

    Args:
        headers: Column headers.
        rows: String cells, one list per row.
        alignments: Per-column ``"left"`` or ``"right"``.
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


def main(
    log_data_directory: Annotated[
        Path,
        typer.Argument(
            help="Path to the pipeline log_data folder (one sub-directory per language).",
        ),
    ] = Path("log_data"),
    view: Annotated[
        FunnelView,
        typer.Option(
            "-v",
            "--view",
            help="Show documents dropped at each stage, or documents surviving each stage.",
        ),
    ] = FunnelView.DROPPED,
    table_format: Annotated[
        TableFormat,
        typer.Option("-f", "--format", help="Output table format."),
    ] = TableFormat.MARKDOWN,
    sort_by: Annotated[
        SortKey,
        typer.Option(
            "-s",
            "--sort-by",
            help="Row ordering: descending FineWiki input, descending documents kept, or language name.",
        ),
    ] = SortKey.LANGUAGE,
    all_stages: Annotated[
        bool,
        typer.Option(
            "--all-stages/--prune-empty-stages",
            help="Keep every filter-stage column, even ones that dropped nothing for any language.",
        ),
    ] = False,
    output_file: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output-file",
            help="Write the table to this file instead of stdout.",
        ),
    ] = None,
) -> None:
    """Summarise per-language document attrition through the pipeline's filter stages.

    Scans ``log_data_directory`` for per-language folders, reads each stage's
    ``stats.json`` to recover the ``total`` / ``forwarded`` document counts of
    the six document-dropping blocks, then prints either the per-stage drop
    counts (``--view dropped``) or the surviving-document funnel
    (``--view survived``).

    Args:
        log_data_directory: The pipeline ``log_data`` folder to scan.
        view: ``dropped`` (default) or ``survived``.
        table_format: ``markdown`` (default) or ``latex``.
        sort_by: Row ordering (``input``, ``kept``, or ``language``).
        all_stages: Keep filter-stage columns that dropped nothing for every
            language (pruned by default).
        output_file: Optional path to write the table to instead of stdout.

    Raises:
        typer.BadParameter: If ``log_data_directory`` is not a directory or
            contains no per-language pipeline folders.
    """
    if not log_data_directory.is_dir():
        raise typer.BadParameter(f"Not a directory: {log_data_directory}")

    language_directories = discover_log_data_language_directories(log_data_directory)
    if not language_directories:
        raise typer.BadParameter(
            f"No per-language pipeline folders found under {log_data_directory}"
        )

    funnels = sort_funnels(
        [build_language_funnel(directory) for directory in language_directories],
        sort_by,
    )
    stage_indices = (
        list(range(len(FILTER_STAGE_SPECS)))
        if all_stages
        else informative_stage_indices(funnels)
    )
    headers, rows, alignments = build_table(funnels, view, stage_indices)
    rendered_table = render_table(headers, rows, alignments, table_format)

    if output_file is None:
        typer.echo(rendered_table)
        return

    output_file.write_text(rendered_table + "\n", encoding="utf-8")
    typer.echo(
        f"Wrote {table_format.value} {view.value} table for {len(funnels)} language(s) to {output_file}"
    )


if __name__ == "__main__":
    typer.run(main)
