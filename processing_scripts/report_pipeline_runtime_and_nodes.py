"""Summarise per-language pipeline wall-clock time and peak Slurm node usage from a ``log_data`` folder.

The DataTrove pipeline writes one sub-directory per language under ``log_data``
(plus a ``driver`` folder for the launch scripts). Each language folder holds one
sub-directory per pipeline stage, and every stage records:

* ``<stage>/logs/task_*.log`` -- per-task loguru logs whose lines start with a
  ``YYYY-MM-DD HH:MM:SS.mmm`` timestamp. The span between the earliest and latest
  timestamp across every stage is the pipeline's wall-clock runtime (it includes
  the Slurm queue gaps between dependent stages, not just compute time).
* ``<stage>/launch_script.slurm`` -- the submitted sbatch script. Its
  ``#SBATCH --array=0-(N-1)%M`` directive fixes how many single-core array tasks
  the stage may run at once. Each array task asks for ``--nodes=1``, so the
  largest array width across all stages is the most compute nodes the language
  can occupy in parallel (the minimum is always 1 -- the stages form a
  dependency chain and every task is an independent single-core job).

This script reports the language name, that peak node count, and the total
wall-clock time as a Markdown table (default) or a LaTeX ``booktabs`` table.

Example:
    Print the Markdown table for the checked-out logs::

        $ uv run processing_scripts/report_pipeline_runtime_and_nodes.py ./log_data

    Write a LaTeX table, ordered by descending node count::

        $ uv run processing_scripts/report_pipeline_runtime_and_nodes.py ./log_data \\
            --format latex --sort-by nodes --output-file table.tex
"""

import dataclasses
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from wikipedia_processing.utils import (
    discover_log_data_language_directories,
    language_display_name,
)

LOG_TIMESTAMP_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})")
LOG_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
SLURM_ARRAY_PATTERN = re.compile(r"--array=(\d+)(?:-(\d+))?(?:%(\d+))?")

TABLE_HEADERS = ("Language", "Max nodes", "Total wall-clock")
_COLUMN_ALIGNMENTS = ("left", "right", "left")


class TableFormat(str, Enum):
    """Supported rendering formats for the summary table."""

    MARKDOWN = "markdown"
    LATEX = "latex"


class SortKey(str, Enum):
    """Supported row orderings for the summary table."""

    RUNTIME = "runtime"
    NODES = "nodes"
    LANGUAGE = "language"


@dataclasses.dataclass(frozen=True)
class LanguagePipelineSummary:
    """Runtime and node-usage summary for a single language's pipeline run.

    Attributes:
        language_code: Wikipedia language code (the ``log_data`` sub-directory
            name), e.g. ``"da"``.
        language_name: Human-readable language name, e.g. ``"Danish"``.
        maximum_node_count: Largest Slurm array width across the language's
            pipeline stages, i.e. the most compute nodes it can use in parallel.
            ``None`` when no ``launch_script.slurm`` files were found.
        wall_clock_seconds: Seconds between the earliest and latest task-log
            timestamp across every stage. ``None`` when no task logs were found.
    """

    language_code: str
    language_name: str
    maximum_node_count: int | None
    wall_clock_seconds: float | None


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a compact ``Hh Mm Ss`` string.

    Leading zero-value units are dropped, except that the minutes field is kept
    whenever an hours field is present. The seconds field is always shown.

    Args:
        seconds: A non-negative duration in seconds.

    Returns:
        The rounded duration, e.g. ``"2h 49m 54s"`` or ``"23m 37s"``.

    Examples:
        >>> format_duration(10193.9)
        '2h 49m 54s'
        >>> format_duration(1416.6)
        '23m 37s'
        >>> format_duration(45.2)
        '45s'
        >>> format_duration(0)
        '0s'
    """
    total_seconds = round(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, whole_seconds = divmod(remainder, 60)

    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes or hours:
        parts.append(f"{minutes}m")
    parts.append(f"{whole_seconds}s")
    return " ".join(parts)


def parse_slurm_array_concurrency(directive: str) -> int:
    """Return the maximum concurrent task count described by an ``#SBATCH --array`` spec.

    A Slurm array directive of the form ``--array=<lo>-<hi>%<throttle>`` runs
    ``hi - lo + 1`` tasks, at most ``throttle`` of them at once. The bounded
    range and the ``%`` throttle are both optional.

    Args:
        directive: Text containing an ``--array=`` specification (typically a
            single ``#SBATCH`` line from a ``launch_script.slurm`` file).

    Returns:
        The number of array tasks that may run simultaneously.

    Raises:
        ValueError: If ``directive`` contains no ``--array=`` specification.

    Examples:
        >>> parse_slurm_array_concurrency("#SBATCH --array=0-14%15")
        15
        >>> parse_slurm_array_concurrency("#SBATCH --array=0-0%1")
        1
        >>> parse_slurm_array_concurrency("#SBATCH --array=0-9")
        10
        >>> parse_slurm_array_concurrency("#SBATCH --array=0-99%4")
        4
    """
    match = SLURM_ARRAY_PATTERN.search(directive)
    if match is None:
        raise ValueError(f"No --array= specification found in: {directive!r}")

    low_index = int(match.group(1))
    high_group = match.group(2)
    throttle_group = match.group(3)

    task_count = int(high_group) - low_index + 1 if high_group is not None else 1
    if throttle_group is not None:
        return min(task_count, int(throttle_group))
    return task_count


def collect_log_timestamps(language_directory: Path) -> list[datetime]:
    """Parse every timestamp from a language's per-task stage logs.

    Args:
        language_directory: A ``log_data/<language>`` folder.

    Returns:
        Every ``YYYY-MM-DD HH:MM:SS.mmm`` timestamp found in
        ``*/logs/task_*.log``, in file-then-line order (unsorted overall).

    Examples:
        >>> collect_log_timestamps(Path("log_data/da"))  # doctest: +SKIP
        [datetime.datetime(2026, 8, 24, 15, 50, 19, 449000), ...]
    """
    timestamps: list[datetime] = []
    for log_file in sorted(language_directory.glob("*/logs/task_*.log")):
        log_text = log_file.read_text(encoding="utf-8", errors="replace")
        for match in LOG_TIMESTAMP_PATTERN.finditer(log_text):
            timestamps.append(datetime.strptime(match.group(1), LOG_TIMESTAMP_FORMAT))
    return timestamps


def pipeline_wall_clock_seconds(language_directory: Path) -> float | None:
    """Return the wall-clock span of a language's whole pipeline run.

    Args:
        language_directory: A ``log_data/<language>`` folder.

    Returns:
        Seconds between the earliest and latest task-log timestamp across every
        stage, or ``None`` when the language has no task logs.

    Examples:
        >>> pipeline_wall_clock_seconds(Path("log_data/da"))  # doctest: +SKIP
        2005.87
    """
    timestamps = collect_log_timestamps(language_directory)
    if not timestamps:
        return None
    return (max(timestamps) - min(timestamps)).total_seconds()


def maximum_node_count(language_directory: Path) -> int | None:
    """Return the largest Slurm array width across a language's pipeline stages.

    Each stage's ``launch_script.slurm`` requests ``--nodes=1`` per array task,
    so the widest array is the most compute nodes the language can occupy at
    once.

    Args:
        language_directory: A ``log_data/<language>`` folder.

    Returns:
        The peak concurrent task count, or ``None`` when the language has no
        ``launch_script.slurm`` files.

    Examples:
        >>> maximum_node_count(Path("log_data/en"))  # doctest: +SKIP
        15
    """
    concurrencies: list[int] = []
    for launch_script in sorted(language_directory.glob("*/launch_script.slurm")):
        script_text = launch_script.read_text(encoding="utf-8", errors="replace")
        if SLURM_ARRAY_PATTERN.search(script_text) is not None:
            concurrencies.append(parse_slurm_array_concurrency(script_text))
    return max(concurrencies) if concurrencies else None


def summarise_language(language_directory: Path) -> LanguagePipelineSummary:
    """Build the :class:`LanguagePipelineSummary` for one language folder.

    Args:
        language_directory: A ``log_data/<language>`` folder.

    Returns:
        The language's runtime and peak node-count summary.

    Examples:
        >>> summarise_language(Path("log_data/da"))  # doctest: +SKIP
        LanguagePipelineSummary(language_code='da', language_name='Danish', ...)
    """
    language_code = language_directory.name
    return LanguagePipelineSummary(
        language_code=language_code,
        language_name=language_display_name(language_code),
        maximum_node_count=maximum_node_count(language_directory),
        wall_clock_seconds=pipeline_wall_clock_seconds(language_directory),
    )


def sort_summaries(
    summaries: list[LanguagePipelineSummary], sort_by: SortKey
) -> list[LanguagePipelineSummary]:
    """Return ``summaries`` ordered according to ``sort_by``.

    Args:
        summaries: The per-language summaries to order.
        sort_by: ``RUNTIME`` and ``NODES`` sort descending (missing values last);
            ``LANGUAGE`` sorts ascending by language name.

    Returns:
        A new, ordered list. The input list is left unchanged.

    Examples:
        >>> a = LanguagePipelineSummary("da", "Danish", 2, 2005.0)
        >>> b = LanguagePipelineSummary("en", "English", 15, 10193.0)
        >>> [s.language_code for s in sort_summaries([a, b], SortKey.RUNTIME)]
        ['en', 'da']
        >>> [s.language_code for s in sort_summaries([b, a], SortKey.LANGUAGE)]
        ['da', 'en']
    """
    ordered = list(summaries)
    match sort_by:
        case SortKey.LANGUAGE:
            ordered.sort(key=lambda summary: summary.language_name)
        case SortKey.NODES:
            ordered.sort(
                key=lambda summary: (
                    summary.maximum_node_count or 0,
                    summary.wall_clock_seconds or 0.0,
                ),
                reverse=True,
            )
        case SortKey.RUNTIME:
            ordered.sort(
                key=lambda summary: summary.wall_clock_seconds or 0.0, reverse=True
            )
    return ordered


def _summary_cells(summary: LanguagePipelineSummary) -> tuple[str, str, str]:
    """Return the ``(language, max nodes, wall-clock)`` display strings for one row."""
    node_count = (
        "n/a"
        if summary.maximum_node_count is None
        else str(summary.maximum_node_count)
    )
    wall_clock = (
        "n/a"
        if summary.wall_clock_seconds is None
        else format_duration(summary.wall_clock_seconds)
    )
    return (summary.language_name, node_count, wall_clock)


def _column_widths(rows: list[tuple[str, str, str]]) -> list[int]:
    """Return the display width of each column, header included."""
    widths: list[int] = []
    for index, header in enumerate(TABLE_HEADERS):
        cell_widths = [len(row[index]) for row in rows]
        widths.append(max(len(header), *cell_widths) if cell_widths else len(header))
    return widths


def _pad(value: str, width: int, alignment: str) -> str:
    """Pad ``value`` to ``width`` using left or right alignment."""
    return value.rjust(width) if alignment == "right" else value.ljust(width)


def render_markdown_table(summaries: list[LanguagePipelineSummary]) -> str:
    """Render the per-language summary as a padded GitHub-flavoured Markdown table.

    Args:
        summaries: The per-language summaries, already in display order.

    Returns:
        The table as a single string, without a trailing newline.

    Examples:
        >>> summary = LanguagePipelineSummary("en", "English", 15, 10193.9)
        >>> print(render_markdown_table([summary]))
        | Language | Max nodes | Total wall-clock |
        | :------- | --------: | :--------------- |
        | English  |        15 | 2h 49m 54s       |
    """
    rows = [_summary_cells(summary) for summary in summaries]
    widths = _column_widths(rows)

    def render_row(cells: tuple[str, str, str]) -> str:
        padded = (
            _pad(cell, width, alignment)
            for cell, width, alignment in zip(cells, widths, _COLUMN_ALIGNMENTS)
        )
        return "| " + " | ".join(padded) + " |"

    separator_cells = []
    for width, alignment in zip(widths, _COLUMN_ALIGNMENTS):
        dashes = "-" * (max(width, 3) - 1)
        separator_cells.append(dashes + ":" if alignment == "right" else ":" + dashes)
    separator_row = "| " + " | ".join(separator_cells) + " |"

    lines = [render_row(TABLE_HEADERS), separator_row]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def render_latex_table(summaries: list[LanguagePipelineSummary]) -> str:
    r"""Render the per-language summary as a LaTeX ``booktabs`` ``tabular``.

    The output requires ``\usepackage{booktabs}`` in the document preamble.

    Args:
        summaries: The per-language summaries, already in display order.

    Returns:
        The ``tabular`` environment as a single string, without a trailing
        newline.

    Examples:
        >>> summary = LanguagePipelineSummary("en", "English", 15, 10193.9)
        >>> print(render_latex_table([summary]))
        \begin{tabular}{lrl}
        \toprule
        Language & Max nodes & Total wall-clock \\
        \midrule
        English & 15 & 2h 49m 54s \\
        \bottomrule
        \end{tabular}
    """
    lines = [
        r"\begin{tabular}{lrl}",
        r"\toprule",
        " & ".join(TABLE_HEADERS) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(_summary_cells(summary)) + r" \\" for summary in summaries)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def render_table(
    summaries: list[LanguagePipelineSummary], table_format: TableFormat
) -> str:
    """Render ``summaries`` in the requested :class:`TableFormat`.

    Args:
        summaries: The per-language summaries, already in display order.
        table_format: Which renderer to use.

    Returns:
        The rendered table, without a trailing newline.

    Examples:
        >>> summary = LanguagePipelineSummary("en", "English", 15, 10193.9)
        >>> render_table([summary], TableFormat.MARKDOWN).splitlines()[0]
        '| Language | Max nodes | Total wall-clock |'
    """
    match table_format:
        case TableFormat.MARKDOWN:
            return render_markdown_table(summaries)
        case TableFormat.LATEX:
            return render_latex_table(summaries)


def main(
    log_data_directory: Annotated[
        Path,
        typer.Argument(
            help="Path to the pipeline log_data folder (one sub-directory per language).",
        ),
    ] = Path("log_data"),
    table_format: Annotated[
        TableFormat,
        typer.Option("-f", "--format", help="Output table format."),
    ] = TableFormat.MARKDOWN,
    sort_by: Annotated[
        SortKey,
        typer.Option(
            "-s",
            "--sort-by",
            help="Row ordering: descending runtime, descending node count, or language name.",
        ),
    ] = SortKey.LANGUAGE,
    output_file: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output-file",
            help="Write the table to this file instead of stdout.",
        ),
    ] = None,
) -> None:
    """Summarise per-language pipeline wall-clock time and peak Slurm node usage.

    Scans ``log_data_directory`` for per-language folders, derives each
    language's total wall-clock runtime (from the ``*/logs/task_*.log``
    timestamps) and its peak parallel node count (from the widest
    ``#SBATCH --array`` directive across its stages), then prints a table of
    language name, max nodes, and total wall-clock.

    Args:
        log_data_directory: The pipeline ``log_data`` folder to scan.
        table_format: ``markdown`` (default) or ``latex``.
        sort_by: Row ordering (``runtime``, ``nodes``, or ``language``).
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

    summaries = sort_summaries(
        [summarise_language(directory) for directory in language_directories],
        sort_by,
    )
    rendered_table = render_table(summaries, table_format)

    if output_file is None:
        typer.echo(rendered_table)
        return

    output_file.write_text(rendered_table + "\n", encoding="utf-8")
    typer.echo(
        f"Wrote {table_format.value} table for {len(summaries)} language(s) to {output_file}"
    )


if __name__ == "__main__":
    typer.run(main)
