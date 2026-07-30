"""Benchmark and plot spaCy Danish model throughput.

Compares ``da_core_news_lg`` against ``da_core_news_trf`` by timing how long
each model takes to process a fixed-length sentence repeated a varying
number of times, then renders the results as a comparison graph.

The fixed sentence length (21 tokens) and one of the benchmarked sentence
counts (362) are taken from the observed statistics of the processed Danish
Wikipedia corpus (see ``log_data/da/post_processing/stats/00000.json``):
a mean of 362.2 sentences per document and 7556.1 tokens per document, i.e.
an average of ~20.9 (~21) tokens per sentence.
"""

import time
from pathlib import Path
from typing import Annotated, Iterable

import matplotlib

matplotlib.use("Agg")

from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import typer
from rich.console import Console
from rich.table import Table

from wikipedia_processing.models_install import pip_install_model

# Observed from log_data/da/post_processing/stats/00000.json:
# "sentences" mean (documents' average sentence count) and
# "tokens" mean / "sentences" mean (average tokens per sentence).
AVERAGE_SENTENCES_PER_DOCUMENT = 362
AVERAGE_TOKENS_PER_SENTENCE = 21


class ModelNames(str, Enum):
    """Danish spaCy models compared by this benchmark."""

    da_core_news_lg = "da_core_news_lg"
    da_core_news_trf = "da_core_news_trf"


MODEL_WHEEL_URLS: dict[ModelNames, str] = {
    ModelNames.da_core_news_lg: "https://github.com/explosion/spacy-models/releases/download/da_core_news_lg-3.8.0/da_core_news_lg-3.8.0-py3-none-any.whl",
    ModelNames.da_core_news_trf: "https://github.com/explosion/spacy-models/releases/download/da_core_news_trf-3.8.0/da_core_news_trf-3.8.0-py3-none-any.whl",
}

# Fixed categorical slots (blue, orange) from the project's palette, assigned
# in this order so the two series never repaint if models are reordered.
MODEL_COLORS: dict[ModelNames, str] = {
    ModelNames.da_core_news_lg: "#2a78d6",
    ModelNames.da_core_news_trf: "#eb6834",
}


class OutputFormat(str, Enum):
    """How to present the benchmark results."""

    plot = "plot"
    table = "table"


def generate_sentences(token_length: int, number_sentences: int) -> Iterable[str]:
    """Yield `number_sentences` identical dummy sentences.

    Args:
        token_length: Number of tokens in each generated sentence.
        number_sentences: Number of sentences to yield.

    Yields:
        A sentence containing `token_length` space-separated "hello" tokens.

    Examples:
        >>> list(generate_sentences(2, 2))
        ['hello hello', 'hello hello']
    """
    sentence = " ".join(["hello"] * token_length)
    for _ in range(number_sentences):
        yield sentence


def build_sentence_counts(max_tokens: int, token_length: int, num_points: int, must_include: int) -> list[int]:
    """Build a log-spaced sweep of sentence counts, capped by a token budget.

    Args:
        max_tokens: Upper bound on `token_length * number_sentences` for any
            single benchmarked point.
        token_length: Fixed number of tokens per sentence.
        num_points: How many sentence-count values to generate.
        must_include: A sentence count that must appear in the result,
            regardless of the log-spaced sweep (e.g. a corpus-observed
            average).

    Returns:
        A sorted list of unique sentence counts, each satisfying
        `count * token_length <= max_tokens`, including `must_include`.

    Raises:
        ValueError: If `must_include` alone would exceed `max_tokens`.

    Examples:
        >>> build_sentence_counts(max_tokens=100, token_length=10, num_points=3, must_include=5)
        [1, 3, 5, 10]
    """
    max_sentences = max_tokens // token_length
    if must_include > max_sentences:
        raise ValueError(
            f"must_include ({must_include}) sentences at {token_length} tokens/sentence "
            f"needs {must_include * token_length} tokens, exceeding max_tokens ({max_tokens})"
        )

    sweep = np.geomspace(1, max_sentences, num=num_points).round().astype(int)
    counts = np.unique(np.append(sweep, must_include))
    return counts.tolist()


def time_model(nlp: spacy.Language, token_length: int, number_sentences: int) -> float:
    """Time how long a loaded spaCy pipeline takes to process generated sentences.

    Args:
        nlp: An already-loaded spaCy pipeline.
        token_length: Fixed number of tokens per sentence.
        number_sentences: Number of sentences to process.

    Returns:
        Elapsed wall-clock time in seconds.
    """
    start_time = time.perf_counter()
    for sentence in generate_sentences(token_length, number_sentences):
        nlp(sentence)
    return time.perf_counter() - start_time


def plot_results(
    sentence_counts: list[int],
    results: dict[ModelNames, list[float]],
    token_length: int,
    output: Path,
) -> None:
    """Render a line chart comparing per-model processing time across sentence counts.

    Args:
        sentence_counts: The x-axis values (number of sentences benchmarked).
        results: Mapping of model to its per-`sentence_counts` elapsed times.
        token_length: Fixed tokens-per-sentence used for the run, shown in the title.
        output: File path the figure is saved to.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)

    for model_name, timings in results.items():
        ax.plot(
            sentence_counts,
            timings,
            label=model_name.value,
            color=MODEL_COLORS[model_name],
            linewidth=2,
            marker="o",
            markersize=6,
        )

    if AVERAGE_SENTENCES_PER_DOCUMENT in sentence_counts:
        ax.axvline(
            AVERAGE_SENTENCES_PER_DOCUMENT,
            color="#898781",
            linestyle="--",
            linewidth=1,
        )
        ax.annotate(
            f"corpus average\n({AVERAGE_SENTENCES_PER_DOCUMENT} sentences/doc)",
            xy=(AVERAGE_SENTENCES_PER_DOCUMENT, ax.get_ylim()[1]),
            xytext=(5, -5),
            textcoords="offset points",
            va="top",
            fontsize=8,
            color="#52514e",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of sentences processed")
    ax.set_ylabel("Time taken (seconds)")
    ax.set_title(f"Danish spaCy model speed comparison ({token_length} tokens/sentence)")
    ax.grid(True, which="both", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def build_results_dataframe(
    sentence_counts: list[int],
    results: dict[ModelNames, list[float]],
    token_length: int,
) -> pd.DataFrame:
    """Build a tidy results table with one row per benchmarked sentence count.

    Args:
        sentence_counts: The row values (number of sentences benchmarked).
        results: Mapping of model to its per-`sentence_counts` elapsed times.
        token_length: Fixed tokens-per-sentence used for the run.

    Returns:
        A DataFrame with "Sentences", "Tokens", and one "<model> (s)" column
        per benchmarked model.

    Examples:
        >>> list(build_results_dataframe([1, 2], {ModelNames.da_core_news_lg: [0.1, 0.2]}, 21).columns)
        ['Sentences', 'Tokens', 'da_core_news_lg (s)']
    """
    data: dict[str, list[int] | list[float]] = {
        "Sentences": sentence_counts,
        "Tokens": [count * token_length for count in sentence_counts],
    }
    for model_name, timings in results.items():
        data[f"{model_name.value} (s)"] = timings
    return pd.DataFrame(data)


def print_results_table(results_df: pd.DataFrame, token_length: int) -> None:
    """Print a results DataFrame as a Rich console table.

    Args:
        results_df: The results table, as returned by `build_results_dataframe`.
        token_length: Fixed tokens-per-sentence used for the run, shown in the title.
    """
    table = Table(title=f"Danish spaCy model speed comparison ({token_length} tokens/sentence)")
    for column in results_df.columns:
        table.add_column(column, justify="right")

    for row in results_df.to_dict(orient="records"):
        sentences = row["Sentences"]
        marker = " *" if sentences == AVERAGE_SENTENCES_PER_DOCUMENT else ""
        formatted = [f"{sentences}{marker}", str(row["Tokens"])]
        formatted.extend(f"{value:.3f}" for column, value in row.items() if column not in ("Sentences", "Tokens"))
        table.add_row(*formatted)

    console = Console()
    console.print(table)
    console.print(f"* {AVERAGE_SENTENCES_PER_DOCUMENT} sentences/doc is the corpus average")


def export_latex_table(results_df: pd.DataFrame, output: Path) -> None:
    """Write a results DataFrame to a file as a LaTeX tabular environment.

    Args:
        results_df: The results table, as returned by `build_results_dataframe`.
        output: File path the LaTeX table is written to.
    """
    output.write_text(results_df.to_latex(index=False, float_format="%.3f"))


def main(
    token_length: Annotated[
        int,
        typer.Option(help="Fixed tokens per sentence (average tokens/sentence observed in the Danish Wikipedia corpus)."),
    ] = AVERAGE_TOKENS_PER_SENTENCE,
    max_tokens: Annotated[
        int,
        typer.Option(help="Upper bound on tokens processed (token_length * number_sentences) for any single benchmarked point."),
    ] = 100_000,
    num_points: Annotated[
        int,
        typer.Option(help="Number of log-spaced sentence-count values to benchmark."),
    ] = 8,
    output_format: Annotated[
        OutputFormat,
        typer.Option("--format", "-f", help="Present results as a plotted graph or a printed table."),
    ] = OutputFormat.plot,
    output: Annotated[
        Path,
        typer.Option(help="Path to write the comparison graph to (plot format only)."),
    ] = Path("speed_test_results.png"),
    latex_output: Annotated[
        Path | None,
        typer.Option(help="If set, also write the results table to this path as a LaTeX tabular, independent of --format."),
    ] = None,
    install_models: Annotated[
        bool,
        typer.Option(help="Pip install the two Danish spaCy models before benchmarking."),
    ] = True,
) -> None:
    """Benchmark and plot da_core_news_lg vs da_core_news_trf processing speed."""
    if install_models:
        for model_name in ModelNames:
            pip_install_model(MODEL_WHEEL_URLS[model_name], model_name.value)

    sentence_counts = build_sentence_counts(
        max_tokens=max_tokens,
        token_length=token_length,
        num_points=num_points,
        must_include=AVERAGE_SENTENCES_PER_DOCUMENT,
    )

    results: dict[ModelNames, list[float]] = {}
    for model_name in ModelNames:
        typer.echo(f"Loading {model_name.value}")
        nlp = spacy.load(model_name.value, exclude=["ner"])

        timings = []
        for number_sentences in sentence_counts:
            elapsed = time_model(nlp, token_length, number_sentences)
            total_tokens = number_sentences * token_length
            typer.echo(f"  {number_sentences} sentences ({total_tokens} tokens): {elapsed:.3f}s")
            timings.append(elapsed)
        results[model_name] = timings

    results_df = build_results_dataframe(sentence_counts, results, token_length)

    match output_format:
        case OutputFormat.plot:
            plot_results(sentence_counts, results, token_length, output)
            typer.echo(f"Wrote graph to {output}")
        case OutputFormat.table:
            print_results_table(results_df, token_length)

    if latex_output is not None:
        export_latex_table(results_df, latex_output)
        typer.echo(f"Wrote LaTeX table to {latex_output}")


if __name__ == "__main__":
    typer.run(main)
