# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture

This repo builds [DataTrove](https://github.com/huggingface/datatrove) pipelines that turn the
HuggingFace [`HuggingFaceFW/finewiki`](https://huggingface.co/datasets/HuggingFaceFW/finewiki)
dataset into a synthetic (silver-labelled) training corpus for USAS semantic tagging and
Multi-Word Expression (MWE) identification, restricted to Wikipedia "Good"/"Featured" articles,
across 8 languages (English, Dutch, Spanish, Danish, Italian, Portuguese, Chinese, Finnish).

### Package layout (`wikipedia_processing/`)

- `filters.py` — DataTrove-compatible filters/formatters: GA/FA article lookup, test-URL
  exclusion, family-tree/LaTeX table stripping, empty-text and min-word filters, plus the
  sentence-splitting and PyMUSAS tagging annotators.
- `markdown_renderer.py` — `FineWikiPlainTextRenderer`, a `mistune.MarkdownRenderer` subclass
  that converts FineWiki markdown to plain text (drops tables/code/images/HTML, keeps inline math).
- `executors.py` — `ExecutorBackend` (local/slurm choice), `SlurmExecutorSettings`, and
  `PipelineExecutorFactory`, which builds either a `LocalPipelineExecutor` or a
  `SlurmPipelineExecutor` per pipeline stage from a single shared backend config.
- `utils.py` — `get_language_information` (loads `data/languages.yaml`), `truncate_to_255_bytes`.
  Note: `load_page_meta_data_file` is an unimplemented stub (returns `{}`).
- `models_util.py` — builds language-specific spaCy pipelines: `get_language_tagger` (spaCy +
  PyMUSAS rule-based tagger) and `get_language_sentence_splitter`.
- `models_install.py` — Typer CLI to download/install the spaCy + PyMUSAS models per language.

### Pipeline orchestration

The pipeline is not run through the package — it's orchestrated in the root-level `test_local.py`
using `datatrove.executor.LocalPipelineExecutor` across 5 dependent stages:

1. Read JSONL → filter to GA/FA page IDs → exclude test-set URLs → strip family-tree
   tables/LaTeX lines → markdown → plaintext → empty-text filter → min-word filter → write +
   exact-dedup signature.
2. Find exact duplicates.
3. Filter exact dupes → write + MinHash signature.
4. MinHash bucket generation.
5. MinHash clustering → filter final dupes → sentence-split annotator → PyMUSAS tagging
   annotator (semantic tags + MWEs) → `JsonlWriter`.

Output records contain: `text`, `id`, `page_id`, `title`, `url`,
`start_end_sentence_character_indexes`, `tokens`, `tags`, `mwes` (see `README.md` for full field
semantics).

### Configuration

`data/languages.yaml` drives per-language metadata (`language`, `iso_639_3`, `wikipedia_code`,
`training` flag, `data_trove_language`) and is loaded via `utils.get_language_information`. It is
the one file under `data/` that is **not** gitignored (see `.gitignore`).

### Known quirks

- `tests/` exists but is currently empty, even though `pyproject.toml` configures
  `testpaths = ["tests"]` — there are no existing test patterns to follow yet.
- `pyproject.toml`'s `[tool.coverage.run] source = ["src"]` does not match the actual package
  location (`wikipedia_processing/`), so coverage reporting from `make test` is likely misleading.
- Root-level `test.py` is a manual demo script (prints PyMUSAS tagger output), not a pytest suite.
- There is no CI configuration (`.github/workflows`) in this repo.

## Python Guidelines

- Use `match`/`case` syntax instead of `if`/`elif`/`else` for pattern matching.
- Use modern type hints with built-in generics (`list`, `dict`) and the union pipe (`|`)
  operator. Do not use deprecated `typing` module aliases (`Optional`, `Union`, `Dict`, `List`).
- Write code compatible with strict static analysis. This project uses
  [ty](https://docs.astral.sh/ty/) — avoid `type: ignore` comments unless absolutely necessary.
- Use `pathlib.Path` for all filesystem operations instead of `os.path`.
- Follow PEP 8. Prefer f-strings, comprehensions, and context managers where they improve clarity.
- Prioritise readability — avoid deeply nested `if` statements or complex one-liner comprehensions.

## Docstrings

Document all public functions and classes using **Google-style docstrings** with
**doctest-style examples**. See @coding_style_format_example.py
for the full reference. A minimal example:
```python
def add(x: int, y: int) -> int:
    """Add two integers.

    Args:
        x: The first integer.
        y: The second integer.

    Returns:
        The sum of x and y.

    Raises:
        ValueError: If x is equal to 5.

    Examples:
        >>> add(2, 3)
        5
    """
    if x == 5:
        raise ValueError("x == 5")
    return x + y
```

## Development Environment

This project uses [`uv`](https://docs.astral.sh/uv/) for environment and dependency management.
**Never invoke `python` or `pip` directly** — always go through `uv`.

### Setup

```bash
uv sync --all-extras
```

### Dependency management

| Task                        | Command                      |
|-----------------------------|------------------------------|
| Add a runtime dependency    | `uv add <package>`           |
| Add a dev dependency        | `uv add --dev <package>`     |
| Remove a dependency         | `uv remove <package>`        |
| Run a script                | `uv run <script.py>`         |

### Linting and tests

Run these via `make` — do not invoke the underlying tools directly:

- `make lint` — `ruff check --fix-only`, then `ruff check`, then `ty check`, scoped to
  `wikipedia_processing tests coding_style_format_example.py`.
- `make test` — `uv run coverage run` (runs `pytest` under the hood via
  `[tool.coverage.run] command_line`) then `uv run coverage report`. See the `tests/` quirk above
  — this currently collects zero tests.

### Running the pipeline

```bash
uv run test_local.py <wikipedia_language_code> <input_dir> <output_dir> <logging_dir>
# e.g.
uv run test_local.py da ./data/wikipedia_pages ./local_da/ ./log_data/
```

### Installing language models

`models_install.py` downloads the spaCy + PyMUSAS models/lexicons needed for tagging:

```bash
uv run wikipedia_processing/models_install.py --all          # install every language
uv run wikipedia_processing/models_install.py -l English -l Dutch
uv run wikipedia_processing/models_install.py --describe     # list models without installing
```

## Context Management

Use the @.gitignore to understand which files should NOT be READ or edited.
