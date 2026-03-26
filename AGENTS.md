# AGENTS.md

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
**doctest-style examples**. See [`coding_style_format_example.py`](./coding_style_format_example.py)
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

### Dependency management

| Task                        | Command                      |
|-----------------------------|------------------------------|
| Add a runtime dependency    | `uv add <package>`           |
| Add a dev dependency        | `uv add --dev <package>`     |
| Remove a dependency         | `uv remove <package>`        |
| Run a script                | `uv run <script.py>`         |

### Linting and tests

Run these via `make` — do not invoke the underlying tools directly:

- `make lint` — runs the linter (ty + ruff)
- `make test` — runs the test suite (pytest + coverage)