from unittest.mock import Mock, patch

import pytest
from datatrove.utils.logging import logger as data_trove_logger

from wikipedia_processing.formatters import (
    RemoveFamilyTreeTableFormatter,
    RemoveLinesWithGivenLatexCommandsFormatter,
    WikipediaMarkdownFormatter,
)


@pytest.fixture
def markdown_formatter() -> WikipediaMarkdownFormatter:
    return WikipediaMarkdownFormatter()


@pytest.mark.parametrize(
    ("markdown_text", "expected"),
    [
        # Docstring example: bold emphasis is stripped, plain text kept.
        ("**bold** text", "bold text\n"),
        # Empty input still renders a trailing blank-line newline.
        ("", "\n"),
        # Tables are dropped entirely.
        ("| a | b |\n|---|---|\n| 1 | 2 |\n", "\n"),
        # Fenced code blocks are dropped entirely.
        ("```python\nprint(1)\n```\n", "\n"),
        # Image alt text is dropped, leaving nothing.
        ("![alt text](http://example.com/img.png)", "\n"),
        # Raw HTML blocks are dropped.
        ("<div>html block</div>", "\n"),
        # Inline math is preserved, as it is plain text between $ symbols
        # rather than real markdown/LaTeX.
        ("Inline $x^2$ math", "Inline x^2 math\n"),
        # Link URLs are dropped but the link label text is kept.
        ("[link text](http://example.com)", "link text\n"),
    ],
    ids=[
        "docstring-example-bold",
        "empty-string",
        "table-dropped",
        "code-block-dropped",
        "image-dropped",
        "html-block-dropped",
        "inline-math-preserved",
        "link-label-kept",
    ],
)
def test_wikipedia_markdown_formatter_format(
    markdown_formatter: WikipediaMarkdownFormatter, markdown_text: str, expected: str
) -> None:
    assert markdown_formatter.format(markdown_text) == expected


def test_wikipedia_markdown_formatter_parser_error_returns_empty_string(
    markdown_formatter: WikipediaMarkdownFormatter,
) -> None:
    # Any exception raised during parsing is caught, logged, and reported as
    # a stat rather than propagated.
    markdown_formatter.claude_markdown_parser = Mock(side_effect=RuntimeError("boom"))

    with patch.object(data_trove_logger, "warning") as mock_warning:
        result = markdown_formatter.format("text")

    assert result == ""
    assert markdown_formatter.stats["wiki_markdown_formatter_error"].n == 1
    mock_warning.assert_called_once()


def test_wikipedia_markdown_formatter_non_string_parser_result_returns_empty_string(
    markdown_formatter: WikipediaMarkdownFormatter,
) -> None:
    # A parser that returns a non-string value hits the same error path (the
    # internally-raised TypeError is caught just like a parsing exception).
    markdown_formatter.claude_markdown_parser = Mock(return_value=123)

    with patch.object(data_trove_logger, "warning") as mock_warning:
        assert markdown_formatter.format("text") == ""
        assert markdown_formatter.stats["wiki_markdown_formatter_error"].n == 1
    mock_warning.assert_called_once()


@pytest.mark.parametrize(
    ("markdown_text", "pipe_threshold", "expected"),
    [
        # Docstring example.
        ("a\n|1|2|3|\nb", 2, "a\nb"),
        # Pipe count exactly at the threshold is kept (the check is strictly `>`).
        ("|1|2|3|", 4, "|1|2|3|"),
        # Pipe count one over the threshold is removed.
        ("|1|2|3|", 3, ""),
        # No pipes at all, nothing removed regardless of threshold.
        ("plain text\nmore text", 0, "plain text\nmore text"),
        # Empty string stays empty.
        ("", 40, ""),
        # Multiple offending lines are all removed, surrounding lines kept.
        ("keep\n|1|2|3|\nkeep2\n|4|5|6|\nkeep3", 2, "keep\nkeep2\nkeep3"),
    ],
    ids=[
        "docstring-example",
        "pipe-count-equals-threshold-kept",
        "pipe-count-over-threshold-removed",
        "no-pipes",
        "empty-string",
        "multiple-offending-lines-removed",
    ],
)
def test_remove_family_tree_tables_static_method(
    markdown_text: str, pipe_threshold: int, expected: str
) -> None:
    assert (
        RemoveFamilyTreeTableFormatter.remove_family_tree_tables(markdown_text, pipe_threshold)
        == expected
    )


def test_remove_family_tree_table_formatter_format_default_threshold() -> None:
    formatter = RemoveFamilyTreeTableFormatter()
    assert formatter.pipe_threshold == 40


def test_remove_lines_with_given_latex_commands_formatter_init_raises_on_empty_set() -> None:
    with pytest.raises(ValueError):
        RemoveLinesWithGivenLatexCommandsFormatter(set())


@pytest.mark.parametrize(
    ("markdown_text", "latex_commands", "expected"),
    [
        # Docstring example.
        ("keep\n{\\pagebreak}\nkeep2", {"\\pagebreak"}, "keep\nkeep2"),
        # Empty string stays empty.
        ("", {"\\pagebreak"}, ""),
        # Line without braces at all is kept.
        ("plain text", {"\\pagebreak"}, "plain text"),
        # Braced content without a leading backslash doesn't match the
        # pattern, so the line is kept even though it looks bracketed.
        ("{plain text}", {"\\pagebreak"}, "{plain text}"),
        # A braced backslash command that isn't in the target set is kept.
        ("{\\other}", {"\\pagebreak"}, "{\\other}"),
        # LATEX_CODE_PATTERN is greedy and DOTALL, so two separate `{...}`
        # spans on one line collapse into a single match capturing only the
        # first command name -- the second command is invisible to the
        # intersection check, so the line survives even though \cmd2 is
        # textually present.
        ("{\\cmd1 foo} bar {\\cmd2 baz}", {"\\cmd2"}, "{\\cmd1 foo} bar {\\cmd2 baz}"),
        # ...but if the *first* command on the line matches, it is removed.
        ("{\\cmd1 foo} bar {\\cmd2 baz}", {"\\cmd1"}, ""),
        # Multiple target commands: a line matching any one of them is removed.
        ("{\\pagebreak}", {"\\pagebreak", "\\other"}, ""),
        # Multiple offending lines are all removed, surrounding lines kept.
        ("keep\n{\\pagebreak}\nkeep2\n{\\pagebreak}\nkeep3", {"\\pagebreak"}, "keep\nkeep2\nkeep3"),
    ],
    ids=[
        "docstring-example",
        "empty-string",
        "no-braces-kept",
        "braces-without-backslash-kept",
        "non-matching-command-kept",
        "second-command-on-line-invisible-to-match",
        "first-command-on-line-matches",
        "multiple-target-commands",
        "multiple-offending-lines-removed",
    ],
)
def test_remove_lines_with_given_latex_commands_class_method(
    markdown_text: str, latex_commands: set[str], expected: str
) -> None:
    assert (
        RemoveLinesWithGivenLatexCommandsFormatter.remove_lines_with_given_latex_commands(
            markdown_text, latex_commands
        )
        == expected
    )
