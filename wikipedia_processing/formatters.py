import re

import mistune
from datatrove.pipeline.formatters.base import BaseFormatter
from datatrove.utils.logging import logger as data_trove_logger

from wikipedia_processing.markdown_renderer import FineWikiPlainTextRenderer


class WikipediaMarkdownFormatter(BaseFormatter):
    """DataTrove formatter that converts FineWiki markdown into plain text.

    Wraps a `mistune` markdown parser configured with
    [`FineWikiPlainTextRenderer`][wikipedia_processing.markdown_renderer.FineWikiPlainTextRenderer]
    so that tables, code blocks, images, and HTML are dropped while inline math is
    preserved. Parsing errors are caught, logged, and reported via the
    `wiki_markdown_formatter_error` stat rather than raised.
    """

    name = "⬇️ Wiki Markdown Formatter"

    def __init__(self):
        """Initializes the mistune markdown parser used for formatting."""
        super().__init__()
        self.claude_markdown_parser =  mistune.create_markdown(
            renderer=FineWikiPlainTextRenderer(),
            plugins=["table", "math", "strikethrough", "abbr", "footnotes", "task_lists", "def_list", "mark", "insert", "spoiler"],  # register so tokens are parsed
        )

    def format(self, text: str) -> str:
        """Converts FineWiki markdown text to plain text.

        Args:
            text: The markdown text to be converted.

        Returns:
            The plain-text rendering of `text`, or an empty string if parsing
            failed.

        Examples:
            >>> WikipediaMarkdownFormatter().format("**bold** text")
            'bold text\\n'
        """
        try:
            # The parsed text always adds a new line at the end of the text if
            # the new line does not already exist
            parsed_text = self.claude_markdown_parser(text)
            if not isinstance(parsed_text, str):
                raise TypeError("The Wiki Markdown Formatter returned a non-string value")
            return parsed_text
        except Exception as e:
            self.stat_update("wiki_markdown_formatter_error", 1)
            data_trove_logger.warning(f"Wiki Markdown Formatter Error: {e}")
            return ""

class RemoveFamilyTreeTableFormatter(BaseFormatter):
    """DataTrove formatter that strips family-tree tables from markdown text.

    Family trees and other very large tables tend to render as lines with an
    unusually high number of pipe (`|`) characters, so any line whose pipe
    count exceeds `pipe_threshold` is dropped.
    """

    name = "🌳 Wiki Family Tree Removal"

    def __init__(self, pipe_threshold=40):
        """Initializes the formatter.

        Args:
            pipe_threshold: The maximum number of pipes allowed in a line
                before it is treated as a family-tree table and removed.
        """
        super().__init__()
        self.pipe_threshold = pipe_threshold

    @staticmethod
    def remove_family_tree_tables(markdown_text: str, pipe_threshold=40) -> str:
        """Removes lines that contain more than `pipe_threshold` pipes.

        This typically removes very large tables and family trees from the text.

        Args:
            markdown_text: The text to be processed.
            pipe_threshold: The maximum number of pipes allowed in a line.

        Returns:
            The remaining text after removing lines with more than pipe_threshold number of pipes.

        Examples:
            >>> RemoveFamilyTreeTableFormatter.remove_family_tree_tables(
            ...     "a\\n|1|2|3|\\nb", pipe_threshold=2
            ... )
            'a\\nb'
        """
        text_lines = markdown_text.split("\n")
        non_family_tree_text: list[str] = []

        for line in text_lines:
            if line.count("|") > pipe_threshold:
                continue
            non_family_tree_text.append(line)

        return "\n".join(non_family_tree_text)

    def format(self, text: str) -> str:
        """Removes family-tree tables from `text`.

        Args:
            text: The markdown text to be processed.

        Returns:
            `text` with any line containing more than `self.pipe_threshold`
            pipes removed.

        Examples:
            >>> RemoveFamilyTreeTableFormatter(pipe_threshold=2).format("a\\n|1|2|3|\\nb")
            'a\\nb'
        """
        return self.remove_family_tree_tables(text, self.pipe_threshold)


class RemoveLinesWithGivenLatexCommandsFormatter(BaseFormatter):
    """DataTrove formatter that strips lines containing given LaTeX commands.

    Any line whose braced LaTeX command (matched by `LATEX_CODE_PATTERN`, e.g.
    `{\\command ...}`) is a member of `latex_commands` is dropped entirely.

    Attributes:
        LATEX_CODE_PATTERN (`re.Pattern`): Matches a braced LaTeX command and
            its contents, e.g. `{\\command ...}`, capturing the command name.
    """

    name= "✂️ Latex Commands Removal"

    LATEX_CODE_PATTERN = re.compile(r"\{(\\[\S]+).*\}", re.DOTALL)

    def __init__(self, latex_commands: set[str]):
        """Initializes the formatter.

        Args:
            latex_commands: The LaTeX commands whose lines should be removed.

        Raises:
            ValueError: If `latex_commands` is empty.
        """
        super().__init__()
        if not latex_commands:
            raise ValueError("latex_commands cannot be empty")
        self.latex_commands = latex_commands

    @classmethod
    def remove_lines_with_given_latex_commands(cls, markdown_text: str, latex_commands: set[str]) -> str:
        """Removes the lines of text that contain any of the given latex commands.

        Args:
            markdown_text: The text to be processed.
            latex_commands: A set of latex commands to be removed.

        Returns:
            The remaining text after removing lines with any of the given latex commands.

        Examples:
            >>> RemoveLinesWithGivenLatexCommandsFormatter.remove_lines_with_given_latex_commands(
            ...     "keep\\n{\\\\pagebreak}\\nkeep2", {"\\\\pagebreak"}
            ... )
            'keep\\nkeep2'
        """
        text_lines = markdown_text.split("\n")
        non_latex_command_text: list[str] = []

        for line in text_lines:
            found_latex_commands = set(cls.LATEX_CODE_PATTERN.findall(line))
            if found_latex_commands and latex_commands.intersection(found_latex_commands):
                continue
            non_latex_command_text.append(line)

        return "\n".join(non_latex_command_text)

    def format(self, text: str) -> str:
        """Removes lines containing any of `self.latex_commands` from `text`.

        Args:
            text: The markdown text to be processed.

        Returns:
            `text` with any line containing one of `self.latex_commands` removed.

        Examples:
            >>> RemoveLinesWithGivenLatexCommandsFormatter({"\\\\pagebreak"}).format(
            ...     "keep\\n{\\\\pagebreak}\\nkeep2"
            ... )
            'keep\\nkeep2'
        """
        return self.remove_lines_with_given_latex_commands(text, self.latex_commands)