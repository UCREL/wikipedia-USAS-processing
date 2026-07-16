import re

import mistune
from datatrove.pipeline.formatters.base import BaseFormatter
from datatrove.utils.logging import logger as data_trove_logger

from wikipedia_processing.markdown_renderer import FineWikiPlainTextRenderer


class WikipediaMarkdownFormatter(BaseFormatter):

    name = "⬇️ Wiki Markdown Formatter"
    def __init__(self):
        super().__init__()
        self.claude_markdown_parser =  mistune.create_markdown(
            renderer=FineWikiPlainTextRenderer(),
            plugins=["table", "math", "strikethrough", "abbr", "footnotes", "task_lists", "def_list", "mark", "insert", "spoiler"],  # register so tokens are parsed
        )

    def format(self, text: str) -> str:
        try:
            parsed_text = self.claude_markdown_parser(text)
            if not isinstance(parsed_text, str):
                raise TypeError("The Wiki Markdown Formatter returned a non-string value")
            return parsed_text
        except Exception as e:
            self.stat_update("wiki_markdown_formatter_error", 1)
            data_trove_logger.warning(f"Wiki Markdown Formatter Error: {e}")
            return ""

class RemoveFamilyTreeTableFormatter(BaseFormatter):

    name = "🌳 Wiki Family Tree Removal"
    def __init__(self, pipe_threshold=40):
        super().__init__()
        self.pipe_threshold = pipe_threshold

    @staticmethod
    def remove_family_tree_tables(markdown_text: str, pipe_threshold=40) -> str:
        """
        Removes the lines of text that contain more than pipe_threshold number of pipes
        and returns the remaining text.

        This typically removes very large tables and family trees from the text.

        Args:
            markdown_text: The text to be processed.
            pipe_threshold: The maximum number of pipes allowed in a line.

        Returns:
            The remaining text after removing lines with more than pipe_threshold number of pipes.
        """
        text_lines = markdown_text.split("\n")
        non_family_tree_text: list[str] = []

        for line in text_lines:
            if line.count("|") > pipe_threshold:
                continue
            non_family_tree_text.append(line)

        return "\n".join(non_family_tree_text)

    def format(self, text: str) -> str:
        return self.remove_family_tree_tables(text, self.pipe_threshold)


class RemoveLinesWithGivenLatexCommandsFormatter(BaseFormatter):

    name= "✂️ Latex Commands Removal"

    LATEX_CODE_PATTERN = re.compile(r"\{(\\[\S]+).*\}", re.DOTALL)
    def __init__(self, latex_commands: set[str]):
        super().__init__()
        if not latex_commands:
            raise ValueError("latex_commands cannot be empty")
        self.latex_commands = latex_commands

    @classmethod
    def remove_lines_with_given_latex_commands(cls, markdown_text: str, latex_commands: set[str]) -> str:
        """
        Removes the lines of text that contain any of the given latex commands.

        Args:
            markdown_text: The text to be processed.
            latex_commands: A set of latex commands to be removed.

        Returns:
            The remaining text after removing lines with any of the given latex commands.
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
        return self.remove_lines_with_given_latex_commands(text, self.latex_commands)