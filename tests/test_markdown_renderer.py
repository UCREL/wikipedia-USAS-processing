import mistune
import pytest
from mistune.markdown import Markdown

from wikipedia_processing.markdown_renderer import FineWikiPlainTextRenderer


@pytest.fixture
def render() -> Markdown:
    return mistune.create_markdown(
        renderer=FineWikiPlainTextRenderer(),
        plugins=[
            "table",
            "math",
            "strikethrough",
            "abbr",
            "footnotes",
            "task_lists",
            "def_list",
            "mark",
            "insert",
            "spoiler",
        ],
    )


@pytest.mark.parametrize(
    ("markdown_text", "expected"),
    [
        # Empty and whitespace-only input render to a single trailing newline.
        ("", "\n"),
        ("   ", "\n"),
        # A single paragraph.
        ("hello world", "hello world\n"),
        # Two paragraphs separated by a blank line.
        ("para1\n\npara2", "para1\n\n\npara2\n"),
        # Headings keep their text, drop the `#` markers.
        ("# Heading", "Heading\n"),
        # Thematic breaks are dropped, leaving no visible content.
        ("---", "\n"),
        # Fenced code blocks are dropped entirely.
        ("```python\nprint(1)\n```", "\n"),
        # Indented code blocks are also dropped.
        ("    print(1)", "\n"),
        # Block quotes keep their inner text, drop the `>` marker.
        ("> quoted text", "quoted text\n"),
        # Unordered lists keep item text with a line break between items.
        ("- item1\n- item2", "item1\n item2\n"),
        # Ordered lists behave the same way, numbering is dropped.
        ("1. item1\n2. item2", "item1\n item2\n"),
        # Nested list items are flattened into the parent's text, each on its
        # own line.
        ("- item1\n  - nested1\n- item2", "item1\n nested1\n item2\n"),
        # Raw HTML blocks are dropped.
        ("<div>html block</div>", "\n"),
        # Tables are dropped entirely.
        ("| a | b |\n|---|---|\n| 1 | 2 |\n", "\n"),
        # Block math (standalone `$$...$$`) is dropped entirely.
        ("$$\nx^2\n$$", "\n"),
        # Inline math is kept, as it is plain text between `$` symbols
        # rather than real markdown/LaTeX.
        ("Inline $x^2$ math", "Inline x^2 math\n"),
        # Emphasis and strong text keep their contents, drop the markers.
        ("*em*", "em\n"),
        ("**bold**", "bold\n"),
        # Codespans are kept as plain text, not treated as code to drop.
        ("`code`", "code\n"),
        # Hard line breaks (trailing double-space or trailing backslash)
        # become a real newline.
        ("line1  \nline2", "line1\nline2\n"),
        ("line1\\\nline2", "line1\nline2\n"),
        # A soft line break (single newline) collapses to a single space.
        ("line1\nline2", "line1 line2\n"),
        # Link label text is kept, the URL is dropped.
        ("[link text](http://example.com)", "link text\n"),
        # Image alt text is dropped.
        ("![alt text](http://example.com/img.png)", "\n"),
        # Inline HTML is kept, as it is treated as plain tokens rather
        # than real HTML for this dataset.
        ("before <span>inline</span> after", "before <span>inline</span> after\n"),
        # Strikethrough text is kept, the `~~` markers are dropped.
        ("~~del~~", "del\n"),
        # Mark and insert text is kept, the `==`/`^^` markers are dropped.
        ("==marked==", "marked\n"),
        ("^^inserted^^", "inserted\n"),
        # Footnote references are dropped; footnote item text is kept
        # inline where the footnotes block appears.
        ("Text[^1]\n\n[^1]: Footnote text", "Text\nFootnote text\n"),
        # Task list items keep their text, the `[ ]`/`[x]` markers are dropped.
        ("- [ ] task1\n- [x] task2", "\n task1\n task2\n"),
        # Definition list terms and definitions are kept, markers dropped.
        # Term and definition share a line; a new term/definition starts a
        # new line.
        ("Term\n: Definition", "Term: Definition\n"),
        # Multiple terms each get their own line.
        ("Term1\n: Definition1\n\nTerm2\n: Definition2", "Term1: Definition1\n Term2: Definition2\n"),
        # Inline spoilers keep their text, the `>!`/`!<` markers are dropped.
        ("before >! hidden text !< after", "before hidden text after\n"),
        # Block spoilers keep their text, the `>!` marker is dropped.
        (">! this is spoiler\n>!\n>! the content", "this is spoiler\n\n\nthe content\n"),
    ],
    ids=[
        "empty-string",
        "whitespace-only",
        "single-paragraph",
        "two-paragraphs",
        "heading",
        "thematic-break-dropped",
        "fenced-code-block-dropped",
        "indented-code-block-dropped",
        "block-quote",
        "unordered-list",
        "ordered-list",
        "nested-list",
        "block-html-dropped",
        "table-dropped",
        "block-math-dropped",
        "inline-math-kept",
        "emphasis",
        "strong",
        "codespan-kept",
        "hard-linebreak-trailing-spaces",
        "hard-linebreak-trailing-backslash",
        "soft-linebreak-collapses-to-space",
        "link-label-kept-url-dropped",
        "image-dropped",
        "inline-html-kept",
        "strikethrough",
        "mark",
        "insert",
        "footnote",
        "task-list",
        "def-list-single-term",
        "def-list-multiple-terms",
        "inline-spoiler",
        "block-spoiler",
    ],
)
def test_fine_wiki_plain_text_renderer(render: Markdown, markdown_text: str, expected: str) -> None:
    assert render(markdown_text) == expected
