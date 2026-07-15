from mistune.renderers.markdown import MarkdownRenderer


class FineWikiPlainTextRenderer(MarkdownRenderer):
    """
    Renders the markdown found in the HuggingFace fine-wiki dataset
    (https://huggingface.co/datasets/HuggingFaceFW/finewiki) as plain text.

    In addition it removes tables, code, and if they do appear alternative text
    for images. 

    The Math (math equations) is not removed as this is rendered within the dataset through the
    LaTeX tags `displaystyle` and `textstyle`. The Math text is not markdown formatted
    hence why it is not removed by this renderer.
    """

    NAME = "fine_wiki_plain_text"

    # --- Block-level elements ---

    def paragraph(self, token, state):
        children = self.render_children(token, state)
        return children + "\n\n"

    def heading(self, token, state):
        children = self.render_children(token, state)
        return children + "\n\n"

    def blank_line(self, token, state):
        return "\n"

    def thematic_break(self, token, state):
        return "\n"

    def block_code(self, token, state):
        return ""

    def block_quote(self, token, state):
        return self.render_children(token, state)

    def list(self, token, state):
        return self.render_children(token, state)

    def list_item(self, token, state):
        return self.render_children(token, state)

    def block_html(self, token, state):
        return ""  # drop raw HTML blocks

    # Drop tables entirely
    def table(self, token, state):
        return ""

    # Drop block math entirely (e.g. $$...$$)
    def block_math(self, token, state):
        return ""

    def block_text(self, token, state):
        return self.render_children(token, state)

    # --- Inline elements ---

    def text(self, token, state):
        return token["raw"]

    def strong(self, token, state):
        return self.render_children(token, state)

    def emphasis(self, token, state):
        return self.render_children(token, state)

    def codespan(self, token, state):
        return token["raw"] # These are not code spans they are normally just some form of formatted text

    def linebreak(self, token, state):
        return "\n"

    def softlinebreak(self, token, state):
        return " "
    
    def softbreak(self, token, state):
        return " "

    def link(self, token, state):
        # Keep the link label text, drop the URL
        return self.render_children(token, state)

    def image(self, token, state):
        # remove the alt text
        return ""

    def inline_html(self, token, state):
        # keep inline HTML as it is not HTML but rather tokens
        # This is when parsing the HuggingFace Fine Wiki dataset:
        # https://huggingface.co/datasets/HuggingFaceFW/finewiki
        return token["raw"]

    def inline_math(self, token, state):
        # Keep inline math as it is not math but rather text that is between two $
        # symbols.
        return token["raw"]

    def strikethrough(self, token, state):
        return self.render_children(token, state)

    def abbr(self, token, state):
        # Could be useful in the future to perhaps change this so that it shows up like:
        # FULL TEXT (ABBR), e.g. World Wide Web Consortium (W3C)
        # At the moment it just renders the ABBR, e.g. W3C
        return self.render_children(token, state)


    def footnote_ref(self, token, state):
        return ""

    def footnote_item(self, token, state):
        return self.render_children(token, state)

    def footnotes(self, token, state):
        # Ideally this will be shown where the footnote is referenced but
        # that is not possible either because of the way this library parses the
        # content or the way the FineWiki dataset has already been processed.
        # For now we will render the footnotes where the footnote appears in the
        # text.
        return self.render_children(token, state)

    def task_list_item(self, token, state):
        return "\n" + self.render_children(token, state)

    def def_list_item(self, token, state):
        return "\n" + self.render_children(token, state)
    
    def def_list_head(self, token, state):
        return self.render_children(token, state)

    def def_list(self, token, state):
        return self.render_children(token, state)

    def mark(self, token, state):
        return self.render_children(token, state)

    def insert(self, token, state):
        return self.render_children(token, state)

    def inline_spoiler(self, token, state):
        return self.render_children(token, state)

    def block_spoiler(self, token, state):
        return self.render_children(token, state)