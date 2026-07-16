from typing import Callable

from datasets import load_dataset
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.text import PUNCTUATION_SET, split_into_words
from datatrove.utils.typeshelper import Languages as DatatroveLanguages

from wikipedia_processing.utils import truncate_to_255_bytes


def is_in_lookup(page_id: int,
                 page_title: str,
                 lookup: dict[int, str],
                 use_title: bool) -> bool:
    """
    Check if page_id is in lookup dataset if so returns True else returns False.

    To note the lookup truncates the document and lookup title to 255 bytes this
    is because Wikipedia titles when saved in the GAFA dataset are truncated to
    255 bytes.

    Args:
        page_id: A page ID to lookup.
        page_title: A page title to lookup if use_title is True.
        lookup: A dictionary mapping page IDs to their corresponding titles.
        use_title: A boolean indicating whether to check the title as well,
            if the title does not match the lookup then it will return False.

    Returns:
        A boolean indicating whether the page_id is in the lookup dataset,
        True it is in the lookup dataset else False.

    Examples:
        >>> lookup = {1: "Cat"}
        >>> is_in_lookup(1, "Cat", lookup, use_title=True)
        True
        >>> is_in_lookup(1, "Dog", lookup, use_title=True)
        False
        >>> is_in_lookup(2, "Cat", lookup, use_title=True)
        False
        >>> is_in_lookup(1, "Dog", lookup, use_title=False)
        True
    """
    if page_id not in lookup:
        return False
    
    if use_title:
        # titles in the lookup are truncated to 255 bytes whereby a character
        # can be up to 4 bytes due to UTF-8 encoding
        title = truncate_to_255_bytes(page_title)
        lookup_title = truncate_to_255_bytes(lookup[page_id])
        if title != lookup_title:
            return False
    return True

def get_language_good_article_id_titles(wikipedia_language_code: str) -> dict[int, str]:
    """
    Given a Wikipedia language code it loads the good and featured Wikipedia article IDs
    from the ucrelnlp/wikipedia-ga-fa-ids dataset and converts them to a dictionary
    mapping page IDs to their corresponding titles.

    The returned dictionary only contains page IDs from good and featured Wikipedia articles
    for the given language.
    
    Args:
        wikipedia_language_code: A string representing the Wikipedia language code.

    Returns:
        A dictionary of good and featured Wikipedia article page IDs to their corresponding titles.

    Raises:
        KeyError: If the dataset contains more than one entry for the same page ID.
    """
    filter_dataset = load_dataset("ucrelnlp/wikipedia-ga-fa-ids", wikipedia_language_code, streaming=True)
    
    good_article_id_titles: dict[int, str] = {}
    for sample in filter_dataset["train"]:
        page_id = sample["page_id"]
        if page_id in good_article_id_titles:
            raise KeyError("Duplicate page id")
        good_article_id_titles[page_id] = sample["page_title"]
    return good_article_id_titles

def get_relevant_page_function(wikipedia_language_code: str, use_title: bool) -> Callable[[Document], bool]:
    """Build a predicate that checks whether a Document is a good/featured article.

    Loads the good/featured article ID-title lookup for the given language once,
    then returns a closure over that lookup so it can be reused to test many
    documents (e.g. as a DataTrove filter function) without reloading the dataset.

    Args:
        wikipedia_language_code: A string representing the Wikipedia language code.
        use_title: A boolean indicating whether the returned predicate should also
            check the document title against the lookup. If the title does not
            match the lookup then it will return False.

    Returns:
        A callable that takes a Document and returns True if its ``page_id``
        (and, if use_title is True, its ``title``) metadata match a good or
        featured Wikipedia article for the given language, else False.
    """
    page_id_to_title: dict[int, str] = get_language_good_article_id_titles(wikipedia_language_code)
    def is_relevant_page(article_data: Document) -> bool:
        page_id = article_data.metadata["page_id"]
        page_title = article_data.metadata["title"]

        return is_in_lookup(page_id, page_title, page_id_to_title, use_title)
    return is_relevant_page


class EmptyTextFilter(BaseFilter):
    """DataTrove filter that drops documents whose text is empty or whitespace-only.

    Attributes:
        name: Human-readable filter name shown in DataTrove pipeline stats.
    """

    name = "🗑 Empty text filter"

    def __init__(self, exclusion_writer: DiskWriter | None = None) -> None:
        """Initialize the filter.

        Args:
            exclusion_writer: Optional writer used to persist documents that
                are dropped by this filter. If None, dropped documents are
                not written anywhere.
        """
        if exclusion_writer is None:
            super().__init__()
        else:
            super().__init__(exclusion_writer=exclusion_writer)

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """Check whether a document's text contains any non-whitespace content.

        Args:
            doc: The document to test.

        Returns:
            True if doc.text has non-whitespace content. Otherwise a tuple of
            (False, "empty_text") identifying the drop reason.

        Examples:
            >>> from datatrove.data import Document
            >>> EmptyTextFilter().filter(Document(text="   ", id="1"))
            (False, 'empty_text')
            >>> EmptyTextFilter().filter(Document(text="hello world", id="1"))
            True
        """
        if not doc.text or not doc.text.strip():
            return False, "empty_text"
        return True

class MinWordsDocumentFilter(BaseFilter):
    """DataTrove filter that drops documents with too few (non-symbol) words.

    Words made up entirely of punctuation/symbol characters are ignored when
    counting towards the minimum.

    Punctuation symbols are determined by :attr:`~datatrove.utils.text.PUNCTUATION_SET`

    Attributes:
        name: Human-readable filter name shown in DataTrove pipeline stats.
    """

    name = "Minimum Words Document Filter"

    def __init__(
        self,
        min_words: int | None = 50,
        exclusion_writer: DiskWriter | None = None,
        language: str = DatatroveLanguages.english) -> None:
        """Initialize the filter.

        Args:
            min_words: The minimum number of non-symbol words a document must
                contain to pass the filter. If None, no document is dropped.
            exclusion_writer: Optional writer used to persist documents that
                are dropped by this filter. If None, dropped documents are
                not written anywhere.
            language: The DataTrove language used to tokenize doc.text into
                words.
        """
        if exclusion_writer is None:
            super().__init__()
        else:
            super().__init__(exclusion_writer=exclusion_writer)
        self.min_words = min_words
        self.language = language

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """Check whether a document meets the minimum non-symbol word count.

        Args:
            doc: The document to test.

        Returns:
            True if the number of non-symbol words in doc.text is at least
            min_words (or min_words is None). Otherwise a tuple of
            (False, "minimum_word_drop") identifying the drop reason.

        Examples:
            >>> from datatrove.data import Document
            >>> f = MinWordsDocumentFilter(min_words=5)
            >>> f.filter(Document(text="hello world", id="1"))
            (False, 'minimum_word_drop')
            >>> f.filter(Document(text="one two three four five six", id="1"))
            True
        """
        text = doc.text
        words = split_into_words(text, self.language)

        non_symbol_words = [w for w in words if any(ch not in PUNCTUATION_SET for ch in w)]
        n_non_symbol_words_words = len(non_symbol_words)

        # words < min_doc_words or words > max_doc_words
        if self.min_words and n_non_symbol_words_words < self.min_words:
            return False, "minimum_word_drop"
        return True

class SimpleURLFilter(BaseFilter):
    """DataTrove filter that drops documents whose URL is in a blocklist.

    A document with no URL metadata (i.e. the configured metadata attribute is
    absent) is always kept.

    Attributes:
        name: Human-readable filter name shown in DataTrove pipeline stats.
    """

    name = "Simple URL Filter"

    def __init__(self,
                 urls_to_filter: set[str],
                 exclusion_writer: DiskWriter | None = None,
                 meta_data_attribute: str = "url") -> None:
        """Initialize the filter.

        Args:
            urls_to_filter: A set of URLs that should cause a document to be
                dropped.
            exclusion_writer: Optional writer used to persist documents that
                are dropped by this filter. If None, dropped documents are
                not written anywhere.
            meta_data_attribute: The key in doc.metadata holding the URL to
                check against urls_to_filter.
        """
        if exclusion_writer is None:
            super().__init__()
        else:
            super().__init__(exclusion_writer=exclusion_writer)
        self.urls_to_filter = urls_to_filter
        self.meta_data_attribute = meta_data_attribute

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """Check whether a document's URL is absent from the blocklist.

        Args:
            doc: The document to test.

        Returns:
            True if doc.metadata has no URL, or the URL is not in
            urls_to_filter. Otherwise a tuple of (False, "filtered_url")
            identifying the drop reason.

        Examples:
            >>> from datatrove.data import Document
            >>> f = SimpleURLFilter(urls_to_filter={"http://example.com/test"})
            >>> f.filter(Document(text="x", id="1", metadata={"url": "http://example.com/test"}))
            (False, 'filtered_url')
            >>> f.filter(Document(text="x", id="1", metadata={"url": "http://example.com/ok"}))
            True
        """
        url = doc.metadata.get(self.meta_data_attribute)
        if url is None:
            return True
        if url in self.urls_to_filter:
            return False, "filtered_url"
        return True