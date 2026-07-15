"""
A module of Datatrove LambdaFilter functions 
(https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/filters/lambda_filter.py) 
whereby each has a corresponding function signature;

Callable[[Any], Callable[[datatrove.data.Document], bool]]

The bool of the filter when True indicates that the Document should be kept.
"""

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

    Args:
        page_id: A page ID to lookup.
        page_title: A page title to lookup if use_title is True.
        lookup: A dictionary mapping page IDs to their corresponding titles.
        use_title: A boolean indicating whether to check the title as well,
            if the title does not match the lookup then it will return False.

    Returns:
        A boolean indicating whether the page_id is in the lookup dataset,
        True it is in the lookup dataset else False.
    """
    if page_id not in lookup:
        return False
    
    if use_title:
        # titles in the lookup are truncated to 255 bytes whereby a character
        # can be up to 4 bytes due to UTF-8 encoding
        title = truncate_to_255_bytes(page_title)
        if title != lookup[page_id]:
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
    page_id_to_title: dict[int, str] = get_language_good_article_id_titles(wikipedia_language_code)
    def is_relevant_page(article_data: Document) -> bool:
        page_id = article_data.metadata["page_id"]
        page_title = article_data.metadata["title"]

        # To remove
        if page_id == 1171348:
            return True

        return is_in_lookup(page_id, page_title, page_id_to_title, use_title)
    return is_relevant_page


class EmptyTextFilter(BaseFilter):
    name = "🗑 Empty text filter"

    def __init__(self, exclusion_writer: DiskWriter | None = None):
        if exclusion_writer is None:
            super().__init__()
        else:
            super().__init__(exclusion_writer=exclusion_writer)

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        
        if not doc.text or not doc.text.strip():
            return False, "empty_text"
        return True

class MinWordsDocumentFilter(BaseFilter):
    name = "Minimum Words Document Filter"

    def __init__(
        self,
        min_words: int | None = 50,
        exclusion_writer: DiskWriter | None = None,
        language: str = DatatroveLanguages.english):
        if exclusion_writer is None:
            super().__init__()
        else:
            super().__init__(exclusion_writer=exclusion_writer)
        self.min_words = min_words
        self.language = language

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """

        Args:
            doc: Applies the heuristics rules to decide if a document should be REMOVED


        Returns: False if sample.text does not pass any of the heuristic tests

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
    name = "Simple URL Filter"

    def __init__(self,
                 urls_to_filter: set[str],
                 exclusion_writer: DiskWriter | None = None,
                 meta_data_attribute: str = "url") -> None:
        if exclusion_writer is None:
            super().__init__()
        else:
            super().__init__(exclusion_writer=exclusion_writer)
        self.urls_to_filter = urls_to_filter
        self.meta_data_attribute = meta_data_attribute

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        url = doc.metadata.get(self.meta_data_attribute)
        if url is None:
            return True
        if url in self.urls_to_filter:
            return False, "filtered_url"
        return True