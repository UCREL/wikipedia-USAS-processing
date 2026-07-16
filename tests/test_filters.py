from unittest.mock import Mock, patch

import pytest
from datatrove.data import Document
from datatrove.pipeline.writers.disk_base import DiskWriter

from wikipedia_processing.filters import (
    EmptyTextFilter,
    MinWordsDocumentFilter,
    SimpleURLFilter,
    get_language_good_article_id_titles,
    get_relevant_page_function,
    is_in_lookup,
)


@pytest.mark.parametrize(
    ("page_id", "page_title", "lookup", "use_title", "expected"),
    [
        # page_id missing from the lookup entirely.
        (2, "Cat", {1: "Cat"}, True, False),
        # page_id present, use_title False so the title is never checked.
        (1, "Dog", {1: "Cat"}, False, True),
        # page_id present, title matches exactly.
        (1, "Cat", {1: "Cat"}, True, True),
        # page_id present, title does not match.
        (1, "Dog", {1: "Cat"}, True, False),
        # Empty lookup.
        (1, "Cat", {}, True, False),
        # Title longer than 255 bytes is truncated before comparison, so it
        # matches a lookup entry that stores the already-truncated form.
        (1, "字" * 90, {1: "字" * 85}, True, True),
        # Does not fail on the title as it is truncated to the first 255 characters
        (1, "a" * 255 + "c", {1: "a" * 255}, True, True),
        # Does not fail on the title as both the lookup title and the document title are truncated
        (1, "a" * 255 + "c", {1: "a" * 255 + "b"}, True, True),
    ],
    ids=[
        "page-id-missing",
        "use-title-false-skips-check",
        "title-matches",
        "title-mismatch",
        "empty-lookup",
        "title-truncated-to-match",
        "title-truncated-and-matches-document-title-mismatch",
        "title-truncated-and-matches-document-lookup-title-mismatch",
    ],
)
def test_is_in_lookup(
    page_id: int, page_title: str, lookup: dict[int, str], use_title: bool, expected: bool
) -> None:
    assert is_in_lookup(page_id, page_title, lookup, use_title) is expected


def test_get_language_good_article_id_titles_builds_dict() -> None:
    fake_dataset = {
        "train": [
            {"page_id": 1, "page_title": "Cat"},
            {"page_id": 2, "page_title": "Dog"},
        ]
    }
    with patch("wikipedia_processing.filters.load_dataset", return_value=fake_dataset) as mock_load:
        result = get_language_good_article_id_titles("en")

    mock_load.assert_called_once_with("ucrelnlp/wikipedia-ga-fa-ids", "en", streaming=True)
    assert result == {1: "Cat", 2: "Dog"}


def test_get_language_good_article_id_titles_empty_dataset() -> None:
    with patch("wikipedia_processing.filters.load_dataset", return_value={"train": []}):
        assert get_language_good_article_id_titles("en") == {}


def test_get_language_good_article_id_titles_duplicate_page_id_raises() -> None:
    fake_dataset = {
        "train": [
            {"page_id": 1, "page_title": "Cat"},
            {"page_id": 1, "page_title": "Cat (duplicate)"},
        ]
    }
    with (
        patch("wikipedia_processing.filters.load_dataset", return_value=fake_dataset),
        pytest.raises(KeyError),
    ):
        get_language_good_article_id_titles("en")


@pytest.mark.parametrize(
    ("page_id", "title", "use_title", "expected"),
    [
        # Known page, title matches, title checking enabled.
        (1, "Cat", True, True),
        # Known page, title differs, title checking enabled -> rejected.
        (1, "Cat (redirect)", True, False),
        # Known page, title differs, title checking disabled -> still accepted.
        (1, "Cat (redirect)", False, True),
        # Unknown page is always rejected, regardless of use_title.
        (99, "Cat", True, False),
    ],
    ids=[
        "known-page-title-matches",
        "known-page-title-mismatch-checked",
        "known-page-title-mismatch-ignored",
        "unknown-page",
    ],
)
def test_get_relevant_page_function(page_id: int, title: str, use_title: bool, expected: bool) -> None:
    with patch(
        "wikipedia_processing.filters.get_language_good_article_id_titles",
        return_value={1: "Cat"},
    ):
        is_relevant_page = get_relevant_page_function("en", use_title)

    doc = Document(text="x", id="1", metadata={"page_id": page_id, "title": title})
    assert is_relevant_page(doc) is expected


def test_get_relevant_page_function_loads_lookup_once() -> None:
    # The lookup dataset should be loaded once per call to
    # get_relevant_page_function, and reused across every call to the
    # returned predicate rather than being reloaded per document.
    with patch(
        "wikipedia_processing.filters.get_language_good_article_id_titles",
        return_value={1: "Cat"},
    ) as mock_lookup:
        is_relevant_page = get_relevant_page_function("en", use_title=True)
        is_relevant_page(Document(text="x", id="1", metadata={"page_id": 1, "title": "Cat"}))
        is_relevant_page(Document(text="y", id="2", metadata={"page_id": 1, "title": "Cat"}))

    mock_lookup.assert_called_once_with("en")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Whitespace-only text is treated as empty.
        ("   ", (False, "empty_text")),
        # Genuinely empty string.
        ("", (False, "empty_text")),
        # Non-empty text passes.
        ("hello world", True),
    ],
    ids=["whitespace-only", "empty-string", "non-empty-text"],
)
def test_empty_text_filter(text: str, expected: bool | tuple[bool, str]) -> None:
    doc = Document(text=text, id="1")
    assert EmptyTextFilter().filter(doc) == expected


@pytest.mark.parametrize(
    ("text", "min_words", "expected"),
    [
        # Below the minimum.
        ("hello world", 5, (False, "minimum_word_drop")),
        # Meets the minimum exactly (the check is strictly `<`, not `<=`).
        ("one two three four five", 5, True),
        # Punctuation-only tokens are not counted towards the minimum.
        ("Hello , world ! -- foo", 3, True),
        ("Hello , world ! -- foo", 4, (False, "minimum_word_drop")),
        # min_words=None disables the check entirely, even for empty text.
        ("", None, True),
    ],
    ids=[
        "below-minimum",
        "exactly-at-minimum",
        "punctuation-not-counted-passes",
        "punctuation-not-counted-fails",
        "min-words-none-disables-check",
    ],
)
def test_min_words_document_filter(
    text: str, min_words: int | None, expected: bool | tuple[bool, str]
) -> None:
    doc = Document(text=text, id="1")
    assert MinWordsDocumentFilter(min_words=min_words).filter(doc) == expected


@pytest.mark.parametrize(
    ("metadata", "urls_to_filter", "meta_data_attribute", "expected"),
    [
        # URL is in the blocklist.
        ({"url": "http://example.com/test"}, {"http://example.com/test"}, "url", (False, "filtered_url")),
        # URL is present but not in the blocklist.
        ({"url": "http://example.com/ok"}, {"http://example.com/test"}, "url", True),
        # Metadata attribute is missing entirely -> always kept.
        ({}, {"http://example.com/test"}, "url", True),
        # Empty blocklist -> always kept.
        ({"url": "http://example.com/test"}, set(), "url", True),
        # A non-default metadata attribute is used to look up the URL.
        (
            {"source_url": "http://example.com/test"},
            {"http://example.com/test"},
            "source_url",
            (False, "filtered_url"),
        ),
    ],
    ids=[
        "url-blocked",
        "url-not-blocked",
        "url-metadata-missing",
        "empty-blocklist",
        "custom-metadata-attribute",
    ],
)
def test_simple_url_filter(
    metadata: dict[str, str],
    urls_to_filter: set[str],
    meta_data_attribute: str,
    expected: bool | tuple[bool, str],
) -> None:
    doc = Document(text="x", id="1", metadata=metadata)
    filter_ = SimpleURLFilter(urls_to_filter, meta_data_attribute=meta_data_attribute)
    assert filter_.filter(doc) == expected


@pytest.mark.parametrize(
    "make_filter",
    [
        # Each filter's __init__ has a branch that forwards a provided
        # exclusion_writer to BaseFilter, distinct from the None default.
        lambda writer: EmptyTextFilter(exclusion_writer=writer),
        lambda writer: MinWordsDocumentFilter(exclusion_writer=writer),
        lambda writer: SimpleURLFilter(urls_to_filter=set(), exclusion_writer=writer),
    ],
    ids=["empty-text-filter", "min-words-document-filter", "simple-url-filter"],
)
def test_filter_stores_provided_exclusion_writer(make_filter) -> None:
    writer = Mock(spec=DiskWriter)
    assert make_filter(writer).exclusion_writer is writer
