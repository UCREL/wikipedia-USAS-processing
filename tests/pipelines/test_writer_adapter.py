from types import SimpleNamespace

import pytest
from datatrove.data import Document

from wikipedia_processing.pipelines.writer_adapter import (
    get_metadata_whitelist_writer_adapter,
)

_FULL_METADATA = {
    "page_id": 1,
    "title": "Cat",
    "url": "https://en.wikipedia.org/wiki/Cat",
    "start_end_sentence_character_indexes": [[0, 3]],
    "tokens": [["Cat"]],
    "tags": [[["Z2"]]],
    "other_tags": [[[]]],
    "mwes": [["0"]],
    "split": "validation",
    "dump": "CC-MAIN-2024",
    "n_words": 1,
}
_KEYS_TO_KEEP = frozenset({
    "page_id",
    "title",
    "url",
    "start_end_sentence_character_indexes",
    "tokens",
    "tags",
    "other_tags",
    "mwes",
})


@pytest.mark.parametrize("expand_metadata", [True, False], ids=["expand-metadata", "no-expand-metadata"])
def test_metadata_whitelist_writer_adapter_restricts_metadata(expand_metadata: bool) -> None:
    adapter = get_metadata_whitelist_writer_adapter(keys_to_keep=_KEYS_TO_KEEP)
    writer = SimpleNamespace(expand_metadata=expand_metadata, save_media_bytes=True)
    doc = Document(text="x", id="1", metadata=dict(_FULL_METADATA))

    data = adapter(writer, doc)  # ty: ignore[invalid-argument-type]

    if expand_metadata:
        assert data["page_id"] == 1
        assert data["title"] == "Cat"
        assert data["url"] == "https://en.wikipedia.org/wiki/Cat"
        assert data["tokens"] == [["Cat"]]
        for dropped_key in ("split", "dump", "n_words"):
            assert dropped_key not in data
    else:
        # Metadata is only restricted when expanded into top-level columns;
        # otherwise it stays untouched inside the nested "metadata" dict.
        assert data["metadata"] == _FULL_METADATA


def test_different_keys_to_keep_produce_independent_adapters() -> None:
    url_only_adapter = get_metadata_whitelist_writer_adapter(keys_to_keep=frozenset({"url"}))
    page_id_only_adapter = get_metadata_whitelist_writer_adapter(keys_to_keep=frozenset({"page_id"}))
    writer = SimpleNamespace(expand_metadata=True, save_media_bytes=True)
    doc = Document(text="x", id="1", metadata={"page_id": 1, "url": "u"})

    assert "page_id" not in url_only_adapter(writer, doc)  # ty: ignore[invalid-argument-type]
    assert "url" not in page_id_only_adapter(writer, doc)  # ty: ignore[invalid-argument-type]
