from datatrove.data import Document

from wikipedia_processing.pipelines.metadata_whitelist import MetadataWhitelistAnnotator


def _run(annotator: MetadataWhitelistAnnotator, docs: list[Document]) -> list[Document]:
    return list(annotator.run(docs))  # ty: ignore[invalid-return-type, invalid-argument-type]


def test_run_keeps_only_whitelisted_keys() -> None:
    annotator = MetadataWhitelistAnnotator(keys_to_keep=frozenset({"page_id", "title", "url"}))
    doc = Document(
        text="x",
        id="1",
        metadata={"page_id": 1, "title": "Cat", "url": "https://en.wikipedia.org/wiki/Cat", "dump": "CC-MAIN-2024"},
    )

    result = _run(annotator, [doc])[0]

    assert result.metadata == {"page_id": 1, "title": "Cat", "url": "https://en.wikipedia.org/wiki/Cat"}


def test_run_keeps_all_documents_but_removes_all_metadata() -> None:
    
    annotator = MetadataWhitelistAnnotator(keys_to_keep=frozenset())
    docs = [Document(text="x", id=str(i), metadata={"junk": i}) for i in range(3)]

    results = _run(annotator, docs)

    assert [doc.id for doc in results] == ["0", "1", "2"]
    assert all(doc.metadata == {} for doc in results)
