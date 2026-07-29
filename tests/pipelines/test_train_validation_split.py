from types import SimpleNamespace

import pytest
from datatrove.data import Document

from wikipedia_processing.pipelines.train_validation_split import (
    TrainValidationSplitAnnotator,
    drop_split_column_writer_adapter,
)


def _run(annotator: TrainValidationSplitAnnotator, docs: list[Document], world_size: int = 1) -> list[Document]:
    return list(annotator.run(docs, world_size=world_size))  # ty: ignore[invalid-return-type, invalid-argument-type]


@pytest.mark.parametrize(
    ("validation_percentage", "expected"),
    [(100, True), (0, False)],
    ids=["100-percent", "0-percent"],
)
def test_is_validation_candidate_at_percentage_extremes(validation_percentage: int, expected: bool) -> None:
    annotator = TrainValidationSplitAnnotator(
        validation_percentage=validation_percentage, max_validation_documents=10, split_hash_metadata_key="page_id"
    )
    assert annotator._is_validation_candidate(1171348) is expected


@pytest.mark.parametrize(
    ("validation_percentage", "expected_split"),
    [(100, "validation"), (0, "train")],
    ids=["100-percent", "0-percent"],
)
def test_run_assigns_split_at_percentage_extremes(validation_percentage: int, expected_split: str) -> None:
    annotator = TrainValidationSplitAnnotator(
        validation_percentage=validation_percentage, max_validation_documents=10, split_hash_metadata_key="page_id"
    )
    docs = [Document(text="x", id=str(page_id), metadata={"page_id": page_id}) for page_id in range(5)]
    results = _run(annotator, docs)

    assert [doc.metadata["split"] for doc in results] == [expected_split] * 5
    assert annotator.stats["validation documents"].total == (5 if expected_split == "validation" else 0)


def test_run_caps_validation_documents_regardless_of_percentage() -> None:
    # 100% would put every document in validation, but the cap should win.
    annotator = TrainValidationSplitAnnotator(
        validation_percentage=100, max_validation_documents=2, split_hash_metadata_key="page_id"
    )
    docs = [Document(text="x", id=str(page_id), metadata={"page_id": page_id}) for page_id in range(5)]
    results = _run(annotator, docs)

    assert [doc.metadata["split"] for doc in results] == ["validation", "validation", "train", "train", "train"]
    assert annotator.stats["validation documents"].total == 2


def test_run_divides_cap_across_world_size() -> None:
    # With world_size=2 each rank should only get half of the global cap.
    annotator = TrainValidationSplitAnnotator(
        validation_percentage=100, max_validation_documents=4, split_hash_metadata_key="page_id"
    )
    docs = [Document(text="x", id=str(page_id), metadata={"page_id": page_id}) for page_id in range(5)]
    results = _run(annotator, docs, world_size=2)

    assert [doc.metadata["split"] for doc in results] == ["validation", "validation", "train", "train", "train"]
    assert annotator.stats["validation documents"].total == 2


def test_run_uses_configurable_split_hash_metadata_key() -> None:
    annotator = TrainValidationSplitAnnotator(
        validation_percentage=100, max_validation_documents=10, split_hash_metadata_key="custom_key"
    )
    docs = [Document(text="x", id="1", metadata={"custom_key": 42})]
    results = _run(annotator, docs)

    assert results[0].metadata["split"] == "validation"


def test_split_assignment_is_deterministic_for_the_same_metadata_value() -> None:
    first_annotator = TrainValidationSplitAnnotator(
        validation_percentage=50, max_validation_documents=100, split_hash_metadata_key="page_id"
    )
    second_annotator = TrainValidationSplitAnnotator(
        validation_percentage=50, max_validation_documents=100, split_hash_metadata_key="page_id"
    )
    docs = [Document(text="x", id=str(page_id), metadata={"page_id": page_id}) for page_id in range(50)]

    first_results = _run(first_annotator, docs)
    second_results = _run(second_annotator, docs)

    assert [doc.metadata["split"] for doc in first_results] == [
        doc.metadata["split"] for doc in second_results
    ]


@pytest.mark.parametrize("expand_metadata", [True, False], ids=["expand-metadata", "no-expand-metadata"])
def test_drop_split_column_writer_adapter_omits_split(expand_metadata: bool) -> None:
    writer = SimpleNamespace(expand_metadata=expand_metadata, save_media_bytes=True)
    doc = Document(text="x", id="1", metadata={"page_id": 1, "split": "validation"})

    data = drop_split_column_writer_adapter(writer, doc)  # ty: ignore[invalid-argument-type]

    if expand_metadata:
        assert "split" not in data
        assert data["page_id"] == 1
    else:
        # split is only stripped when metadata is expanded into top-level
        # columns; otherwise it stays untouched inside the metadata dict.
        assert data["metadata"] == {"page_id": 1, "split": "validation"}
