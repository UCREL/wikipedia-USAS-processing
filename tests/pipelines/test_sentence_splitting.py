from unittest.mock import Mock, patch

import pytest
from datatrove.data import Document

from wikipedia_processing.models_install import Languages
from wikipedia_processing.pipelines.sentence_splitting import SentenceSplitterAnnotator


def _run_with_fake_splitter(
    annotator: SentenceSplitterAnnotator, docs: list[Document], fake_splitter: Mock
) -> list[Document]:
    with patch(
        "wikipedia_processing.pipelines.sentence_splitting.get_language_sentence_splitter",
        return_value=fake_splitter,
    ):
        # Same `DocumentsPipeline` NewType limitation ty has with `run`'s
        # declared return type (see the ignore comment on `run` itself) --
        # a plain list is a valid `DocumentsPipeline` at runtime but doesn't
        # structurally match the NewType.
        return list(annotator.run(docs))  # ty: ignore[invalid-return-type, invalid-argument-type]


def test_sentence_splitter_annotator_init_invalid_language_raises() -> None:
    with pytest.raises(ValueError):
        SentenceSplitterAnnotator("not_a_real_language_code")


def test_sentence_splitter_annotator_init_valid_language() -> None:
    annotator = SentenceSplitterAnnotator("en")
    assert annotator.language is Languages.en
    # Model is lazily loaded, so it should not be set on construction.
    assert annotator._nlp is None


def test_load_model_lazy_loads_once() -> None:
    # The sentence splitter should be built once per annotator and reused
    # across repeated calls to _load_model, rather than rebuilt each time.
    fake_splitter = Mock()
    with patch(
        "wikipedia_processing.pipelines.sentence_splitting.get_language_sentence_splitter",
        return_value=fake_splitter,
    ) as mock_get_splitter:
        annotator = SentenceSplitterAnnotator("en")
        first = annotator._load_model()
        second = annotator._load_model()

    mock_get_splitter.assert_called_once_with(Languages.en)
    assert first is fake_splitter
    assert second is fake_splitter


@pytest.mark.parametrize(
    ("sentences", "expected_indexes"),
    [
        # Docstring-style example: two sentences.
        (
            [("Hello world.", (0, 12)), ("Bye.", (13, 17))],
            [(0, 12), (13, 17)],
        ),
        # A single sentence.
        ([("Just one sentence.", (0, 19))], [(0, 19)]),
        # No sentences at all, e.g. an empty document.
        ([], []),
    ],
    ids=["two-sentences", "single-sentence", "no-sentences"],
)
def test_run_annotates_document_with_sentence_indexes(
    sentences: list[tuple[str, tuple[int, int]]], expected_indexes: list[tuple[int, int]]
) -> None:
    fake_splitter = Mock(return_value=iter(sentences))
    annotator = SentenceSplitterAnnotator("en")
    doc = Document(text="irrelevant, splitter is faked", id="1")
    (result,) = _run_with_fake_splitter(annotator, [doc], fake_splitter)

    assert result is doc
    assert result.metadata["start_end_sentence_character_indexes"] == expected_indexes
    assert annotator.stats["sentences"].total == len(expected_indexes)
    assert annotator.stats["sentences"].n == 1


def test_run_multiple_documents_accumulates_stats() -> None:
    # Each document may have a different sentence count; the "sentences"
    # stat should accumulate the total across all documents while metadata
    # remains distinct per document.
    per_doc_sentences = [
        [("A.", (0, 2)), ("B.", (3, 5))],
        [("C.", (0, 2))],
    ]
    fake_splitter = Mock(side_effect=[iter(sentences) for sentences in per_doc_sentences])
    annotator = SentenceSplitterAnnotator("en")
    docs = [Document(text="x", id="1"), Document(text="y", id="2")]
    results = _run_with_fake_splitter(annotator, docs, fake_splitter)

    assert results[0].metadata["start_end_sentence_character_indexes"] == [(0, 2), (3, 5)]
    assert results[1].metadata["start_end_sentence_character_indexes"] == [(0, 2)]
    assert annotator.stats["sentences"].total == 3
    assert annotator.stats["sentences"].n == 2
