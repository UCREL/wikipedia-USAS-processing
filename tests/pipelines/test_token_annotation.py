from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from datatrove.data import Document

from wikipedia_processing.pipelines.token_annotation import TokenPyMUSASAnnotator


def _make_token(
    text: str,
    is_space: bool = False,
    pymusas_tags: list[str] | None = None,
    pymusas_mwe_indexes: list[tuple[int, int]] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        is_space=is_space,
        _=SimpleNamespace(
            pymusas_tags=pymusas_tags or [],
            pymusas_mwe_indexes=pymusas_mwe_indexes or [],
        ),
    )


def _run_with_fake_tagger(
    annotator: TokenPyMUSASAnnotator, docs: list[Document], sentence_tokens: dict[str, list[SimpleNamespace]]
) -> list[Document]:
    fake_nlp = Mock(side_effect=lambda sentence: sentence_tokens[sentence])
    with patch(
        "wikipedia_processing.pipelines.token_annotation.get_language_tagger",
        return_value=fake_nlp,
    ):
        # ty infers generator functions as `types.GeneratorType`, which it won't
        # match against `DocumentsPipeline`'s `NewType(Generator[...] | None)`.
        return list(annotator.run(docs))  # ty: ignore[invalid-return-type, invalid-argument-type]


@pytest.fixture
def annotator() -> TokenPyMUSASAnnotator:
    return TokenPyMUSASAnnotator("en")


def test_init_raises_on_invalid_language_code() -> None:
    with pytest.raises(ValueError):
        TokenPyMUSASAnnotator("xx")


@pytest.mark.parametrize(
    ("text", "sentence_indexes", "expected"),
    [
        # Docstring example.
        ("Hello world. Bye.", [(0, 12), (13, 17)], ["Hello world.", "Bye."]),
        # No sentence boundaries at all -> nothing is yielded.
        ("Hello world.", [], []),
    ],
    ids=["docstring-example", "empty-sentence-indexes"],
)
def test_get_sentences(
    annotator: TokenPyMUSASAnnotator,
    text: str,
    sentence_indexes: list[tuple[int, int]],
    expected: list[str],
) -> None:
    doc = Document(text=text, id="1", metadata={"start_end_sentence_character_indexes": sentence_indexes})
    assert list(annotator.get_sentences(doc)) == expected


def test_get_sentences_raises_without_sentence_indexes_metadata(annotator: TokenPyMUSASAnnotator) -> None:
    # Docstring example: metadata is missing the required key entirely.
    doc = Document(text="x", id="1")
    with pytest.raises(ValueError, match="requires `start_end_sentence_character_indexes`"):
        list(annotator.get_sentences(doc))


def test_run_tags_tokens_skips_whitespace_and_handles_empty_sentences(
    annotator: TokenPyMUSASAnnotator,
) -> None:
    # Deterministic tag validity independent of the real USAS mapper data.
    annotator.valid_usas_tags = {"Z2"}

    sentence_tokens = {
        "Cats sit.": [
            # Valid, most-likely tag -> kept and counted.
            _make_token("Cats", pymusas_tags=["Z2"], pymusas_mwe_indexes=[(0, 1)]),
            # Whitespace tokens are dropped entirely, not even counted.
            _make_token(" ", is_space=True),
            # Tag is present but not a valid USAS tag -> filtered to [].
            _make_token("sit", pymusas_tags=["Q1"], pymusas_mwe_indexes=[(1, 2)]),
            # No PyMUSAS tags at all -> [].
            _make_token(".", pymusas_tags=[], pymusas_mwe_indexes=[(2, 3)]),
        ],
    }

    doc = Document(
        text="Cats sit. ",
        id="1",
        metadata={"start_end_sentence_character_indexes": [(0, 9), (9, 10)]},
    )
    (result,) = _run_with_fake_tagger(annotator, [doc], sentence_tokens)

    assert result is doc
    assert result.metadata["tokens"] == [["Cats", "sit", "."], []]
    assert result.metadata["tags"] == [[["Z2"], [], []], []]
    # None of the tokens share an MWE index slice, so no MWEs are found.
    assert result.metadata["mwes"] == [[[], [], []], []]

    assert annotator.stats["tokens"].total == 3
    assert annotator.stats["tagged tokens"].total == 1
    assert annotator.stats["PyMUSAS tags"].total == 1
    assert annotator.stats["MWEs"].total == 0


def test_run_applies_tag_mapper() -> None:
    annotator = TokenPyMUSASAnnotator("en", tag_mapper={"Z2": "Z2_MAPPED"})
    annotator.valid_usas_tags = {"Z2"}
    sentence_tokens = {
        "Cats": [_make_token("Cats", pymusas_tags=["Z2"], pymusas_mwe_indexes=[(0, 1)])],
    }

    doc = Document(text="Cats", id="1", metadata={"start_end_sentence_character_indexes": [(0, 4)]})
    (result,) = _run_with_fake_tagger(annotator, [doc], sentence_tokens)

    assert result.metadata["tags"] == [[["Z2_MAPPED"]]]


def test_run_labels_multi_word_expressions(annotator: TokenPyMUSASAnnotator) -> None:
    # Docstring example from `mwe_labels_from_pymusas_indexes`: tokens 0 and 1
    # form a single MWE (label 1), token 2 is not part of any MWE.
    sentence_tokens = {
        "A B C": [
            _make_token("A", pymusas_mwe_indexes=[(0, 2)]),
            _make_token("B", pymusas_mwe_indexes=[(0, 2)]),
            _make_token("C", pymusas_mwe_indexes=[(2, 3)]),
        ],
    }

    doc = Document(text="A B C", id="1", metadata={"start_end_sentence_character_indexes": [(0, 5)]})
    (result,) = _run_with_fake_tagger(annotator, [doc], sentence_tokens)

    assert result.metadata["mwes"] == [[[1], [1], []]]
    assert annotator.stats["MWEs"].total == 1


def test_run_mwes_stat_when_last_token_belongs_to_an_earlier_mwe(
    annotator: TokenPyMUSASAnnotator,
) -> None:
    sentence_tokens = {
        "A B C D": [
            _make_token("A", pymusas_mwe_indexes=[(0, 1), (3, 4)]),
            _make_token("B", pymusas_mwe_indexes=[(1, 2), (2, 3)]),
            _make_token("C", pymusas_mwe_indexes=[(1, 2), (2, 3)]),
            _make_token("D", pymusas_mwe_indexes=[(0, 1), (3, 4)]),
        ],
    }

    doc = Document(text="A B C D", id="1", metadata={"start_end_sentence_character_indexes": [(0, 7)]})
    (result,) = _run_with_fake_tagger(annotator, [doc], sentence_tokens)

    assert result.metadata["mwes"] == [[[1], [2], [2], [1]]]
    assert annotator.stats["MWEs"].total == 2


def test_run_processes_multiple_documents_in_order_and_accumulates_stats(
    annotator: TokenPyMUSASAnnotator,
) -> None:
    sentence_tokens = {
        "one": [_make_token("one")],
        "two": [_make_token("two")],
    }

    doc1 = Document(text="one", id="1", metadata={"start_end_sentence_character_indexes": [(0, 3)]})
    doc2 = Document(text="two", id="2", metadata={"start_end_sentence_character_indexes": [(0, 3)]})

    result = _run_with_fake_tagger(annotator, [doc1, doc2], sentence_tokens)

    assert result == [doc1, doc2]
    assert result[0].metadata["tokens"] == [["one"]]
    assert result[1].metadata["tokens"] == [["two"]]
    assert annotator.stats["tokens"].n == 2
    assert annotator.stats["tokens"].total == 2
