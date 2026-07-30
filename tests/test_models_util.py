from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from wikipedia_processing.models_install import (
    LANGUAGE_2_PYMUSAS_SPACY_MODEL,
    PYMUSAS_SPACY_MODEL_2_URL,
    SPACY_MODEL_2_URL,
    Languages,
    SpacyModel,
)
from wikipedia_processing.models_util import (
    get_language_sentence_splitter,
    get_language_tagger,
    spacy_sentence_splitter,
)


@pytest.mark.parametrize(
    ("language", "expected_spacy_model"),
    [
        (Languages.zh, SpacyModel.zh_lg),
        (Languages.da, SpacyModel.da_lg),
        (Languages.nl, SpacyModel.nl_lg),
        (Languages.en, SpacyModel.en_lg),
        (Languages.fi, SpacyModel.fi_lg),
        (Languages.it, SpacyModel.it_lg),
        (Languages.pt, SpacyModel.pt_lg),
        (Languages.es, SpacyModel.es_lg),
    ],
    ids=["zh", "da", "nl", "en", "fi", "it", "pt", "es"],
)
def test_get_language_tagger_dispatches_to_expected_spacy_model(
    language: Languages, expected_spacy_model: SpacyModel
) -> None:
    fake_nlp = Mock()
    fake_pymusas_pipe = Mock()
    pymusas_model_name = LANGUAGE_2_PYMUSAS_SPACY_MODEL[language]

    with (
        patch("wikipedia_processing.models_util.pip_install_model") as mock_pip_install,
        patch(
            "wikipedia_processing.models_util.spacy.load",
            side_effect=[fake_nlp, fake_pymusas_pipe],
        ) as mock_load,
    ):
        result = get_language_tagger(language)

    mock_pip_install.assert_any_call(SPACY_MODEL_2_URL[expected_spacy_model], expected_spacy_model.value)
    mock_pip_install.assert_any_call(PYMUSAS_SPACY_MODEL_2_URL[pymusas_model_name], pymusas_model_name)
    # "ner" is excluded for every language, since only the PyMUSAS tagger is added.
    mock_load.assert_any_call(expected_spacy_model.value, exclude=["ner"])
    mock_load.assert_any_call(pymusas_model_name)
    fake_nlp.add_pipe.assert_called_once_with("pymusas_rule_based_tagger", source=fake_pymusas_pipe)
    assert result is fake_nlp


def test_get_language_tagger_raises_for_unsupported_language() -> None:
    with pytest.raises(ValueError, match="not supported"):
        get_language_tagger("xx")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("language", "expected_spacy_model"),
    [
        (Languages.zh, SpacyModel.zh_md),
        (Languages.da, SpacyModel.da_lg),
        (Languages.nl, SpacyModel.nl_lg),
        (Languages.en, SpacyModel.en_lg),
        (Languages.fi, SpacyModel.fi_sm),
        (Languages.it, SpacyModel.it_sm),
        (Languages.pt, SpacyModel.pt_lg),
        (Languages.es, SpacyModel.es_lg),
    ],
    ids=["zh", "da", "nl", "en", "fi", "it", "pt", "es"],
)
def test_get_language_sentence_splitter_dispatches_to_expected_spacy_model(
    language: Languages, expected_spacy_model: SpacyModel
) -> None:
    fake_full_pipeline = Mock(pipe_names=["tok2vec", "parser", "ner", "lemmatizer"])
    fake_excluded_pipeline = Mock()
    sentinel_splitter = Mock()

    with (
        patch("wikipedia_processing.models_util.pip_install_model") as mock_pip_install,
        patch(
            "wikipedia_processing.models_util.spacy.load",
            side_effect=[fake_full_pipeline, fake_excluded_pipeline],
        ) as mock_load,
        patch(
            "wikipedia_processing.models_util.spacy_sentence_splitter",
            return_value=sentinel_splitter,
        ) as mock_splitter_factory,
    ):
        result = get_language_sentence_splitter(language)

    mock_pip_install.assert_called_once_with(SPACY_MODEL_2_URL[expected_spacy_model], expected_spacy_model.value)
    mock_load.assert_any_call(expected_spacy_model.value)
    # Only "ner" and "lemmatizer" are not in {transformer, tok2vec, parser}, so
    # those are the pipes excluded from the reloaded pipeline.
    mock_load.assert_any_call(expected_spacy_model.value, exclude=["ner", "lemmatizer"])
    mock_splitter_factory.assert_called_once_with(fake_excluded_pipeline)
    assert result is sentinel_splitter


def test_get_language_sentence_splitter_raises_for_unsupported_language() -> None:
    with pytest.raises(ValueError, match="not supported"):
        get_language_sentence_splitter("xx")  # ty: ignore[invalid-argument-type]


def _make_sentence(text: str, start_char: int, end_char: int) -> SimpleNamespace:
    return SimpleNamespace(text=text, start_char=start_char, end_char=end_char)


@pytest.mark.parametrize(
    ("sentences", "expected"),
    [
        # Docstring-style example: two sentences.
        (
            [_make_sentence("Hello world.", 0, 12), _make_sentence("Bye.", 13, 17)],
            [("Hello world.", (0, 12)), ("Bye.", (13, 17))],
        ),
        # A single sentence.
        ([_make_sentence("Just one.", 0, 9)], [("Just one.", (0, 9))]),
        # No sentence boundaries at all -> nothing is yielded.
        ([], []),
    ],
    ids=["two-sentences", "single-sentence", "no-sentences"],
)
def test_spacy_sentence_splitter_yields_text_and_char_offsets(
    sentences: list[SimpleNamespace], expected: list[tuple[str, tuple[int, int]]]
) -> None:
    fake_doc = SimpleNamespace(sents=sentences)
    fake_pipeline = Mock(return_value=fake_doc)

    splitter = spacy_sentence_splitter(fake_pipeline)
    result = list(splitter("irrelevant, doc is faked"))

    fake_pipeline.assert_called_once_with("irrelevant, doc is faked")
    assert result == expected


class _NoSentenceBoundariesDoc:
    """Fake `Doc` mirroring spaCy's error when no sentence-boundary component ran."""

    @property
    def sents(self) -> list[SimpleNamespace]:
        raise ValueError("Sentence boundaries have not been set")


def test_spacy_sentence_splitter_raises_when_consumed_without_sentence_boundaries() -> None:
    fake_pipeline = Mock(return_value=_NoSentenceBoundariesDoc())
    splitter = spacy_sentence_splitter(fake_pipeline)

    with pytest.raises(ValueError):
        list(splitter("some text"))
