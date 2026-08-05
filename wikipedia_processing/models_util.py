from typing import Callable, Iterable

import spacy
from spacy.tokens import Doc

from wikipedia_processing.models_install import (
    LANGUAGE_2_PYMUSAS_SPACY_MODEL,
    PYMUSAS_SPACY_MODEL_2_URL,
    SPACY_MODEL_2_URL,
    Languages,
    SpacyModel,
    pip_install_model,
)

# Some FineWiki documents contain very long stretches of text with no sentence
# boundaries (e.g. malformed tables that slip past markdown/table stripping),
# so both spaCy pipelines below need a raised `nlp.max_length` to avoid E088.
MAX_SPACY_TEXT_LENGTH = 25_000_000


def get_language_tagger(language: Languages) -> spacy.Language:
    """
    Returns a spaCy pipeline with a PyMUSAS rule-based tagger pipeline component added.

    The rule-based tagger pipeline component is added to the end of the pipeline. The pipeline is constructed
    by loading a language specific spaCy model and then adding the PyMUSAS rule-based tagger pipeline component.

    The language specific spaCy model is loaded with the following pipeline components excluded:
    - 'ner' and 'parser' for all languages, since the PyMUSAS rule-based tagger only requires
      `token.pos` and `token.lemma`.

    Args:
        language: The language that the spaCy pipeline should be constructed for.

    Returns:
        A language specific spaCy pipeline with a PyMUSAS rule-based tagger pipeline component added.

    Raises:
        ValueError: If the given language is not supported.
    """
    def get_tagger(language: Languages,
                   spacy_model_name: SpacyModel,
                   spacy_pipes_to_exclude: list[str]) -> spacy.Language:
        pip_install_model(SPACY_MODEL_2_URL[spacy_model_name], spacy_model_name.value)
        pymusas_spacy_model_name = LANGUAGE_2_PYMUSAS_SPACY_MODEL[language]
        pip_install_model(PYMUSAS_SPACY_MODEL_2_URL[pymusas_spacy_model_name], pymusas_spacy_model_name)

        nlp = spacy.load(spacy_model_name.value, exclude=spacy_pipes_to_exclude)
        nlp.max_length = MAX_SPACY_TEXT_LENGTH
        pymusas_pipe = spacy.load(LANGUAGE_2_PYMUSAS_SPACY_MODEL[language])
        nlp.add_pipe("pymusas_rule_based_tagger", source=pymusas_pipe)
        return nlp

    match language:
        case Languages.zh:
            return get_tagger(language, SpacyModel.zh_lg, ['ner', 'parser'])
        case Languages.da:
            return get_tagger(language, SpacyModel.da_lg, ['ner', 'parser'])
        case Languages.nl:
            return get_tagger(language, SpacyModel.nl_lg, ['ner', 'parser'])
        case Languages.en:
            return get_tagger(language, SpacyModel.en_lg, ['ner', 'parser'])
        case Languages.fi:
            return get_tagger(language, SpacyModel.fi_lg, ['ner', 'parser'])
        case Languages.it:
            return get_tagger(language, SpacyModel.it_lg, ['ner', 'parser'])
        case Languages.pt:
            return get_tagger(language, SpacyModel.pt_lg, ['ner', 'parser'])
        case Languages.es:
            return get_tagger(language, SpacyModel.es_lg, ['ner', 'parser'])
        case _:
            raise ValueError(f"Language {language} not supported")


def spacy_sentence_splitter(spacy_pipeline: spacy.Language) -> Callable[[str], Iterable[tuple[str, tuple[int, int]]]]:
    """
    Returns a function that splits a given text into sentences using the given
    spaCy pipeline.

    Args:
        spacy_pipeline: A spaCy pipeline to use for sentence splitting. We assume
            that the pipeline includes a component (e.g. `parser`, `senter`, or
            `sentencizer`) that sets sentence boundaries on the resulting `Doc`.

    Returns:
        A function that takes a string and returns an iterable of `(sentence_text,
        (start_char, end_char))` tuples, one per sentence in the input text, in
        order.

    Raises:
        ValueError: When the iterable returned by the function is consumed, if
            `spacy_pipeline` does not include a component that sets sentence
            boundaries.
    """
    def _sentence_splitter(text: str) -> Iterable[tuple[str, tuple[int, int]]]:
        doc: Doc = spacy_pipeline(text)
        for sentence in doc.sents:
            yield (sentence.text, (sentence.start_char, sentence.end_char))

    return _sentence_splitter


def get_language_sentence_splitter(language: Languages) -> Callable[[str], Iterable[tuple[str, tuple[int, int]]]]:
    """
    Returns a function that splits a given text into sentences using a
    language specific spaCy model.

    The language specific spaCy model is loaded twice: once with its full pipeline
    to determine which components are not needed for sentence boundary detection
    (i.e. anything other than `transformer`, `tok2vec`, or `parser`), and once more
    with those unneeded components excluded, so that the returned function only
    runs the pipeline components required for sentence splitting.

    Args:
        language: The language of the text to split into sentences.

    Returns:
        A function that takes a string and returns an iterable of `(sentence_text,
        (start_char, end_char))` tuples, one per sentence in the input text, in
        order.

    Raises:
        ValueError: If the given language is not supported.
    """
    def get_pipes_to_exclude(spacy_pipeline: spacy.Language) -> list[str]:
        pipes_to_include = set({"transformer", "tok2vec", "parser"})
        return [pipe_name for pipe_name in spacy_pipeline.pipe_names if pipe_name not in pipes_to_include]

    def get_spacy_sentence_splitter(spacy_model_name: SpacyModel) -> Callable[[str], Iterable[tuple[str, tuple[int, int]]]]:
        pip_install_model(SPACY_MODEL_2_URL[spacy_model_name], spacy_model_name.value)
        full_nlp_pipeline = spacy.load(spacy_model_name.value)
        pipes_to_exclude = get_pipes_to_exclude(full_nlp_pipeline)
        nlp = spacy.load(spacy_model_name.value, exclude=pipes_to_exclude)
        nlp.max_length = MAX_SPACY_TEXT_LENGTH
        return spacy_sentence_splitter(nlp)

    match language:
        case Languages.zh:
            return get_spacy_sentence_splitter(SpacyModel.zh_md)
        case Languages.da:
            return get_spacy_sentence_splitter(SpacyModel.da_lg)
        case Languages.nl:
            return get_spacy_sentence_splitter(SpacyModel.nl_lg)
        case Languages.en:
            return get_spacy_sentence_splitter(SpacyModel.en_lg)
        case Languages.fi:
            return get_spacy_sentence_splitter(SpacyModel.fi_sm)
        case Languages.it:
            return get_spacy_sentence_splitter(SpacyModel.it_sm)
        case Languages.pt:
            return get_spacy_sentence_splitter(SpacyModel.pt_lg)
        case Languages.es:
            return get_spacy_sentence_splitter(SpacyModel.es_lg)
        case _:
            raise ValueError(f"Language {language} not supported")


