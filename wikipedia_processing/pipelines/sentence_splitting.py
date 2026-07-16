from typing import Callable, Iterable, cast

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep

from wikipedia_processing.models_install import Languages as ModelInstallLanguages
from wikipedia_processing.models_util import get_language_sentence_splitter


class SentenceSplitterAnnotator(PipelineStep):
    """DataTrove pipeline step that annotates documents with sentence boundaries.

    For each document it runs a language-specific spaCy sentence splitter over
    doc.text and stores the character start/end offsets of every sentence in
    doc.metadata under the "start_end_sentence_character_indexes" key.

    Attributes:
        name: Human-readable step name shown in DataTrove pipeline stats.
        type: DataTrove pipeline step category shown in DataTrove pipeline stats.
    """

    name = "🏷 Sentence Splitter Annotator"
    type = "🏷 - ANNOTATE"

    def __init__(self, wikipedia_language_code: str):
        """Initialize the annotator.

        Args:
            wikipedia_language_code: A Wikipedia language code (e.g. "en",
                "da") identifying a member of
                :class:`~wikipedia_processing.models_install.Languages`.

        Raises:
            ValueError: If wikipedia_language_code is not a supported
                language code.
        """
        super().__init__()
        _language = getattr(ModelInstallLanguages, wikipedia_language_code, None)
        supported_languages = list(ModelInstallLanguages)
        if _language is None:
            raise ValueError(f"Invalid language code: {wikipedia_language_code!r} Supported languages: {supported_languages!r}")
        self.language = cast(ModelInstallLanguages, _language)
        self._nlp: None | Callable[[str], Iterable[tuple[str, tuple[int, int]]]] = None  # lazy-loaded per worker process

    def _load_model(self) -> Callable[[str], Iterable[tuple[str, tuple[int, int]]]]:
        if self._nlp is None:
            self._nlp = get_language_sentence_splitter(self.language)
        return self._nlp

    # ty infers generator functions as `types.GeneratorType`, which it won't match against
    # `DocumentsPipeline`'s `NewType(Generator[Document, None, None] | None)` alias.
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:  # ty: ignore[invalid-return-type]
        """Annotate each document in data with sentence boundary indexes.

        Lazily loads the language-specific sentence splitter on first use, then
        for every document sets
        doc.metadata["start_end_sentence_character_indexes"] to a list of
        (start, end) character offset tuples, one per sentence in doc.text.

        Args:
            data: An iterable of documents to annotate.
            rank: The rank of the current worker process (unused, present for
                DataTrove's PipelineStep interface).
            world_size: The total number of worker processes (unused, present
                for DataTrove's PipelineStep interface).

        Yields:
            Each input document, with its metadata updated in place to
            include "start_end_sentence_character_indexes".
        """
        sentence_splitter = self._load_model()

        for doc in cast(Iterable[Document], data):
            with self.track_time():
                doc.metadata["start_end_sentence_character_indexes"] = []
                for _, (start_sentence_index, end_sentence_index) in sentence_splitter(doc.text):
                    doc.metadata["start_end_sentence_character_indexes"].append((start_sentence_index, end_sentence_index))
            self.stat_update("sentences", value=len(doc.metadata["start_end_sentence_character_indexes"]))
            yield doc

