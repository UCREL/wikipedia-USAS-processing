from typing import Callable, Iterable, cast

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep

from wikipedia_processing.models_install import Languages as ModelInstallLanguages
from wikipedia_processing.models_util import get_language_sentence_splitter


class SentenceSplitterAnnotator(PipelineStep):
    name = "🏷 Sentence Splitter Annotator"
    type = "🏷 - ANNOTATE"

    def __init__(self, wikipedia_language_code: str):
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

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        sentence_splitter = self._load_model()
        
        for doc in data:
            doc = cast(Document, doc)
            with self.track_time():
                doc.metadata["start_end_sentence_character_indexes"] = []
                for _, (start_sentence_index, end_sentence_index) in sentence_splitter(doc.text):
                    doc.metadata["start_end_sentence_character_indexes"].append((start_sentence_index, end_sentence_index))
            self.stat_update("sentences", value=len(doc.metadata["start_end_sentence_character_indexes"]))
            yield doc

