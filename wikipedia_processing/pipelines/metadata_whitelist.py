from typing import Iterable, cast

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep


class MetadataWhitelistAnnotator(PipelineStep):
    """DataTrove pipeline step that prunes doc.metadata down to a fixed set of keys.

    Readers and stats blocks (e.g. `HuggingFaceDatasetReader`, `WordStats`) can
    stuff `doc.metadata` with fields that are never used downstream (unused
    source-dataset columns, per-document word statistics, etc.). Unlike a
    DataTrove filter, which drops whole documents, this step never drops
    documents -- it only removes metadata keys that are not in `keys_to_keep`
    from every document that passes through it, so unwanted metadata does not
    get carried through (or written by) later pipeline stages.

    Attributes:
        name: Human-readable step name shown in DataTrove pipeline stats.
        type: DataTrove pipeline step category shown in DataTrove pipeline stats.
    """

    name = "🧹 Metadata Whitelist Annotator"
    type = "🏷 - ANNOTATE"

    def __init__(self, keys_to_keep: frozenset[str]) -> None:
        """Initialize the annotator.

        Args:
            keys_to_keep: The doc.metadata keys to retain; every other key is
                dropped from doc.metadata.
        """
        super().__init__()
        self.keys_to_keep = keys_to_keep

    # ty infers generator functions as `types.GeneratorType`, which it won't match against
    # `DocumentsPipeline`'s `NewType(Generator[Document, None, None] | None)` alias.
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:  # ty: ignore[invalid-return-type]
        """Prune doc.metadata down to `keys_to_keep` for each document in data.

        Args:
            data: An iterable of documents to prune.
            rank: The rank of the current worker process (unused, present for
                DataTrove's PipelineStep interface).
            world_size: The total number of parallel ranks this step is
                running across (unused, present for DataTrove's PipelineStep
                interface).

        Yields:
            Each input document, with doc.metadata replaced by a copy
            containing only keys present in `keys_to_keep`.

        Examples:
            >>> from datatrove.data import Document
            >>> step = MetadataWhitelistAnnotator(keys_to_keep=frozenset({"url"}))
            >>> doc = Document(text="a", id="1", metadata={"url": "u", "junk": 1})
            >>> list(step.run([doc]))[0].metadata
            {'url': 'u'}
        """
        for doc in cast(Iterable[Document], data):
            doc.metadata = {key: val for key, val in doc.metadata.items() if key in self.keys_to_keep}
            yield doc
