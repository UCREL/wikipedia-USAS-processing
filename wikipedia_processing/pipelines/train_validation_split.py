import dataclasses
import hashlib
from typing import cast

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers.disk_base import DiskWriter


def drop_split_column_writer_adapter(writer: DiskWriter, document: Document) -> dict:
    """DataTrove writer adapter that omits the "split" metadata field from written rows.

    Behaves like `DiskWriter`'s own default adapter (flattening metadata into
    top-level fields when `writer.expand_metadata` is True), except it drops
    "split" from the output. This is meant to be paired with
    :class:`TrainValidationSplitAnnotator` and a writer whose
    `output_filename` already routes documents into `train`/`validation`
    subfolders via a "${split}" placeholder: that routing reads
    `document.metadata` directly before this adapter runs, so dropping
    "split" here only removes the redundant column from the written data --
    the split is still recoverable from which subfolder a row is written to.

    NOTE: "split" is only dropped when `writer.expand_metadata` is True (i.e.
    metadata is flattened into top-level columns). When `writer.expand_metadata`
    is False, "split" is left untouched inside the nested "metadata" dict,
    matching how `DiskWriter`'s own default adapter otherwise leaves metadata
    alone in that mode.

    Args:
        writer: The `DiskWriter` instance this adapter is bound to (passed
            automatically by DataTrove).
        document: The document being written.

    Returns:
        A dictionary of the document's fields to write, with "split" omitted
        from the (optionally expanded) metadata.
    """
    data = {key: val for key, val in dataclasses.asdict(document).items() if val}
    if writer.expand_metadata and "metadata" in data:
        metadata = data.pop("metadata")
        metadata.pop("split", None)
        data |= metadata
    if not writer.save_media_bytes and "media" in data:
        data["media"] = [{**media, "media_bytes": None} for media in data["media"]]
    return data


class TrainValidationSplitAnnotator(PipelineStep):
    """DataTrove pipeline step that assigns each document to a train/validation split.

    Each document is deterministically routed to the "validation" split if a
    stable hash of its `split_hash_metadata_key` metadata value falls within
    `validation_percentage` of the hash space, up to a
    `max_validation_documents` cap; every other document (including anything
    over the cap) is routed to "train". The assigned split is stored in
    doc.metadata["split"].

    Because this step may run across multiple parallel DataTrove ranks, the
    cap is divided evenly across ranks (via `world_size`) rather than
    enforced as a single global counter, giving an approximate but
    deterministic, coordination-free total cap of `max_validation_documents`
    documents.

    Attributes:
        name: Human-readable step name shown in DataTrove pipeline stats.
        type: DataTrove pipeline step category shown in DataTrove pipeline stats.
    """

    name = "🔀 Train/Validation Split"
    type = "🔀 - ANNOTATE"

    def __init__(self, validation_percentage: float, max_validation_documents: int, split_hash_metadata_key: str):
        """Initialize the annotator.

        Args:
            validation_percentage: Target percentage (0-100) of documents
                assigned to the "validation" split.
            max_validation_documents: Absolute cap, across all ranks, on the
                number of documents assigned to the "validation" split,
                regardless of validation_percentage.
            split_hash_metadata_key: The doc.metadata key whose value is
                hashed to deterministically decide a document's split (e.g.
                "page_id"). The value must be stable across pipeline re-runs.
        """
        super().__init__()
        self.validation_percentage = validation_percentage
        self.max_validation_documents = max_validation_documents
        self.split_hash_metadata_key = split_hash_metadata_key

    def _is_validation_candidate(self, metadata_value: object) -> bool:
        """Deterministically decide if a metadata value falls within the validation percentage.

        Args:
            metadata_value: The document's `split_hash_metadata_key` metadata
                value.

        Returns:
            True if the value hashes into the validation percentage of the
            hash space, False otherwise.
        """
        digest = hashlib.md5(str(metadata_value).encode("utf-8")).hexdigest()
        return int(digest, 16) % 100 < self.validation_percentage

    # ty infers generator functions as `types.GeneratorType`, which it won't match against
    # `DocumentsPipeline`'s `NewType(Generator[Document, None, None] | None)` alias.
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:  # ty: ignore[invalid-return-type]
        """Assign each document in data to the "train" or "validation" split.

        Args:
            data: An iterable of documents to annotate.
            rank: The rank of the current worker process (unused, present for
                DataTrove's PipelineStep interface).
            world_size: The total number of parallel ranks this step is
                running across, used to divide max_validation_documents into
                a per-rank share.

        Yields:
            Each input document, with its metadata updated in place to
            include "split" set to either "train" or "validation".
        """
        local_validation_cap = max(1, round(self.max_validation_documents / world_size))
        validation_count = 0

        for doc in cast(list[Document], data):
            is_validation = (
                validation_count < local_validation_cap
                and self._is_validation_candidate(doc.metadata[self.split_hash_metadata_key])
            )
            if is_validation:
                doc.metadata["split"] = "validation"
                validation_count += 1
            else:
                doc.metadata["split"] = "train"
            yield doc

        self.stat_update("validation documents", value=validation_count)
