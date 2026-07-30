import dataclasses
from typing import Callable

from datatrove.data import Document
from datatrove.pipeline.writers.disk_base import DiskWriter


def get_metadata_whitelist_writer_adapter(keys_to_keep: frozenset[str]) -> Callable[[DiskWriter, Document], dict]:
    """Build a DataTrove writer adapter that restricts written metadata to `keys_to_keep`.

    Readers and stats blocks (e.g. `HuggingFaceDatasetReader`, `WordStats`) can
    stuff `doc.metadata` with fields that are never meant to reach the final
    output (unused source-dataset columns, per-document word statistics,
    etc.). The returned adapter behaves like `DiskWriter`'s own default
    adapter (flattening metadata into top-level fields when
    `writer.expand_metadata` is True), except any metadata key not in
    `keys_to_keep` is dropped from that flattened output.

    This only filters metadata when `writer.expand_metadata` is True; otherwise
    it leaves metadata untouched.

    Args:
        keys_to_keep: The doc.metadata keys allowed into the written output
            when `writer.expand_metadata` is True.

    Returns:
        A writer adapter function suitable for `DiskWriter(..., adapter=...)`.

    Examples:
        >>> from types import SimpleNamespace
        >>> from datatrove.data import Document
        >>> adapter = get_metadata_whitelist_writer_adapter(keys_to_keep=frozenset({"url"}))
        >>> writer = SimpleNamespace(expand_metadata=True, save_media_bytes=True)
        >>> doc = Document(text="x", id="1", metadata={"url": "u", "junk": 1})
        >>> adapter(writer, doc)["url"]
        'u'
        >>> "junk" in adapter(writer, doc)
        False
    """

    def adapter(writer: DiskWriter, document: Document) -> dict:
        data = {key: val for key, val in dataclasses.asdict(document).items() if val}
        if writer.expand_metadata and "metadata" in data:
            metadata = data.pop("metadata")
            data |= {key: val for key, val in metadata.items() if key in keys_to_keep}
        if not writer.save_media_bytes and "media" in data:
            data["media"] = [{**media, "media_bytes": None} for media in data["media"]]
        return data

    return adapter
