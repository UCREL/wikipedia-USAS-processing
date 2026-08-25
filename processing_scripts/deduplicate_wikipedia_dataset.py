"""Deduplicate the Multilingual USAS Wikipedia dataset by `id`, per language."""

import hashlib
import math
import tempfile
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Annotated, TypedDict

import pyarrow as pa
import pyarrow.parquet as pq
import typer
from datasets import (
    Dataset,
    concatenate_datasets,
    get_dataset_config_names,
    load_dataset,
)
from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
from rich import print as rprint
from rich.table import Table

from wikipedia_processing.utils import (
    get_valid_usas_language_processing_wikipedia_codes,
)

WikipediaLanguageCode = Enum("WikipediaLanguageCode", [(value, value) for value in get_valid_usas_language_processing_wikipedia_codes()], type=str)


class LanguageDedupPlan(TypedDict):
    """Per-language plan for deduplicating `train`/`validation` splits by `id`.

    Attributes:
        train_keep_indexes: Indexes into the original `train` split to keep
            in the new `train` split.
        validation_keep_indexes: Indexes into the original `validation`
            split to keep in the new `validation` split.
        train_move_to_validation_indexes: Indexes into the original `train`
            split whose rows are relocated to the new `validation` split
            (the highest-version copy of an `id` that also appears in
            `validation`).
        removed_from_train: Number of `train` rows dropped as lower-version
            duplicates.
        removed_from_validation: Number of `validation` rows dropped as
            lower-version duplicates.
        cross_split_ids: Number of `id`s duplicated across both `train` and
            `validation`.
    """

    train_keep_indexes: list[int]
    validation_keep_indexes: list[int]
    train_move_to_validation_indexes: list[int]
    removed_from_train: int
    removed_from_validation: int
    cross_split_ids: int


def compute_language_dedup_plan(train: Dataset, validation: Dataset) -> LanguageDedupPlan:
    """Plan how to deduplicate one language's `train`/`validation` splits by `id`.

    Groups every row from `train` and `validation` (combined) by its `id`.
    Groups with a single row are kept unchanged. For groups with more than
    one row, only the row with the highest `version` is kept; the rest are
    dropped. If a group's rows span both splits, the kept row always ends up
    in `validation` (moved out of `train` if that's where the highest
    -version copy originally was); otherwise the kept row stays in its
    original split.

    Args:
        train: The language's `train` split. Must have `id` and `version`
            columns.
        validation: The language's `validation` split. Must have `id` and
            `version` columns.

    Returns:
        A `LanguageDedupPlan` describing which original row indexes to
        keep, move, or drop.

    Examples:
        >>> from datasets import Dataset
        >>> train = Dataset.from_dict({"id": ["a", "b"], "version": [1, 5]})
        >>> validation = Dataset.from_dict({"id": ["b"], "version": [3]})
        >>> plan = compute_language_dedup_plan(train, validation)
        >>> plan["train_keep_indexes"]
        [0]
        >>> plan["train_move_to_validation_indexes"]
        [1]
        >>> plan["validation_keep_indexes"]
        []
        >>> plan["cross_split_ids"]
        1
    """
    locations: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for index, (id_, version) in enumerate(zip(train["id"], train["version"])):
        locations[id_].append(("train", index, version))
    for index, (id_, version) in enumerate(zip(validation["id"], validation["version"])):
        locations[id_].append(("validation", index, version))

    train_keep_indexes: list[int] = []
    validation_keep_indexes: list[int] = []
    train_move_to_validation_indexes: list[int] = []
    removed_from_train = 0
    removed_from_validation = 0
    cross_split_ids = 0

    for locs in locations.values():
        is_cross_split = len({split for split, _, _ in locs}) > 1
        if is_cross_split:
            cross_split_ids += 1
        keep_split, keep_index, _ = max(locs, key=lambda loc: (loc[2], loc[0] == "validation"))
        target_split = "validation" if is_cross_split else keep_split

        for split, index, _ in locs:
            if split == keep_split and index == keep_index:
                match (target_split, split):
                    case ("train", _):
                        train_keep_indexes.append(index)
                    case (_, "validation"):
                        validation_keep_indexes.append(index)
                    case _:
                        train_move_to_validation_indexes.append(index)
            elif split == "train":
                removed_from_train += 1
            else:
                removed_from_validation += 1

    return LanguageDedupPlan(
        train_keep_indexes=sorted(train_keep_indexes),
        validation_keep_indexes=sorted(validation_keep_indexes),
        train_move_to_validation_indexes=sorted(train_move_to_validation_indexes),
        removed_from_train=removed_from_train,
        removed_from_validation=removed_from_validation,
        cross_split_ids=cross_split_ids,
    )


def apply_dedup_plan(train: Dataset, validation: Dataset, plan: LanguageDedupPlan) -> tuple[Dataset, Dataset]:
    """Apply a `LanguageDedupPlan` to produce deduplicated `train`/`validation` splits.

    Args:
        train: The original `train` split the plan was computed from.
        validation: The original `validation` split the plan was computed
            from.
        plan: The plan returned by `compute_language_dedup_plan` for `train`
            and `validation`.

    Returns:
        A `(new_train, new_validation)` tuple of deduplicated datasets.
    """
    new_train = train.select(plan["train_keep_indexes"])
    validation_parts = [validation.select(plan["validation_keep_indexes"])]
    if plan["train_move_to_validation_indexes"]:
        validation_parts.append(train.select(plan["train_move_to_validation_indexes"]))
    new_validation = concatenate_datasets(validation_parts) if len(validation_parts) > 1 else validation_parts[0]
    return new_train, new_validation


def compute_split_hash(value: object) -> int:
    """Compute the same deterministic hash `TrainValidationSplitAnnotator` uses for split assignment.

    See `wikipedia_processing.pipelines.train_validation_split.TrainValidationSplitAnnotator._is_validation_candidate`,
    which hashes a document's `split_hash_metadata_key` metadata value
    (e.g. `page_id`) the same way. Re-deriving it here lets a validation
    split be trimmed deterministically after the fact, without needing to
    know the original per-rank processing order.

    Args:
        value: The metadata value to hash, e.g. a `page_id`.

    Returns:
        The value's MD5 digest, as an integer.

    Examples:
        >>> compute_split_hash(1234) == compute_split_hash(1234)
        True
    """
    return int(hashlib.md5(str(value).encode("utf-8")).hexdigest(), 16)


def compute_validation_overflow_plan(validation: Dataset, max_validation_documents: int, split_hash_metadata_key: str = "page_id") -> tuple[list[int], list[int]]:
    """Plan how to cap a validation split at `max_validation_documents` rows.

    `TrainValidationSplitAnnotator` (see `wikipedia_processing.pipelines
    .train_validation_split`) is supposed to cap the whole validation split
    at `max_validation_documents` documents, but divides that cap evenly
    across DataTrove ranks with a `max(1, ...)` floor -- so whenever a
    language's rank/task count exceeds `max_validation_documents`, every
    rank gets a local cap of at least 1, and the real total ends up close
    to the rank count instead of `max_validation_documents`. This plans a
    post-hoc fix: if `validation` already has more than
    `max_validation_documents` rows, only the rows with the smallest
    `compute_split_hash` value (over `split_hash_metadata_key`) are kept --
    the same deterministic ordering the annotator itself uses -- and the
    rest are moved back to `train`.

    Args:
        validation: The (already-deduplicated) `validation` split to cap.
        max_validation_documents: The intended cap on the number of rows in
            `validation`.
        split_hash_metadata_key: The column hashed to break ties
            deterministically, matching `TrainValidationSplitAnnotator`'s
            `split_hash_metadata_key`.

    Returns:
        A `(keep_indexes, move_to_train_indexes)` tuple of indexes into
        `validation`. `move_to_train_indexes` is empty if `validation` is
        already at or under `max_validation_documents` rows.

    Examples:
        >>> from datasets import Dataset
        >>> validation = Dataset.from_dict({"page_id": [1, 2, 3]})
        >>> keep, move = compute_validation_overflow_plan(validation, max_validation_documents=2)
        >>> len(keep), len(move)
        (2, 1)
        >>> sorted(keep + move)
        [0, 1, 2]
    """
    if len(validation) <= max_validation_documents:
        return list(range(len(validation))), []

    ranked_indexes = sorted(range(len(validation)), key=lambda index: compute_split_hash(validation[split_hash_metadata_key][index]))
    keep_indexes = sorted(ranked_indexes[:max_validation_documents])
    move_to_train_indexes = sorted(ranked_indexes[max_validation_documents:])
    return keep_indexes, move_to_train_indexes


def compute_target_validation_documents(total_documents: int, max_validation_percentage: float, max_validation_documents: int) -> int:
    """Compute the intended validation split size: `min(X% of total, N)`.

    Mirrors the train/validation split rule described in `README.md`'s
    "Train/validation split" section and used by
    `build_usas_wikipedia_dataset.py`'s `-v/--validation-percentage` and
    `-n/--max-validation-documents` options: a language's validation split
    should be `max_validation_percentage`% of its total document count, or
    `max_validation_documents`, whichever is smaller. Percentage rounding
    uses Python's built-in `round` (banker's rounding), matching
    `TrainValidationSplitAnnotator`'s own
    `round(self.max_validation_documents / world_size)`.

    Args:
        total_documents: The language's total (post-dedup) document count,
            across `train` and `validation` combined.
        max_validation_percentage: Target percentage (0-100) of
            `total_documents` to assign to validation.
        max_validation_documents: Absolute cap on the validation split size,
            regardless of `max_validation_percentage`.

    Returns:
        The smaller of the percentage-based count and
        `max_validation_documents`, floored at 0.

    Examples:
        >>> compute_target_validation_documents(200, max_validation_percentage=10, max_validation_documents=20)
        20
        >>> compute_target_validation_documents(150, max_validation_percentage=10, max_validation_documents=20)
        15
    """
    percentage_based = round(total_documents * max_validation_percentage / 100)
    return max(0, min(percentage_based, max_validation_documents))


class ValidationRebalancePlan(TypedDict):
    """Plan for rebalancing `train`/`validation` splits to a target validation size.

    Attributes:
        train_keep_indexes: Indexes into the original `train` split to keep
            in the new `train` split.
        train_move_to_validation_indexes: Indexes into the original `train`
            split whose rows are relocated to the new `validation` split.
        validation_keep_indexes: Indexes into the original `validation`
            split to keep in the new `validation` split.
        validation_move_to_train_indexes: Indexes into the original
            `validation` split whose rows are relocated to the new `train`
            split.
        target_validation_documents: The target validation document count
            actually used, after clamping the requested target to
            `[0, len(train) + len(validation)]`.
    """

    train_keep_indexes: list[int]
    train_move_to_validation_indexes: list[int]
    validation_keep_indexes: list[int]
    validation_move_to_train_indexes: list[int]
    target_validation_documents: int


def compute_validation_rebalance_plan(train: Dataset, validation: Dataset, target_validation_documents: int, split_hash_metadata_key: str = "page_id") -> ValidationRebalancePlan:
    """Plan how to rebalance `train`/`validation` to an exact target validation size.

    Unlike `compute_validation_overflow_plan`, which only ever shrinks an
    oversized `validation` split, this ranks every row from `train` and
    `validation` (combined) by `compute_split_hash` -- the same deterministic
    ordering `TrainValidationSplitAnnotator` and `compute_validation_overflow_plan`
    both use -- and assigns the `target_validation_documents` rows with the
    smallest hash to `validation`, the rest to `train`. Ranking the pooled
    rows once, rather than only ever removing rows from `validation`, lets
    the same plan either grow `validation` (pulling rows from `train`) or
    shrink it, depending on how `target_validation_documents` compares to
    `validation`'s current size.

    Args:
        train: The language's `train` split. Must have
            `split_hash_metadata_key` as a column.
        validation: The language's `validation` split. Must have
            `split_hash_metadata_key` as a column.
        target_validation_documents: The intended size of the new
            `validation` split. Clamped to `[0, len(train) + len(validation)]`
            if out of range.
        split_hash_metadata_key: The column hashed to rank rows
            deterministically, matching `TrainValidationSplitAnnotator`'s
            `split_hash_metadata_key`.

    Returns:
        A `ValidationRebalancePlan` describing which original row indexes to
        keep or move.

    Examples:
        >>> from datasets import Dataset
        >>> train = Dataset.from_dict({"page_id": [1, 2, 3, 4]})
        >>> validation = Dataset.from_dict({"page_id": [5]})
        >>> plan = compute_validation_rebalance_plan(train, validation, target_validation_documents=2)
        >>> plan["target_validation_documents"]
        2
        >>> len(plan["validation_keep_indexes"]) + len(plan["train_move_to_validation_indexes"])
        2
    """
    clamped_target = max(0, min(target_validation_documents, len(train) + len(validation)))

    combined = [("train", index, compute_split_hash(value)) for index, value in enumerate(train[split_hash_metadata_key])]
    combined += [("validation", index, compute_split_hash(value)) for index, value in enumerate(validation[split_hash_metadata_key])]
    combined.sort(key=lambda item: item[2])

    train_keep_indexes: list[int] = []
    train_move_to_validation_indexes: list[int] = []
    validation_keep_indexes: list[int] = []
    validation_move_to_train_indexes: list[int] = []

    for rank, (split, index, _) in enumerate(combined):
        is_validation = rank < clamped_target
        match (split, is_validation):
            case ("train", True):
                train_move_to_validation_indexes.append(index)
            case ("train", False):
                train_keep_indexes.append(index)
            case ("validation", True):
                validation_keep_indexes.append(index)
            case _:
                validation_move_to_train_indexes.append(index)

    return ValidationRebalancePlan(
        train_keep_indexes=sorted(train_keep_indexes),
        train_move_to_validation_indexes=sorted(train_move_to_validation_indexes),
        validation_keep_indexes=sorted(validation_keep_indexes),
        validation_move_to_train_indexes=sorted(validation_move_to_train_indexes),
        target_validation_documents=clamped_target,
    )


def apply_validation_rebalance_plan(train: Dataset, validation: Dataset, plan: ValidationRebalancePlan) -> tuple[Dataset, Dataset]:
    """Apply a `ValidationRebalancePlan` to produce rebalanced `train`/`validation` splits.

    Args:
        train: The original `train` split the plan was computed from.
        validation: The original `validation` split the plan was computed
            from.
        plan: The plan returned by `compute_validation_rebalance_plan` for
            `train` and `validation`.

    Returns:
        A `(new_train, new_validation)` tuple of rebalanced datasets.
    """
    train_parts = [train.select(plan["train_keep_indexes"])]
    if plan["validation_move_to_train_indexes"]:
        train_parts.append(validation.select(plan["validation_move_to_train_indexes"]))
    new_train = concatenate_datasets(train_parts) if len(train_parts) > 1 else train_parts[0]

    validation_parts = [validation.select(plan["validation_keep_indexes"])]
    if plan["train_move_to_validation_indexes"]:
        validation_parts.append(train.select(plan["train_move_to_validation_indexes"]))
    new_validation = concatenate_datasets(validation_parts) if len(validation_parts) > 1 else validation_parts[0]

    return new_train, new_validation


def compute_number_of_shards(table: pa.Table, max_output_file_size_gb: float) -> int:
    """Estimate how many Parquet shards a table needs to stay under a target size.

    Takes an already-materialized `pa.Table` (see `write_language_parquet`)
    rather than a `Dataset`, since a `Dataset`'s `.data.nbytes` reflects the
    size of whatever underlying Arrow table(s) it's a view into -- after a
    `Dataset.select`/`concatenate_datasets` call, that can be the entire
    original (e.g. multi-shard, tens-of-thousands-of-rows) table, wildly
    overstating a small selection's real size. A materialized `pa.Table`
    doesn't have this problem, so its own `.nbytes` is accurate.

    Args:
        table: The (already-materialized) Arrow table to size.
        max_output_file_size_gb: Target maximum shard size, in GB.

    Returns:
        The number of shards needed. At least 1, and never more than
        `table.num_rows` (a shard can't usefully hold zero rows).

    Examples:
        >>> import pyarrow as pa
        >>> compute_number_of_shards(pa.table({"x": [1, 2, 3]}), max_output_file_size_gb=1.0)
        1
    """
    if table.num_rows == 0:
        return 1
    # from GB to bytes
    max_output_file_size_bytes = max_output_file_size_gb * 1024 * 1024 * 1024
    return min(table.num_rows, max(1, math.ceil(table.nbytes / max_output_file_size_bytes)))


def write_language_parquet(base_dir: Path, wikipedia_language_code: str, split: str, dataset: Dataset, max_output_file_size_gb: float) -> list[Path]:
    """Write one language/split's dataset to one or more zstd-compressed Parquet shards.

    `dataset` is first fully materialized into a single `pa.Table` (via
    `Dataset.with_format("arrow")[:]`, which resolves any indices mapping
    left over from `Dataset.select`/`concatenate_datasets` into real data),
    then split into contiguous shards each targeting at most
    `max_output_file_size_gb` (see `compute_number_of_shards`), and written
    under the dataset's `data/<language>/<split>/` layout.

    Args:
        base_dir: Directory to write under.
        wikipedia_language_code: Wikipedia language code, e.g. `"en"`.
        split: Either `"train"` or `"validation"`.
        dataset: The dataset to write.
        max_output_file_size_gb: Target maximum shard size, in GB.

    Returns:
        The paths of the written Parquet shard files.
    """
    split_dir = base_dir / "data" / wikipedia_language_code / split
    split_dir.mkdir(parents=True, exist_ok=True)
    table = dataset.with_format("arrow")[:]
    number_of_shards = compute_number_of_shards(table, max_output_file_size_gb)
    rows_per_shard = math.ceil(table.num_rows / number_of_shards) if table.num_rows else 0
    output_paths = []
    for shard_index in range(number_of_shards):
        shard_table = table.slice(shard_index * rows_per_shard, rows_per_shard)
        output_path = split_dir / f"000_{shard_index:05d}.parquet"
        # datasets>=5's Dataset.to_parquet always passes its own `compression` kwarg to
        # pyarrow internally, so forwarding compression="zstd" through it raises
        # "got multiple values for keyword argument 'compression'" -- write via pyarrow directly instead.
        pq.write_table(shard_table, output_path, compression="zstd")
        output_paths.append(output_path)
    return output_paths


def push_language_to_hub(
    api: HfApi,
    hf_dataset_repo_id: str,
    wikipedia_language_code: str,
    train_paths: list[Path],
    validation_paths: list[Path],
    revision: str | None,
) -> None:
    """Replace a language's `train`/`validation` Parquet shards on the Hub in one commit.

    Deletes every existing file under `data/<wikipedia_language_code>/` in
    the Hub dataset repo and adds the given (already-deduplicated) Parquet
    shard files in their place, so no stale duplicate shards are left
    behind.

    Args:
        api: An authenticated `HfApi` client.
        hf_dataset_repo_id: HuggingFace Hub dataset repository
            (`namespace/name`) to write to.
        wikipedia_language_code: Wikipedia language code, e.g. `"en"`.
        train_paths: Local paths to the deduplicated `train` Parquet
            shards.
        validation_paths: Local paths to the deduplicated `validation`
            Parquet shards.
        revision: Branch (or other revision) of the Hub dataset repo to
            write to. `None` uses the repo's default branch.
    """
    existing_files = [
        path
        for path in api.list_repo_files(hf_dataset_repo_id, repo_type="dataset", revision=revision)
        if path.startswith(f"data/{wikipedia_language_code}/")
    ]
    operations: list[CommitOperationAdd | CommitOperationDelete] = [CommitOperationDelete(path_in_repo=path) for path in existing_files]
    for path in train_paths:
        operations.append(CommitOperationAdd(path_in_repo=f"data/{wikipedia_language_code}/train/{path.name}", path_or_fileobj=str(path)))
    for path in validation_paths:
        operations.append(CommitOperationAdd(path_in_repo=f"data/{wikipedia_language_code}/validation/{path.name}", path_or_fileobj=str(path)))
    api.create_commit(
        repo_id=hf_dataset_repo_id,
        repo_type="dataset",
        operations=operations,
        revision=revision,
        commit_message=f"Deduplicate {wikipedia_language_code} by id, keeping the highest version",
    )


def main(
    languages: Annotated[list[WikipediaLanguageCode] | None, typer.Option("-l", "--language", help="Language(s) to deduplicate. Repeatable. Defaults to every config found in --hf-dataset-repo-id.")] = None,
    hf_dataset_repo_id: Annotated[str, typer.Option("--hf-dataset-repo-id", help="HuggingFace Hub dataset repository (`namespace/name`) to read from, and, with --push, write back to.")] = "ucrelnlp/Multilingual-USAS-Labelled-Silver-Wikipedia",
    hf_dataset_revision: Annotated[str | None, typer.Option("--hf-dataset-revision", help="Branch (or other revision) of the Hub dataset repo to read/write. Defaults to the repo's default branch.")] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir", help="Local directory to write the deduplicated Parquet output to, in `output_dir/data/<language>/{train,validation}/` subfolders. Omit to skip writing local output.")] = None,
    push: Annotated[bool, typer.Option("--push/--no-push", help="Whether to commit the deduplicated data back to --hf-dataset-repo-id, replacing each processed language's existing train/validation Parquet shards. Defaults to False, a dry run that only reports what would change.")] = False,
    max_output_file_size: Annotated[float, typer.Option("-e", "--max-output-file-size", help="Target maximum size in GB per output Parquet shard pre-compression (the actually file size will be a lot smaller due to the compression); larger splits are written as multiple shard files instead of one.")] = 1.0,
    max_validation_documents: Annotated[int, typer.Option("-n", "--max-validation-documents", help="Cap each language's post-dedup validation split at this many rows, moving any excess back to train (deterministically, via the same page_id hash TrainValidationSplitAnnotator uses). Fixes the known per-rank validation cap bug in train_validation_split.py, where a language whose task/rank count exceeds this value ends up with closer to that rank count of validation rows instead of this cap, without needing to re-run the whole pipeline. Should match the --max-validation-documents value the data was originally built with.")] = 20,
    max_validation_percentage: Annotated[float | None, typer.Option("-p", "--max-validation-percentage", help="Target percentage (0-100) of each language's post-dedup total documents assigned to validation, mirroring build_usas_wikipedia_dataset.py's --validation-percentage. When given, each language's validation split is rebalanced to exactly min(round(total_post_dedup_documents * max_validation_percentage / 100), max_validation_documents) -- unlike --max-validation-documents alone, this can grow validation (pulling rows from train) as well as shrink it. Omit (default) to keep the existing shrink-only --max-validation-documents cap behavior unchanged.")] = None,
) -> None:
    """Deduplicate the Multilingual USAS Wikipedia dataset by `id`, per language.

    For each selected language, every row of `train` and `validation`
    (combined) is grouped by `id`. Where an `id` occurs more than once, only
    the row with the highest `version` is kept. If the duplicate copies span
    both `train` and `validation`, the surviving row always ends up in
    `validation`; duplicates confined to a single split keep the surviving
    row in that same split. With neither --output-dir nor --push, this only
    prints a report of what would change.

    Examples:
        Report duplicate counts for every language without writing anything:

        $ uv run processing_scripts/deduplicate_wikipedia_dataset.py

        Write deduplicated Parquet for Danish only to a local directory:

        $ uv run processing_scripts/deduplicate_wikipedia_dataset.py \\
              -l da --output-dir ./local_dedup

        Deduplicate every language and push the result back to the Hub:

        $ uv run processing_scripts/deduplicate_wikipedia_dataset.py --push

        Also rebalance validation to 10% of each language's documents (capped
        at --max-validation-documents), growing an undersized split as well
        as shrinking an oversized one:

        $ uv run processing_scripts/deduplicate_wikipedia_dataset.py \\
              -p 10 --push
    """
    load_dotenv()
    wikipedia_language_codes = [language.value for language in languages] if languages else get_dataset_config_names(hf_dataset_repo_id, revision=hf_dataset_revision)

    api = HfApi() if push else None
    summary_table = Table(title="Deduplication report")
    for column in ("language", "train (before)", "validation (before)", "cross-split ids", "removed from train", "removed from validation", "validation target", "train → validation", "validation → train", "train (after)", "validation (after)"):
        summary_table.add_column(column)

    with tempfile.TemporaryDirectory() as tmp_dir:
        staging_dir = Path(output_dir) if output_dir is not None else Path(tmp_dir)
        for wikipedia_language_code in wikipedia_language_codes:
            train = load_dataset(hf_dataset_repo_id, wikipedia_language_code, split="train", revision=hf_dataset_revision)
            validation = load_dataset(hf_dataset_repo_id, wikipedia_language_code, split="validation", revision=hf_dataset_revision)
            plan = compute_language_dedup_plan(train, validation)

            new_train, new_validation = apply_dedup_plan(train, validation, plan)

            match max_validation_percentage:
                case None:
                    validation_keep_indexes, validation_overflow_indexes = compute_validation_overflow_plan(new_validation, max_validation_documents)
                    if validation_overflow_indexes:
                        overflow_to_train = new_validation.select(validation_overflow_indexes)
                        new_validation = new_validation.select(validation_keep_indexes)
                        new_train = concatenate_datasets([new_train, overflow_to_train])
                    target_display = "—"
                    moved_to_validation = 0
                    moved_to_train = len(validation_overflow_indexes)
                case _:
                    total_documents = len(new_train) + len(new_validation)
                    target_validation_documents = compute_target_validation_documents(total_documents, max_validation_percentage, max_validation_documents)
                    rebalance_plan = compute_validation_rebalance_plan(new_train, new_validation, target_validation_documents)
                    new_train, new_validation = apply_validation_rebalance_plan(new_train, new_validation, rebalance_plan)
                    target_display = str(rebalance_plan["target_validation_documents"])
                    moved_to_validation = len(rebalance_plan["train_move_to_validation_indexes"])
                    moved_to_train = len(rebalance_plan["validation_move_to_train_indexes"])

            summary_table.add_row(
                wikipedia_language_code,
                str(len(train)),
                str(len(validation)),
                str(plan["cross_split_ids"]),
                str(plan["removed_from_train"]),
                str(plan["removed_from_validation"]),
                target_display,
                str(moved_to_validation),
                str(moved_to_train),
                str(len(new_train)),
                str(len(new_validation)),
            )

            if output_dir is None and not push:
                continue

            train_paths = write_language_parquet(staging_dir, wikipedia_language_code, "train", new_train, max_output_file_size)
            validation_paths = write_language_parquet(staging_dir, wikipedia_language_code, "validation", new_validation, max_output_file_size)

            if push and api is not None:
                rprint(f"Pushing deduplicated {wikipedia_language_code!r} data to {hf_dataset_repo_id!r}")
                push_language_to_hub(api, hf_dataset_repo_id, wikipedia_language_code, train_paths, validation_paths, hf_dataset_revision)

    rprint(summary_table)


if __name__ == "__main__":
    typer.run(main)
