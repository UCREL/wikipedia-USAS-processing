"""Deduplicate the Multilingual USAS Wikipedia dataset by `id`, per language."""

import tempfile
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Annotated, TypedDict

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


def write_language_parquet(base_dir: Path, wikipedia_language_code: str, split: str, dataset: Dataset) -> Path:
    """Write one language/split's dataset to a single zstd-compressed Parquet file.

    Args:
        base_dir: Directory to write under, mirroring the dataset's
            `data/<language>/<split>/` layout.
        wikipedia_language_code: Wikipedia language code, e.g. `"en"`.
        split: Either `"train"` or `"validation"`.
        dataset: The dataset to write.

    Returns:
        The path of the written Parquet file.
    """
    split_dir = base_dir / "data" / wikipedia_language_code / split
    split_dir.mkdir(parents=True, exist_ok=True)
    output_path = split_dir / "000_00000.parquet"
    dataset.to_parquet(str(output_path), compression="zstd")
    return output_path


def push_language_to_hub(
    api: HfApi,
    hf_dataset_repo_id: str,
    wikipedia_language_code: str,
    train_path: Path,
    validation_path: Path,
    revision: str | None,
) -> None:
    """Replace a language's `train`/`validation` Parquet shards on the Hub in one commit.

    Deletes every existing file under `data/<wikipedia_language_code>/` in
    the Hub dataset repo and adds the two given (already-deduplicated,
    single-shard) Parquet files in their place, so no stale duplicate
    shards are left behind.

    Args:
        api: An authenticated `HfApi` client.
        hf_dataset_repo_id: HuggingFace Hub dataset repository
            (`namespace/name`) to write to.
        wikipedia_language_code: Wikipedia language code, e.g. `"en"`.
        train_path: Local path to the deduplicated `train` Parquet file.
        validation_path: Local path to the deduplicated `validation`
            Parquet file.
        revision: Branch (or other revision) of the Hub dataset repo to
            write to. `None` uses the repo's default branch.
    """
    existing_files = [
        path
        for path in api.list_repo_files(hf_dataset_repo_id, repo_type="dataset", revision=revision)
        if path.startswith(f"data/{wikipedia_language_code}/")
    ]
    operations: list[CommitOperationAdd | CommitOperationDelete] = [CommitOperationDelete(path_in_repo=path) for path in existing_files]
    operations.append(CommitOperationAdd(path_in_repo=f"data/{wikipedia_language_code}/train/000_00000.parquet", path_or_fileobj=str(train_path)))
    operations.append(CommitOperationAdd(path_in_repo=f"data/{wikipedia_language_code}/validation/000_00000.parquet", path_or_fileobj=str(validation_path)))
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
    """
    load_dotenv()
    wikipedia_language_codes = [language.value for language in languages] if languages else get_dataset_config_names(hf_dataset_repo_id, revision=hf_dataset_revision)

    api = HfApi() if push else None
    summary_table = Table(title="Deduplication report")
    for column in ("language", "train (before)", "validation (before)", "cross-split ids", "removed from train", "removed from validation", "train (after)", "validation (after)"):
        summary_table.add_column(column)

    with tempfile.TemporaryDirectory() as tmp_dir:
        staging_dir = Path(output_dir) if output_dir is not None else Path(tmp_dir)
        for wikipedia_language_code in wikipedia_language_codes:
            train = load_dataset(hf_dataset_repo_id, wikipedia_language_code, split="train", revision=hf_dataset_revision)
            validation = load_dataset(hf_dataset_repo_id, wikipedia_language_code, split="validation", revision=hf_dataset_revision)
            plan = compute_language_dedup_plan(train, validation)

            new_train, new_validation = apply_dedup_plan(train, validation, plan)
            summary_table.add_row(
                wikipedia_language_code,
                str(len(train)),
                str(len(validation)),
                str(plan["cross_split_ids"]),
                str(plan["removed_from_train"]),
                str(plan["removed_from_validation"]),
                str(len(new_train)),
                str(len(new_validation)),
            )

            if output_dir is None and not push:
                continue

            train_path = write_language_parquet(staging_dir, wikipedia_language_code, "train", new_train)
            validation_path = write_language_parquet(staging_dir, wikipedia_language_code, "validation", new_validation)

            if push and api is not None:
                rprint(f"Pushing deduplicated {wikipedia_language_code!r} data to {hf_dataset_repo_id!r}")
                push_language_to_hub(api, hf_dataset_repo_id, wikipedia_language_code, train_path, validation_path, hf_dataset_revision)

    rprint(summary_table)


if __name__ == "__main__":
    typer.run(main)
