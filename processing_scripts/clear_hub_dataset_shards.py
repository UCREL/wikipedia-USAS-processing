"""Delete stale Parquet shards from a HuggingFace Hub dataset repo before re-running the pipeline.

`HuggingFaceDatasetWriter` (see `ReusableHuggingFaceDatasetWriter` in
`wikipedia_processing/pipelines/hf_writer.py`) only ever adds/overwrites the
specific shard files it writes -- it never deletes pre-existing files in the
repo. If a previous run wrote more shards than the current run produces
(e.g. `data/da/train/003.parquet` from an older, larger run), those extra
shards are left behind and silently included in the dataset. This script
clears a language's (or the whole repo's) existing shards first, so a
re-run starts from a clean slate.
"""

from enum import Enum
from typing import Annotated

import typer
from dotenv import load_dotenv
from huggingface_hub import CommitOperationDelete, HfApi
from rich import print as rprint
from rich.table import Table

from wikipedia_processing.utils import (
    get_valid_usas_language_processing_wikipedia_codes,
)

WikipediaLanguageCode = Enum("WikipediaLanguageCode", [(value, value) for value in get_valid_usas_language_processing_wikipedia_codes()], type=str)


def matching_shard_paths(
    api: HfApi,
    hf_dataset_repo_id: str,
    revision: str | None,
    path_in_repo: str,
    languages: list[str] | None,
) -> list[str]:
    """Find existing repo files under `path_in_repo`, optionally scoped to specific languages.

    Args:
        api: An authenticated `HfApi` client.
        hf_dataset_repo_id: HuggingFace Hub dataset repository
            (`namespace/name`) to inspect.
        revision: Branch (or other revision) of the Hub dataset repo to
            read. `None` uses the repo's default branch.
        path_in_repo: Repo-relative folder prefix shards are written under,
            e.g. `"data"`.
        languages: Wikipedia language codes to scope the search to, matching
            files under `path_in_repo/<language>/`. `None` matches every
            file under `path_in_repo/`.

    Returns:
        Repo-relative paths of every matching file, in the order returned by
        the Hub.
    """
    prefixes = [f"{path_in_repo}/{language}/" for language in languages] if languages else [f"{path_in_repo}/"]
    existing_files = api.list_repo_files(hf_dataset_repo_id, repo_type="dataset", revision=revision)
    return [path for path in existing_files if any(path.startswith(prefix) for prefix in prefixes)]


def main(
    languages: Annotated[list[WikipediaLanguageCode] | None, typer.Option("-l", "--language", help="Language(s) to clear shards for, matching `<path-in-repo>/<language>/`. Repeatable. Defaults to clearing every file under --path-in-repo.")] = None,
    hf_dataset_repo_id: Annotated[str, typer.Option("--hf-dataset-repo-id", help="HuggingFace Hub dataset repository (`namespace/name`) to clear shards from.")] = "ucrelnlp/Multilingual-USAS-Labelled-Silver-Wikipedia",
    hf_dataset_revision: Annotated[str | None, typer.Option("--hf-dataset-revision", help="Branch (or other revision) of the Hub dataset repo to clear shards from. Defaults to the repo's default branch.")] = None,
    path_in_repo: Annotated[str, typer.Option("--path-in-repo", help="Repo-relative folder prefix shards are written under.")] = "data",
    delete: Annotated[bool, typer.Option("--delete/--dry-run", help="Whether to actually delete the matched files. Defaults to a dry run that only reports what would be deleted.")] = False,
    yes: Annotated[bool, typer.Option("-y", "--yes", help="Skip the confirmation prompt before deleting. Only used with --delete.")] = False,
) -> None:
    """Delete stale Parquet shards from a HuggingFace Hub dataset repo before re-running the pipeline.

    `HuggingFaceDatasetWriter` only ever adds/overwrites the specific shard
    files it writes, so a run that produces fewer shards than a previous run
    leaves the extra old shards behind. This clears them out first, either
    for specific --language(s) or, by default, everything under
    --path-in-repo. With --delete, the removal is a single commit; since the
    Hub dataset repo is git-backed, the deleted files remain recoverable
    from the repo's commit history (via `revision=<sha>`) as long as that
    commit stays reachable.

    Examples:
        Report what would be deleted for Danish, without deleting anything:

        $ uv run processing_scripts/clear_hub_dataset_shards.py -l da

        Actually delete Danish's existing shards, with a confirmation prompt:

        $ uv run processing_scripts/clear_hub_dataset_shards.py -l da --delete

        Delete every language's shards without a confirmation prompt:

        $ uv run processing_scripts/clear_hub_dataset_shards.py --delete --yes
    """
    load_dotenv()
    wikipedia_language_codes = [language.value for language in languages] if languages else None

    api = HfApi()
    matched_paths = matching_shard_paths(api, hf_dataset_repo_id, hf_dataset_revision, path_in_repo, wikipedia_language_codes)

    if not matched_paths:
        rprint(f"No files found under {path_in_repo!r} in {hf_dataset_repo_id!r}. Nothing to do.")
        return

    table = Table(title=f"Shards under {path_in_repo!r} in {hf_dataset_repo_id!r}")
    table.add_column("Path in repo")
    for path in matched_paths:
        table.add_row(path)
    rprint(table)

    if not delete:
        rprint(f"Dry run: {len(matched_paths)} file(s) would be deleted. Pass --delete to actually delete them.")
        return

    if not yes and not typer.confirm(f"Delete these {len(matched_paths)} file(s) from {hf_dataset_repo_id!r}?"):
        rprint("Aborted.")
        raise typer.Exit(code=1)

    operations = [CommitOperationDelete(path_in_repo=path) for path in matched_paths]
    api.create_commit(
        repo_id=hf_dataset_repo_id,
        repo_type="dataset",
        operations=operations,
        revision=hf_dataset_revision,
        commit_message=f"Clear {len(operations)} stale shard(s) under {path_in_repo}",
    )
    rprint(f"Deleted {len(operations)} file(s) from {hf_dataset_repo_id!r}.")


if __name__ == "__main__":
    typer.run(main)
