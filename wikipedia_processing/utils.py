import json
import math
import os
import time
from importlib.resources import files
from pathlib import Path
from typing import Callable, Generator, TypedDict, cast

import datasets
import typer
from datatrove.data import Document, DocumentsPipeline
from datatrove.utils.logging import logger as data_trove_logger
from yaml import Loader
from yaml import load as yaml_load


def truncate_to_255_bytes(string_to_truncate: str) -> str:
    """
    Truncate a string to at most 255 UTF-8 bytes, respecting character boundaries.

    Args:
        string_to_truncate: The string to truncate.

    Returns:
        The truncated string.
    """
    encoded = string_to_truncate.encode("utf-8")
    if len(encoded) <= 255:
        return string_to_truncate
    # Decode with errors='ignore' drops any incomplete multi-byte sequence at the cut
    return encoded[:255].decode("utf-8", errors="ignore")

def get_available_cpu_count() -> int:
    """Get the number of CPUs available to the current process.

    Prefers `os.sched_getaffinity`, which respects CPU affinity masks and
    container/cgroup CPU limits (the same semantics as the 3.13+-only
    `os.process_cpu_count()`), falling back to `os.cpu_count()` on
    platforms where `sched_getaffinity` doesn't exist (e.g. macOS).

    Returns:
        The number of available CPUs, or 1 if this cannot be determined.

    Examples:
        >>> get_available_cpu_count() >= 1
        True
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def load_page_meta_data_file(meta_data_file: Path) -> dict[str, int]:

    return {}






def convert_dataset_to_lookup(dataset: datasets.IterableDataset) -> dict[int, str]:
    """
    Convert an iterable dataset to a dictionary mapping page IDs to their corresponding titles.

    Args:
        dataset: An iterable dataset containing "page_id" and "page_title" keys.

    Returns:
        A dictionary mapping page IDs to their corresponding titles.
    """
    page_id_to_title: dict[int, str] = {}
    for entry in dataset:
        page_id = entry["page_id"]
        title = entry["page_title"]
        page_id_to_title[page_id] = title
    return page_id_to_title


def get_number_of_shards(dataset: datasets.IterableDataset) -> int:
    """Get the number of shards backing an iterable dataset.

    Args:
        dataset: An iterable dataset to inspect.

    Returns:
        The number of shards in the dataset.
    """
    return dataset.n_shards


def create_sub_directory(parent_directory: Path, sub_directory_name: str) -> str:
    """Create a sub-directory under a parent directory, including any missing parents.

    If the sub-directory already exists, it is left untouched.

    Args:
        parent_directory: The directory under which the sub-directory should be created.
        sub_directory_name: The name of the sub-directory to create.

    Returns:
        The resolved, absolute path to the sub-directory, as a string.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     sub_dir = create_sub_directory(Path(tmp), "sub")
        ...     Path(sub_dir).is_dir()
        True
    """
    sub_directory = parent_directory /sub_directory_name
    sub_directory.mkdir(parents=True, exist_ok=True)
    return str(sub_directory.resolve())


class UsasLanguageProcessingInformation(TypedDict):
    """Wikipedia and USAS processing metadata for a single language.

    Attributes:
        language: The English name of the language, e.g. "English".
        iso_639_3: The ISO 639-3 code for the language, e.g. "eng".
        wikipedia_code: The Wikipedia language code, e.g. "en".
        training: Whether this language is included in the training data.
        data_trove_language: The DataTrove language identifier used for
            tokenization, e.g. "english".
    """

    language: str
    iso_639_3: str
    wikipedia_code: str
    training: bool
    data_trove_language: str


def _load_usas_language_processing_entries(
    language_data_file: Path | None,
) -> list[UsasLanguageProcessingInformation]:
    """Load the "languages" list from a USAS processing YAML file.

    Args:
        language_data_file: Path to a YAML file containing a "languages" list.
            If None, the packaged
            ``wikipedia_processing/data/usas_wikipedia_processing.yaml`` file
            is used instead.

    Returns:
        The "languages" list, as parsed from the YAML file.
    """
    if language_data_file is None:
        usas_wikipedia_processing_file_str = str(files("wikipedia_processing").joinpath("data/usas_wikipedia_processing.yaml"))
        language_data_file_path = Path(usas_wikipedia_processing_file_str)
    else:
        language_data_file_path = language_data_file

    with language_data_file_path.open("r", encoding="utf-8") as fp:
        yaml_data = yaml_load(fp, Loader=Loader)
    return yaml_data["languages"]


def get_valid_usas_language_processing_wikipedia_codes(
    language_data_file: Path | None = None,
) -> list[str]:
    """Get all Wikipedia language codes with USAS processing information.

    Args:
        language_data_file: Path to a YAML file containing a "languages" list.
            If None, the packaged
            ``wikipedia_processing/data/usas_wikipedia_processing.yaml`` file
            is used instead.

    Returns:
        The "wikipedia_code" of every entry in the "languages" list of the
        YAML file.

    Examples:
        >>> "en" in get_valid_usas_language_processing_wikipedia_codes()
        True
    """
    entries = _load_usas_language_processing_entries(language_data_file)
    return [language["wikipedia_code"] for language in entries]


def get_usas_language_processing_information(
    wikipedia_language_code: str, language_data_file: Path | None = None
) -> UsasLanguageProcessingInformation:
    """
    Look up USAS processing metadata for a given Wikipedia language code.

    Loads a YAML file containing a "languages" list, where each entry describes
    one language's Wikipedia and USAS processing metadata, and returns the
    entry whose "wikipedia_code" matches wikipedia_language_code.

    Args:
        wikipedia_language_code: A string representing the Wikipedia language
            code to look up, e.g. "en" for English.
        language_data_file: Path to a YAML file containing a "languages" list.
            If None, the packaged
            ``wikipedia_processing/data/usas_wikipedia_processing.yaml`` file
            is used instead.

    Returns:
        The UsasLanguageProcessingInformation for the matching language, as
        found in the "languages" list of the YAML file.

    Raises:
        ValueError: If no entry in the "languages" list has a "wikipedia_code"
            matching wikipedia_language_code.
    """
    entries = _load_usas_language_processing_entries(language_data_file)
    for language in entries:
        if language["wikipedia_code"] == wikipedia_language_code:
            return language
    valid_codes = [language["wikipedia_code"] for language in entries]
    raise ValueError(
        f"Language {wikipedia_language_code!r} not found in {language_data_file}. "
        f"Valid Wikipedia language codes are: {valid_codes}"
    )

def get_hashes_per_bucket(num_buckets: int, threshold: float) -> int:
    """Estimate the number of hashes per bucket (r) for MinHash LSH banding.

    Given a fixed number of buckets (bands) and a target Jaccard similarity
    threshold, solves for the number of hashes per bucket (rows per band)
    that places the LSH S-curve's inflection point at that threshold, using
    the closed-form relationship:

        threshold ≈ (1 / num_buckets) ** (1 / r)

    This anchors the approximate 50%-detection-probability point of the
    S-curve to the given threshold. It does not account for the steepness
    of the curve or optimize the false positive / false negative tradeoff --
    for that, use a weighted search over false_positive_probability and
    false_negative_probability instead.

    Args:
        num_buckets: Number of buckets (bands), i.e. the `b` parameter in
            standard LSH banding notation. Must be a positive integer
            greater than 1 (num_buckets == 1 makes threshold undefined,
            since 1/num_buckets == 1).
        threshold: Target Jaccard similarity threshold, in the open
            interval (0, 1), at which candidate pairs should start being
            flagged as duplicates.

    Returns:
        The estimated number of hashes per bucket (r), rounded to the
        nearest integer. Note this can round to 0 for very loose thresholds
        or large num_buckets -- callers should clamp to a minimum of 1.

    Raises:
        ValueError: If num_buckets < 2, or if threshold is not strictly
            between 0 and 1.

    Examples:
        >>> get_hashes_per_bucket(num_buckets=14, threshold=0.72)
        8
    """
    if num_buckets < 2:
        raise ValueError(f"num_buckets must be >= 2, got {num_buckets}")
    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")

    r = math.log(1 / num_buckets) / math.log(threshold)
    return round(r)


def compute_shard_scaled_tasks_and_workers(
    number_of_shards: int,
    shard_tasks_multiplier: float,
    min_tasks: int,
    max_tasks: int,
    max_workers: int,
) -> tuple[int, int]:
    """Derive a (workers, tasks_multiplier) pair sized off a dataset's shard count.

    Intended for driving `build_usas_wikipedia_dataset.py`'s `-w`/`-t` options
    per language: rather than using the same fixed task count for every
    language regardless of dataset size, this scales the target task count
    with `number_of_shards` (a proxy for data volume), so small languages
    don't get more (mostly empty) tasks than they have data for, and large
    languages get enough parallelism.

    The returned pair satisfies
    `number_of_workers * tasks_multiplier >= target_tasks`, where
    `target_tasks = clamp(round(number_of_shards * shard_tasks_multiplier), min_tasks, max_tasks)`,
    while never exceeding `max_workers` concurrent workers.

    Args:
        number_of_shards: The number of shards backing the language's
            dataset, e.g. from `get_number_of_shards`.
        shard_tasks_multiplier: Target number of tasks per shard.
        min_tasks: Minimum target task count, regardless of shard count.
        max_tasks: Maximum target task count, regardless of shard count.
        max_workers: Maximum number of concurrent workers to use.

    Returns:
        A `(number_of_workers, tasks_multiplier)` tuple.

    Raises:
        ValueError: If `min_tasks`, `max_tasks`, or `max_workers` is not a
            positive integer, or if `min_tasks` is greater than `max_tasks`.

    Examples:
        >>> compute_shard_scaled_tasks_and_workers(
        ...     number_of_shards=2, shard_tasks_multiplier=3.0,
        ...     min_tasks=4, max_tasks=200, max_workers=16,
        ... )
        (6, 1)
        >>> compute_shard_scaled_tasks_and_workers(
        ...     number_of_shards=1000, shard_tasks_multiplier=3.0,
        ...     min_tasks=4, max_tasks=200, max_workers=16,
        ... )
        (16, 13)
    """
    if min_tasks < 1 or max_tasks < 1 or max_workers < 1:
        raise ValueError(f"min_tasks, max_tasks, and max_workers must all be >= 1, got {min_tasks!r}, {max_tasks!r}, {max_workers!r}")
    if min_tasks > max_tasks:
        raise ValueError(f"min_tasks must be <= max_tasks, got min_tasks={min_tasks!r}, max_tasks={max_tasks!r}")

    target_tasks = min(max(round(number_of_shards * shard_tasks_multiplier), min_tasks), max_tasks)
    number_of_workers = min(max_workers, target_tasks)
    tasks_multiplier = math.ceil(target_tasks / number_of_workers)
    return number_of_workers, tasks_multiplier


def scale_workers_to_budget(workers: list[int], budget: int) -> list[int]:
    """Shrink a list of worker counts so their sum fits within a shared budget.

    Intended for capping the total number of concurrently-running Slurm
    tasks across multiple independently-launched language pipelines: each
    entry in `workers` is one language's own concurrency throttle (its
    `-w`), and this redistributes them so their sum no longer exceeds
    `budget`.

    Every entry keeps a floor of 1. When shrinking is required, each
    entry's *reduction* is taken out of its "extra" allocation above that
    floor (`workers[i] - 1`), proportionally to that extra amount, using
    the largest-remainder method to round the proportional shares to whole
    workers -- so entries with a larger original allocation lose more
    workers, in proportion, than entries already close to the floor, and
    the returned total is exactly `budget` (when `sum(workers) > budget`).

    Args:
        workers: Each entry's original worker count. Every value must be
            >= 1.
        budget: Maximum total to distribute across all entries combined.
            Must be >= len(workers), since every entry needs at least 1
            worker.

    Returns:
        A new list, same length and order as `workers`, where every value
        is `>= 1` and `<= workers[i]`, and the total is `<= budget`
        (exactly `budget` whenever `sum(workers) > budget`).

    Raises:
        ValueError: If `budget < len(workers)`, or if any entry in
            `workers` is < 1.

    Examples:
        >>> scale_workers_to_budget([16, 16, 16, 16], budget=40)
        [10, 10, 10, 10]
        >>> scale_workers_to_budget([4, 4, 4], budget=20)
        [4, 4, 4]
    """
    if any(worker_count < 1 for worker_count in workers):
        raise ValueError(f"All worker counts must be >= 1, got {workers!r}")
    if budget < len(workers):
        raise ValueError(f"budget must be >= len(workers) ({len(workers)}) so every entry keeps >= 1 worker, got budget={budget!r}")

    total = sum(workers)
    if total <= budget:
        return list(workers)

    extra_weights = [worker_count - 1 for worker_count in workers]
    total_extra_weight = sum(extra_weights)
    remaining_budget = budget - len(workers)

    if total_extra_weight == 0:
        return [1] * len(workers)

    raw_extra_shares = [remaining_budget * extra_weight / total_extra_weight for extra_weight in extra_weights]
    extra_scaled = [math.floor(share) for share in raw_extra_shares]
    leftover = remaining_budget - sum(extra_scaled)

    remainder_order = sorted(range(len(workers)), key=lambda index: raw_extra_shares[index] - extra_scaled[index], reverse=True)
    for index in remainder_order:
        if leftover <= 0:
            break
        if extra_scaled[index] < extra_weights[index]:
            extra_scaled[index] += 1
            leftover -= 1

    return [1 + extra for extra in extra_scaled]


def get_progress_logger_function(step_name: str, log_every: int = 1000) -> Callable[[DocumentsPipeline, int, int], Generator[Document, None, None]]:
    """Build a DataTrove pipeline step that periodically logs document throughput.

    Yields every document unchanged, logging a running count every `log_every`
    documents so that long-running, otherwise-silent steps (e.g. streaming a
    dataset from HuggingFace) show visible, per-task progress in the task log
    files.

    Args:
        step_name: Label included in each log line, to identify which pipeline
            step is reporting progress (e.g. "reading").
        log_every: Number of documents between progress log lines.

    Returns:
        A callable with the DataTrove custom pipeline step signature
        `(data, rank, world_size) -> DocumentsPipeline`.
    """
    def log_progress(data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> Generator[Document, None, None]:
        count = 0
        if data:
            for document in cast(Generator[Document, None, None], data):
                count += 1
                if count % log_every == 0:
                    data_trove_logger.info(f"[{step_name}] rank={rank} processed {count} documents so far")
                yield document
        data_trove_logger.info(f"[{step_name}] rank={rank} finished, processed {count} documents total")
    return log_progress


def time_elapsed(start_time: float) -> float:
    """Compute the elapsed time since a starting `time.perf_counter()` reading.

    Args:
        start_time: A starting timestamp, as returned by `time.perf_counter()`.

    Returns:
        The elapsed time in seconds since `start_time`.
    """
    return time.perf_counter() - start_time


def parse_json_object(value: str) -> dict:
    """Parse a JSON object string into a dictionary.

    Args:
        value: A JSON-encoded object, e.g. `{"account": "myaccount"}`.

    Returns:
        The decoded dictionary.

    Raises:
        typer.BadParameter: If value is not valid JSON or does not decode to
            a JSON object.

    Examples:
        >>> parse_json_object('{"account": "myaccount"}')
        {'account': 'myaccount'}
    """
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise typer.BadParameter(f"Invalid JSON: {value!r}") from error
    if not isinstance(parsed, dict):
        raise typer.BadParameter(f"Expected a JSON object, got: {value!r}")
    return parsed