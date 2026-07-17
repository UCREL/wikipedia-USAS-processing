from importlib.resources import files
from pathlib import Path

import datasets
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


def number_of_shards(dataset: datasets.IterableDataset) -> int:
    return dataset.n_shards


def create_sub_directory(parent_directory: Path, sub_directory_name: str) -> str:
    """Create a sub-directory under a parent directory, including any missing parents.

    If the sub-directory already exists, it is left untouched.

    Args:
        parent_directory: The directory under which the sub-directory should be created.
        sub_directory_name: The name of the sub-directory to create.

    Returns:
        The resolved, absolute path to the sub-directory, as a string.
    """
    sub_directory = parent_directory /sub_directory_name
    sub_directory.mkdir(parents=True, exist_ok=True)
    return str(sub_directory.resolve())


def get_usas_language_processing_information(wikipedia_language_code: str, language_data_file: Path | None = None) -> dict[str, str]:
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
        The dictionary of metadata for the matching language, as found in the
        "languages" list of the YAML file (e.g. "language", "iso_639_3",
        "wikipedia_code", "training", "data_trove_language").

    Raises:
        ValueError: If no entry in the "languages" list has a "wikipedia_code"
            matching wikipedia_language_code.
    """
    if language_data_file is None:
        usas_wikipedia_processing_file_str = str(files("wikipedia_processing").joinpath("data/usas_wikipedia_processing.yaml"))
        language_data_file_path = Path(usas_wikipedia_processing_file_str)
    else:
        language_data_file_path = language_data_file

    with language_data_file_path.open("r", encoding="utf-8") as fp:
        yaml_data = yaml_load(fp, Loader=Loader)
    for language in yaml_data["languages"]:
        if language["wikipedia_code"] == wikipedia_language_code:
            return language
    raise ValueError(f"Language {wikipedia_language_code} not found in {language_data_file}")