from importlib.resources import files
from pathlib import Path
from typing import TypedDict

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