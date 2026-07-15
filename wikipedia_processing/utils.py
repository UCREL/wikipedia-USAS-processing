from pathlib import Path

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


def get_language_information(wikipedia_language_code: str, language_data_file: Path) -> dict[str, str]:
    """
    
    """
    with language_data_file.open("r", encoding="utf-8") as fp:
        yaml_data = yaml_load(fp, Loader=Loader)
    for language in yaml_data["languages"]:
        if language["wikipedia_code"] == wikipedia_language_code:
            return language
    raise ValueError(f"Language {wikipedia_language_code} not found in {language_data_file}")