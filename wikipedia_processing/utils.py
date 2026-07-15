from pathlib import Path

from yaml import Loader, load as yaml_load

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