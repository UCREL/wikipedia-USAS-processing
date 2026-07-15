import json
from pathlib import Path
from time import perf_counter
import multiprocessing
from yaml import Loader, load as yaml_load

from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()

def get_number_of_shards(dataset_id: str, dataset_name: str, split: str) -> int:
    dataset = load_dataset(dataset_id, name=dataset_name, split=split, streaming=True)
    return dataset.n_shards


def get_language_to_wiki_code() -> dict[str, str]:
    yaml_file = Path("data/languages.yaml")
    with yaml_file.open("r", encoding="utf-8") as fp:
        yaml_data = yaml_load(fp, Loader=Loader)
    return {language["language"]: language["wikipedia_code"] for language in yaml_data["languages"]}

languages_requires = [
    "English",
    "Dutch",
    "Spanish",
    "Danish",
    "Italian",
    "Portuguese",
    "Chinese",
    "Finnish",
    "Irish",
    "Welsh",
]

def main():
    raise ValueError()
    huggingface_dataset_id = "HuggingFaceFW/finewiki"
    languages_to_wiki_code = get_language_to_wiki_code()
    for language in languages_requires:
        print(f"Processing {language}")
        start_time = perf_counter() 
        dataset_name = languages_to_wiki_code[language]
        dataset_split = "train"
        
        wiki_dataset = load_dataset("HuggingFaceFW/finewiki", name=dataset_name, split=dataset_split, streaming=True, columns=["text", "id", "title", "url", "page_id"])
        data_folder = Path(f"data/wikipedia_pages")
        data_folder.mkdir(parents=True, exist_ok=True)
        data_file = Path(data_folder, f"{language}_page_data.jsonl")
        count = 0
        with data_file.open("w", encoding="utf-8") as fp:
            for data in wiki_dataset:
                fp.write(json.dumps(data) + "\n")
                count += 1
                if count == 50000:
                    break
        end_time = perf_counter()
        print(f"Time taken: {end_time - start_time}")
        print(f"Finished {language}")
        print()

if __name__ == "__main__":
    main()