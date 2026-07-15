import logging
from numpy import number
from typing import Annotated
import json
from pathlib import Path
from time import perf_counter
from typing import Callable, Any
import tempfile
import re

import typer
from datasets import load_dataset, Value
from datasets import IterableDataset
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.filters import LambdaFilter
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.dedup.exact_dedup import ExactDedupConfig, ExactDedupSignature, ExactFindDedups, ExactDedupFilter
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupSignature, MinhashDedupBuckets, MinhashDedupCluster, MinhashDedupFilter 
from datatrove.utils.logging import logger as data_trove_logger
from datatrove.data import Document
from datatrove.utils.typeshelper import Languages as DatatroveLanguages
from datatrove.pipeline.formatters.base import BaseFormatter
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.text import PUNCTUATION_SET, split_into_words
from datatrove.utils.typeshelper import Languages
from dotenv import load_dotenv
import mistune
import spacy
from usas_validator.utils import keep_valid_usas_tags, load_usas_mapper, mwe_token_indexes_from_slices, mwe_token_labels_from_indexes, mwe_labels_from_pymusas_indexes

from wikipedia_processing.markdown_renderer import FineWikiPlainTextRenderer
from wikipedia_processing.utils import get_language_information

import datasets

load_dotenv()

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


def number_of_shards(dataset: IterableDataset) -> int:
    return dataset.n_shards


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

def is_in_lookup(page_id: int,
                 page_title: str,
                 lookup: dict[int, str],
                 use_title: bool) -> bool:
    """
    Check if page_id is in lookup dataset if so returns True else returns False.

    Args:
        page_id: A page ID to lookup.
        page_title: A page title to lookup if use_title is True.
        lookup: A dictionary mapping page IDs to their corresponding titles.
        use_title: A boolean indicating whether to check the title as well,
            if the title does not match the lookup then it will return False.

    Returns:
        A boolean indicating whether the page_id is in the lookup dataset,
        True it is in the lookup dataset else False.
    """
    if page_id not in lookup:
        return False
    
    if use_title:
        # titles in the lookup are truncated to 255 bytes whereby a character
        # can be up to 4 bytes due to UTF-8 encoding
        title = truncate_to_255_bytes(page_title)
        if title != lookup[page_id]:
            return False
    return True


def get_language_good_article_id_titles(wikipedia_language_code: str) -> dict[int, str]:
    """
    Given a Wikipedia language code it loads the good and featured Wikipedia article IDs
    from the ucrelnlp/wikipedia-ga-fa-ids dataset and converts them to a dictionary
    mapping page IDs to their corresponding titles.

    The returned dictionary only contains page IDs from good and featured Wikipedia articles
    for the given language.
    
    Args:
        wikipedia_language_code: A string representing the Wikipedia language code.

    Returns:
        A dictionary of good and featured Wikipedia article page IDs to their corresponding titles.
    """
    filter_dataset = load_dataset("ucrelnlp/wikipedia-ga-fa-ids", wikipedia_language_code, streaming=True)
    
    good_article_id_titles: dict[int, str] = {}
    for sample in filter_dataset["train"]:
        page_id = sample["page_id"]
        if page_id in good_article_id_titles:
            raise KeyError("Duplicate page id")
        good_article_id_titles[page_id] = sample["page_title"]
    return good_article_id_titles

def get_relevant_page_function(wikipedia_language_code: str, use_title: bool) -> Callable[[Document], bool]:
    page_id_to_title: dict[int, str] = get_language_good_article_id_titles(wikipedia_language_code)
    def is_relevant_page(article_data: Document) -> bool:
        page_id = article_data.metadata["page_id"]
        page_title = article_data.metadata["title"]

        # To remove
        if page_id == 1171348:
            return True

        return is_in_lookup(page_id, page_title, page_id_to_title, use_title)
    return is_relevant_page

def create_sub_directory(parent_directory: Path, sub_directory_name: str) -> str:
    """
    """
    sub_directory = parent_directory /sub_directory_name
    sub_directory.mkdir(parents=True, exist_ok=True)
    return str(sub_directory.resolve())


class WikipediaMarkdownFormatter(BaseFormatter):

    name = "⬇️ Wiki Markdown Formatter"
    def __init__(self):
        super().__init__()
        self.claude_markdown_parser =  mistune.create_markdown(
            renderer=FineWikiPlainTextRenderer(),
            plugins=["table", "math", "strikethrough", "abbr", "footnotes", "task_lists", "def_list", "mark", "insert", "spoiler"],  # register so tokens are parsed
        )

    def format(self, text: str) -> str:
        try:
            parsed_text = self.claude_markdown_parser(text)
            if not isinstance(parsed_text, str):
                raise TypeError("The Wiki Markdown Formatter returned a non-string value")
            return parsed_text
        except Exception as e:
            self.stat_update("wiki_markdown_formatter_error", 1)
            data_trove_logger.warning(f"Wiki Markdown Formatter Error: {e}")
            return ""

class RemoveFamilyTreeTableFormatter(BaseFormatter):

    name = "🌳 Wiki Family Tree Removal"
    def __init__(self, pipe_threshold=40):
        super().__init__()
        self.pipe_threshold = pipe_threshold

    @staticmethod
    def remove_family_tree_tables(markdown_text: str, pipe_threshold=40) -> str:
        """
        Removes the lines of text that contain more than pipe_threshold number of pipes
        and returns the remaining text.

        This typically removes very large tables and family trees from the text.

        Args:
            markdown_text: The text to be processed.
            pipe_threshold: The maximum number of pipes allowed in a line.

        Returns:
            The remaining text after removing lines with more than pipe_threshold number of pipes.
        """
        text_lines = markdown_text.split("\n")
        non_family_tree_text: list[str] = []

        for line in text_lines:
            if line.count("|") > pipe_threshold:
                continue
            non_family_tree_text.append(line)

        return "\n".join(non_family_tree_text)

    def format(self, text: str) -> str:
        return self.remove_family_tree_tables(text, self.pipe_threshold)


class RemoveLinesWithGivenLatexCommandsFormatter(BaseFormatter):

    name= "✂️ Latex Commands Removal"

    LATEX_CODE_PATTERN = re.compile(r"\{(\\[\S]+).*\}", re.DOTALL)
    def __init__(self, latex_commands: set[str]):
        super().__init__()
        if not latex_commands:
            raise ValueError("latex_commands cannot be empty")
        self.latex_commands = latex_commands

    @classmethod
    def remove_lines_with_given_latex_commands(cls, markdown_text: str, latex_commands: set[str]) -> str:
        """
        Removes the lines of text that contain any of the given latex commands.

        Args:
            markdown_text: The text to be processed.
            latex_commands: A set of latex commands to be removed.

        Returns:
            The remaining text after removing lines with any of the given latex commands.
        """
        text_lines = markdown_text.split("\n")
        non_latex_command_text: list[str] = []

        for line in text_lines:
            found_latex_commands = set(cls.LATEX_CODE_PATTERN.findall(line))
            if found_latex_commands and latex_commands.intersection(found_latex_commands):
                continue
            non_latex_command_text.append(line)

        return "\n".join(non_latex_command_text)

    def format(self, text: str) -> str:
        return self.remove_lines_with_given_latex_commands(text, self.latex_commands)


class EmptyTextFilter(BaseFilter):
    name = "🗑 Empty text filter"

    def __init__(self, exclusion_writer: DiskWriter | None = None):
        if exclusion_writer is None:
            super().__init__()
        else:
            super().__init__(exclusion_writer=exclusion_writer)

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        
        if not doc.text or not doc.text.strip():
            return False, "empty_text"
        return True

class MinWordsDocumentFilter(BaseFilter):
    name = "Minimum Words Document Filter"

    def __init__(
        self,
        min_words: int | None = 50,
        exclusion_writer: DiskWriter | None = None,
        language: str = Languages.english):
        if exclusion_writer is None:
            super().__init__()
        else:
            super().__init__(exclusion_writer=exclusion_writer)
        self.min_words = min_words
        self.language = language

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """

        Args:
            doc: Applies the heuristics rules to decide if a document should be REMOVED


        Returns: False if sample.text does not pass any of the heuristic tests

        """
        text = doc.text
        words = split_into_words(text, self.language)

        non_symbol_words = [w for w in words if any(ch not in PUNCTUATION_SET for ch in w)]
        n_non_symbol_words_words = len(non_symbol_words)

        # words < min_doc_words or words > max_doc_words
        if self.min_words and n_non_symbol_words_words < self.min_words:
            return False, "minimum_word_drop"
        return True

class SimpleURLFilter(BaseFilter):
    name = "Simple URL Filter"

    def __init__(self,
                 urls_to_filter: set[str],
                 exclusion_writer: DiskWriter | None = None,
                 meta_data_attribute: str = "url") -> None:
        if exclusion_writer is None:
            super().__init__()
        else:
            super().__init__(exclusion_writer=exclusion_writer)
        self.urls_to_filter = urls_to_filter
        self.meta_data_attribute = meta_data_attribute

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        url = doc.metadata.get(self.meta_data_attribute)
        if url is None:
            return True
        if url in self.urls_to_filter:
            return False, "filtered_url"
        return True

from typing import cast, Callable, Iterable, Generator, NewType
from wikipedia_processing.models_install import Languages as ModelInstallLanguages
from wikipedia_processing.models_util import get_language_sentence_splitter, get_language_tagger
from datatrove.data import DocumentsPipeline, Document
from datatrove.pipeline.base import PipelineStep



class SentenceSplitterAnnotator(PipelineStep):
    name = "🏷 Sentence Splitter Annotator"
    type = "🏷 - ANNOTATE"

    def __init__(self, wikipedia_language_code: str):
        super().__init__()
        _language = getattr(ModelInstallLanguages, wikipedia_language_code, None)
        supported_languages = list(ModelInstallLanguages)
        if _language is None:
            raise ValueError(f"Invalid language code: {wikipedia_language_code!r} Supported languages: {supported_languages!r}")
        self.language = cast(ModelInstallLanguages, _language)
        self._nlp: None | Callable[[str], Iterable[tuple[str, tuple[int, int]]]] = None  # lazy-loaded per worker process

    def _load_model(self) -> Callable[[str], Iterable[tuple[str, tuple[int, int]]]]:
        if self._nlp is None:
            self._nlp = get_language_sentence_splitter(self.language)
        return self._nlp

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        sentence_splitter = self._load_model()
        
        for doc in data:
            doc = cast(Document, doc)
            with self.track_time():
                doc.metadata["start_end_sentence_character_indexes"] = []
                for _, (start_sentence_index, end_sentence_index) in sentence_splitter(doc.text):
                    doc.metadata["start_end_sentence_character_indexes"].append((start_sentence_index, end_sentence_index))
            self.stat_update("sentences", value=len(doc.metadata["start_end_sentence_character_indexes"]))
            yield doc


class TokenPyMUSASAnnotator(PipelineStep):
    name = "🏷 Token + PyMUSAS Tag/MWE"
    type = "🏷 - ANNOTATE"

    def __init__(self, wikipedia_language_code: str):
        super().__init__()
        _language = getattr(ModelInstallLanguages, wikipedia_language_code, None)
        supported_languages = list(ModelInstallLanguages)
        if _language is None:
            raise ValueError(f"Invalid language code: {wikipedia_language_code!r} Supported languages: {supported_languages!r}")
        self.language = cast(ModelInstallLanguages, _language)
        self._nlp = None  # lazy-loaded per worker process

        usas_tags_to_filter_out = set({"Z99"})
        self.valid_usas_tags = set(load_usas_mapper(None, usas_tags_to_filter_out).keys())

    def _load_model(self) -> spacy.Language:
        if self._nlp is None:
            # disable components you don't need for speed (e.g. parser, ner)
            self._nlp = get_language_tagger(self.language)
        return self._nlp


    def get_sentences(self, doc: Document) -> Iterable[str]:
        sentence_start_end_indexes = doc.metadata.get("start_end_sentence_character_indexes", None)
        if sentence_start_end_indexes is None:
            raise ValueError(f"{self.name} requires `start_end_sentence_character_indexes` "
                                "within each document's metadata of which this can be generated by "
                                "the `SentenceSplitterAnnotator` pipeline step")
        for start_index, end_index in sentence_start_end_indexes:
            yield doc.text[start_index: end_index]

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        nlp = self._load_model()
        batch = []

        for doc in data:
            doc = cast(Document, doc)
            sentence_tokens = []
            sentence_tags = []
            sentence_mwe_labels = []
            number_tokens = 0
            number_tagged_tokens = 0
            number_pymusas_tags = 0
            number_mwes = 0

            with self.track_time():
                for sentence in self.get_sentences(doc):
                    sentence = sentence.strip()
                    tokens = []
                    tags = []
                    mwe_labels = []
                    
                    # Some sentences do not contain any text content, in these
                    # cases some of the spaCy pipelines will fail, but to keep
                    # number of values in the tokens etc lists and sentence indexes 
                    # list the same we append an empty token list for empty sentences
                    if sentence:
                        all_pymusas_mwe_indexes: list[list[tuple[int, int]]] = []
                        for token in nlp(sentence):
                            tokens.append(token.text)
                            pymusas_tags = token._.pymusas_tags
                            all_pymusas_mwe_indexes.append(token._.pymusas_mwe_indexes)

                            most_likely_pymusas_tag: str = ""
                            if pymusas_tags and len(pymusas_tags) > 0:
                                most_likely_pymusas_tag = pymusas_tags[0]
                            else:
                                tags.append([])
                                continue

                            valid_pymusas_tags = keep_valid_usas_tags(most_likely_pymusas_tag, self.valid_usas_tags)
                            valid_most_likely_pymusas_tags: list[str] = []
                            if valid_pymusas_tags and len(valid_pymusas_tags) > 0:
                                valid_most_likely_pymusas_tags = valid_pymusas_tags[0].tag_strings
                                number_tagged_tokens += 1
                            
                            number_pymusas_tags += len(valid_most_likely_pymusas_tags)
                            tags.append(valid_most_likely_pymusas_tags)
                    
                        number_sentence_tokens = len(tokens)
                        tmp_mwe_labels = mwe_labels_from_pymusas_indexes(all_pymusas_mwe_indexes)
                        
                        mwe_labels_json_serializable = []
                        number_mwes_in_sentence = 0
                        for mwe_labels in tmp_mwe_labels:
                            if mwe_labels:
                                number_mwes_in_sentence = max(mwe_labels)
                            mwe_labels_json_serializable.append(list(mwe_labels))
                        mwe_labels = mwe_labels_json_serializable
                        number_mwes += number_mwes_in_sentence

                        number_tokens += number_sentence_tokens
                    
                    sentence_tokens.append(tokens)
                    sentence_tags.append(tags)
                    sentence_mwe_labels.append(mwe_labels)
            doc.metadata["tokens"] = sentence_tokens
            doc.metadata["tags"] = sentence_tags
            doc.metadata["mwes"] = sentence_mwe_labels

            self.stat_update("tokens", value=number_tokens)
            self.stat_update("tagged tokens", value=number_tagged_tokens)
            self.stat_update("PyMUSAS tags", value=number_pymusas_tags)
            self.stat_update("MWEs", value=number_mwes)
            yield doc


def main(wikipedia_language_code: Annotated[str, typer.Argument(help="Wikipedia language code for the language you want to download and process data for.")],
         input_dir: Annotated[Path, typer.Argument(help="Directory that will contains the input data. The directory should contain JSONL files that will be read.`")],
         output_dir: Annotated[Path, typer.Argument(help="Directory that will contain the output data. The output data will be written to `output_dir/wikipedia_language_code.jsonl.gz`")],
         logging_dir: Annotated[Path, typer.Argument(help="Directory to save the language specific log too. Log folder will be `logging_dir/wikipedia_language_code`")],
         max_number_cpus: Annotated[int, typer.Option("-c", "--max-number-cpus", help="Maximum number of CPUs to use for processing data. The number of CPUs actually used is the minimum of (max_number_cpus, number of shards the dataset is split into). However, if the number of shards is 1 then max_number_cpus is set to max_number_cpus.")] = 1,
         overwrite: Annotated[bool, typer.Option("-o", "--overwrite", help="Whether to overwrite existing data.")] = False):
    
    start_time = perf_counter()
    dataset_id = "HuggingFaceFW/finewiki"
    dataset_load_kwargs = {
        "name": wikipedia_language_code,
        "split": "train",
    }

    language_meta_data = get_language_information(wikipedia_language_code, Path(__file__).parent / "data" / "languages.yaml")
    data_trove_language = getattr(DatatroveLanguages, language_meta_data["data_trove_language"])
    assert isinstance(data_trove_language, str)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"{wikipedia_language_code}" + "${rank}.jsonl"

    #wiki_dataset: IterableDataset = load_dataset(dataset_id, split="train", name=wikipedia_language_code, streaming=True)
    
    randomize_start_duration = 5
    main_logging_dir_str = create_sub_directory(logging_dir, wikipedia_language_code)

    #number_of_tasks = number_of_shards(wiki_dataset)
    # The dataset can also come in shards whereby each data shard is bucketed 
    # into one task shard thus a task shard can have one or more data shards.
    # (if the number of tasks is greater than the number of data shards then some task shards will be empty)
    number_of_tasks = 1 # This determines the number of shards the data is chunked into whereby each task shard is processed by one worker
    # Number of CPUs we will use
    number_of_workers: int = 1 # Determines the number of shards that are processed in parallel
    #number_of_workers: int = max_number_cpus
    #if number_of_tasks > 1:
    #    number_of_workers = min(max_number_cpus, number_of_tasks)
    # 50 MB
    max_file_size_in_bytes = int(5e7)
    # Number of buckets has to be divisible by the number of tasks
    minhash_number_of_buckets = 1

    

    #reader_pipe = HuggingFaceDatasetReader(dataset=dataset_id, dataset_options=dataset_load_kwargs, streaming=True)
    #output_pipe = JsonlWriter(output_folder=str(output_dir.resolve()), output_filename=output_filename, compression="gzip", expand_metadata=False, max_file_size=max_file_size_in_bytes)

    wikipedia_urls_to_filter = set({
        "https://en.wikipedia.org/wiki/Cancer",
        "https://en.wikipedia.org/wiki/Breast_cancer",
        "https://en.wikipedia.org/wiki/Melanoma",
        "https://en.wikipedia.org/wiki/Prostate_cancer",
        "https://en.wikipedia.org/wiki/Palliative_care",
        "https://en.wikipedia.org/wiki/Chemotherapy",
        "https://en.wikipedia.org/wiki/Radiation_therapy",
        "https://nl.wikipedia.org/wiki/Kanker",
        "https://nl.wikipedia.org/wiki/Borstkanker",
        "https://nl.wikipedia.org/wiki/Melanoom",
        "https://nl.wikipedia.org/wiki/Prostaatkanker",
        "https://nl.wikipedia.org/wiki/Palliatieve_zorg",
        "https://nl.wikipedia.org/wiki/Chemotherapie",
        "https://nl.wikipedia.org/wiki/Radiotherapie",
        "https://da.wikipedia.org/wiki/Kr%C3%A6ft",
        "https://da.wikipedia.org/wiki/Brystkr%C3%A6ft",
        "https://da.wikipedia.org/wiki/Moderm%C3%A6rkekr%C3%A6ft",
        "https://da.wikipedia.org/wiki/Prostatakr%C3%A6ft",
        "https://da.wikipedia.org/wiki/Palliativ_behandling",
        "https://da.wikipedia.org/wiki/Kemoterapi",
        "https://da.wikipedia.org/wiki/Str%C3%A5lebehandling",
        "https://es.wikipedia.org/wiki/C%C3%A1ncer",
        "https://es.wikipedia.org/wiki/C%C3%A1ncer_de_mama",
        "https://es.wikipedia.org/wiki/Melanoma",
        "https://es.wikipedia.org/wiki/C%C3%A1ncer_de_pr%C3%B3stata",
        "https://es.wikipedia.org/wiki/Cuidados_paliativos",
        "https://es.wikipedia.org/wiki/Quimioterapia",
        "https://es.wikipedia.org/wiki/Radioterapia",
        "https://hi.wikipedia.org/wiki/%E0%A4%95%E0%A4%B0%E0%A5%8D%E0%A4%95%E0%A4%9F_%E0%A4%B0%E0%A5%8B%E0%A4%97",
        "https://hi.wikipedia.org/wiki/%E0%A4%B8%E0%A5%8D%E0%A4%A4%E0%A4%A8_%E0%A4%95%E0%A5%88%E0%A4%A8%E0%A5%8D%E0%A4%B8%E0%A4%B0",
        "https://hi.wikipedia.org/wiki/%E0%A4%AE%E0%A5%87%E0%A4%B2%E0%A5%87%E0%A4%A8%E0%A5%8B%E0%A4%AE%E0%A4%BE",
        "https://hi.wikipedia.org/wiki/%E0%A4%AA%E0%A5%8D%E0%A4%B0%E0%A5%8B%E0%A4%B8%E0%A5%8D%E0%A4%9F%E0%A5%87%E0%A4%9F_%E0%A4%95%E0%A5%88%E0%A4%82%E0%A4%B8%E0%A4%B0",
        "https://hi.wikipedia.org/wiki/%E0%A4%AA%E0%A5%8D%E0%A4%B0%E0%A4%B6%E0%A4%BE%E0%A4%AE%E0%A4%95_%E0%A4%89%E0%A4%AA%E0%A4%9A%E0%A4%BE%E0%A4%B0",
        "https://hi.wikipedia.org/wiki/%E0%A4%95%E0%A5%80%E0%A4%AE%E0%A5%8B%E0%A4%A5%E0%A5%87%E0%A4%B0%E0%A5%87%E0%A4%AA%E0%A5%80",
        "https://hi.wikipedia.org/wiki/%E0%A4%B5%E0%A4%BF%E0%A4%95%E0%A4%BF%E0%A4%B0%E0%A4%A3_%E0%A4%9A%E0%A4%BF%E0%A4%95%E0%A4%BF%E0%A4%A4%E0%A5%8D%E0%A4%B8%E0%A4%BE",
        "https://ig.wikipedia.org/wiki/Oru_ugbo",
        "https://ig.wikipedia.org/wiki/Iri_Ji_%E1%BB%8Dh%E1%BB%A5r%E1%BB%A5_ndi_Igbo",
    })

    reader_pipe = JsonlReader(str(input_dir.resolve()), glob_pattern="Danish_page_data.jsonl", compression=None)
    remove_family_tree_formatter = RemoveFamilyTreeTableFormatter(pipe_threshold=40)
    remove_lines_with_given_latex_commands_formatter = RemoveLinesWithGivenLatexCommandsFormatter(latex_commands={"\\displaystyle", "\\textstyle"})
    min_words_document_filter = MinWordsDocumentFilter(min_words=50, language=data_trove_language)
    url_filter = SimpleURLFilter(urls_to_filter=wikipedia_urls_to_filter)
    wikipedia_markdown_formatter = WikipediaMarkdownFormatter()

    page_id_title_filter = LambdaFilter(filter_function=get_relevant_page_function(wikipedia_language_code, use_title=True))
    output_pipe = JsonlWriter(output_folder=str(output_dir.resolve()), output_filename=output_filename, compression=None, expand_metadata=False, max_file_size=max_file_size_in_bytes)
    
    

    # Exact match deduplication
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        exact_intermediate_data_dir = create_sub_directory(tmp_dir_path, "exact_intermediate")
        exact_intermediate_output_pipe = JsonlWriter(output_folder=exact_intermediate_data_dir, output_filename=output_filename, compression=None, expand_metadata=False, max_file_size=max_file_size_in_bytes)
        exact_intermediate_read_pipe = JsonlReader(exact_intermediate_data_dir, glob_pattern="*.jsonl", compression=None)
        
        exact_dedup_config = ExactDedupConfig(content_getter=lambda x: x.text)
        exact_dedup_sigs_dir = create_sub_directory(tmp_dir_path, "exact_dedup_sigs")
        exact_dedup_finds_dir = create_sub_directory(tmp_dir_path, "exact_dedup_finds")
        
        exact_dedup_sig = ExactDedupSignature(output_folder=exact_dedup_sigs_dir, config=exact_dedup_config, finder_workers=number_of_workers)
        exact_dedup_finds = ExactFindDedups(exact_dedup_sigs_dir, exact_dedup_finds_dir, config=exact_dedup_config)
        exact_dedup_filter = ExactDedupFilter(exact_dedup_finds_dir, config=exact_dedup_config)

        logging_dir_initial_process_str = create_sub_directory(Path(main_logging_dir_str), "initial_process_and_sig")
        logging_dir_exact_finds_str = create_sub_directory(Path(main_logging_dir_str), "exact_dedup_finds")
        logging_dir_exact_filter_str = create_sub_directory(Path(main_logging_dir_str), "exact_dedup_filter")

        minhash_intermediate_data_dir = create_sub_directory(tmp_dir_path, "minhash_intermediate")
        minhash_intermediate_output_pipe = JsonlWriter(output_folder=minhash_intermediate_data_dir, output_filename=output_filename, compression=None, expand_metadata=False, max_file_size=max_file_size_in_bytes)
        minhash_intermediate_read_pipe = JsonlReader(minhash_intermediate_data_dir, glob_pattern="*.jsonl", compression=None)
        
        minhash_dedup_config = MinhashConfig(num_buckets=minhash_number_of_buckets, hashes_per_bucket=8, n_grams=5)
        minhash_dedup_sigs_dir = create_sub_directory(tmp_dir_path, "minhash_dedup_sigs")
        minhash_dedup_buckets_dir = create_sub_directory(tmp_dir_path, "minhash_dedup_buckets")
        minhash_dedup_clusters_dir = create_sub_directory(tmp_dir_path, "minhash_dedup_clusters")

        minhash_dedup_sig = MinhashDedupSignature(output_folder=minhash_dedup_sigs_dir, config=minhash_dedup_config, language=data_trove_language)
        minhash_dedup_buckets = MinhashDedupBuckets(input_folder=minhash_dedup_sigs_dir, output_folder=minhash_dedup_buckets_dir, config=minhash_dedup_config)
        minhash_dedup_clusters = MinhashDedupCluster(input_folder=minhash_dedup_buckets_dir, output_folder=minhash_dedup_clusters_dir, config=minhash_dedup_config)
        minhash_dedup_filter = MinhashDedupFilter(input_folder=minhash_dedup_clusters_dir)

        logging_dir_minhash_buckets_str = create_sub_directory(Path(main_logging_dir_str), "minhash_dedup_buckets")
        logging_dir_minhash_clusters_str = create_sub_directory(Path(main_logging_dir_str), "minhash_dedup_clusters")
        logging_dir_minhash_filter_str = create_sub_directory(Path(main_logging_dir_str), "minhash_dedup_filter")
        
        initial_process_and_sig_stage = LocalPipelineExecutor(
            pipeline=[reader_pipe, page_id_title_filter, url_filter, remove_family_tree_formatter, remove_lines_with_given_latex_commands_formatter, wikipedia_markdown_formatter, EmptyTextFilter(), min_words_document_filter, exact_intermediate_output_pipe, exact_dedup_sig],
            tasks=number_of_tasks,
            workers=number_of_workers,
            randomize_start_duration=randomize_start_duration,
            skip_completed=overwrite,
            logging_dir=logging_dir_initial_process_str)
        exact_finds_stage = LocalPipelineExecutor(
            pipeline=[exact_dedup_finds],
            tasks=number_of_tasks,
            workers=number_of_workers,
            randomize_start_duration=randomize_start_duration,
            skip_completed=overwrite,
            logging_dir=logging_dir_exact_finds_str,
            depends=initial_process_and_sig_stage)
        exact_filter_stage = LocalPipelineExecutor(
            pipeline=[exact_intermediate_read_pipe, exact_dedup_filter, minhash_intermediate_output_pipe, minhash_dedup_sig],
            tasks=number_of_tasks,
            workers=number_of_workers,
            randomize_start_duration=randomize_start_duration,
            skip_completed=overwrite,
            logging_dir=logging_dir_exact_filter_str,
            depends=exact_finds_stage)
        minhash_buckets_stage = LocalPipelineExecutor(
            pipeline=[minhash_dedup_buckets],
            tasks=number_of_tasks,
            workers=number_of_workers,
            randomize_start_duration=randomize_start_duration,
            skip_completed=overwrite,
            logging_dir=logging_dir_minhash_buckets_str,
            depends=exact_filter_stage)
        minhash_clusters_stage = LocalPipelineExecutor(
            pipeline=[minhash_dedup_clusters],
            tasks=number_of_tasks,
            workers=number_of_workers,
            randomize_start_duration=randomize_start_duration,
            skip_completed=overwrite,
            logging_dir=logging_dir_minhash_clusters_str,
            depends=minhash_buckets_stage)
        minhash_filter_stage = LocalPipelineExecutor(
            pipeline=[minhash_intermediate_read_pipe, minhash_dedup_filter, SentenceSplitterAnnotator(wikipedia_language_code), TokenPyMUSASAnnotator(wikipedia_language_code), output_pipe],
            tasks=number_of_tasks,
            workers=number_of_workers,
            randomize_start_duration=randomize_start_duration,
            skip_completed=overwrite,
            logging_dir=logging_dir_minhash_filter_str,
            depends=minhash_clusters_stage)
        
        minhash_filter_stage.run()

    end_time = perf_counter()
    print(f"Time taken: {end_time - start_time}")

if __name__ == "__main__":
    typer.run(main)