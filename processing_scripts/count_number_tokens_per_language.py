import os
import tempfile
import time
from enum import Enum
from pathlib import Path
from typing import Annotated
import shutil

import datasets
import typer
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.dedup.exact_dedup import (
    ExactDedupConfig,
    ExactDedupFilter,
    ExactDedupSignature,
    ExactFindDedups,
)
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
    MinhashDedupSignature,
)
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.readers import HuggingFaceDatasetReader, JsonlReader
from datatrove.pipeline.stats import WordStats
from datatrove.pipeline.stats.merger import StatsMerger
from datatrove.pipeline.writers import JsonlWriter
from datatrove.utils.logging import logger as data_trove_logger
from dotenv import load_dotenv

from wikipedia_processing.filters import (
    EmptyTextFilter,
    MinWordsDocumentFilter,
    SimpleURLFilter,
    get_relevant_page_function,
)
from wikipedia_processing.formatters import (
    RemoveFamilyTreeTableFormatter,
    RemoveLinesWithGivenLatexCommandsFormatter,
    WikipediaMarkdownFormatter,
)
from wikipedia_processing.pipelines.sentence_splitting import SentenceSplitterAnnotator
from wikipedia_processing.pipelines.token_annotation import TokenPyMUSASAnnotator
from wikipedia_processing.utils import (
    create_sub_directory,
    get_hashes_per_bucket,
    get_number_of_shards,
    get_progress_logger_function,
    get_usas_language_processing_information,
    get_valid_usas_language_processing_wikipedia_codes,
    time_elapsed,
)

TEST_SET_WIKIPEDIA_URLS = set({
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


load_dotenv()
WikipediaLanguageCode = Enum("WikipediaLanguageCode", [(value, value) for value in get_valid_usas_language_processing_wikipedia_codes()], type=str)

def main(wikipedia_language_code: Annotated[WikipediaLanguageCode, typer.Argument(help="Wikipedia language code for the language you want to download and process data for.")],
         logging_dir: Annotated[Path, typer.Argument(help="Directory to save the language specific log too. Log folder will be `logging_dir/wikipedia_language_code`")],
         number_of_workers: Annotated[int, typer.Option("-w", "--number-of-workers", help="The number of workers, whereby one worker is one CPU core, this value is capped by the number of CPUs.")] = 1,
         tasks_multiplier: Annotated[int, typer.Option("-t", "--tasks-multiplier", help="Multiplier for the number of tasks to use for processing data based on the maximum number of workers.")] = 5,
         overwrite: Annotated[bool, typer.Option("-o", "--overwrite", help="Whether to overwrite existing data, this will also delete the existing log directory for the language if it exists.")] = False,
         min_hash_threshold: Annotated[float, typer.Option("-m", "--min-hash-threshold", help="Approximate Jaccard similarity threshold for minhash, to determine if a document is a duplicate, default value is what FineWeb choose.")] = 0.72,
         min_words_filter_threshold: Annotated[int, typer.Option("-f", "--min-words-filter-threshold", help="Minimum number of words in a document for it to be processed, anything less then the document is filtered out, default value is what Google Deepmind Gopher LLM (2022) choose.")] = 50,
         print_number_of_shards: Annotated[bool, typer.Option("-g", "--get-number-of-shards", help="Get the number of shards that the Wikipedia dataset is split into, this can be useful to determine the number of workers to assign when running the pipeline, this number is printed to stdout.")] = False,
         max_output_file_size: Annotated[int, typer.Option("-s", "--max-output-file-size", help="Maximum output file size in MB, default value is 100MB, when the output is larger than this value it is split into multiple output files of up to this size.")] = 200,
         randomize_start_duration: Annotated[int, typer.Option("-r", "--randomize-start-duration", help="The maximum number of seconds to delay the start of each task to prevent all tasks from starting simultaneously and potentially overloading the system.")] = 5,):
    
    pipeline_start_time = time.perf_counter()
    dataset_id = "HuggingFaceFW/finewiki"
    wikipedia_language_code_str: str = wikipedia_language_code.value
    dataset_load_kwargs = {
        "name": wikipedia_language_code_str,
        "split": "train",
    }

    wiki_dataset: datasets.IterableDataset = datasets.load_dataset(dataset_id, split="train", name=wikipedia_language_code_str, streaming=True)
    number_of_shards = get_number_of_shards(wiki_dataset)

    if print_number_of_shards:
        print(number_of_shards)
        typer.Exit(0)

    number_of_workers = min(number_of_workers, os.process_cpu_count())

    number_processing_tasks = number_of_workers * tasks_multiplier
    number_data_downloading_tasks = min(number_of_shards, number_processing_tasks)
    number_downloading_workers = min(number_of_workers, number_data_downloading_tasks)
    
    minhash_number_of_buckets = number_processing_tasks # The number of number of buckets must be divisible by the number of tasks
    minhash_hashes_per_bucket = get_hashes_per_bucket(minhash_number_of_buckets, min_hash_threshold)
    
    main_logging_dir_str = create_sub_directory(logging_dir, wikipedia_language_code_str)
    if overwrite and Path(main_logging_dir_str).exists():
        data_trove_logger.info(f"Deleting existing log directory: {main_logging_dir_str!r}")
        shutil.rmtree(main_logging_dir_str, ignore_errors=False)

    stats_logging_dir_str = create_sub_directory(Path(main_logging_dir_str), "stats")
    merged_stats_logging_dir_str = create_sub_directory(Path(main_logging_dir_str), "merged_stats")

    data_trove_logger.info(f"Wikipedia language code: {wikipedia_language_code_str!r}")
    data_trove_logger.info(f"Number of shards in the dataset: {number_of_shards!r}")
    data_trove_logger.info(f"Number of workers: {number_of_workers!r}")
    data_trove_logger.info("Number of task for data downloading (if the number of "
                           "shards is less than the number of tasks then the number "
                           f"of tasks is set to the number of shards): {number_data_downloading_tasks!r}")
    data_trove_logger.info(f"Number of data processing tasks: {number_processing_tasks!r}")
    data_trove_logger.info(f"Number of buckets and hashes per bucket for MinHash: {minhash_number_of_buckets!r}, {minhash_hashes_per_bucket!r}")
    data_trove_logger.info(f"MinHash threshold: {min_hash_threshold!r}")
    data_trove_logger.info(f"Minimum number of words filter threshold: {min_words_filter_threshold!r}")

    language_meta_data = get_usas_language_processing_information(wikipedia_language_code)
    data_trove_language = language_meta_data["data_trove_language"]

    reader_pipe = HuggingFaceDatasetReader(dataset=dataset_id, dataset_options=dataset_load_kwargs, streaming=True)
    page_id_title_filter = LambdaFilter(filter_function=get_relevant_page_function(wikipedia_language_code_str, use_title=True))
    remove_family_tree_formatter = RemoveFamilyTreeTableFormatter(pipe_threshold=40)
    remove_lines_with_given_latex_commands_formatter = RemoveLinesWithGivenLatexCommandsFormatter(latex_commands={"\\displaystyle", "\\textstyle"})
    min_words_document_filter = MinWordsDocumentFilter(min_words=min_words_filter_threshold, language=data_trove_language)
    url_filter = SimpleURLFilter(urls_to_filter=TEST_SET_WIKIPEDIA_URLS)
    wikipedia_markdown_formatter = WikipediaMarkdownFormatter()
    word_statistics = WordStats(stats_logging_dir_str, language=data_trove_language)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        generic_output_filename = f"{wikipedia_language_code_str}" + "${rank}.jsonl.gz"
        # Converting to MB
        max_file_size_in_bytes = int(max_output_file_size * 10e5)

        reading_intermediate_data_dir = create_sub_directory(tmp_dir_path, "reading_intermediate")
        reading_stage_output_pipe = JsonlWriter(reading_intermediate_data_dir, output_filename=generic_output_filename, compression="infer", expand_metadata=False, max_file_size=max_file_size_in_bytes)
        initial_processing_input_pipe = JsonlReader(reading_intermediate_data_dir, glob_pattern="*.jsonl.gz", compression="infer")

        exact_intermediate_data_dir = create_sub_directory(tmp_dir_path, "exact_intermediate")
        exact_intermediate_output_pipe = JsonlWriter(output_folder=exact_intermediate_data_dir, output_filename=generic_output_filename, compression="infer", expand_metadata=False, max_file_size=max_file_size_in_bytes)
        exact_intermediate_read_pipe = JsonlReader(exact_intermediate_data_dir, glob_pattern="*.jsonl.gz", compression="infer")
        
        exact_dedup_config = ExactDedupConfig(content_getter=lambda x: x.text)
        exact_dedup_sigs_dir = create_sub_directory(tmp_dir_path, "exact_dedup_sigs")
        exact_dedup_finds_dir = create_sub_directory(tmp_dir_path, "exact_dedup_finds")
        
        exact_dedup_sig = ExactDedupSignature(output_folder=exact_dedup_sigs_dir, config=exact_dedup_config, finder_workers=number_of_workers)
        exact_dedup_finds = ExactFindDedups(exact_dedup_sigs_dir, exact_dedup_finds_dir, config=exact_dedup_config)
        exact_dedup_filter = ExactDedupFilter(exact_dedup_finds_dir, config=exact_dedup_config)

        logging_reading_dir = create_sub_directory(Path(main_logging_dir_str), "reading")
        logging_dir_initial_process_str = create_sub_directory(Path(main_logging_dir_str), "initial_process")
        logging_dir_exact_sigs_str = create_sub_directory(Path(main_logging_dir_str), "exact_dedup_sigs")
        logging_dir_exact_finds_str = create_sub_directory(Path(main_logging_dir_str), "exact_dedup_finds")
        logging_dir_exact_filter_str = create_sub_directory(Path(main_logging_dir_str), "exact_dedup_filter")

        minhash_intermediate_data_dir = create_sub_directory(tmp_dir_path, "minhash_intermediate")
        minhash_intermediate_output_pipe = JsonlWriter(output_folder=minhash_intermediate_data_dir, output_filename=generic_output_filename, compression="infer", expand_metadata=False, max_file_size=max_file_size_in_bytes)
        minhash_intermediate_read_pipe = JsonlReader(minhash_intermediate_data_dir, glob_pattern="*.jsonl.gz", compression="infer")
        
        minhash_filter_data_dir = create_sub_directory(tmp_dir_path, "minhash_filter")
        minhash_filter_read_pipe = JsonlReader(minhash_filter_data_dir, glob_pattern="*.jsonl.gz", compression="infer")
        minhash_filter_output_pipe = JsonlWriter(output_folder=minhash_filter_data_dir, output_filename=generic_output_filename, compression="infer", expand_metadata=False, max_file_size=max_file_size_in_bytes)
        
        minhash_dedup_config = MinhashConfig(num_buckets=minhash_number_of_buckets, hashes_per_bucket=minhash_hashes_per_bucket, n_grams=5)
        minhash_dedup_sigs_dir = create_sub_directory(tmp_dir_path, "minhash_dedup_sigs")
        minhash_dedup_buckets_dir = create_sub_directory(tmp_dir_path, "minhash_dedup_buckets")
        minhash_dedup_clusters_dir = create_sub_directory(tmp_dir_path, "minhash_dedup_clusters")

        minhash_dedup_sig = MinhashDedupSignature(output_folder=minhash_dedup_sigs_dir, config=minhash_dedup_config, language=data_trove_language)
        minhash_dedup_buckets = MinhashDedupBuckets(input_folder=minhash_dedup_sigs_dir, output_folder=minhash_dedup_buckets_dir, config=minhash_dedup_config)
        minhash_dedup_clusters = MinhashDedupCluster(input_folder=minhash_dedup_buckets_dir, output_folder=minhash_dedup_clusters_dir, config=minhash_dedup_config)
        minhash_dedup_filter = MinhashDedupFilter(input_folder=minhash_dedup_clusters_dir)

        logging_dir_minhash_sigs_str = create_sub_directory(Path(main_logging_dir_str), "minhash_dedup_sigs")
        logging_dir_minhash_buckets_str = create_sub_directory(Path(main_logging_dir_str), "minhash_dedup_buckets")
        logging_dir_minhash_clusters_str = create_sub_directory(Path(main_logging_dir_str), "minhash_dedup_clusters")
        logging_dir_minhash_filter_str = create_sub_directory(Path(main_logging_dir_str), "minhash_dedup_filter")
        logging_dir_post_processing_str = create_sub_directory(Path(main_logging_dir_str), "post_processing")
        logging_dir_merged_stats_processing_str = create_sub_directory(Path(main_logging_dir_str), "merged_stats_processing")

        reading_stage = LocalPipelineExecutor(
            pipeline=[reader_pipe, get_progress_logger_function("reading"), page_id_title_filter, url_filter, reading_stage_output_pipe],
            tasks=number_data_downloading_tasks,
            workers=number_downloading_workers,
            randomize_start_duration=randomize_start_duration,
            skip_completed=overwrite,
            logging_dir=logging_reading_dir
        )
        initial_process_stage = LocalPipelineExecutor(
            pipeline=[initial_processing_input_pipe, remove_family_tree_formatter, remove_lines_with_given_latex_commands_formatter, wikipedia_markdown_formatter, EmptyTextFilter(), min_words_document_filter, exact_intermediate_output_pipe],
            tasks=number_processing_tasks,
            workers=number_of_workers,
            randomize_start_duration=randomize_start_duration,
            skip_completed=overwrite,
            logging_dir=logging_dir_initial_process_str,
            depends=reading_stage)
        exact_sigs_stage = LocalPipelineExecutor(
            pipeline=[exact_intermediate_read_pipe, exact_dedup_sig],
            tasks=number_processing_tasks,
            workers=number_of_workers,
            randomize_start_duration=randomize_start_duration,
            skip_completed=overwrite,
            logging_dir=logging_dir_exact_sigs_str,
            depends=initial_process_stage
        )
        exact_finds_stage = LocalPipelineExecutor(
            pipeline=[exact_dedup_finds],
            tasks=number_of_workers, # Has to be the same as finder_workers
            workers=number_of_workers, # Has to be the same as finder_workers
            randomize_start_duration=randomize_start_duration,
            skip_completed=overwrite,
            logging_dir=logging_dir_exact_finds_str,
            depends=exact_sigs_stage)
        exact_filter_stage = LocalPipelineExecutor(
            pipeline=[exact_intermediate_read_pipe, exact_dedup_filter, minhash_intermediate_output_pipe],
            tasks=number_processing_tasks, # This has to match exact_sigs_stage
            workers=number_of_workers,
            randomize_start_duration=randomize_start_duration,
            skip_completed=overwrite,
            logging_dir=logging_dir_exact_filter_str,
            depends=exact_finds_stage)
        minhash_sigs_stage = LocalPipelineExecutor(
            pipeline=[minhash_intermediate_read_pipe, minhash_dedup_sig],
            tasks=number_processing_tasks,
            workers=number_of_workers,
            randomize_start_duration=randomize_start_duration,
            skip_completed=overwrite,
            logging_dir=logging_dir_minhash_sigs_str,
            depends=exact_filter_stage)
        minhash_buckets_stage = LocalPipelineExecutor(
            pipeline=[minhash_dedup_buckets],
            tasks=minhash_dedup_config.num_buckets,
            workers=number_of_workers,
            randomize_start_duration=randomize_start_duration,
            skip_completed=overwrite,
            logging_dir=logging_dir_minhash_buckets_str,
            depends=minhash_sigs_stage)
        minhash_clusters_stage = LocalPipelineExecutor(
            pipeline=[minhash_dedup_clusters],
            tasks=1,
            workers=1,
            randomize_start_duration=randomize_start_duration,
            skip_completed=overwrite,
            logging_dir=logging_dir_minhash_clusters_str,
            depends=minhash_buckets_stage)
        minhash_filter_stage = LocalPipelineExecutor(
            pipeline=[minhash_intermediate_read_pipe, minhash_dedup_filter, word_statistics, minhash_filter_output_pipe],
            tasks=number_processing_tasks, # This has to match minhash_sigs_stage
            workers=number_of_workers,
            randomize_start_duration=randomize_start_duration,
            skip_completed=overwrite,
            logging_dir=logging_dir_minhash_filter_str,
            depends=minhash_clusters_stage)
        post_processing_stage = LocalPipelineExecutor(
            pipeline=[minhash_filter_read_pipe, SentenceSplitterAnnotator(wikipedia_language_code_str), TokenPyMUSASAnnotator(wikipedia_language_code_str), word_statistics],
            tasks=number_processing_tasks, # This has to match minhash_sigs_stage
            workers=number_of_workers,
            randomize_start_duration=randomize_start_duration,
            skip_completed=overwrite,
            logging_dir=logging_dir_post_processing_str,
            depends=minhash_filter_stage)
        merge_stage = LocalPipelineExecutor(
            pipeline=[
                StatsMerger(
                    input_folder=stats_logging_dir_str,
                    output_folder=merged_stats_logging_dir_str,
                )
            ],
            tasks=1,
            workers=1,
            logging_dir=logging_dir_merged_stats_processing_str,
            depends=post_processing_stage,
        )
        
        merge_stage.run()
        data_trove_logger.info(f"PIPELINE_RUNTIME_SECONDS={time_elapsed(pipeline_start_time):.2f}")


if __name__ == "__main__":
    typer.run(main)