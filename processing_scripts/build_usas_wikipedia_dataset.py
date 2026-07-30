import json
import os
import tempfile
import time
from enum import Enum
from pathlib import Path
from typing import Annotated
import shutil

import datasets
import typer
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
from datatrove.pipeline.writers import HuggingFaceDatasetWriter, JsonlWriter, ParquetWriter
from datatrove.utils.logging import logger as data_trove_logger
from dotenv import load_dotenv

from wikipedia_processing.executors import (
    ExecutorBackend,
    PipelineExecutorFactory,
    SlurmExecutorSettings,
)
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
from wikipedia_processing.pipelines.metadata_whitelist import MetadataWhitelistAnnotator
from wikipedia_processing.pipelines.sentence_splitting import SentenceSplitterAnnotator
from wikipedia_processing.pipelines.token_annotation import TokenPyMUSASAnnotator
from wikipedia_processing.pipelines.train_validation_split import TrainValidationSplitAnnotator
from wikipedia_processing.pipelines.writer_adapter import get_metadata_whitelist_writer_adapter
from wikipedia_processing.utils import (
    create_sub_directory,
    get_hashes_per_bucket,
    get_number_of_shards,
    get_progress_logger_function,
    get_usas_language_processing_information,
    get_valid_usas_language_processing_wikipedia_codes,
    parse_json_object,
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
         number_of_workers: Annotated[int, typer.Option("-w", "--number-of-workers", help="The number of workers, whereby one worker is one CPU core. With --executor=local this value is capped by the number of CPUs on the machine running this script; with --executor=slurm it is only a concurrency throttle on each stage's Slurm job array and is not capped locally.")] = 1,
         tasks_multiplier: Annotated[int, typer.Option("-t", "--tasks-multiplier", help="Multiplier for the number of tasks to use for processing data based on the maximum number of workers.")] = 5,
         overwrite: Annotated[bool, typer.Option("-o", "--overwrite", help="Whether to overwrite existing data, this will also delete the existing log directory for the language if it exists.")] = False,
         min_hash_threshold: Annotated[float, typer.Option("-m", "--min-hash-threshold", help="Approximate Jaccard similarity threshold for minhash, to determine if a document is a duplicate, default value is what FineWeb choose.")] = 0.72,
         min_words_filter_threshold: Annotated[int, typer.Option("-f", "--min-words-filter-threshold", help="Minimum number of words in a document for it to be processed, anything less then the document is filtered out, default value is what Google Deepmind Gopher LLM (2022) choose.")] = 50,
         print_number_of_shards: Annotated[bool, typer.Option("-g", "--get-number-of-shards", help="Get the number of shards that the Wikipedia dataset is split into, this can be useful to determine the number of workers to assign when running the pipeline, this number is printed to stdout.")] = False,
         max_output_file_size: Annotated[int, typer.Option("-s", "--max-output-file-size", help="Maximum output file size in MB for the intermediate temporary files, the smaller this is the more tasks and potential parallelism there will be, default value is 100MB, when the output is larger than this value it is split into multiple output files of up to this size.")] = 100,
         output_dir: Annotated[Path | None, typer.Option("--output-dir", help="Local directory to write the final Parquet output data to, in `output_dir/data/<language>/{train,validation}/` subfolders. Mutually exclusive with --hf-dataset-repo-id; exactly one of the two must be given.")] = None,
         hf_dataset_repo_id: Annotated[str | None, typer.Option("--hf-dataset-repo-id", help="HuggingFace Hub dataset repository (`namespace/name`) to upload the final Parquet output to directly, e.g. `ucrelnlp/wikipedia-usas-mwe`. Mutually exclusive with --output-dir; exactly one of the two must be given.")] = None,
         hf_dataset_private: Annotated[bool, typer.Option("--private/--public", help="Whether to create the Hub dataset repo as private if it does not already exist. Only used with --hf-dataset-repo-id.")] = False,
         hf_local_working_dir: Annotated[Path | None, typer.Option("--hf-local-working-dir", help="Local staging directory used before uploading to the Hub. Only used with --hf-dataset-repo-id; defaults to a temporary directory that is cleaned up after upload.")] = None,
         hf_dataset_revision: Annotated[str | None, typer.Option("--hf-dataset-revision", help="Branch (or other revision) of the Hub dataset repo to upload to, e.g. `main` or a custom branch name. Only used with --hf-dataset-repo-id; defaults to the repo's default branch.")] = None,
         max_final_output_file_size: Annotated[int, typer.Option("-e", "--max-final-output-file-size", help="Maximum size in MB of the final Parquet output shards, when the output is larger than this value it is split into multiple output files of up to this size. Distinct from --max-output-file-size, which only governs intermediate staging files.")] = 200,
         validation_percentage: Annotated[float, typer.Option("-v", "--validation-percentage", help="Target percentage (0-100) of a language's documents assigned to the validation split; the rest go to train.")] = 10,
         max_validation_documents: Annotated[int, typer.Option("-n", "--max-validation-documents", help="Absolute cap on the number of documents in the validation split, regardless of --validation-percentage. The smaller of the percentage-based and absolute-cap counts wins.")] = 20,
         randomize_start_duration: Annotated[int, typer.Option("-r", "--randomize-start-duration", help="The maximum number of seconds to delay the start of each task to prevent all tasks from starting simultaneously and potentially overloading the system.")] = 5,
         tag_mapper_json: Annotated[str, typer.Option("--tag-mapper", help="JSON object mapping USAS tag strings to replacement tag strings, applied to each token's tags during PyMUSAS annotation. Tags with no entry are kept unchanged.")] = '{"PUNCT": "Z9"}',
         executor_backend: Annotated[ExecutorBackend, typer.Option("--executor", help="Which DataTrove executor backend to run pipeline stages with: 'local' runs stages as local multiprocessing workers; 'slurm' submits each stage as a Slurm job array.")] = ExecutorBackend.local,
         slurm_partition: Annotated[str | None, typer.Option("--slurm-partition", help="Slurm partition to submit jobs to. Required when --executor=slurm.")] = None,
         slurm_time: Annotated[str | None, typer.Option("--slurm-time", help="Slurm job time limit, e.g. '2:00:00'. Required when --executor=slurm.")] = None,
         slurm_cpus_per_task: Annotated[int, typer.Option("--slurm-cpus-per-task", help="Number of CPUs to request per Slurm task. Only used with --executor=slurm.")] = 1,
         slurm_mem_per_cpu_gb: Annotated[int, typer.Option("--slurm-mem-per-cpu-gb", help="Memory in GB to request per CPU for Slurm tasks. Only used with --executor=slurm.")] = 2,
         slurm_qos: Annotated[str, typer.Option("--slurm-qos", help="Slurm QOS to submit jobs under. Only used with --executor=slurm.")] = "normal",
         slurm_venv_path: Annotated[Path | None, typer.Option("--slurm-venv-path", help="Path to a virtualenv to activate in each Slurm job. Mutually exclusive with --slurm-condaenv. Only used with --executor=slurm.")] = None,
         slurm_condaenv: Annotated[str | None, typer.Option("--slurm-condaenv", help="Name of a conda environment to activate in each Slurm job. Mutually exclusive with --slurm-venv-path. Only used with --executor=slurm.")] = None,
         slurm_mail_user: Annotated[str | None, typer.Option("--slurm-mail-user", help="Email address for Slurm job notifications. Only used with --executor=slurm.")] = None,
         slurm_mail_type: Annotated[str, typer.Option("--slurm-mail-type", help="Slurm mail notification type(s), e.g. 'ALL', 'FAIL'. Only used with --executor=slurm.")] = "ALL",
         slurm_sbatch_args_json: Annotated[str | None, typer.Option("--slurm-sbatch-args", help="JSON object of additional raw sbatch arguments to pass through, e.g. '{\"account\": \"myaccount\"}'. Only used with --executor=slurm.")] = None,):
    """Build the USAS/MWE-tagged Wikipedia training corpus for a single language.

    Streams the `HuggingFaceFW/finewiki` dataset for `wikipedia_language_code`,
    restricts it to Good/Featured articles, cleans and deduplicates the text
    (exact-match then MinHash near-duplicate removal), then runs sentence
    splitting and PyMUSAS semantic/MWE tagging before splitting into
    train/validation and writing Parquet shards either to a local directory or
    directly to a HuggingFace Hub dataset repo.

    The pipeline runs as a chain of dependent DataTrove executor stages
    (reading, initial processing, exact dedup signature/find/filter, MinHash
    dedup signature/buckets/clusters/filter, post-processing/tagging, and
    stats merging), using either local multiprocessing workers or Slurm job
    arrays depending on `executor_backend`.

    Per-option details are shown in `--help`, generated from each option's own
    `typer.Option`/`typer.Argument` help text; they are not repeated here.

    Raises:
        typer.BadParameter: If neither or both of `output_dir` and
            `hf_dataset_repo_id` are given; if `hf_dataset_private` or
            `hf_dataset_revision` is set without `hf_dataset_repo_id`; if
            `executor_backend` is `slurm` and `slurm_partition` or
            `slurm_time` is missing; if `executor_backend` is `slurm` and
            both `slurm_venv_path` and `slurm_condaenv` are given; or if any
            Slurm-only option is given while `executor_backend` is `local`.

    Examples:
        Process Danish Wikipedia locally and write Parquet to a directory:

        $ uv run processing_scripts/build_usas_wikipedia_dataset.py \\
              da ./log_data --output-dir ./local_da/ -w 4

        Just report how many shards the dataset has, without processing 
        (this does not use the logging or output directories or create them):

        $ uv run processing_scripts/build_usas_wikipedia_dataset.py \\
              da ./log_data --output-dir ./test --get-number-of-shards
    """
    pipeline_start_time = time.perf_counter()
    tag_mapper = parse_json_object(tag_mapper_json)
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
        raise typer.Exit(0)

    if (output_dir is None) == (hf_dataset_repo_id is None):
        raise typer.BadParameter("Exactly one of --output-dir or --hf-dataset-repo-id must be provided.")
    if hf_dataset_repo_id is None and hf_dataset_private:
        raise typer.BadParameter("--private can only be used with --hf-dataset-repo-id.")
    if hf_dataset_repo_id is None and hf_dataset_revision is not None:
        raise typer.BadParameter("--hf-dataset-revision can only be used with --hf-dataset-repo-id.")

    slurm_settings: SlurmExecutorSettings | None = None
    match executor_backend:
        case ExecutorBackend.slurm:
            if slurm_partition is None or slurm_time is None:
                raise typer.BadParameter("--slurm-partition and --slurm-time are required when --executor=slurm.")
            if slurm_venv_path is not None and slurm_condaenv is not None:
                raise typer.BadParameter("--slurm-venv-path and --slurm-condaenv are mutually exclusive.")
            slurm_settings = SlurmExecutorSettings(
                partition=slurm_partition,
                time=slurm_time,
                cpus_per_task=slurm_cpus_per_task,
                mem_per_cpu_gb=slurm_mem_per_cpu_gb,
                qos=slurm_qos,
                venv_path=slurm_venv_path,
                condaenv=slurm_condaenv,
                mail_user=slurm_mail_user,
                mail_type=slurm_mail_type,
                sbatch_args=parse_json_object(slurm_sbatch_args_json) if slurm_sbatch_args_json is not None else None,
            )
        case ExecutorBackend.local:
            slurm_only_options = {
                "--slurm-partition": slurm_partition,
                "--slurm-time": slurm_time,
                "--slurm-venv-path": slurm_venv_path,
                "--slurm-condaenv": slurm_condaenv,
                "--slurm-mail-user": slurm_mail_user,
                "--slurm-sbatch-args": slurm_sbatch_args_json,
            }
            provided_slurm_only_options = [name for name, value in slurm_only_options.items() if value is not None]
            if provided_slurm_only_options:
                raise typer.BadParameter(f"{', '.join(provided_slurm_only_options)} can only be used with --executor=slurm.")

    if executor_backend is ExecutorBackend.local:
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
    data_trove_logger.info(f"Executor backend: {executor_backend.value!r}")

    executor_factory = PipelineExecutorFactory(
        backend=executor_backend,
        randomize_start_duration=randomize_start_duration,
        skip_completed=overwrite,
        slurm_settings=slurm_settings,
    )

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
    reader_metadata_whitelist = MetadataWhitelistAnnotator(keys_to_keep=frozenset({"page_id", "title", "url", "version"}))
    output_writer_adapter = get_metadata_whitelist_writer_adapter(keys_to_keep=frozenset({
        "page_id",
        "title",
        "url",
        "version",
        "start_end_sentence_character_indexes",
        "tokens",
        "tags",
        "other_tags",
        "mwes",
    }))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        generic_output_filename = f"{wikipedia_language_code_str}" + "${rank}.jsonl.gz"
        # Converting to MB
        max_file_size_in_bytes = int(max_output_file_size * 10e5)

        final_output_filename = f"data/{wikipedia_language_code_str}" + "/${split}/${rank}.parquet"
        max_final_output_file_size_in_bytes = max_final_output_file_size * 1024 * 1024
        if output_dir is not None:
            final_output_pipe = ParquetWriter(
                output_folder=str(output_dir.resolve()),
                output_filename=final_output_filename,
                compression="zstd",
                expand_metadata=True,
                adapter=output_writer_adapter,
                max_file_size=max_final_output_file_size_in_bytes,
            )
        else:
            assert hf_dataset_repo_id is not None
            hf_writer_kwargs = {}
            if hf_local_working_dir is not None:
                hf_writer_kwargs["local_working_dir"] = str(hf_local_working_dir.resolve())
            else:
                hf_local_working_dir_str = create_sub_directory(tmp_dir_path, "hf_local_working_dir")
                hf_writer_kwargs["local_working_dir"] = hf_local_working_dir_str
            if hf_dataset_revision is not None:
                hf_writer_kwargs["revision"] = hf_dataset_revision
            final_output_pipe = HuggingFaceDatasetWriter(
                dataset=hf_dataset_repo_id,
                private=hf_dataset_private,
                output_filename=final_output_filename,
                compression="zstd",
                expand_metadata=True,
                adapter=output_writer_adapter,
                max_file_size=max_final_output_file_size_in_bytes,
                **hf_writer_kwargs,
            )

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

        train_validation_split_annotator = TrainValidationSplitAnnotator(
            validation_percentage=validation_percentage,
            max_validation_documents=max_validation_documents,
            split_hash_metadata_key="page_id",
        )

        logging_dir_minhash_sigs_str = create_sub_directory(Path(main_logging_dir_str), "minhash_dedup_sigs")
        logging_dir_minhash_buckets_str = create_sub_directory(Path(main_logging_dir_str), "minhash_dedup_buckets")
        logging_dir_minhash_clusters_str = create_sub_directory(Path(main_logging_dir_str), "minhash_dedup_clusters")
        logging_dir_minhash_filter_str = create_sub_directory(Path(main_logging_dir_str), "minhash_dedup_filter")
        logging_dir_post_processing_str = create_sub_directory(Path(main_logging_dir_str), "post_processing")
        logging_dir_merged_stats_processing_str = create_sub_directory(Path(main_logging_dir_str), "merged_stats_processing")

        reading_stage = executor_factory.create(
            pipeline=[reader_pipe, get_progress_logger_function("reading"), page_id_title_filter, url_filter, reader_metadata_whitelist, reading_stage_output_pipe],
            tasks=number_data_downloading_tasks,
            workers=number_downloading_workers,
            logging_dir=logging_reading_dir,
            job_name="reading",
        )
        initial_process_stage = executor_factory.create(
            pipeline=[initial_processing_input_pipe, remove_family_tree_formatter, remove_lines_with_given_latex_commands_formatter, wikipedia_markdown_formatter, EmptyTextFilter(), min_words_document_filter, exact_intermediate_output_pipe],
            tasks=number_processing_tasks,
            workers=number_of_workers,
            logging_dir=logging_dir_initial_process_str,
            depends=reading_stage,
            job_name="initial_process")
        exact_sigs_stage = executor_factory.create(
            pipeline=[exact_intermediate_read_pipe, exact_dedup_sig],
            tasks=number_processing_tasks,
            workers=number_of_workers,
            logging_dir=logging_dir_exact_sigs_str,
            depends=initial_process_stage,
            job_name="exact_dedup_sigs",
        )
        exact_finds_stage = executor_factory.create(
            pipeline=[exact_dedup_finds],
            tasks=number_of_workers, # Has to be the same as finder_workers
            workers=number_of_workers, # Has to be the same as finder_workers
            logging_dir=logging_dir_exact_finds_str,
            depends=exact_sigs_stage,
            job_name="exact_dedup_finds")
        exact_filter_stage = executor_factory.create(
            pipeline=[exact_intermediate_read_pipe, exact_dedup_filter, minhash_intermediate_output_pipe],
            tasks=number_processing_tasks, # This has to match exact_sigs_stage
            workers=number_of_workers,
            logging_dir=logging_dir_exact_filter_str,
            depends=exact_finds_stage,
            job_name="exact_dedup_filter")
        minhash_sigs_stage = executor_factory.create(
            pipeline=[minhash_intermediate_read_pipe, minhash_dedup_sig],
            tasks=number_processing_tasks,
            workers=number_of_workers,
            logging_dir=logging_dir_minhash_sigs_str,
            depends=exact_filter_stage,
            job_name="minhash_dedup_sigs")
        minhash_buckets_stage = executor_factory.create(
            pipeline=[minhash_dedup_buckets],
            tasks=minhash_dedup_config.num_buckets,
            workers=number_of_workers,
            logging_dir=logging_dir_minhash_buckets_str,
            depends=minhash_sigs_stage,
            job_name="minhash_dedup_buckets")
        minhash_clusters_stage = executor_factory.create(
            pipeline=[minhash_dedup_clusters],
            tasks=1,
            workers=1,
            logging_dir=logging_dir_minhash_clusters_str,
            depends=minhash_buckets_stage,
            job_name="minhash_dedup_clusters")
        minhash_filter_stage = executor_factory.create(
            pipeline=[minhash_intermediate_read_pipe, minhash_dedup_filter, word_statistics, minhash_filter_output_pipe],
            tasks=number_processing_tasks, # This has to match minhash_sigs_stage
            workers=number_of_workers,
            logging_dir=logging_dir_minhash_filter_str,
            depends=minhash_clusters_stage,
            job_name="minhash_dedup_filter")
        post_processing_stage = executor_factory.create(
            pipeline=[minhash_filter_read_pipe, SentenceSplitterAnnotator(wikipedia_language_code_str), TokenPyMUSASAnnotator(wikipedia_language_code_str, tag_mapper=tag_mapper), train_validation_split_annotator, final_output_pipe],
            tasks=number_processing_tasks, # This has to match minhash_sigs_stage
            workers=number_of_workers,
            logging_dir=logging_dir_post_processing_str,
            depends=minhash_filter_stage,
            job_name="post_processing")
        merge_stage = executor_factory.create(
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
            job_name="merged_stats",
        )

        merge_stage.run()
        data_trove_logger.info(f"PIPELINE_RUNTIME_SECONDS={time_elapsed(pipeline_start_time):.2f}")


if __name__ == "__main__":
    typer.run(main)