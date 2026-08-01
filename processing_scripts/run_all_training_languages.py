import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated

import datasets
import typer
from rich import print as rprint

from wikipedia_processing.utils import (
    compute_shard_scaled_tasks_and_workers,
    create_sub_directory,
    get_number_of_shards,
    get_usas_language_processing_information,
    get_valid_usas_language_processing_wikipedia_codes,
    scale_workers_to_budget,
)

DATASET_ID = "HuggingFaceFW/finewiki"

app = typer.Typer(add_completion=False)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def main(
    ctx: typer.Context,
    logging_dir: Annotated[Path, typer.Argument(help="Shared parent directory for every language's logs. Each language writes to its own `logging_dir/<wikipedia_code>/` subdirectory, same as a single build_usas_wikipedia_dataset.py run.")],
    languages_file: Annotated[Path | None, typer.Option("--languages-file", help="YAML file with a top-level 'languages' list to read training languages from. Defaults to the packaged wikipedia_processing/data/usas_wikipedia_processing.yaml.")] = None,
    shard_tasks_multiplier: Annotated[float, typer.Option("--shard-tasks-multiplier", help="Target number of processing tasks per dataset shard, used to size -w/-t per language off that language's shard count.")] = 3.0,
    min_tasks_per_language: Annotated[int, typer.Option("--min-tasks-per-language", help="Minimum target task count for any language, regardless of shard count.")] = 4,
    max_tasks_per_language: Annotated[int, typer.Option("--max-tasks-per-language", help="Maximum target task count for any language, regardless of shard count. Kept well below Slurm's default job array size limit (1001).")] = 200,
    max_workers_per_language: Annotated[int, typer.Option("--max-workers-per-language", help="Maximum number of concurrent workers (-w) to use for any single language.")] = 16,
    max_number_of_parallel_tasks: Annotated[int | None, typer.Option("--max-number-of-parallel-tasks", help="Optional cap on the total number of Slurm tasks allowed to run concurrently across ALL training languages combined, i.e. the sum of every language's -w. With --executor=slurm, -w is Slurm's own per-stage concurrency throttle (--array=...%W), and each language's own stages run one at a time, so this sum is an exact upper bound on total concurrently-running Slurm tasks at any instant. When set and the shard-scaled -w values would sum above this, each language's -w is shrunk proportionally to fit (see wikipedia_processing.utils.scale_workers_to_budget), and -t is recomputed so each language's total task count stays close to its original shard-scaled target. Must be >= the number of training languages. Only meaningful with --executor=slurm; with --executor=local, concurrency is already separately capped by the CPUs available to each language's own subprocess.")] = None,
    hf_dataset_repo_id: Annotated[str, typer.Option("--hf-dataset-repo-id", help="Shared HuggingFace Hub dataset repository every language uploads its Parquet output to, under its own data/<wikipedia_code>/ path.")] = "ucrelnlp/Multilingual-USAS-Labelled-Silver-Wikipedia",
    stagger_seconds: Annotated[int, typer.Option("--stagger-seconds", help="Delay between launching each language's subprocess, to avoid submitting every language's Slurm jobs and HuggingFace dataset requests simultaneously.")] = 10,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print each language's computed shard count, -w, -t, and the command that would run, without launching anything.")] = False,
) -> None:
    """Run `build_usas_wikipedia_dataset.py` once per `training: true` language within the `languages_file`.

    For each language in `languages_file` with `training: true`, determines
    the language's HuggingFace `finewiki` dataset shard count, derives
    `-w`/`-t` values scaled to that shard count via
    `compute_shard_scaled_tasks_and_workers` (so tiny languages don't get
    more Slurm tasks than they have data for, and large languages get more
    parallelism), then launches `build_usas_wikipedia_dataset.py` for every
    language concurrently, as independent background subprocesses, using
    the same Python interpreter (`sys.executable`) this command is itself
    running under -- so it works wherever this script was launched from.

    Any options not recognized by this command (e.g. `--executor`,
    `--slurm-partition`, `--slurm-time`, `--overwrite`, ...) are forwarded
    verbatim to every language's `build_usas_wikipedia_dataset.py`
    invocation, after this command's own computed `-w`/`-t`/
    `--hf-dataset-repo-id` options.

    Each language's subprocess blocks until its own dependent chain of
    pipeline stages fully finishes (same as a single-language run), so this
    command itself blocks until every language finishes -- run it under
    `tmux`/`screen`/`nohup` for a real multi-hour run.

    Per-option details are shown in `--help`, generated from each option's own
    `typer.Option`/`typer.Argument` help text; they are not repeated here.

    Raises:
        typer.Exit: With code 1 if no training languages are found, if
            `--max-number-of-parallel-tasks` is set below the number of
            training languages, or if any language's subprocess exits
            non-zero. With code 0 after a `--dry-run` preview.

    Examples:
        Preview sizing without launching anything:

        $ uv run processing_scripts/run_all_training_languages.py \\
              ./log_data --dry-run --executor slurm \\
              --slurm-partition gpu --slurm-time 6:00:00

        Real run:

        $ uv run processing_scripts/run_all_training_languages.py \\
              ./log_data --executor slurm --slurm-partition gpu \\
              --slurm-time 6:00:00 --slurm-cpus-per-task 2 \\
              --slurm-mem-per-cpu-gb 4
    """
    wikipedia_codes = get_valid_usas_language_processing_wikipedia_codes(language_data_file=languages_file)
    languages = [get_usas_language_processing_information(code, language_data_file=languages_file) for code in wikipedia_codes]
    training_languages = [language for language in languages if language["training"]]

    if not training_languages:
        rprint(f"[red]No languages with training=true found in {languages_file}[/red]")
        raise typer.Exit(1)

    rprint(f"Found {len(training_languages)} training languages: {[language['wikipedia_code'] for language in training_languages]}")

    if max_number_of_parallel_tasks is not None and max_number_of_parallel_tasks < len(training_languages):
        rprint(
            f"[red]--max-number-of-parallel-tasks ({max_number_of_parallel_tasks}) is below the number of "
            f"training languages ({len(training_languages)}); every language needs at least 1 worker.[/red]"
        )
        raise typer.Exit(1)

    driver_logging_dir = logging_dir / "driver"

    language_sizes: list[tuple[str, int, int, int]] = []
    for language in training_languages:
        wikipedia_code = language["wikipedia_code"]
        dataset: datasets.IterableDataset = datasets.load_dataset(DATASET_ID, split="train", name=wikipedia_code, streaming=True)
        number_of_shards = get_number_of_shards(dataset)
        number_of_workers, tasks_multiplier = compute_shard_scaled_tasks_and_workers(
            number_of_shards=number_of_shards,
            shard_tasks_multiplier=shard_tasks_multiplier,
            min_tasks=min_tasks_per_language,
            max_tasks=max_tasks_per_language,
            max_workers=max_workers_per_language,
        )
        language_sizes.append((wikipedia_code, number_of_shards, number_of_workers, tasks_multiplier))

    if max_number_of_parallel_tasks is not None:
        original_workers = [number_of_workers for _, _, number_of_workers, _ in language_sizes]
        scaled_workers = scale_workers_to_budget(original_workers, budget=max_number_of_parallel_tasks)
        rescaled_sizes: list[tuple[str, int, int, int]] = []
        for (wikipedia_code, number_of_shards, number_of_workers, tasks_multiplier), new_number_of_workers in zip(language_sizes, scaled_workers):
            if new_number_of_workers != number_of_workers:
                original_total_tasks = number_of_workers * tasks_multiplier
                tasks_multiplier = math.ceil(original_total_tasks / new_number_of_workers)
                number_of_workers = new_number_of_workers
            rescaled_sizes.append((wikipedia_code, number_of_shards, number_of_workers, tasks_multiplier))
        language_sizes = rescaled_sizes

    planned_runs: list[tuple[str, list[str], Path]] = []
    for wikipedia_code, number_of_shards, number_of_workers, tasks_multiplier in language_sizes:
        command = [
            sys.executable, "processing_scripts/build_usas_wikipedia_dataset.py",
            wikipedia_code, str(logging_dir),
            "-w", str(number_of_workers),
            "-t", str(tasks_multiplier),
            "--hf-dataset-repo-id", hf_dataset_repo_id,
            *ctx.args,
        ]
        log_file = driver_logging_dir / f"{wikipedia_code}.log"
        planned_runs.append((wikipedia_code, command, log_file))
        rprint(
            f"[cyan]{wikipedia_code}[/cyan]: {number_of_shards} shards -> "
            f"-w {number_of_workers} -t {tasks_multiplier} "
            f"(~{number_of_workers * tasks_multiplier} processing tasks)"
        )
        rprint(f"  command: {' '.join(command)}")

    if max_number_of_parallel_tasks is not None:
        total_workers = sum(number_of_workers for _, _, number_of_workers, _ in language_sizes)
        rprint(f"Total workers across all languages: {total_workers} / {max_number_of_parallel_tasks}")

    if dry_run:
        rprint("[yellow]--dry-run set, not launching anything.[/yellow]")
        raise typer.Exit(0)

    create_sub_directory(logging_dir, "driver")

    processes: list[tuple[str, subprocess.Popen, Path]] = []
    for index, (wikipedia_code, command, log_file) in enumerate(planned_runs):
        if index > 0 and stagger_seconds > 0:
            time.sleep(stagger_seconds)
        rprint(f"[green]Launching {wikipedia_code}[/green] (log: {log_file})")
        with log_file.open("w", encoding="utf-8") as fp:
            process = subprocess.Popen(command, stdout=fp, stderr=subprocess.STDOUT)
        processes.append((wikipedia_code, process, log_file))

    failed_languages: list[str] = []
    for wikipedia_code, process, log_file in processes:
        return_code = process.wait()
        if return_code == 0:
            rprint(f"[green]{wikipedia_code} finished successfully[/green]")
        else:
            rprint(f"[red]{wikipedia_code} failed (exit code {return_code}), see {log_file}[/red]")
            failed_languages.append(wikipedia_code)

    if failed_languages:
        rprint(f"[red]{len(failed_languages)} language(s) failed: {failed_languages}[/red]")
        raise typer.Exit(1)
    rprint("[green]All languages finished successfully[/green]")


if __name__ == "__main__":
    app()