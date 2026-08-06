# wikipedia-USAS-processing

This repository contains various [DataTrove](https://github.com/huggingface/datatrove) pipelines, filters, formatters, and helper functions for processing the [HuggingFaceFW finewiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki) dataset, in various languages listed in the [languages section](#languages), to create a synthetic (silver labelled) training dataset for USAS semantic tags and Multi Word Expression (MWE) identification for some languages.

For more information on the filtering and processing, see the [filtering and processing section below](#filtering-and-processing) and for more information about the data we use see [the data section below.](#data)


## Setup

You can either use the dev container with your favourite editor, e.g. VSCode. Or you can create your setup locally below we demonstrate both.

In both cases they share the same tools, of which these tools are:
* [uv](https://docs.astral.sh/uv/) for Python packaging and development
* [make](https://www.gnu.org/software/make/) (OPTIONAL) for automation of tasks, not strictly required but makes life easier.

### Dev Container

A [dev container](https://containers.dev/) uses a docker container to create the required development environment, the Dockerfile we use for this dev container can be found at [./.devcontainer/Dockerfile](./.devcontainer/Dockerfile). To run it locally it requires docker to be installed, you can also run it in a cloud based code editor, for a list of supported editors/cloud editors see [the following webpage.](https://containers.dev/supporting)

To run for the first time on a local VSCode editor (a slightly more detailed and better guide on the [VSCode website](https://code.visualstudio.com/docs/devcontainers/tutorial)):
1. Ensure docker is running.
2. Ensure the VSCode [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension is installed in your VSCode editor.
3. Open the command pallete `CMD + SHIFT + P` and then select `Dev Containers: Rebuild and Reopen in Container`

You should now have everything you need to develop, `uv`, `make`, for VSCode various extensions like `Pylance`, etc.

If you have any trouble see the [VSCode website.](https://code.visualstudio.com/docs/devcontainers/tutorial).

### Local

To run locally first ensure you have the following tools installted locally:
* [uv](https://docs.astral.sh/uv/getting-started/installation/) for Python packaging and development. (version `0.9.6`)
* [make](https://www.gnu.org/software/make/) (OPTIONAL) for automation of tasks, not strictly required but makes life easier.
  * Ubuntu: `apt-get install make`
  * Mac: [Xcode command line tools](https://mac.install.guide/commandlinetools/4) includes `make` else you can use [brew.](https://formulae.brew.sh/formula/make)
  * Windows: Various solutions proposed in this [blog post](https://earthly.dev/blog/makefiles-on-windows/) on how to install on Windows, inclduing `Cygwin`, and `Windows Subsystem for Linux`.

When developing on the project you will want to install the Python package locally in editable format with all the extra requirements, this can be done like so:

```bash
uv sync --all-extras
```

### Linting

Linting and formatting with [ruff](https://docs.astral.sh/ruff/) it is a replacement for tools like Flake8, isort, Black etc, and we us [ty](https://github.com/astral-sh/ty) for type checking.

To run the linting:

``` bash
make lint
```

### Tests

To run the tests (uses pytest and coverage) and generate a coverage report:

``` bash
make test
```

## HuggingFace Authentication

Before processing or uploading to the HuggingFace hub please authenticate using a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens);

``` bash
hf auth login
```

or by using a token that is set within [./.env](./.env), read using [dotenv](https://github.com/theskumar/python-dotenv), e.g.

```bash
HF_TOKEN="HUGGINGFACE_TOKEN_KEY_VALUE"
```

Set the relevant permissions, the minimum for this is repository is "read" only permission, if you want to upload the created synthetic silver labelled dataset to HuggingFace please ensure that you have allowed write permission to the namespace/repository you are going to upload too on HuggingFace.


## Data

The data will be coming from [HuggingFaceFW finewiki dataset](https://huggingface.co/datasets/HuggingFaceFW/finewiki) and will be filtered so that each Wikipedia article is either rated as a "Good Articles" (GA) or "Featured Articles" (FA) by an editor, we hope that this will remove articles that might be incomplete or require additional editing. This filtering is inspired by [Conia et al. 2024](https://aclanthology.org/2024.naacl-long.442.pdf) whereby they found training on data from only "featured" and "good" articles performed similarly to training on the far larger Wikipedia articles that contained non-good and non-featured articles thus showing that training on smaller amounts of data is as affective and more efficient. The "Featured" and "Good" article can be defined differently for each Wikipedia language site as stated in the English site definition within the following [article](https://en.wikipedia.org/wiki/Wikipedia:Content_assessment). The list of **GA** and **FA** can be found at [the HuggingFace dataset ucrelnlp/wikipedia-ga-fa-ids](https://huggingface.co/datasets/ucrelnlp/wikipedia-ga-fa-ids).


## Languages

The languages that this repository covers and supports, of which this table is also available in machine readable format at [./data/languages.yaml](./data/languages.yaml) (languages that have the value of `True` for the key `training`). These languages have been selected based on semantic tagging support for the given language whereby in some cases setting up the semantic tagger for a given language can be difficult within a large scale tagging pipeline in addition some languages have very few to none **GA** or **FA** articles.

| Language | ISO 639-3 |
|----------|-----------|
| English | eng |
| Dutch | nld |
| Spanish | spa |
| Danish | dan |
| Italian | ita |
| Portuguese | por |
| Chinese | zho |
| Finnish | fin |

## Filtering and Processing

Each Wikipedia article from [HuggingFaceFW finewiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki) goes through the following pipeline;
* The Wikipedia article ID and title match an article within the **GA** or **FA** articles (taken from the [ucrelnlp/wikipedia-ga-fa-ids dataset](https://huggingface.co/datasets/ucrelnlp/wikipedia-ga-fa-ids)).
* The Wikipedia article is not an article that is in the test data. (manual list of Wikipedia article URLs)
* Remove Wikipedia family tree tables, mathematical equations, and tables from the article text.
* Remove any Markdown formatting like headers using [mistune Python package](https://github.com/lepture/mistune) from the article text.
* Remove articles that contain less than 50 tokens based off a language specific tokenizer.
* Apply exact and then MinHash de-duplication.
* Sentence split using language specific sentence splitters (spaCy sentence splitters are installed when the processing script is running).
* USAS semantic and when the tagger supports it MWE identification using [PyMUSAS Rule Based languages specific taggers](https://ucrel.github.io/pymusas/#rule-based) per sentence (The spaCy tokenizers, lemmatizers, and POS taggers as well as the PyMUSAS tagger are installed when the processing script is running).
  * Each sentence has all leading and starting whitespace removed, any tokens identified as whitespace are kept as tokens but with no USAS tags, and all `PUNCT` tags are mapped onto `Z9` tags. If a `Z99` USAS tag (the unmatched tag) is produced by any of the rule based taggers this USAS tag is not kept, it is removed.

The processed data will contain the following fields:
* `text` - the processed article text.
* `id` - dawiki/1171348
* `page_id` - Page ID `1171348`
* `title` - Article title.
* `url` - https://da.wikipedia.org/wiki/El_Salvador_ved_sommer-OL_2024
* `version` (int|string) - revision/version identifier of the page (comes from [HuggingFaceFW finewiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki)) `1167219203`
* `start_end_sentence_character_indexes` - list of start and end character offsets for each sentence, e.g. `[[0, 10], [11, 15]]` the first sentence is between `text[0:10]`.
* `tokens` - list of a list of tokens whereby the inner list represents the tokens for a given sentence, e.g. `tokens[0]` would contain all of the tokens in the first sentence.
* `tags` - list of a list of a list of USAS tags that were predicted by the PyMUSAS Rule Based languages specific tagger. The inner list represents the most likely USAS tags for the given token, e.g. `tags[0][0]` will contain a list of most likely USAS tags for the first token in the first sentence, in most cases it will only contain one USAS tag. When it contains more than one USAS tag this represents a token in which the meaning is a combination of the given predicted USAS tags. Some tokens will contain no USAS tags as the Rule Based tagger cannot make prediction for all tokens.
* `other_tags` - list of a list of a list of USAS tags with the same shape as `tags`, but containing every valid USAS tag for a token that was **not** its most likely tag(s), e.g. `other_tags[0][0]` will contain any other valid USAS tags for the first token in the first sentence. Most tokens will contain no other USAS tags, in which case the inner list is empty.
* `mwes` - list of a list of MWE labels that were predicted by the PyMUSAS Rule Based languages specific tagger, these always relate to the most likely USAS tags. The MWE labels denote at the sentence level which tokens are MWEs, e.g. `mwes[0][0]` represent all of the MWE labels for the first token in the first sentence, if it contains `1` and `mwes[0][1]` also contains `1` then the first token and second token in the first sentence are a MWE. If more than one label occurs then MWEs are overlapping which should not be the case with PyMUSAS taggers. MWEs can be dis-continuous. The index of MWE labels always start at 1 and reset per sentence, e.g. the first sentence can contain a MWE label of `1` and so can the second sentence, but they will be different MWEs as MWEs are constrained to occur within a single sentence; they cannot span sentence boundaries.

The final data is written as [zstd](https://github.com/facebook/zstd)-compressed [Parquet](https://parquet.apache.org/) files rather than JSONL, as Parquet gives better compression on the repetitive, deeply nested `tokens`/`tags`/`other_tags`/`mwes` fields and native [HuggingFace Hub Dataset Viewer](https://huggingface.co/docs/hub/en/datasets-viewer) support.

### Train/validation split

Each language's documents are split into `train` and `validation` subsets, written to separate `train`/`validation` subfolders (there is no `split` field/column in the data itself, the split is entirely determined by which subfolder a file is in). The validation split is **X% of the language's documents, or N documents, whichever is reached first** (controlled by the `--validation-percentage`/`-v` and `--max-validation-documents`/`-n` options of [build_usas_wikipedia_dataset.py](processing_scripts/build_usas_wikipedia_dataset.py) below) rather than a flat percentage. A flat percentage would give under-resourced languages (some have as few as ~200 articles) a tiny handful of validation documents while giving well-resourced languages like English potentially thousands, which is more than needed for a useful validation set and just eats into training data. Capping at `min(X% of total, N)` keeps small languages at their natural percentage-based split (they will never reach the `N` cap) while bounding well-resourced languages' validation set to a sane absolute size. The split assignment is deterministic (hashed from each document's page ID), so re-running the pipeline reproduces the same split.

### Filtering and Processing script

[processing_scripts/build_usas_wikipedia_dataset.py](processing_scripts/build_usas_wikipedia_dataset.py) is the pipeline entry point. It writes the final output either to a local directory or directly to a HuggingFace Hub dataset repository — pass exactly one of `--output-dir` or `--hf-dataset-repo-id`.

Writing locally, as Parquet files under `./local_da/data/da/{train,validation}/`:

``` bash
uv run processing_scripts/build_usas_wikipedia_dataset.py da ./log_data/ --output-dir ./local_da/
```

Uploading directly to a HuggingFace Hub dataset repository (see [HuggingFace Authentication](#huggingface-authentication) above):

``` bash
uv run processing_scripts/build_usas_wikipedia_dataset.py da ./log_data/ --hf-dataset-repo-id ucrelnlp/Multilingual-USAS-Labelled-Silver-Wikipedia
```

Some options worth knowing about (run `--help` for the full list):
* `--max-final-output-file-size`/`-e` - maximum size in MB of the final Parquet shards (default 200MB); distinct from `--max-output-file-size`/`-s`, which only governs intermediate staging files used during processing.
* `--validation-percentage`/`-v` and `--max-validation-documents`/`-n` - control the train/validation split described above.
* `--private`/`--public` - whether a Hub repository created by `--hf-dataset-repo-id` is private (default: public).

An example of processing the Danish Wikipedia data locally but saving the data to the HuggingFace `ucrelnlp/Multilingual-USAS-Labelled-Silver-Wikipedia` repository.

``` bash
uv run processing_scripts/build_usas_wikipedia_dataset.py da ./log_data/ -o -w 5 -t 2 -m 0.85 --hf-dataset-repo-id ucrelnlp/Multilingual-USAS-Labelled-Silver-Wikipedia --public
```

#### Running on Slurm

By default the script runs each pipeline stage (reading, dedup, tagging, etc.) as local multiprocessing workers. Pass `--executor slurm` to instead submit each stage as a Slurm job array via [DataTrove's `SlurmPipelineExecutor`](https://github.com/huggingface/datatrove), which is useful when a single machine doesn't have enough CPUs/memory to process a language in reasonable time. Stages still run in the same dependency order — one stage's Slurm job array only starts once the previous stage's finishes, enforced via Slurm `--dependency` chaining between the stages' `sbatch` calls, not by this script waiting in-process.

Because of that chaining, `SlurmPipelineExecutor` submits every stage's job and then returns immediately — it does not poll Slurm for completion. So with `--executor slurm` this script itself exits as soon as all stages are submitted, well before the actual Slurm jobs finish; a clean exit (code 0) only means submission succeeded, not that processing has completed. Track real progress with `squeue`/`sacct` or by watching `logging_dir/`.

`--slurm-partition` and `--slurm-time` are required when `--executor slurm` is set; every other `--slurm-*` option is optional and only used with `--executor slurm` (passing any of them with the default `--executor local` is an error).

``` bash
uv run processing_scripts/build_usas_wikipedia_dataset.py da ./log_data/ \
    --output-dir ./local_da/ \
    --executor slurm \
    --slurm-partition compute \
    --slurm-time 4:00:00 \
    --slurm-cpus-per-task 4 \
    --slurm-mem-per-cpu-gb 4 \
    -w 32
```

Some other Slurm options worth knowing about (run `--help` for the full list):
* `--slurm-venv-path`/`--slurm-condaenv` - activate a virtualenv or conda environment in each Slurm job before running; mutually exclusive with each other.
* `--slurm-qos` - Slurm QOS to submit jobs under (default `normal`).
* `--slurm-mail-user`/`--slurm-mail-type` - get emailed on job events, e.g. `--slurm-mail-type FAIL`.
* `--slurm-sbatch-args` - JSON object of any additional raw `sbatch` arguments, e.g. `'{"account": "myaccount"}'`.

##### Understanding tasks, workers, and resources on Slurm

The pipeline runs as a chain of dependent stages (reading → initial processing → exact dedup → MinHash dedup → post-processing/tagging → stats merge). With `--executor slurm`, **each stage is submitted as its own Slurm job array**, and a stage's job array is only submitted once the previous stage's array has fully finished — stages never run concurrently with each other.

Within a single stage's job array:
* `tasks` (derived from `-w`/`--number-of-workers` × `-t`/`--tasks-multiplier`) is the **array size** — the number of independent work units (array indices) that stage is split into, e.g. `sbatch --array=0-159` for 160 tasks.
* `workers` (essentially `-w`/`--number-of-workers`) is a **concurrency throttle** on that array, e.g. `--array=0-159%32` means at most 32 array indices run at the same instant. It is **not** a node count, and `-w 32` does **not** reserve 32 nodes.
* `--slurm-cpus-per-task` and `--slurm-mem-per-cpu-gb` are the resources requested **per array index** (per task), e.g. 4 CPUs and 4GB/CPU = 16GB for that one task. Slurm's scheduler then places each task on whatever node has free capacity — it may pack many concurrently-running tasks onto a single large node, or spread them across several smaller ones, entirely independently of `workers`. If you need specific node placement, pass the relevant raw flags via `--slurm-sbatch-args`.
* `--slurm-time` is a **per-task limit, not a shared budget**: every task gets its own fresh clock starting when *that task* begins running, regardless of when the array was submitted or how long earlier tasks took. A task started 2 hours after the array began still gets the full time limit from its own start.
* Because of the concurrency throttle, a stage's total wall-clock time can exceed the per-task time limit: e.g. 160 tasks at 32 concurrent, each taking close to the 4-hour limit, is roughly `ceil(160 / 32) × 4h ≈ 20h` for that stage to fully drain, even though no single task exceeds 4 hours.
* With `--executor local`, `-w`/`--number-of-workers` is additionally capped by the CPU count of the machine running the script (`os.process_cpu_count()`), since local workers are real concurrent processes on that machine. This cap does **not** apply with `--executor slurm` — there, `-w` is only a concurrency throttle on the job array (see above) and is not limited by the submission host's core count.
* `-j`/`--tasks-per-job` (Slurm-only, default `1`) shrinks the **array size** itself: instead of one array index per task, each array index runs `tasks_per_job` tasks sequentially, one after another, inside the same allocation. A stage with 160 tasks and `-j 5` submits `sbatch --array=0-31` (32 array indices) instead of `0-159` — the same 160 tasks still all get run, just 5-at-a-time-in-series per array index rather than each getting its own index. This doesn't change `-w`/`-t`, concurrency, or resource requests per index at all; it only reduces how many array indices Slurm has to create in the first place, at the cost of each index taking up to `tasks_per_job` times as long. Useful when a Slurm admin caps the total number of array indices you can have submitted (running or pending) at once — see `--max-number-of-parallel-tasks` below, which sets this automatically across a multi-language run.

#### Uploading multiple languages to the same Hub repository

All languages can share a single Hub dataset repository so users can pick a language, by running the script once per language against the same `--hf-dataset-repo-id`. Each run writes into its own `data/<wikipedia_language_code>/{train,validation}/` path within the repo, so nothing is overwritten between languages.

For the Hub Dataset Viewer to expose each language as a selectable config with its train/validation splits, add a `configs:` block to the dataset repository's own README (its "dataset card"), keyed by `wikipedia_language_code` for consistency with [data/languages.yaml](data/languages.yaml) and the CLI, e.g.:

```yaml
configs:
  - config_name: da
    data_files:
      - split: train
        path: "data/da/train/*.parquet"
      - split: validation
        path: "data/da/validation/*.parquet"
  - config_name: en
    data_files:
      - split: train
        path: "data/en/train/*.parquet"
      - split: validation
        path: "data/en/validation/*.parquet"
```

#### Running every training language at once

[processing_scripts/run_all_training_languages.py](processing_scripts/run_all_training_languages.py)
automates the "run once per language" pattern above: it reads every language with
`training: true` (from the packaged
[wikipedia_processing/data/usas_wikipedia_processing.yaml](wikipedia_processing/data/usas_wikipedia_processing.yaml)
by default, or a custom file via `--languages-file`), and for each one launches
`build_usas_wikipedia_dataset.py` as an independent, concurrently-running subprocess,
using the same Python interpreter this script is itself running under (`sys.executable`)
so it works from a Slurm login node or any other
environment, as long as this script was launched with an
interpreter that already has the project's dependencies (e.g. via `uv run`, or an
activated venv). This only affects the process that submits/polls each language's Slurm
job array; the environment each pipeline stage actually runs in on the compute nodes is
still controlled independently via `--slurm-venv-path`/`--slurm-condaenv`.

Rather than using the same fixed `-w`/`-t` for every language, it first checks each
language's `finewiki` dataset shard count and scales `-w`/`-t` off that (via
`--shard-tasks-multiplier`, clamped between `--min-tasks-per-language` and
`--max-tasks-per-language`, and capped at `--max-workers-per-language` concurrent
workers) — so small languages aren't handed more Slurm tasks than they have data to
fill, and large languages get proportionally more parallelism. Any options this script
doesn't declare itself (`--executor`, `--slurm-partition`, `--slurm-time`, `--overwrite`,
etc.) are forwarded verbatim to every language's invocation.

``` bash
# Preview the computed shard counts / -w / -t / commands without launching anything:
uv run processing_scripts/run_all_training_languages.py ./log_data --dry-run \
    --executor slurm --slurm-partition compute --slurm-time 6:00:00

python processing_scripts/run_all_training_languages.py ./log_data --executor slurm --slurm-partition cpu-6h --slurm-time 4:00:00 --hf-dataset-repo-id ucrelnlp/Multilingual-USAS-Labelled-Silver-Wikipedia --slurm-mem-per-cpu-gb 2 --slurm-venv-path /mnt/nfs/homes/mooreap1/wikipedia-USAS-processing/venv/bin/python --slurm-cpus-per-task 2

--dry-run

uv run processing_scripts/run_all_training_languages.py ./log_data --executor slurm --slurm-partition cpu-6h --slurm-time 4:00:00 --hf-dataset-repo-id ucrelnlp/Multilingual-USAS-Labelled-Silver-Wikipedia --slurm-mem-per-cpu-gb 2 --slurm-venv-path /mnt/nfs/homes/mooreap1/wikipedia-USAS-processing/venv/bin/python --slurm-cpus-per-task 2 --dry-run --slurm-sbatch-args "'{\"nice\": 100}'"


# Real run, uploading every language to the same shared Hub repo:
uv run processing_scripts/run_all_training_languages.py ./log_data \
    --executor slurm --slurm-partition compute --slurm-time 6:00:00 \
    --slurm-cpus-per-task 4 --slurm-mem-per-cpu-gb 4
```

With `--executor local`, each language's process blocks until its own full stage chain
finishes (same as a single-language run), so this command blocks until every language
finishes — run it under `tmux`/`screen`/`nohup` for a real multi-hour run.

With `--executor slurm` this is **not** the case: DataTrove's `SlurmPipelineExecutor`
submits each stage as a chained `sbatch` job array and returns as soon as submission
succeeds, without waiting for the jobs to actually run (see [Running on
Slurm](#running-on-slurm) above). So this command exits almost immediately after every
stage for every language has been submitted — the `finished successfully`/`failed`
messages it prints only reflect whether submission succeeded, not whether the Slurm jobs
themselves have completed. Use `squeue`/`sacct`, or tail `logging_dir/<wikipedia_code>/`,
to track the actual runs. Per-language stdout/stderr from the driver process itself is
captured under `logging_dir/driver/<wikipedia_code>.log`.

##### Staying under a Slurm submitted-jobs quota with `--max-number-of-parallel-tasks`

Because every stage of a language's pipeline is chained by Slurm `--dependency` rather
than by this script waiting in-process, launching several languages within seconds of
each other (see `--stagger-seconds`) means **all of their stages get submitted to Slurm
essentially at once** — most stay pending on their dependency, but a Slurm admin's
`MaxSubmitJobsPerUser`-style quota typically counts pending and running array indices
alike. That total is not something `-w`/`-t` can control (they only throttle
concurrency of already-submitted indices, see above); the only way to actually shrink
how many array indices get created is `-j`/`--tasks-per-job`.

`--max-number-of-parallel-tasks` automates this: given a budget, it estimates each
training language's own peak simultaneously-submitted array-index count (summed across
its ~11 chained stages), proportionally shrinks each language's share of that budget
when it doesn't fit, and works out a `-j`/`--tasks-per-job` value per language that
brings its peak down to its share — appending `--tasks-per-job` to that language's
forwarded `build_usas_wikipedia_dataset.py` command. It never touches `-w`/`-t` or the
actual shard-scaled data-processing parallelism; the only effect is on how many Slurm
array indices get created to do that same work. Every language has a fixed floor of 11
array indices (once `-j` is large enough that every stage collapses to a single index),
so the budget must be at least `11 × (number of training languages)`.

``` bash
# Cap the combined peak submitted array indices across all languages at 300:
uv run processing_scripts/run_all_training_languages.py ./log_data --dry-run \
    --executor slurm --slurm-partition compute --slurm-time 6:00:00 \
    --max-number-of-parallel-tasks 300
```

Only meaningful with `--executor slurm`; it has no effect with `--executor local`.


## Deduplicating an existing dataset

[processing_scripts/deduplicate_wikipedia_dataset.py](processing_scripts/deduplicate_wikipedia_dataset.py) is a post-hoc fix-up script for a dataset already built and uploaded by `build_usas_wikipedia_dataset.py` (e.g. `ucrelnlp/Multilingual-USAS-Labelled-Silver-Wikipedia`) — it does not re-run any of the filtering/tagging pipeline above, it only reads and rewrites each language's already-processed `train`/`validation` Parquet output. It fixes two issues that can occur, especially after re-running the pipeline for a language (e.g. to add more articles) or on a highly-parallel Slurm run:

* **Duplicate `id`s.** For each language, every row of `train` and `validation` (combined) is grouped by `id`. Where an `id` occurs more than once, only the row with the highest `version` is kept and the rest are dropped. If the duplicate copies span both `train` and `validation`, the surviving row always ends up in `validation` (moved out of `train` if that's where the highest-version copy originally was); duplicates confined to a single split just keep the surviving row in that same split.
* **Validation split overflow.** `TrainValidationSplitAnnotator` (see [Train/validation split](#trainvalidation-split) above) is meant to cap the whole validation split at `--max-validation-documents`, but divides that cap evenly across DataTrove ranks with a `max(1, ...)` floor — so whenever a language's rank/task count exceeds `--max-validation-documents`, every rank keeps at least 1 validation document regardless, and the real total ends up close to the rank count instead of the intended cap (English hit this after a 27-task Slurm run capped at 20: it ended up with 27 validation documents, not 20). This script re-caps the post-dedup validation split at `--max-validation-documents`/`-n` (default 20), moving any excess back into `train` deterministically (same `page_id` hash the annotator itself uses), without needing to re-run the whole pipeline.

By default the script only **reports** what it would change — pass `--output-dir` to also write the deduplicated Parquet locally, and/or `--push` to commit the result back to the Hub repository (this replaces each processed language's existing `train`/`validation` Parquet shards in one commit per language, so no stale duplicate shards are left behind).

``` bash
# Report duplicate/overflow counts for every language without writing anything:
uv run processing_scripts/deduplicate_wikipedia_dataset.py

# Write deduplicated Parquet for Danish only to a local directory, without pushing:
uv run processing_scripts/deduplicate_wikipedia_dataset.py -l da --output-dir ./local_dedup

# Deduplicate every language and push the result back to the Hub:
uv run processing_scripts/deduplicate_wikipedia_dataset.py --push
```

Some options worth knowing about (run `--help` for the full list):
* `-l`/`--language` - restrict processing to specific language(s) (repeatable); defaults to every config found in `--hf-dataset-repo-id`.
* `-n`/`--max-validation-documents` - the validation split cap to enforce (default 20); should match the value the data was originally built with.
* `-e`/`--max-output-file-size` - target maximum size in GB per output Parquet shard, pre-compression (default 1.0; the actual file size will be a lot smaller due to compression); larger splits are written as multiple shard files instead of one.

## Dataset statistics

[processing_scripts/dataset_statistics.py](processing_scripts/dataset_statistics.py) reports per-language, per-split statistics for a dataset already built and uploaded by `build_usas_wikipedia_dataset.py` (e.g. `ucrelnlp/Multilingual-USAS-Labelled-Silver-Wikipedia`) — like `deduplicate_wikipedia_dataset.py` above, it only reads the already-processed `train`/`validation` Parquet output, it does not re-run any of the filtering/tagging pipeline. For each language's `train` and `validation` split, plus a `"Total"` language aggregating every language together (again broken down into `train`, `validation`, and the overall total), it reports:

* Number of articles.
* Average article size in tokens, and average number of sentences per article.
* Number of tokens.
* Number of labelled tokens (tokens with at least one USAS tag) and Multi Tag Membership (%) — the percentage of labelled tokens that are "multi tag membership" tokens, i.e. tokens with more than one USAS tag, e.g. `tags[0][0]` is `["A3", "M6"]`; unlabelled tokens are excluded from this percentage.
* Number of unique USAS tags.
* Number of Multi-Word Expressions (MWEs).

It reads `HF_TOKEN` from the environment the same way as [HuggingFace Authentication](#huggingface-authentication) above (via `.env`/`python-dotenv`), needed if `--hf-dataset-repo-id` is private.

``` bash
# Print a table for every language in the default dataset:
uv run processing_scripts/dataset_statistics.py

# Report statistics for a single language and also export to CSV and LaTeX:
uv run processing_scripts/dataset_statistics.py -l da --output-csv ./stats.csv --output-latex ./stats.tex
```

Some options worth knowing about (run `--help` for the full list):
* `-l`/`--language` - restrict to specific language(s) (repeatable); defaults to every config found in `--hf-dataset-repo-id`.
* `--output-csv` - also write the table to a CSV file, with raw unformatted numeric values (unlike the console table, which adds `,` thousands separators).
* `--output-latex` - also write the table as a LaTeX `tabular` environment (`booktabs`-style rules), with the same human-readable, escaped headers shown in the console table.

## Clearing stale shards from a Hub dataset repository

`HuggingFaceDatasetWriter` (used by `build_usas_wikipedia_dataset.py` when uploading directly to the Hub) only ever adds/overwrites the specific Parquet shard files it writes — it never deletes pre-existing files in the repo. If a previous run for a language wrote more shards than a later re-run produces (e.g. an earlier, larger run left `data/da/train/003.parquet` behind), those extra shards are silently left in the repo and included in the dataset by anyone loading it.

[processing_scripts/clear_hub_dataset_shards.py](processing_scripts/clear_hub_dataset_shards.py) clears a language's (or the whole repository's) existing shards out first, so a re-run starts from a clean slate. It defaults to a **dry run** that only lists what would be deleted — pass `--delete` to actually remove the files, which happens as a single commit (`create_commit` with a `CommitOperationDelete` per file). Since Hub dataset repositories are Git-backed, deleted files remain recoverable from the repo's commit history (e.g. via `revision=<commit-sha>` when loading, or `HfApi.list_repo_commits`) as long as that commit stays reachable — i.e. until the branch is force-pushed/rewritten or the commit is otherwise garbage-collected.

``` bash
# Report what would be deleted for Danish, without deleting anything:
uv run processing_scripts/clear_hub_dataset_shards.py -l da

# Actually delete Danish's existing shards, with a confirmation prompt:
uv run processing_scripts/clear_hub_dataset_shards.py -l da --delete

# Delete every language's shards without a confirmation prompt:
uv run processing_scripts/clear_hub_dataset_shards.py --delete --yes
```

Some options worth knowing about (run `--help` for the full list):
* `-l`/`--language` - restrict deletion to specific language(s) (repeatable), matching `<path-in-repo>/<language>/`; defaults to clearing every file under `--path-in-repo`.
* `--path-in-repo` - repo-relative folder prefix shards are written under (default `data`, matching the layout described in [Uploading multiple languages to the same Hub repository](#uploading-multiple-languages-to-the-same-hub-repository) above).
* `-y`/`--yes` - skip the confirmation prompt before deleting; only used with `--delete`.

## Commands used to create the original `ucrelnlp/Multilingual-USAS-Labelled-Silver-Wikipedia` dataset

These commands were used to create the original [ucrelnlp/Multilingual-USAS-Labelled-Silver-Wikipedia](https://huggingface.co/datasets/ucrelnlp/Multilingual-USAS-Labelled-Silver-Wikipedia) datasets that was used in the journal paper;

``` bash
python processing_scripts/run_all_training_languages.py ./log_data --executor slurm --slurm-partition cpu-48h --slurm-time 30:00:00 --hf-dataset-repo-id ucrelnlp/Multilingual-USAS-Labelled-Silver-Wikipedia --slurm-mem-per-cpu-gb 4 --slurm-venv-path /mnt/nfs/homes/mooreap1/wikipedia-USAS-processing/venv/bin/python --slurm-cpus-per-task 2 --max-workers-per-language 100 --max-tasks-per-language 30 --min-tasks-per-language 1 --languages-file ./usas_wikipedia_processing.yaml --slurm-sbatch-args "{\"nice\": 100}" --max-number-of-parallel-tasks 100 --shard-tasks-multiplier 3 --randomize-start-duration 65 --overwrite --min-hash-threshold 0.85
uv run processing_scripts/deduplicate_wikipedia_dataset.py -e 0.4 --push
```

The first command used `python` rather than `uv` as we ran it on our SLURM cluster, in essence most of the time as we had a hard limit on the number of tasks that a user could submit to SLURM in one go (inclduing tasks that are scheduled but not running), we ended up running this command multiple times but processing different languages using `--languages-file ./usas_wikipedia_processing.yaml` file to state which languages ran via setting `training` to `False` for languages that we did not want to process data for. This command used 2 CPUs with 8GB of RAM in total per task which is more than enough for this processing setup.

<details>

<summary> Output after running the deduplicate command </summary>

``` bash
┏━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Language ┃ Split      ┃ Articles ┃ Avg. Tokens / Article ┃ Avg. Sentences / Article ┃ Tokens      ┃ Labelled Tokens ┃ Multi Tag Membership (%) ┃ Unique Tags ┃ MWEs       ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ da       │ train      │ 177      │ 7,739.73              │ 371.74                   │ 1,369,932   │ 871,407         │ 7.48                     │ 210         │ 35,083     │
│ da       │ validation │ 10       │ 4,621.00              │ 232.20                   │ 46,210      │ 28,562          │ 6.70                     │ 201         │ 1,084      │
│ nl       │ train      │ 368      │ 7,287.37              │ 417.04                   │ 2,681,753   │ 1,513,916       │ 3.81                     │ 204         │ 0          │
│ nl       │ validation │ 10       │ 7,743.40              │ 468.70                   │ 77,434      │ 43,976          │ 3.96                     │ 195         │ 0          │
│ fi       │ train      │ 855      │ 3,964.76              │ 255.59                   │ 3,389,867   │ 1,992,394       │ 11.64                    │ 207         │ 0          │
│ fi       │ validation │ 10       │ 3,843.10              │ 254.10                   │ 38,431      │ 22,524          │ 11.81                    │ 200         │ 0          │
│ it       │ train      │ 1,145    │ 8,342.54              │ 289.94                   │ 9,552,206   │ 6,641,591       │ 4.48                     │ 214         │ 94,621     │
│ it       │ validation │ 16       │ 5,469.00              │ 173.94                   │ 87,504      │ 63,275          │ 4.84                     │ 208         │ 889        │
│ pt       │ train      │ 3,456    │ 5,156.83              │ 220.26                   │ 17,821,992  │ 11,546,969      │ 6.44                     │ 210         │ 132,634    │
│ pt       │ validation │ 12       │ 5,304.92              │ 201.75                   │ 63,659      │ 41,313          │ 6.67                     │ 202         │ 495        │
│ es       │ train      │ 4,573    │ 6,587.80              │ 200.96                   │ 30,126,019  │ 20,557,991      │ 0.61                     │ 214         │ 66,928     │
│ es       │ validation │ 8        │ 2,899.50              │ 95.75                    │ 23,196      │ 16,397          │ 0.61                     │ 203         │ 37         │
│ zh       │ train      │ 2,787    │ 5,539.20              │ 240.26                   │ 15,437,755  │ 7,058,881       │ 6.02                     │ 212         │ 47,474     │
│ zh       │ validation │ 20       │ 8,739.45              │ 377.55                   │ 174,789     │ 72,346          │ 5.84                     │ 207         │ 442        │
│ en       │ train      │ 49,195   │ 3,714.85              │ 145.04                   │ 182,752,078 │ 145,924,630     │ 4.96                     │ 215         │ 13,872,660 │
│ en       │ validation │ 20       │ 4,954.95              │ 176.30                   │ 99,099      │ 73,536          │ 4.86                     │ 208         │ 7,858      │
│ Total    │ train      │ 62,556   │ 4,206.34              │ 163.93                   │ 263,131,602 │ 196,107,779     │ 4.68                     │ 220         │ 14,249,400 │
│ Total    │ validation │ 106      │ 5,757.75              │ 250.92                   │ 610,322     │ 361,929         │ 5.53                     │ 220         │ 10,805     │
│ Total    │ total      │ 62,662   │ 4,208.96              │ 164.08                   │ 263,741,924 │ 196,469,708     │ 4.68                     │ 220         │ 14,260,205 │
└──────────┴────────────┴──────────┴───────────────────────┴──────────────────────────┴─────────────┴─────────────────┴──────────────────────────┴─────────────┴────────────┘
```

</details>

## Benchmarking spaCy model speed

[processing_scripts/benchmark_da_spacy_model_speed.py](processing_scripts/benchmark_da_spacy_model_speed.py) compares the processing speed of the two Danish spaCy models (`da_core_news_lg` and `da_core_news_trf`) across a sweep of sentence counts, at a fixed sentence length of 21 tokens — both figures derived from the observed corpus statistics (an average of 362 sentences and 7556 tokens per document, i.e. ~21 tokens/sentence). It caps the tokens processed for any single benchmarked point at 100,000 by default.

Results can be shown as a graph, a console table, or both, and the table can additionally be exported as a LaTeX `tabular`:

``` bash
# Save a comparison graph (default: ./da_spacy_model_speed.png)
uv run processing_scripts/benchmark_da_spacy_model_speed.py

# Print a table to the console instead
uv run processing_scripts/benchmark_da_spacy_model_speed.py --format table

# Also export the table as LaTeX, independent of --format
uv run processing_scripts/benchmark_da_spacy_model_speed.py --latex-output ./results.tex
```

Run `--help` for the full list of options, including `--token-length`, `--max-tokens`, and `--num-points`.

NOTE: the arguments used to produce the table and plot in the paper;

``` bash
uv run processing_scripts/benchmark_da_spacy_model_speed.py --token-length 21 --max-tokens 100000 --num-points 8 --format plot --output data/plots/da_spacy_model_speed.png --latex-output data/tables/da_spacy_model_speed.tex
```

## License

The code is licensed under [Apache License Version 2.0](./LICENSE).

## Claude settings

For those that use [Anthropic's Claude](https://www.anthropic.com/) we have shared some suggested settings, see [./.claude folder](./.claude) that are enforced within this project but can be easily adjusted or removed if you prefer to use your own settings or the default settings of Claude. The project level settings for Claude, can be found at [./.claude/settings.json](./.claude/settings.json) are auto generated by running the following script;

``` bash
cd .claude/hooks && uv run generate_settings.py > ../settings.json
```

This script creates a settings file with;
* Numerous Deny permissions that have come from the list of files, stated in [./.claude/hooks/sensitive_patterns.py](./.claude/hooks/sensitive_patterns.py), that you do not want Claude to write/edit/read.
* A pre-hook, [./.claude/hooks/block_sensitive_files.py](./.claude/hooks/block_sensitive_files.py), that catches any write/edit/read to the list of files that the Deny permissions might miss, e.g. a call to Python using Bash.

**To note** this pre-hook and Deny permissions would not stop Claude from write/edit/read if Claude requests the file through an unusual regex pattern like `e*v` to get the `.env` file, but this is a best effort try to reduce Claude's access to these more sensitive files. Generally speaking if you are using API keys reduce the scope as much as possible and limit the time and resource access while developing.