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

Each Wikipedia articles goes through the following pipeline;
* The Wikipedia article ID and title match an article within the **GA** or **FA** articles.
* The Wikipedia article is not an article that is in the test data. (manual list of Wikipedia article URLs)
* Remove Wikipedia family tree tables, mathematical equations, and tables from the article text.
* Remove any Markdown formatting like headers using [mistune Python package](https://github.com/lepture/mistune) from the article text.
* Remove articles that contain less than 50 tokens based off a language specific tokenizer.
* Apply exact and then MinHash de-duplication.
* Sentence split using language specific sentence splitters
* USAS semantic and when the tagger supports it MWE identification using [PyMUSAS Rule Based languages specific taggers](https://ucrel.github.io/pymusas/#rule-based) per sentence.
  * Each sentence has all leading and starting whitespace removed, any tokens identified as whitespace are removed as tokens, and all `PUNCT` tags are mapped onto `Z9` tags.

The processed data will contain the following fields:
* `text` - the processed article text.
* `id` - dawiki/1171348
* `page_id` - Page ID `1171348`
* `title` - Article title.
* `url` - https://da.wikipedia.org/wiki/El_Salvador_ved_sommer-OL_2024
* `start_end_sentence_character_indexes` - list of start and end character offsets for each sentence, e.g. `[[0, 10], [11, 15]]` the first sentence is between `text[0:10]`.
* `tokens` - list of a list of tokens whereby the inner list represents the tokens for a given sentence, e.g. `tokens[0]` would contain all of the tokens in the first sentence.
* `tags` - list of a list of a list of USAS tags that were predicted by the PyMUSAS Rule Based languages specific tagger. The inner list represents the most likely USAS tags for the given token, e.g. `tags[0][0]` will contain a list of most likely USAS tags for the first token in the first sentence, in most cases it will only contain one USAS tag. When it contains more than one USAS tag this represents a token in which the meaning is a combination of the given predicted USAS tags. Some tokens will contain no USAS tags as the Rule Based tagger cannot make prediction for all tokens.
* `other_tags` - list of a list of a list of USAS tags with the same shape as `tags`, but containing every valid USAS tag for a token that was **not** its most likely tag(s), e.g. `other_tags[0][0]` will contain any other valid USAS tags for the first token in the first sentence. Most tokens will contain no other USAS tags, in which case the inner list is empty.
* `mwes` - list of a list of MWE labels that were predicted by the PyMUSAS Rule Based languages specific tagger, these always relate to the most likely USAS tags. The MWE labels denote at the sentence level which tokens are MWEs, e.g. `mwes[0][0]` represent all of the MWE labels for the first token in the first sentence, if it contains `1` and `mwes[0][1]` also contains `1` then the first token and second token in the first sentence are a MWE. If more than one label occurs then MWEs are overlapping which should not be the case with PyMUSAS taggers. MWEs can be dis-continuous. The index of MWE labels always start at 1 and reset per sentence, e.g. the first sentence can contain a MWE label of `1` and so can the second sentence, but they will be different MWEs as MWEs are constrained to occur within a single sentence; they cannot span sentence boundaries.

## Example script

``` bash
uv run test_local.py da ./data/wikipedia_pages ./local_da/ ./log_data/
```


### Models to install (models will also be installed before processing within the processing pipeline)

> [!NOTE]
> Only download models for languages that you are tagging text for.

The [models_install.py script](./wikipedia_processing/models_install.py) allows you to install all of the required models and lexicons for all languages, these are tokenizer, sentence splitters, and semantic tagging models, it also allows you to install language specific models, and it will detail what models will be installed before using the `--describe` flag:

``` bash
uv run wikipedia_processing/models_install.py --help
                                                                                                                                                                                          
 Usage: models_install.py [OPTIONS]                                                                                                                                                       
                                                                                                                                                                                          
 Install the language specific models. You can either select the languages you want to install or use the --all flag to install all language specific models.                             
                                                                                                                                                                                          
 If you want to describe the models that will be installed use the --describe flag.                                                                                                       
                                                                                                                                                                                          
 Example:                                                                                                                                                                                 
                                                                                                                                                                                          
 To install all language specific models run:                                                                                                                                             
 python models_install.py --all                                                                                                                                                           
                                                                                                                                                                                          
 To install only the English and Dutch language specific models run:                                                                                                                      
 python models_install.py -l English -l Dutch                                                                                                                                             
                                                                                                                                                                                          
 To describe the models that will be installed run:                                                                                                                                       
 python models_install.py --describe                                                                                                                                                      
                                                                                                                                                                                          
 To describe English specific models:                                                                                                                                                     
 python models_install.py -l English --describe                                                                                                                                           
                                                                                                                                                                                          
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --languages  -l      [Chinese|Danish|Dutch|English|Finnish|Italian|Portuguese|Spanish]  Install the language specific models for the given languages.                                  │
│ --all        -a                                                                         Install all language specific models.                                                          │
│ --describe   -d                                                                         Describe the models that will be installed and exit.                                           │
│ --help                                                                                  Show this message and exit.                                                                    │
╰────
```

The following will download all of the models for all languages:

```bash
uv run wikipedia_processing/models_install.py --all
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