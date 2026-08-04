---
language:
- en
- nl
- pt
- es
- da
- it
- fi
- zh
license:
- cc-by-sa-4.0
- gfdl
multilinguality: multilingual
size_categories:
- 10k<n<100K
pretty_name: Multilingual USAS Silver Labelled Wikipedia Articles
configs:
- config_name: en
  data_files:
  - split: train
    path: data/en/train/*.parquet
  - split: validation
    path: data/en/validation/*.parquet
- config_name: nl
  data_files:
  - split: train
    path: data/nl/train/*.parquet
  - split: validation
    path: data/nl/validation/*.parquet
- config_name: pt
  data_files:
  - split: train
    path: data/pt/train/*.parquet
  - split: validation
    path: data/pt/validation/*.parquet
- config_name: es
  data_files:
  - split: train
    path: data/es/train/*.parquet
  - split: validation
    path: data/es/validation/*.parquet
- config_name: da
  data_files:
  - split: train
    path: data/da/train/*.parquet
  - split: validation
    path: data/da/validation/*.parquet
- config_name: it
  data_files:
  - split: train
    path: data/it/train/*.parquet
  - split: validation
    path: data/it/validation/*.parquet
- config_name: fi
  data_files:
  - split: train
    path: data/fi/train/*.parquet
  - split: validation
    path: data/fi/validation/*.parquet
- config_name: zh
  data_files:
  - split: train
    path: data/zh/train/*.parquet
  - split: validation
    path: data/zh/validation/*.parquet
viewer: true
---
# Multilingual USAS Silver Labelled Wikipedia Articles

Silver-labelled Wikipedia article text for training USAS semantic taggers and Multi-Word
Expression (MWE) identifiers, covering 8 Wikipedia language sites. The source text comes from the
HuggingFace [`HuggingFaceFW/finewiki`](https://huggingface.co/datasets/HuggingFaceFW/finewiki)
dataset, restricted to articles rated Good (GA) or Featured (FA) — using the article ID list from
[ucrelnlp/wikipedia-ga-fa-ids](https://huggingface.co/datasets/ucrelnlp/wikipedia-ga-fa-ids) — and
then sentence split and automatically tagged with USAS semantic tags and MWEs using
[PyMUSAS](https://ucrel.github.io/pymusas/#rule-based) rule-based taggers. For more information on
how the dataset was generated, including the full filtering/processing pipeline, see
[https://github.com/UCREL/wikipedia-USAS-processing](https://github.com/UCREL/wikipedia-USAS-processing).


- **Curated by:** [University Centre for Computer Corpus Research on Language (UCREL) group](https://ucrel.lancs.ac.uk/) at [Lancaster University](https://www.lancaster.ac.uk/)
- **Multi-lingual**
- **Repository:** [https://github.com/UCREL/wikipedia-USAS-processing](https://github.com/UCREL/wikipedia-USAS-processing)


## Uses

It can be used to train USAS semantic taggers and MWE identifiers.

## Filtering and Processing

Each Wikipedia article goes through the following pipeline before being included in this dataset:

* The article ID and title must match an article rated as Good (GA) or Featured (FA) (taken from
  [ucrelnlp/wikipedia-ga-fa-ids](https://huggingface.co/datasets/ucrelnlp/wikipedia-ga-fa-ids)).
* Articles that are part of a manually curated test set (by URL) are excluded.
* Wikipedia family-tree tables, mathematical equations, and other tables are removed from the
  article text.
* Markdown formatting (e.g. headers like `#` but not the header text) is stripped from the article text.
* Articles with fewer than 50 tokens, based on a language-specific tokenizer, are removed.
* Exact and then MinHash de-duplication is applied.
* Remaining articles are sentence split using language-specific spaCy sentence splitters.
* Each sentence is tagged with USAS semantic tags and, where the tagger supports it, MWEs, using
  [PyMUSAS Rule Based language-specific taggers](https://ucrel.github.io/pymusas/#rule-based).

## Dataset Structure

Each row is a single article, unique per `id`/`page_id` within a language config. The data is
split per language into `train` and `validation` subsets (see below).

* `text` - the processed article text.
* `id` - unique identifier for the article, e.g. `dawiki/1171348`.
* `page_id` - the Wikipedia page ID, e.g. `1171348`.
* `title` - the article title.
* `url` - the article URL, e.g. `https://da.wikipedia.org/wiki/El_Salvador_ved_sommer-OL_2024`.
* `version` (int|string) - revision/version identifier of the page, as provided by
  [`HuggingFaceFW/finewiki`](https://huggingface.co/datasets/HuggingFaceFW/finewiki), e.g.
  `1167219203`.
* `start_end_sentence_character_indexes` - list of `[start, end]` character offsets for each
  sentence, e.g. `[[0, 10], [11, 15]]` — the first sentence is `text[0:10]`.
* `tokens` - list of a list of tokens, where the inner list is the tokens for a given sentence,
  e.g. `tokens[0]` contains all tokens in the first sentence.
* `tags` - list of a list of a list of USAS tags predicted by the PyMUSAS rule-based tagger. The
  innermost list holds the most likely USAS tag(s) for a given token, e.g. `tags[0][0]` is the
  most likely USAS tag(s) for the first token of the first sentence — usually a single tag, but
  more than one when the token's meaning is a combination of the predicted tags. Some tokens have
  no USAS tags, as the rule-based tagger cannot make a prediction for every token.
* `other_tags` - list of a list of a list of USAS tags, same shape as `tags`, containing every
  other valid USAS tag for a token that was **not** among its most likely tag(s), e.g.
  `other_tags[0][0]`. Most tokens have no other tags, in which case the inner list is empty.
* `mwes` - list of a list of MWE labels predicted by the PyMUSAS rule-based tagger, relating to
  the most likely USAS tags. Labels denote, at the sentence level, which tokens form an MWE, e.g.
  if `mwes[0][0]` and `mwes[0][1]` both contain `1`, the first and second tokens of the first
  sentence are part of the same MWE. MWEs can be discontinuous but should not overlap. MWE label
  indexes start at `1` and reset per sentence — MWEs cannot span sentence boundaries.

The data is stored as [zstd](https://github.com/facebook/zstd)-compressed
[Parquet](https://parquet.apache.org/) files.

Example of a record (shown as JSON for readability):

``` JSON
{
  "text": "El Salvador deltog i sommer-OL 2024 i Paris.",
  "id": "dawiki/1171348",
  "page_id": 1171348,
  "title": "El Salvador ved sommer-OL 2024",
  "url": "https://da.wikipedia.org/wiki/El_Salvador_ved_sommer-OL_2024",
  "version": 1167219203,
  "start_end_sentence_character_indexes": [[0, 45]],
  "tokens": [["El", "Salvador", "deltog", "i", "sommer-OL", "2024", "i", "Paris", "."]],
  "tags": [[["Z2"], ["Z2"], ["M6"], ["Z5"], ["K5.1"], ["T1.3"], ["Z5"], ["Z2"], ["Z9"]]],
  "other_tags": [[[], [], [], [], [], [], [], [], []]],
  "mwes": [[1, 1, 0, 0, 0, 0, 0, 0, 0]]
}
```

### Train/validation split

Each language's documents are split into `train` and `validation` subsets, written to separate
`train`/`validation` subfolders. The validation split is capped at
whichever is reached first: a percentage of the language's documents, or a fixed maximum number of
documents — this keeps under-resourced languages (some have as few as ~200 articles) at a
sensible percentage-based split, while bounding well-resourced languages' validation set to a sane
absolute size. The split assignment is deterministic (hashed from each document's page ID), so
re-running the pipeline reproduces the same split.

## Dataset Statistics

The table below shows per language the number of entries/articles that are either Good or Featured (Total), Good, or Featured:


## License

This dataset contains text from Wikipedia, licensed under [Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.en) (CC BY-SA 4.0) and also available under [GFDL](https://www.gnu.org/licenses/fdl-1.3.html). See Wikipedia’s licensing and Terms of Use: [https://dumps.wikimedia.org/legal.html](https://dumps.wikimedia.org/legal.html)

We release this data under the same license;  [Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.en) (CC BY-SA 4.0) and also available under [GFDL](https://www.gnu.org/licenses/fdl-1.3.html).


## Dataset Card Authors

* UCREL (ucrel@lancaster.ac.uk)
* Andrew Moore / apmoore1 (a.p.moore@lancaster.ac.uk / andrew.p.moore94@gmail.com)
* Paul Rayson (p.rayson@lancaster.ac.uk)

## Dataset Card Contact

* UCREL (ucrel@lancaster.ac.uk)
* Andrew Moore / apmoore1 (a.p.moore@lancaster.ac.uk / andrew.p.moore94@gmail.com)
* Paul Rayson (p.rayson@lancaster.ac.uk)
