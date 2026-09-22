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
  - config_name: da
    data_files:
      - split: train
        path: "data/da/train/*.parquet"
      - split: validation
        path: "data/da/validation/*.parquet"
  - config_name: nl
    data_files:
      - split: train
        path: "data/nl/train/*.parquet"
      - split: validation
        path: "data/nl/validation/*.parquet"
  - config_name: fi
    data_files:
      - split: train
        path: "data/fi/train/*.parquet"
      - split: validation
        path: "data/fi/validation/*.parquet"
  - config_name: it
    data_files:
      - split: train
        path: "data/it/train/*.parquet"
      - split: validation
        path: "data/it/validation/*.parquet"
  - config_name: pt
    data_files:
      - split: train
        path: "data/pt/train/*.parquet"
      - split: validation
        path: "data/pt/validation/*.parquet"
  - config_name: es
    data_files:
      - split: train
        path: "data/es/train/*.parquet"
      - split: validation
        path: "data/es/validation/*.parquet"
  - config_name: zh
    data_files:
      - split: train
        path: "data/zh/train/*.parquet"
      - split: validation
        path: "data/zh/validation/*.parquet"
  - config_name: en
    data_files:
      - split: train
        path: "data/en/train/*.parquet"
      - split: validation
        path: "data/en/validation/*.parquet"
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
* `id` -  unique identifier for the article, e.g. `enwiki/23146210`
* `page_id` - the Wikipedia page ID `23146210`
* `title` - Article title.
* `url` - the article URL, e.g. `https://en.wikipedia.org/wiki/Bill_Gutteron`
* `version`- (integer) revision/version identifier of the page (comes from [HuggingFaceFW finewiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki)) `1230438345`
* `start_end_sentence_character_indexes` - list of `[start, end]` character offsets for each sentence, e.g. `[[0, 10], [11, 15]]` the first sentence is between `text[0:10]`.
* `tokens` - list of a list of tokens whereby the inner list represents the tokens for a given sentence, e.g. `tokens[0]` would contain all of the tokens in the first sentence.
* `tags` - list of a list of a list of USAS tags that were predicted by the PyMUSAS Rule Based languages specific tagger. The inner list represents the most likely USAS tags for the given token, e.g. `tags[0][0]` will contain a list of most likely USAS tags for the first token in the first sentence, in most cases it will only contain one USAS tag. When it contains more than one USAS tag this represents a token in which the meaning is a combination of the given predicted USAS tags. Some tokens will contain no USAS tags as the Rule Based tagger cannot make prediction for all tokens. Tags within each group are de-duplicated.
* `other_tags` - list of a list of a list of a list of USAS tags, one level deeper than `tags`, containing every other valid USAS tag group for a token that was **not** its most likely tag group, e.g. `other_tags[0][0]` will contain a list of the other valid USAS tag groups for the first token in the first sentence, and `other_tags[0][0][0]` the tags within the first of those groups. Keeping each group as its own inner list (rather than merging them together) preserves which tags PyMUSAS considered part of the same combined meaning. Most tokens will contain no other USAS tag groups, in which case the outer list is empty. As with `tags`, tags within each group are de-duplicated. They are ordered by the most likely USAS tag group.
* `mwes` - list of a list of MWE labels that were predicted by the PyMUSAS Rule Based languages specific tagger, these always relate to the most likely USAS tags. The MWE labels denote at the sentence level which tokens are MWEs, e.g. `mwes[0][0]` represent all of the MWE labels for the first token in the first sentence, if it contains `1` and `mwes[0][1]` also contains `1` then the first token and second token in the first sentence are a MWE. If more than one label occurs then MWEs are overlapping which should not be the case with PyMUSAS taggers. MWEs can be dis-continuous. The index of MWE labels always start at 1 and reset per sentence, e.g. the first sentence can contain a MWE label of `1` and so can the second sentence, but they will be different MWEs as MWEs are constrained to occur within a single sentence; they cannot span sentence boundaries.

The data is stored as [zstd](https://github.com/facebook/zstd)-compressed [Parquet](https://parquet.apache.org/) files.

Example of a record (shown as JSON for readability), taken directly from the English config of this dataset (`enwiki/23146210`, the full [Bill Gutteron](https://en.wikipedia.org/wiki/Bill_Gutteron) article). **Note** that `text` reflects the article revision captured in the underlying FineWiki snapshot (June 2024 version of the article) — the live Wikipedia article has since been substantially expanded, so it is now much longer than the short, complete example shown here:

``` JSON
{
  "text": "Bill Gutteron\n\nWilliam Alexander Gutteron (November 26, 1899 – May 30, 1987) was a professional football player in the National Football League (NFL). He made his NFL debut in 1926 with the Los Angeles Buccaneers. He played only one season in the league. A quarterback, Gutteron played college football for the Nevada Wolf Pack.\n",
  "id": "enwiki/23146210",
  "page_id": 23146210,
  "title": "Bill Gutteron",
  "url": "https://en.wikipedia.org/wiki/Bill_Gutteron",
  "version": 1230438345,
  "start_end_sentence_character_indexes": [[0, 150], [151, 213], [214, 254], [255, 329]],
  "tokens": [["Bill", "Gutteron", "\n\n", "William", "Alexander", "Gutteron", "(", "November", "26", ",", "1899", "–", "May", "30", ",", "1987", ")", "was", "a", "professional", "football", "player", "in", "the", "National", "Football", "League", "(", "NFL", ")", "."], ["He", "made", "his", "NFL", "debut", "in", "1926", "with", "the", "Los", "Angeles", "Buccaneers", "."], ["He", "played", "only", "one", "season", "in", "the", "league", "."], ["A", "quarterback", ",", "Gutteron", "played", "college", "football", "for", "the", "Nevada", "Wolf", "Pack", "."]],
  "tags": [[["Z1"], ["Z1"], [], ["Z1"], ["Z1"], ["Z1"], ["Z9"], ["Z2"], ["Z2"], ["Z9"], ["N1"], ["Z9"], ["Z2"], ["Z2"], ["Z9"], ["N1"], ["Z9"], ["A3"], ["Z5"], ["I3.2"], ["K5.1", "S2"], ["K5.1", "S2"], ["Z5"], ["Z5"], ["Z1"], ["Z1"], ["Z1"], ["Z9"], ["G3"], ["Z9"], ["Z9"]], [["Z8"], ["A5.4"], ["A5.4"], ["T1.3"], ["T1.3"], ["Z5"], ["N1"], ["Z5"], ["Z5"], ["Z1"], ["Z1"], ["Z1"], ["Z9"]], [["Z8"], ["K1"], ["A14"], ["N1"], ["T1.3"], ["Z5"], ["Z5"], ["S5"], ["Z9"]], [["Z5"], [], ["Z9"], [], ["K1"], ["P1", "H1"], ["K5.2"], ["Z5"], ["Z5"], ["Z1"], ["Z1"], ["Z1"], ["Z9"]]],
  "other_tags": [[[["Z3"]], [["Z3"]], [], [["Z3"]], [["Z3"]], [["Z3"]], [], [["Z1"], ["T1.3"]], [["Z1"], ["T1.3"]], [], [], [], [["Z1"], ["T1.3"]], [["Z1"], ["T1.3"]], [], [], [], [["Z5"]], [], [["I3.1"], ["X9.1"], ["A5.1"]], [], [], [], [], [["Z3"]], [["Z3"]], [["Z3"]], [], [], [], []], [[], [], [], [], [], [], [], [], [], [["Z3"]], [["Z3"]], [["Z3"]], []], [[], [["K5.1"], ["K5.2"], ["K2"], ["K3"], ["A1.1.1"], ["K6"]], [], [["T3"], ["T1.2"]], [], [], [], [["K5.1", "S5"], ["S7.3"], ["N3.3"]], []], [[], [], [], [], [["K5.1"], ["K5.2"], ["K2"], ["K3"], ["A1.1.1"], ["K6"]], [], [], [], [], [["Z3"]], [["Z3"]], [["Z3"]], []]],
  "mwes": [[[1], [1], [], [2], [2], [2], [], [3], [3], [], [], [], [4], [4], [], [], [], [], [], [], [5], [5], [], [], [6], [6], [6], [], [], [], []], [[], [1], [1], [2], [2], [], [], [], [], [3], [3], [3], []], [[], [], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], [], [], [1], [1], [1], []]]
}
```

The first sentence contains six MWEs: `["Bill", "Gutteron"]` (label `1`), `["William",
"Alexander", "Gutteron"]` (label `2`), `["November", "26"]` (label `3`), `["May", "30"]` (label
`4`), `["football", "player"]` (label `5`), and `["National", "Football", "League"]` (label `6`).
Labels reset per sentence, so the second sentence starts again from `1` with its own three MWEs:
`["made", "his"]` (label `1`), `["NFL", "debut"]` (label `2`), and `["Los", "Angeles",
"Buccaneers"]` (label `3`).

### Train/validation split

Each language's documents are split into `train` and `validation` subsets, written to separate
`train`/`validation` subfolders. The validation split is capped at
whichever is reached first: a percentage of the language's documents (10%), or a fixed maximum number of
documents (20 documents) — this keeps under-resourced languages (some have as few as ~200 articles) at a
sensible percentage-based split, while bounding well-resourced languages' validation set to a sane
absolute size. The split assignment is deterministic (hashed from each document's page ID), so
re-running the pipeline reproduces the same split.

## Dataset Statistics

Statistics for the full dataset (every language config combined), generated using the
`dataset_statistics.py`, `token_count_distribution.py`, and `usas_tag_distribution.py` scripts from
the [processing GitHub repository](https://github.com/UCREL/wikipedia-USAS-processing). See the repositories README and the journal paper for more information about these tables.

### Overview

| Language | Articles | Sentences (M) | Tokens (M) | Labelled Tokens (M) | Labels per Labelled Token | Multi Tag Membership (%) | Unique Tags | MWEs (M) | MWE Tokens (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Chinese | 2,807 | 0.677 | 15.613 | 10.111 | 4.48 | 21.13 | 215 | 0.048 | 0.62 |
| Danish | 187 | 0.068 | 1.416 | 1.070 | 1.69 | 13.41 | 213 | 0.036 | 6.10 |
| Dutch | 378 | 0.158 | 2.759 | 1.853 | 2.37 | 10.90 | 211 | 0.000 | 0.00 |
| English | 49,218 | 7.138 | 182.840 | 170.129 | 1.81 | 11.10 | 217 | 13.793 | 17.57 |
| Finnish | 865 | 0.221 | 3.428 | 2.507 | 1.85 | 16.14 | 209 | 0.000 | 0.00 |
| Italian | 1,161 | 0.335 | 9.640 | 7.904 | 2.04 | 12.01 | 219 | 0.096 | 2.13 |
| Portuguese | 3,469 | 0.764 | 17.886 | 13.875 | 2.78 | 15.24 | 218 | 0.133 | 1.61 |
| Spanish | 4,581 | 0.920 | 30.150 | 24.164 | 1.89 | 1.03 | 219 | 0.067 | 0.47 |
| **Total** | **62,666** | **10.281** | **263.732** | **231.611** | **2.01** | **11.52** | **220** | **14.173** | **12.49** |

"Labelled Tokens" are tokens with at least one USAS tag; "Labels per Labelled Token" and "Multi Tag
Membership (%)" count labels from both the `tags` and `other_tags` columns, since both are positive
labels when training (see [Dataset Structure](#dataset-structure) above). "MWE Tokens (%)" is the
percentage of tokens that are part of at least one Multi-Word Expression.

<details>

<summary>Per-split breakdown (train / validation / total)</summary>

| Language | Split | Articles | Sentences (M) | Tokens (M) | Labelled Tokens (M) | Labels per Labelled Token | Multi Tag Membership (%) | Unique Tags | MWEs (M) | MWE Tokens (%) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Chinese | train | 2,787 | 0.670 | 15.463 | 10.010 | 4.48 | 21.12 | 215 | 0.047 | 0.62 |
| Chinese | validation | 20 | 0.007 | 0.149 | 0.100 | 4.53 | 22.01 | 214 | 0.000 | 0.64 |
| Chinese | total | 2,807 | 0.677 | 15.613 | 10.111 | 4.48 | 21.13 | 215 | 0.048 | 0.62 |
| Danish | train | 168 | 0.062 | 1.278 | 0.968 | 1.69 | 13.42 | 213 | 0.033 | 6.14 |
| Danish | validation | 19 | 0.006 | 0.138 | 0.102 | 1.68 | 13.37 | 211 | 0.003 | 5.74 |
| Danish | total | 187 | 0.068 | 1.416 | 1.070 | 1.69 | 13.41 | 213 | 0.036 | 6.10 |
| Dutch | train | 358 | 0.149 | 2.599 | 1.743 | 2.37 | 10.87 | 211 | 0.000 | 0.00 |
| Dutch | validation | 20 | 0.009 | 0.160 | 0.110 | 2.39 | 11.38 | 211 | 0.000 | 0.00 |
| Dutch | total | 378 | 0.158 | 2.759 | 1.853 | 2.37 | 10.90 | 211 | 0.000 | 0.00 |
| English | train | 49,198 | 7.134 | 182.734 | 170.030 | 1.81 | 11.10 | 217 | 13.785 | 17.57 |
| English | validation | 20 | 0.004 | 0.106 | 0.099 | 1.76 | 11.05 | 212 | 0.008 | 17.65 |
| English | total | 49,218 | 7.138 | 182.840 | 170.129 | 1.81 | 11.10 | 217 | 13.793 | 17.57 |
| Finnish | train | 845 | 0.216 | 3.356 | 2.454 | 1.85 | 16.21 | 209 | 0.000 | 0.00 |
| Finnish | validation | 20 | 0.005 | 0.072 | 0.053 | 1.82 | 12.95 | 207 | 0.000 | 0.00 |
| Finnish | total | 865 | 0.221 | 3.428 | 2.507 | 1.85 | 16.14 | 209 | 0.000 | 0.00 |
| Italian | train | 1,141 | 0.329 | 9.485 | 7.776 | 2.04 | 11.99 | 219 | 0.094 | 2.13 |
| Italian | validation | 20 | 0.005 | 0.154 | 0.127 | 2.11 | 12.83 | 216 | 0.002 | 2.25 |
| Italian | total | 1,161 | 0.335 | 9.640 | 7.904 | 2.04 | 12.01 | 219 | 0.096 | 2.13 |
| Portuguese | train | 3,449 | 0.760 | 17.784 | 13.795 | 2.78 | 15.24 | 218 | 0.132 | 1.61 |
| Portuguese | validation | 20 | 0.004 | 0.102 | 0.080 | 2.74 | 16.67 | 215 | 0.001 | 1.45 |
| Portuguese | total | 3,469 | 0.764 | 17.886 | 13.875 | 2.78 | 15.24 | 218 | 0.133 | 1.61 |
| Spanish | train | 4,561 | 0.917 | 30.044 | 24.078 | 1.89 | 1.03 | 219 | 0.067 | 0.47 |
| Spanish | validation | 20 | 0.003 | 0.106 | 0.086 | 1.90 | 1.13 | 219 | 0.000 | 0.52 |
| Spanish | total | 4,581 | 0.920 | 30.150 | 24.164 | 1.89 | 1.03 | 219 | 0.067 | 0.47 |
| **Total** | **train** | **62,507** | **10.237** | **262.745** | **230.853** | **2.00** | **11.51** | **220** | **14.158** | **12.52** |
| **Total** | **validation** | **159** | **0.044** | **0.987** | **0.758** | **2.39** | **14.21** | **220** | **0.014** | **3.35** |
| **Total** | **total** | **62,666** | **10.281** | **263.732** | **231.611** | **2.01** | **11.52** | **220** | **14.173** | **12.49** |

</details>

### Token Length Distribution

Quantiles of token counts per sentence and per article (`train` + `validation` combined), plus the
unweighted average across languages (`Macro Avg`).

**Tokens per sentence**

| Language | P25 | P50 | P75 | P90 | P95 | P99 | Max |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Chinese | 9.0 | 19.0 | 31.0 | 45.0 | 57.0 | 93.0 | 2,604.0 |
| Danish | 11.0 | 19.0 | 28.0 | 37.0 | 44.0 | 65.0 | 708.0 |
| Dutch | 10.0 | 16.0 | 24.0 | 31.0 | 37.0 | 51.4 | 355.0 |
| English | 15.0 | 23.0 | 32.0 | 43.0 | 51.0 | 76.0 | 49,287.0 |
| Finnish | 9.0 | 13.0 | 18.0 | 24.0 | 29.0 | 46.0 | 1,032.0 |
| Italian | 16.0 | 25.0 | 37.0 | 51.0 | 61.0 | 91.0 | 2,241.0 |
| Portuguese | 11.0 | 21.0 | 32.0 | 44.0 | 53.0 | 76.0 | 927.0 |
| Spanish | 19.0 | 28.0 | 40.0 | 55.0 | 68.0 | 111.0 | 5,629.0 |
| **Macro Avg** | **12.5** | **20.5** | **30.2** | **41.2** | **50.0** | **76.2** | **7,847.9** |

**Tokens per article**

| Language | P25 | P50 | P75 | P90 | P95 | P99 | Max |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Chinese | 2,805.0 | 4,249.0 | 6,642.0 | 10,834.2 | 14,255.2 | 22,480.4 | 50,943.0 |
| Danish | 4,754.5 | 6,948.0 | 9,420.5 | 13,227.4 | 14,707.0 | 18,841.6 | 23,163.0 |
| Dutch | 4,484.8 | 6,613.5 | 9,342.0 | 12,636.8 | 14,737.7 | 19,327.8 | 22,395.0 |
| English | 1,661.0 | 2,701.0 | 4,640.0 | 7,596.0 | 10,019.0 | 15,595.6 | 190,775.0 |
| Finnish | 2,174.0 | 3,244.0 | 4,694.0 | 6,475.4 | 7,872.4 | 14,011.9 | 134,271.0 |
| Italian | 4,365.0 | 7,181.0 | 11,134.0 | 15,575.0 | 17,940.0 | 23,001.0 | 45,295.0 |
| Portuguese | 1,856.0 | 3,517.0 | 7,009.0 | 11,653.6 | 14,543.8 | 19,877.7 | 60,246.0 |
| Spanish | 2,735.0 | 4,716.0 | 8,336.0 | 13,590.0 | 17,705.0 | 29,546.2 | 132,178.0 |
| **Macro Avg** | **3,104.4** | **4,896.2** | **7,652.2** | **11,448.5** | **13,972.5** | **20,335.3** | **82,408.2** |

### USAS Tag Distribution

Tags are counted from both the `tags` and `other_tags` columns (`train` + `validation` combined),
since both are positive labels when training.

**Major tag distribution (%)** — percentage share of the first character of each USAS tag (e.g. `A3`
and `A1` both count towards major tag `A`):

| Tag | Chinese | Danish | Dutch | English | Finnish | Italian | Portuguese | Spanish | Macro Avg |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Z | 13.2 | 33.9 | 27.8 | 38.8 | 26.0 | 28.1 | 24.3 | 39.9 | 29.0 |
| A | 17.5 | 16.0 | 17.9 | 14.7 | 18.9 | 14.1 | 15.8 | 15.2 | 16.2 |
| S | 12.0 | 7.7 | 8.3 | 7.5 | 11.0 | 11.0 | 10.5 | 8.5 | 9.6 |
| N | 9.0 | 7.7 | 7.8 | 7.0 | 8.9 | 9.3 | 8.1 | 7.1 | 8.1 |
| M | 6.1 | 5.7 | 7.1 | 3.6 | 6.2 | 5.5 | 5.9 | 3.1 | 5.4 |
| T | 4.2 | 6.0 | 4.9 | 5.9 | 4.4 | 3.9 | 5.4 | 5.3 | 5.0 |
| X | 6.3 | 4.2 | 4.3 | 3.7 | 4.3 | 6.3 | 3.7 | 3.9 | 4.6 |
| Q | 5.3 | 3.6 | 3.2 | 3.4 | 3.1 | 4.4 | 3.9 | 2.6 | 3.7 |
| O | 4.4 | 2.1 | 4.4 | 2.4 | 3.0 | 3.1 | 4.4 | 2.2 | 3.3 |
| G | 3.6 | 2.3 | 2.5 | 2.1 | 2.1 | 2.2 | 3.4 | 2.4 | 2.6 |
| K | 2.9 | 1.8 | 1.9 | 2.6 | 2.7 | 1.3 | 2.8 | 1.4 | 2.2 |
| I | 3.0 | 1.6 | 1.7 | 1.8 | 1.7 | 1.6 | 2.4 | 1.6 | 1.9 |
| B | 2.9 | 1.2 | 1.9 | 1.4 | 1.8 | 2.1 | 2.0 | 1.0 | 1.8 |
| E | 2.3 | 1.4 | 1.4 | 1.1 | 1.1 | 1.1 | 1.2 | 1.1 | 1.3 |
| H | 2.0 | 1.7 | 1.0 | 1.0 | 1.0 | 1.1 | 1.5 | 0.9 | 1.3 |
| F | 1.4 | 0.6 | 1.0 | 0.6 | 0.5 | 1.4 | 0.9 | 0.4 | 0.9 |
| W | 0.9 | 0.7 | 0.6 | 0.5 | 1.2 | 0.8 | 0.9 | 0.8 | 0.8 |
| L | 0.9 | 0.6 | 0.8 | 0.7 | 1.0 | 0.7 | 0.6 | 1.0 | 0.8 |
| P | 1.0 | 0.6 | 0.7 | 0.5 | 0.4 | 0.9 | 1.0 | 1.0 | 0.8 |
| C | 0.7 | 0.3 | 0.5 | 0.4 | 0.4 | 0.7 | 0.5 | 0.3 | 0.5 |
| Y | 0.6 | 0.5 | 0.2 | 0.3 | 0.3 | 0.3 | 0.6 | 0.3 | 0.4 |

**5 most frequent individual tags (%)**

The USAS tag categories are described and defined within the [Introduction to the USAS category system](https://ucrel.lancs.ac.uk/usas/usas_guide.pdf), of which this can to some degree explain the reason for these tags being in the top-5;
* `Z5` - Prepositions/adverbs/conjunctions etc.
* `Z9` - Punctuation
* `Z8` - Pronouns
* `N1` - Numbers
* `S2` - People

| Tag | Chinese | Danish | Dutch | English | Finnish | Italian | Portuguese | Spanish | Macro Avg |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Z5 | 4.5 | 15.8 | 17.9 | 15.9 | 7.7 | 15.7 | 14.1 | 25.8 | 14.7 |
| Z9 | 6.6 | 9.4 | 6.7 | 7.8 | 10.6 | 7.4 | 5.9 | 7.7 | 7.8 |
| Z8 | 0.6 | 2.9 | 1.6 | 1.9 | 2.8 | 2.9 | 2.8 | 4.0 | 2.4 |
| N1 | 2.3 | 2.5 | 1.8 | 2.4 | 2.6 | 2.4 | 2.5 | 3.0 | 2.4 |
| S2 | 2.6 | 1.7 | 1.5 | 1.6 | 3.1 | 2.1 | 2.5 | 1.8 | 2.1 |

**5 least frequent individual tags (%)** — shown in scientific notation, as these percentages are
usually too small for a fixed decimal place to show meaningfully:

| Tag | Chinese | Danish | Dutch | English | Finnish | Italian | Portuguese | Spanish | Macro Avg |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| X9 | 4.3×10⁻⁴ | 0 | 0 | 2.6×10⁻⁴ | 1.0×10⁻³ | 1.6×10⁻² | 0 | 3.0×10⁻² | 6.0×10⁻³ |
| O4 | 1.8×10⁻⁵ | 5.5×10⁻⁵ | 0 | 3.0×10⁻⁵ | 0 | 2.7×10⁻⁴ | 2.2×10⁻⁴ | 5.7×10⁻² | 7.2×10⁻³ |
| G2 | 0 | 0 | 0 | 1.1×10⁻⁵ | 0 | 6.2×10⁻⁴ | 1.1×10⁻⁴ | 8.8×10⁻² | 1.1×10⁻² |
| Q2 | 3.6×10⁻⁴ | 1.1×10⁻⁴ | 0 | 1.3×10⁻⁵ | 0 | 5.0×10⁻⁵ | 5.4×10⁻⁴ | 9.0×10⁻² | 1.1×10⁻² |
| A1.5 | 8.3×10⁻³ | 0 | 2.1×10⁻³ | 9.4×10⁻⁶ | 0 | 1.3×10⁻² | 1.4×10⁻³ | 7.1×10⁻² | 1.2×10⁻² |

**Tag frequency spread** — five-number summary (raw count, with percentage in brackets) of how
spread out individual tags' frequencies are within each language:

| Statistic | Chinese | Danish | Dutch | English | Finnish | Italian | Portuguese | Spanish | Macro Avg |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Min | 8 (0.0%) | 1 (0.0%) | 91 (0.0%) | 29 (0.0%) | 47 (0.0%) | 8 (0.0%) | 8 (0.0%) | 2,681 (0.0%) | 359 (0.0%) |
| P25 | 57,126 (0.1%) | 1,312 (0.1%) | 4,048 (0.1%) | 219,950 (0.1%) | 3,759 (0.1%) | 14,254 (0.1%) | 38,321 (0.1%) | 33,540 (0.1%) | 46,539 (0.1%) |
| P50 | 126,862 (0.3%) | 3,325 (0.2%) | 8,365 (0.2%) | 555,499 (0.2%) | 8,815 (0.2%) | 31,188 (0.2%) | 75,596 (0.2%) | 71,384 (0.2%) | 110,129 (0.2%) |
| P75 | 253,922 (0.6%) | 7,524 (0.4%) | 19,326 (0.4%) | 1,131,444 (0.4%) | 19,582 (0.4%) | 72,256 (0.4%) | 168,709 (0.4%) | 153,408 (0.3%) | 228,271 (0.4%) |
| Max | 2,975,981 (6.6%) | 286,242 (15.8%) | 784,221 (17.9%) | 49,001,275 (15.9%) | 491,750 (10.6%) | 2,536,198 (15.7%) | 5,432,492 (14.1%) | 11,778,807 (25.8%) | 9,160,871 (15.3%) |

This table shows that on average the most frequent tag accounts for 15.3% of the USAS labels and the 75% least frequent USAS tag classes make up per USAS tag class at most 0.4% of the USAS labels on average.

### Filtering Funnel (FineWiki → Final Dataset)

Of every FineWiki article considered per language, the vast majority are removed for not being
rated Good/Featured; the remainder are then thinned further by the held-out test-set filter,
minimum-word filter, and exact/MinHash de-duplication:

| Language | Good/Featured | Test URL | Min words | Exact dedup | MinHash dedup | Total removed | Kept |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Chinese | 1,291,263 | 0 | 1,526 | 196 | 155 | 1,293,140 | 2,815 (0.217%) |
| Danish | 291,764 | 0 | 0 | 7 | 3 | 291,774 | 187 (0.064%) |
| Dutch | 2,072,477 | 0 | 0 | 5 | 5 | 2,072,487 | 378 (0.0182%) |
| English | 6,562,224 | 4 | 6 | 1,608 | 1,571 | 6,565,413 | 49,242 (0.744%) |
| Finnish | 571,963 | 0 | 0 | 30 | 42 | 572,035 | 865 (0.151%) |
| Italian | 1,798,167 | 0 | 0 | 178 | 252 | 1,798,597 | 1,162 (0.0646%) |
| Portuguese | 1,131,646 | 0 | 0 | 167 | 100 | 1,131,913 | 3,470 (0.306%) |
| Spanish | 1,943,867 | 0 | 0 | 265 | 243 | 1,944,375 | 4,590 (0.236%) |

A final post-hoc de-duplication, based on th Wikipedia Page ID (keeping the highest-version
duplicate and rebalancing the train/validation split) creates the final published article counts shown in [Overview](#overview) above:

| Language | Documents After Filtering | Final Articles | Dropped | Dropped (%) |
| :--- | ---: | ---: | ---: | ---: |
| Chinese | 2,815 | 2,807 | 8 | 0.28 |
| Danish | 187 | 187 | 0 | 0.00 |
| Dutch | 378 | 378 | 0 | 0.00 |
| English | 49,242 | 49,218 | 24 | 0.05 |
| Finnish | 865 | 865 | 0 | 0.00 |
| Italian | 1,162 | 1,161 | 1 | 0.09 |
| Portuguese | 3,470 | 3,469 | 1 | 0.03 |
| Spanish | 4,590 | 4,581 | 9 | 0.20 |
| **Total (matched languages)** | **62,709** | **62,666** | **43** | **0.07** |

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
