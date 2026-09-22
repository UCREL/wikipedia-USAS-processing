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
