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
- 100<n<100K
pretty_name: Multilingual USAS Silver Labelled Wikipedia Articles
configs:
- config_name: en
  data_files:
  - split: train
    path: data/en.jsonl
- config_name: nl
  data_files:
  - split: train
    path: data/nl.jsonl
- config_name: pt
  data_files:
  - split: train
    path: data/pt.jsonl
- config_name: es
  data_files:
  - split: train
    path: data/es.jsonl
- config_name: da
  data_files:
  - split: train
    path: data/da.jsonl
- config_name: it
  data_files:
  - split: train
    path: data/it.jsonl
- config_name: fi
  data_files:
  - split: train
    path: data/fi.jsonl
- config_name: zh
  data_files:
  - split: train
    path: data/zh.jsonl
viewer: true
---
# Multilingual USAS Silver Labelled Wikipedia Articles


Contains the Wikipedia Article IDs and page titles for Good and Featured articles on Wikipedia for a given timestamped data dump, whereby the data was extracted from [Wikipedia/Wikimedia SQL table dumps](https://dumps.wikimedia.org/). This dataset covers 9 Language Wikipedia sites. For more information on how the dataset was generated see [https://github.com/UCREL/wikipedia-ga-fa-extraction](https://github.com/UCREL/wikipedia-ga-fa-extraction).

The data is specific to a given data dump timestamp, the main tag of the repository relates to the most up to date timestamp that has been tagged (versioned). Each timestamp of data that is uploaded will also be uploaded to it's Git tagged specific timestamp (version). This upload relates to timestamp: 2025-12-01


- **Curated by:** [University Centre for Computer Corpus Research on Language (UCREL) group](https://ucrel.lancs.ac.uk/) at [Lancaster University](https://www.lancaster.ac.uk/)
- **Multi-lingual**
- **Repository:** [https://github.com/UCREL/wikipedia-USAS-processing](https://github.com/UCREL/wikipedia-USAS-processing)


## Uses

It can be used to train USAS semantic taggers and MWE identifiers.

## Dataset Structure

Each JSONL line contains the following information for each article, each line is unique, a `page_id` value only occurs once in the file, and each article has either `ga` or `fa` as True;

* `page_id` - The unique per Wikipedia language site article ID. (INT).
* `page_title` - The 255 byte text string of the article title. This is a string but it can be truncated if the original title was longer than 255 bytes, of which a character can be up to 4 bytes. (STRING).
* `ga` - False if the article is not a Good Article (GA), otherwise True. (BOOL).
* `fa` - False if the article is not a Featured Article (FA), otherwise True. (BOOL).

Example of a JSONL file;

``` JSON
{"page_id": 130, "page_title": "Norge", "ga": true, "fa": false}
{"page_id": 167, "page_title": "Sverige", "ga": true, "fa": false}
```

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