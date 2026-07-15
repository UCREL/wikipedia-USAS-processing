import subprocess
import sys
from enum import Enum
from typing import Annotated, List

import spacy
import typer
from rich import print as rprint


from wikipedia_processing.models_install import Languages
from wikipedia_processing.models_util import get_language_tagger, get_language_sentence_splitter


from usas_validator.utils import mwe_token_indexes_from_slices, mwe_token_labels_from_indexes


a_sentence = "New York, London. Kick a bucket. Nothing."

nlp = get_language_tagger(Languages.en)

for sentence in get_language_sentence_splitter(Languages.en)(a_sentence):
    all_mwe_sets = set()
    number_tokens = 0
    for token in nlp(sentence[0]):
        print(token.text)
        c = mwe_token_indexes_from_slices(token._.pymusas_mwe_indexes)
        all_mwe_sets.add(c)
        number_tokens += 1
    print(mwe_token_labels_from_indexes(list(all_mwe_sets), number_tokens))


print(f"{list(Languages)}")