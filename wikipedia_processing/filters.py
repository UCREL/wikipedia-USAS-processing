"""
A module of Datatrove LambdaFilter functions 
(https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/filters/lambda_filter.py) 
whereby each has a corresponding function signature;

Callable[[Any], Callable[[datatrove.data.Document], bool]]

The bool of the filter when True indicates that the Document should be kept.
"""

import logging
from pathlib import Path
from typing import Callable

from datatrove.data import Document
from utils import load_page_meta_data_file

logger = logging.getLogger(__name__)

def get_relevant_wikipedia_page_function(relevant_page_meta_data_file: Path) -> Callable[[Document], bool]:
    dutch_page_id_to_title: dict[int, str] = load_page_meta_data_file(relevant_page_meta_data_file)
    def is_relevant_page(article_data: Document) -> bool:
        page_id = article_data.metadata["page_id"]
        page_title = article_data.metadata["title"][:250]
        if page_id in dutch_page_id_to_title:
            saved_page_title = dutch_page_id_to_title[page_id][:250]
            if page_title == saved_page_title:
                return True
        return False
    return is_relevant_page