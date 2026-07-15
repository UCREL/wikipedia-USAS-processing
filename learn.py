from datasets import load_dataset_builder
from datasets import load_dataset
import re
from typing import cast

#ds_builder = load_dataset_builder("HuggingFaceFW/finewiki")

#print(ds_builder.info)

#wiki_dataset = load_dataset("HuggingFaceFW/finewiki", name="nl", split="train", streaming=True)

#import pdb
#pdb.set_trace()

import json
from pathlib import Path
import mistune

from wikipedia_processing.markdown_renderer import FineWikiPlainTextRenderer



# Spanish data contains a mark block

# To remove all sentences that contain \displaystyle as this is how math content is displayed.

# We need to check on a wider sample that problem of block and in line math content.
import re

latex_code_pattern = re.compile(r"\{(\\[\S]+).*\}", re.DOTALL)


claude_markdown_parser =  mistune.create_markdown(
    renderer=FineWikiPlainTextRenderer(),
    plugins=["table", "math", "strikethrough", "abbr", "footnotes", "task_lists", "def_list", "mark", "insert", "spoiler"],  # register so tokens are parsed
)


from collections import Counter
latex_commands = Counter()

languages_requires = [
    "English",
    "Dutch",
    "Spanish",
    "Danish",
    "Italian",
    "Portuguese",
    "Chinese",
    "Finnish",
    "Irish",
    "Welsh",
]

import time

def remove_family_tree_tables(markdown_text: str, pipe_threshold=40) -> str:
    """
    Removes the lines of text that contain more than pipe_threshold number of pipes
    and returns the remaining text.

    This typically removes very large tables and family trees from the text.

    Args:
        markdown_text: The text to be processed.
        pipe_threshold: The maximum number of pipes allowed in a line.

    Returns:
        The remaining text after removing lines with more than pipe_threshold number of pipes.
    """
    text_lines = markdown_text.split("\n")
    non_family_tree_text: list[str] = []

    for line in text_lines:
        if line.count("|") > pipe_threshold:
            continue
        non_family_tree_text.append(line)

    return "\n".join(non_family_tree_text)

def remove_lines_with_given_latex_commands(markdown_text: str, latex_commands: set[str]) -> str:
    """
    Removes the lines of text that contain any of the given latex commands.

    Args:
        markdown_text: The text to be processed.
        latex_commands: A set of latex commands to be removed.

    Returns:
        The remaining text after removing lines with any of the given latex commands.
    """
    text_lines = markdown_text.split("\n")
    non_latex_command_text: list[str] = []

    for line in text_lines:
        found_latex_commands = set(latex_code_pattern.findall(line))
        if found_latex_commands and latex_commands.intersection(found_latex_commands):
            continue
        non_latex_command_text.append(line)

    return "\n".join(non_latex_command_text)

data_folder = Path(f"data/wikipedia_pages")
output_folder = Path(f"data/wikipedia_pages_filtered")
output_folder.mkdir(parents=True, exist_ok=True)
latex_commands_to_remove = set({"\\displaystyle", "\\textstyle"})

for language in languages_requires:
    print(f"{language}")
    start_time = time.perf_counter()

    data_file = Path(data_folder, f"{language}_page_data.jsonl")
    output_file = Path(output_folder, f"{language}_page_data.md")
    with output_file.open("w", encoding="utf-8") as write_fp:
        with data_file.open("r", encoding="utf-8") as fp:
            for index, line in enumerate(fp):
                if index < 34007:
                    continue
                text = json.loads(line)['text']

                text = remove_family_tree_tables(text)
                text = remove_lines_with_given_latex_commands(text, latex_commands_to_remove)
                #text = square_bracket_page_preview_pattern.sub("", text)
                ast_text = cast(str, claude_markdown_parser(text))
                if not isinstance(ast_text, str):
                    print(f"Error: {data_file!r}")
                    continue
                write_fp.write(ast_text)
                

    end_time = time.perf_counter()
    print(f"Time taken for {language}: {end_time - start_time}")
    print(f"Current latex commands: {latex_commands}")
    print()
    print()

"""
for language in languages_requires:

    data_file = Path(data_folder, f"{language}_page_data.jsonl")
    with data_file.open("r", encoding="utf-8") as fp:
        for line in fp:
            text = json.loads(line)['text']
            #text = square_bracket_page_preview_pattern.sub("", text)
            ast_text = cast(str, claude_markdown_parser(text))
        
            latex_code_pattern_match = latex_code_pattern.findall(ast_text)
            if latex_code_pattern_match:
                continue
            #if square_bracket_page_preview_pattern.findall(ast_text):
            #    print(ast_text)
            #    import sys
            #    sys.exit(1)
            for line in ast_text.split("\n"):
                latex_code_pattern_match = re.findall(latex_code_pattern, line)
                if latex_code_pattern_match:
                    if latex_code_pattern_match[0] == "\\textstyle":
                        print(line)
                    latex_commands.update(latex_code_pattern_match)
            if "&&&&&&&&&&&&&&&" in ast_text:
                print(text)
                #import pdb
                #pdb.set_trace()


with Path("a_page.md").open("w", encoding="utf-8") as fp:
    with Path("data/wikipedia_pages/Dutch_page_data.jsonl").open("r", encoding="utf-8") as fr:
        for data in fr:
            data_ = json.loads(data)
            text = data_['text']
            ast_text = claude_markdown_parser(text)
            if not isinstance(ast_text, str):
                print("Error")
                continue
            for line in ast_text.split("\n"):
                latex_code_pattern_match = latex_code_pattern.findall(line)
                if latex_code_pattern_match:
                    print(latex_code_pattern_match)
                    if latex_code_pattern_match[0] == "\\textstyle":
                        print(line)
                    latex_commands.update(latex_code_pattern_match)
            #if "&&&&&&&&&&&&&&&" in ast_text:
            #    print(text)
                #import pdb
                #pdb.set_trace()
            #text = square_bracket_page_preview_pattern.sub("", data_['text'])
            #fp.write(text)
            #fp.write("--"* 100)
            #fp.write(ast_text)
            #fp.write("##"*100)
            #fp.write("\n\n\n")
            #print(text)
            
            #if re.findall(latex_code_pattern, text):
            #    print("%")
            #    print("\n")
            #    print("latex")
            #    for line in ast_text.split("\n"):
            #        if re.findall(latex_code_pattern, line):
            #            if re.findall(latex_code_pattern, line):
            #                latex_commands.update(latex_code_pattern.findall(line))
            #    print(data_.keys())
            #    #print(data_['url'])
            #    #print(data_['has_math'])
            #    print("%")

print(latex_commands)
"""