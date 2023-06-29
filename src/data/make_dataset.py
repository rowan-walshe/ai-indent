import hashlib
import math
import os
import random
import re
import shutil
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np
import tensorflow as tf

# Set a fixed seed for reproducibility, for the random module, numpy, and tensorflow
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

PROJECT_ROOT = Path(__file__).parent.absolute().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
INTERIM_DATA_DIR = DATA_DIR / 'interim'
FILES_DIR = INTERIM_DATA_DIR / 'complete_files'
CODE_BLOCKS_DIR = INTERIM_DATA_DIR / 'code_blocks'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

ADS_DIR = FILES_DIR / 'ads'
ADB_DIR = FILES_DIR / 'adb'
GPR_DIR = FILES_DIR / 'gpr'
ADA_DIR = FILES_DIR / 'ada'
WEIRD_DIR = FILES_DIR / 'weird'

TRAINABLE_FILE_TYPES = {".ads", ".adb", ".gpr", ".ada"}

def remove_visible_files(dir: Path):
    # Remove all files in the target directory that end with category
    for file in dir.iterdir():
        if file.is_file() and not file.name.startswith("."):
            file.unlink()


def file_hash(file_path: str):
    # Calculate the hash of a file
    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def is_file_mostly_space_indented(file_path: str):
    # Returns True if the file is mostly space indented
    # Returns False if the file is mostly tab indented
    # Defaults to False if the file is empty
    space_indent_count = 0
    tab_indent_count = 0
    with open(file_path, "rb") as f:
        file_contents = f.readlines()
        for line in file_contents:
            whitespace_count = len(line) - len(line.lstrip())
            whitespaces = line[:whitespace_count]
            space_indent_count += whitespaces.count(b" ")
            tab_indent_count += whitespaces.count(b"\t")

    # In ada, the convention is to use 3 spaces for indentation
    space_indent_count = math.ceil(space_indent_count / 3)
    return space_indent_count > tab_indent_count or tab_indent_count == 0


def get_files_to_process(data_dir: str, skip_non_utf8_files: bool = True) -> Tuple[List[str], List[str]]:
    # returns a list of unique ada files in the data ada_code_bases directory
    hashes = set()
    files_to_process = []
    weird_files = []

    for root, _, files in os.walk(data_dir):
        for file in files:
            file_type = os.path.splitext(file)[1]
            if file_type in TRAINABLE_FILE_TYPES:
                file_path = os.path.join(root, file)
                hash = file_hash(file_path)
                if hash not in hashes:
                    hashes.add(hash)
                    # If the file is not UTF-8, skip it
                    if skip_non_utf8_files:
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                f.read()
                        except UnicodeDecodeError:
                            weird_files.append(file_path)
                            continue
                    # We only want to process files that are mostly space indented
                    if is_file_mostly_space_indented(file_path):
                        files_to_process.append(file_path)
                    else:
                        weird_files.append(file_path)
    return files_to_process, weird_files


def copy_file_to_dir(file: str, dest_dir: Path, i: Optional[int] = None):
    # Copies files to a destination directory
    file_name = os.path.basename(file)
    if i is None:
        shutil.copy(file, str(dest_dir / file_name))
    else:
        shutil.copy(file, str(dest_dir / f"{i}_{file_name}"))


def seperate_files():
    for dir in [ADS_DIR, ADB_DIR, GPR_DIR, ADA_DIR, WEIRD_DIR]:
        remove_visible_files(dir)
    # Find any file that contains training data in data/raw and copy it to the structure under data/interim
    files_to_process, weird_files = get_files_to_process(str(RAW_DATA_DIR))
    for i, file in enumerate(files_to_process):
        if file.endswith(".ads"):
            copy_file_to_dir(file, ADS_DIR, i)
        elif file.endswith(".adb"):
            copy_file_to_dir(file, ADB_DIR, i)
        elif file.endswith(".gpr"):
            copy_file_to_dir(file, GPR_DIR, i)
        elif file.endswith(".ada"):
            copy_file_to_dir(file, ADA_DIR, i)
    for i, file in enumerate(weird_files):
        copy_file_to_dir(file, WEIRD_DIR, i)



UNINDENT_LINES = re.compile(r'^\s*(begin\s*$|end(;|\s+\S+))')
GENERIC_BLOCK_START = re.compile(r'^\s*generic($|\s+[^;]*$)')
GENERIC_BLOCK_END = re.compile(r'(^|;)[^\S\r\n]*(function|procedure|package)($|\s+.*)')


def split_file_into_clode_blocks(file_path: str) -> Generator[str, None, None]:
    # Read a file and return the contents as a string
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Split the List of strings into a List of List of Strings
    # Each inner List is delimited by a line that matches the regex UNINDENT_LINES
    # This is used to split the file into predictable blocks of code
    block = []
    generic_block = False
    for line in lines:
        if len(line.strip()) == 0:
            continue

        if not generic_block and GENERIC_BLOCK_START.match(line):
            generic_block = True
        if generic_block and GENERIC_BLOCK_END.match(line):
            generic_block = False
            if len(block) > 0:
                yield ''.join(block)
                block = []

        if UNINDENT_LINES.match(line):
            if len(block) > 0:
                yield ''.join(block)
                block = []
        else:
            block.append(line)
    if len(block) > 0:
        yield ''.join(block)


def trainable_file_in_dir(dir: Path) -> Generator[Path, None, None]:
    # Returns a generator that yields all the files in a directory that are trainable
    for file in dir.iterdir():
        if file.is_file() and file.suffix in TRAINABLE_FILE_TYPES:
            yield file


def create_code_blocks():
    # For every file in data/interim/complete_files, split the file's into
    # interesting code blocks and write them to data/interim/code_blocks
    for category in ['ads', 'adb', 'gpr', 'ada']:
        src_dir = FILES_DIR / category
        tgt_dir = CODE_BLOCKS_DIR / category
        remove_visible_files(tgt_dir)
        block_count = 0
        for file in trainable_file_in_dir(src_dir):
            for block in split_file_into_clode_blocks(str(file)):
                with open(str(tgt_dir / f"{block_count}_{file.name}"), "w", encoding="utf-8") as f:
                    f.write(block)
                block_count += 1
    

if __name__ == "__main__":
    seperate_files()
    create_code_blocks()
