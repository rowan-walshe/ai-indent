import hashlib
import math
import multiprocessing as mp
import os
import shutil
from functools import partial
from pathlib import Path
from typing import Generator, List, Optional, Tuple

from features.tokenizer import Tokenizer


PROJECT_ROOT = Path(__file__).parent.absolute().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
INTERIM_DATA_DIR = DATA_DIR / 'interim'
FILES_DIR = INTERIM_DATA_DIR / 'complete_files'
TOKENIZED_FILES_DIR = INTERIM_DATA_DIR / 'tokenized_files'
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


def copy_file_without_blank_lines(file: str, dest_dir: Path, i: Optional[int] = None):
    # Copies files to a destination directory
    file_name = os.path.basename(file)
    if i is None:
        dest_path = dest_dir / file_name
    else:
        dest_path = dest_dir / f"{i}_{file_name}"
    with open(file, "r", encoding="utf-8") as f:
        # Remove lines which are just whitespace
        lines = list(filter(lambda line: len(line.strip()) > 0, f.readlines()))
    with open(dest_path, "w+", encoding="utf-8") as f:
        f.writelines(lines)


def separate_files():
    for dir in [ADS_DIR, ADB_DIR, GPR_DIR, ADA_DIR, WEIRD_DIR]:
        remove_visible_files(dir)
    # Find any file that contains training data in data/raw and copy it to the structure under data/interim
    files_to_process, weird_files = get_files_to_process(str(RAW_DATA_DIR))
    for i, file in enumerate(files_to_process):
        if file.endswith(".ads"):
            copy_file_without_blank_lines(file, ADS_DIR, i)
        elif file.endswith(".adb"):
            copy_file_without_blank_lines(file, ADB_DIR, i)
        elif file.endswith(".gpr"):
            copy_file_without_blank_lines(file, GPR_DIR, i)
        elif file.endswith(".ada"):
            copy_file_without_blank_lines(file, ADA_DIR, i)
    for i, file in enumerate(weird_files):
        copy_file_to_dir(file, WEIRD_DIR, i)


def trainable_file_in_dir(dir: Path) -> Generator[Path, None, None]:
    # Returns a generator that yields all the files in a directory that are trainable
    for file in dir.iterdir():
        if file.is_file() and file.suffix in TRAINABLE_FILE_TYPES:
            yield file


def tokenize_file(tgt_dir: Path, file: Path, ):
    with file.open('r', encoding="utf-8") as fr:
        with (tgt_dir / file.name).open("wb") as fw:
            for line in fr.readlines():
                tokens = Tokenizer.encode(line)
                for tok in tokens:
                    fw.write(tok.to_bytes(2, 'little'))

def tokenize_files():
    for category in ['ads', 'adb', 'gpr', 'ada']:
        src_dir = FILES_DIR / category
        tgt_dir = TOKENIZED_FILES_DIR / category
        remove_visible_files(tgt_dir)

        tokenize_file_with_dir = partial(tokenize_file, tgt_dir)

        pool = mp.Pool(mp.cpu_count())
        pool.map(tokenize_file_with_dir, trainable_file_in_dir(src_dir))
        pool.close()
        pool.join()

if __name__ == "__main__":
    separate_files()
    tokenize_files()
