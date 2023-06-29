from pathlib import Path
from typing import Generator, List, Tuple

import numpy as np
import tensorflow as tf
from tokenizer import Tokenizer

PROJECT_ROOT = Path(__file__).parent.absolute().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
INTERIM_DATA_DIR = DATA_DIR / 'interim'
CODE_BLOCKS_DIR = INTERIM_DATA_DIR / 'code_blocks'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

TRAINABLE_FILE_TYPES = {".ads", ".adb", ".gpr", ".ada"}

MAX_TOKENS = 256
MAX_INDENTATION = 256

def get_sub_blocks() -> Generator[Tuple[str, str, str], None, None]:
    code_block_dirs = [CODE_BLOCKS_DIR / category[1:] for category in TRAINABLE_FILE_TYPES]
    for dir in code_block_dirs:
        for file in dir.iterdir():
            if file.is_file() and file.suffix in TRAINABLE_FILE_TYPES:
                with open(str(file), "r", encoding="utf-8") as f:
                    lines = f.readlines()
                last_end = 0
                for i in range(len(lines)-1):
                    line = lines[i]
                    if line.strip().endswith(';'):
                        yield file.name, ''.join(lines[last_end:i+1]), lines[i+1]
                        last_end = i+1


def count_leading_spaces(block):
    count = 0
    for token in block:
        if token == Tokenizer.space_token():
            count += 1
        else:
            break
    return count


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def create_example(file_name, block, next_line):
    block_tokens = Tokenizer.encode(block)
    block_tokens = Tokenizer.resize(block_tokens, MAX_TOKENS)
    block_tokens = np.array(block_tokens, dtype=np.int32)
    label = count_leading_spaces(next_line)
    label = tf.keras.utils.to_categorical(label, num_classes=MAX_INDENTATION)
    feature = {
        'file_name': file_name,
        'block': block_tokens,
        'label': label
    }


def create_dataset():
    for file_name, block, next_line in get_sub_blocks():
        example = create_example(file_name, block, next_line)



if __name__ == "__main__":
    create_dataset()
