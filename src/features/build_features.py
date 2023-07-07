from itertools import chain
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


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_example(file_name: str, block: List[int], label: int):
    block_tokens = Tokenizer.resize(block, MAX_TOKENS)
    block_tokens = np.array(block_tokens, dtype=np.uint8)
    label = tf.keras.utils.to_categorical(label, num_classes=MAX_INDENTATION)
    feature = {
        # 'file_name': _bytes_feature(file_name.encode('utf-8')),
        'block': _bytes_feature(tf.io.serialize_tensor(block_tokens)),
        'label': _bytes_feature(tf.io.serialize_tensor(label))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def trainable_file_in_dir(dir: Path) -> Generator[Path, None, None]:
    # Returns a generator that yields all the files in a directory that are trainable
    for file in dir.iterdir():
        if file.is_file() and file.suffix in TRAINABLE_FILE_TYPES:
            yield file


def create_sub_blocks(file: Path) -> Generator[Tuple[str, str, int], None, None]:
    with open(str(file), "r", encoding="utf-8") as f:
        lines = f.readlines()
    encoded_lines = [Tokenizer.encode(line) for line in lines]
    for i in range(len(encoded_lines)-1):
        sub_block = encoded_lines[:i+1]
        label = count_leading_spaces(encoded_lines[i+1])
        sub_block = list(chain.from_iterable(encoded_lines[:i+1]))
        yield file.name, sub_block, label


def parse_tfr_element(element):
    data = {
        'block': tf.io.FixedLenFeature([], tf.string),
        'label':tf.io.FixedLenFeature([], tf.string)
    }
    content = tf.io.parse_single_example(element, data)
    block = tf.io.parse_tensor(content['block'], out_type=tf.uint8)
    label = tf.io.parse_tensor(content['label'], out_type=tf.float32)
    return (block, label)

def create_dataset():
    file_count = 0
    for category in ['ads', 'adb', 'gpr', 'ada']:
            src_dir = CODE_BLOCKS_DIR / category
            for file in trainable_file_in_dir(src_dir):
                file_count += 1
    
    i = 0
    sub_block_count = 0
    with tf.io.TFRecordWriter(str(PROCESSED_DATA_DIR / 'dataset.tfrecord')) as writer:
        for category in ['ads', 'adb', 'gpr', 'ada']:
            src_dir = CODE_BLOCKS_DIR / category
            for file in trainable_file_in_dir(src_dir):
                if i % 1000 == 0:
                    print(f'Processed {i} files out of {file_count}')
                i += 1
                for file_name, block, label in create_sub_blocks(file):
                    sub_block_count += 1
                    example = create_example(file_name, block, label)
                    writer.write(example.SerializeToString())
                    # dataset = tf.data.TFRecordDataset('test.tfrecord')
                    # dataset = dataset.map(parse_tfr_element)
                    # print('-------------------')
                    # for sample in dataset.take(1):
                    #     print(sample[0])
                    #     print(sample[1])
    print(f'Sub-block count: {sub_block_count}')


if __name__ == "__main__":
    create_dataset()
