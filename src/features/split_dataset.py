import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import random
from typing import Tuple
from pathlib import Path

import numpy as np
import tensorflow as tf

# Set a fixed seed for reproducibility, for the random module, numpy, and tensorflow
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

PROJECT_ROOT = Path(__file__).parent.absolute().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
DATASET_FILES = tf.io.gfile.glob(str(PROCESSED_DATA_DIR / '*.tfrecord'))


def read_tfrecord(element):
    data = {
        'block': tf.io.FixedLenFeature([], tf.string),
        'label':tf.io.FixedLenFeature([], tf.string)
    }
    content = tf.io.parse_single_example(element, data)
    block = tf.io.parse_tensor(content['block'], out_type=tf.uint8)
    label = tf.io.parse_tensor(content['label'], out_type=tf.float32)
    block = tf.reshape(block, (256,))
    label = tf.reshape(label, (256,))
    return (block, label)

def read_dataset(shuffle: bool = True) -> Tuple[tf.data.Dataset, int]:
    dataset = tf.data.TFRecordDataset(filenames=DATASET_FILES)
    dataset = dataset.map(read_tfrecord)
    dataset_size = len(list(dataset))
    if shuffle:
        dataset = dataset.shuffle(10000)
    return dataset, dataset_size

def get_dataset_partitions_tf(ds, ds_size, train_split=0.75, val_split=0.15, test_split=0.1):
    assert (train_split + test_split + val_split) == 1
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds

if __name__ == "__main__":
    dataset, dataset_size = read_dataset()
    train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset, dataset_size)
    train_ds.save(str(Path(PROCESSED_DATA_DIR) / 'train_ds'))
    val_ds.save(str(Path(PROCESSED_DATA_DIR) / 'val_ds'))
    test_ds.save(str(Path(PROCESSED_DATA_DIR) / 'test_ds'))
