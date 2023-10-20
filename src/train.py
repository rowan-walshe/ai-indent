import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from features.tokenizer import Tokenizer

# Set a fixed seed for reproducibility, for the random module, numpy, and tensorflow
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

PROJECT_ROOT = Path(__file__).parent.absolute().parent
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
    print(block.shape)
    print(label.shape)
    block = tf.reshape(block, (256,))
    label = tf.reshape(label, (256,))
    print(block.shape)
    print(label.shape)
    return (block, label)

BATCH_SIZE = 32

dataset = tf.data.TFRecordDataset(filenames=DATASET_FILES)
dataset = dataset.map(read_tfrecord)
dataset = dataset.shuffle(10000)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.apply(tf.data.experimental.assert_cardinality(87307))


def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(Tokenizer.n_vocab(), 64, input_length=256))
    model.add(tf.keras.layers.LSTM(64, return_sequences=True))
    model.add(tf.keras.layers.LSTM(64))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, input_shape=(256,), activation="softmax"))
    adam = tf.keras.optimizers.Adam(learning_rate=0.0003)
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


model = create_model()
model.summary()


# checkpoint_path = "checkpoints/indentation_prediction_v2_100_epoch.ckpt"
# callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=0)

# if os.path.exists(checkpoint_path + ".index"):
#     print("Loading weights from checkpoint")
#     model.load_weights(checkpoint_path)


model.fit(dataset, batch_size=BATCH_SIZE, epochs=1, verbose=1, validation_split=0.2)

tf.saved_model.save(model, 'models/hello_world')

# loss, accuracy = model.evaluate(dataset)
# print("Loss :", loss)
# print("Accuracy :", accuracy)
