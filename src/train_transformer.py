import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from features.tokenizer import Tokenizer
from models.transformers import Transformer

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
    block = tf.reshape(block, (256,))
    label = tf.reshape(label, (256,))
    return (block, label)

BATCH_SIZE = 256

dataset = tf.data.TFRecordDataset(filenames=DATASET_FILES)
dataset = dataset.map(read_tfrecord)
dataset = dataset.shuffle(10000)
dataset = dataset.batch(BATCH_SIZE)

# DATASET_SIZE = 8_904_448
# # TODO: Fix this. This is hardcoded for now but shouldn't be
# dataset = dataset.apply(tf.data.experimental.assert_cardinality(DATASET_SIZE))

# train_size = int(0.7 * DATASET_SIZE)
# val_size = int(0.15 * DATASET_SIZE)
# test_size = int(0.15 * DATASET_SIZE)

# train_dataset = dataset.take(train_size)
# test_dataset = dataset.skip(train_size)
# val_dataset = test_dataset.skip(val_size)
# test_dataset = test_dataset.take(test_size)

# train_dataset = train_dataset.apply(tf.data.experimental.assert_cardinality(train_size))
# val_dataset = val_dataset.apply(tf.data.experimental.assert_cardinality(val_size))
# test_dataset = test_dataset.apply(tf.data.experimental.assert_cardinality(test_size))

# train_dataset = train_dataset.batch(BATCH_SIZE)
# val_dataset = val_dataset.batch(BATCH_SIZE)
# test_dataset = test_dataset.batch(BATCH_SIZE)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


num_layers = 2
d_model = 32
dff = 128
num_heads = 4
dropout_rate = 0.1


model = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    vocab_size=Tokenizer.n_vocab(),
    dropout_rate=dropout_rate,
    max_indentation=256)

learning_rate = CustomSchedule(d_model)
adam = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
# adam = tf.keras.optimizers.Adam(learning_rate=0.0003)

model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])

for x, y in dataset.take(1):
   model(x)

model.summary()

checkpoint_path = "checkpoints/indentation_prediction_v5.ckpt"
callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=0)


print("Memory usage: ", tf.config.experimental.get_memory_info('GPU:0')['current'])

class MemoryPrintingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
      gpu_dict = tf.config.experimental.get_memory_info('GPU:0')
      tf.print('\n GPU memory details [current: {} gb, peak: {} gb]'.format(
          float(gpu_dict['current']) / (1024 ** 3), 
          float(gpu_dict['peak']) / (1024 ** 3)))


# if os.path.exists(checkpoint_path + ".index"):
#     print("Loading weights from checkpoint")
#     model.load_weights(checkpoint_path)

model.fit(dataset, batch_size=BATCH_SIZE, epochs=25, verbose=1, callbacks=[MemoryPrintingCallback(), callback])

# loss, accuracy = model.evaluate(test_dataset, verbose=1)
# print("Loss :", loss)
# print("Accuracy :", accuracy)
