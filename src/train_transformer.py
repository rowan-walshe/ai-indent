import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import random
from pathlib import Path

import wandb
import numpy as np
import tensorflow as tf

from features.build_features import MAX_INDENTATION, MAX_TOKENS
from features.tokenizer import Tokenizer
from models.transformers import Transformer
from transformer import train_transformer

# Set a fixed seed for reproducibility, for the random module, numpy, and tensorflow
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

PROJECT_ROOT = Path(__file__).parent.absolute().parent
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

FEATURES = {
    'toks': tf.io.FixedLenFeature([], tf.string),
    'pre': tf.io.FixedLenFeature([], tf.int64),
    'post': tf.io.FixedLenFeature([], tf.int64)
}
def parse_tfr_element(element):
    content = tf.io.parse_single_example(element, FEATURES)
    tokens = tf.io.parse_tensor(content['toks'], tf.uint8)
    tokens = tf.reshape(tokens, (MAX_TOKENS,))
    label = tf.cast(tf.stack([content['pre'], content['post']]), tf.int8)
    return (tokens, label)

train_ds = tf.data.Dataset.load(str(Path(PROCESSED_DATA_DIR) / 'train_ds'))
val_ds = tf.data.Dataset.load(str(Path(PROCESSED_DATA_DIR) / 'val_ds'))
test_ds = tf.data.Dataset.load(str(Path(PROCESSED_DATA_DIR) / 'test_ds'))

train_ds = train_ds.map(parse_tfr_element)
val_ds = val_ds.map(parse_tfr_element)
test_ds = test_ds.map(parse_tfr_element)

BATCH_SIZE = 64

train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.cache()
val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.cache()
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.cache()

# sweep_configuration = {
#     'method': 'random',
#     'metric': {
#         'name': 'val_loss',
#         'goal': 'minimize',
#     },
#     'parameters': {
#         'epochs': {
#             'value': 20,
#         },
#         'batch_size': {
#             'value': BATCH_SIZE,
#         },
#         'num_layers': {
#             'value': 3,
#         },
#         'd_model': {
#             'value': 64,
#         },
#         'dff': {
#             'value': 256,
#         },
#         'num_heads': {
#             'value': 4,
#         },
#         'dropout_rate': {
#             'value': 0.05,
#         },
#         'warmup_steps': {
#             'min': 100,
#             'max': 1000,
#             'distribution': 'log_uniform_values',
#         },
#         'beta_1': {
#             'min': 0.9,
#             'max': 0.98,
#             'distribution': 'uniform',
#         },
#         'beta_2': {
#             'min': 0.9,
#             'max': 0.99,
#             'distribution': 'uniform',
#         },
#         'epsilon': {
#             'min': 1e-9,
#             'max': 1e-8,
#             'distribution': 'log_uniform_values',
#         },
#     }      
# }

def main():
    wandb.init(project='ai-indent')
    train_transformer(
        train_ds, 
        val_ds,
        epochs=wandb.config.epochs,
        batch_size=wandb.config.batch_size,
        num_layers=wandb.config.num_layers,
        d_model=wandb.config.d_model,
        dff=wandb.config.dff,
        num_heads=wandb.config.num_heads,
        dropout_rate=wandb.config.dropout_rate,
        warmup_steps=wandb.config.warmup_steps,
        beta_1=wandb.config.beta_1,
        beta_2=wandb.config.beta_2,
        epsilon=wandb.config.epsilon,
    )

if __name__ == '__main__':
    # sweep_id = wandb.sweep(sweep_configuration, project='ai-indent_transformer_half')
    # wandb.agent(sweep_id, function=main)
    train_transformer(
        train_ds, 
        val_ds,
        epochs=5,
        batch_size=BATCH_SIZE,
        num_layers=3,
        d_model=32,
        dff=128,
        num_heads=4,
        dropout_rate=0.0005,
        warmup_steps=200,
        beta_1=0.95,
        beta_2=0.99,
        epsilon=1e-9,
    )
