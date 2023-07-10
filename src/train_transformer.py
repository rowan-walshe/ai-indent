import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import random
from pathlib import Path

import wandb
import numpy as np
import tensorflow as tf

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

train_ds = tf.data.Dataset.load(str(Path(PROCESSED_DATA_DIR) / 'train_ds'))
val_ds = tf.data.Dataset.load(str(Path(PROCESSED_DATA_DIR) / 'val_ds'))
test_ds = tf.data.Dataset.load(str(Path(PROCESSED_DATA_DIR) / 'test_ds'))

BATCH_SIZE = 64

train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.cache()
val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.cache()
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.cache()

sweep_configuration = {
    'method': 'random',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize',
    },
    'parameters': {
        'epochs': {
            'value': 5,
        },
        'batch_size': {
            'value': BATCH_SIZE,
        },
        'num_layers': {
            'min': 1,
            'max': 4,
        },
        'd_model': {
            'values': [32, 64, 128, 256],
        },
        'dff': {
            'values': [128, 256, 512, 1024],
        },
        'num_heads': {
            'values': [1, 2, 4, 8],
        },
        'dropout_rate': {
            'min': 0.0,
            'max': 0.2,
            'distribution': 'uniform',
        },
        'warmup_steps': {
            'min': 1,
            'max': 10000,
            'distribution': 'log_uniform_values',
        },
        'beta_1': {
            'min': 0.8,
            'max': 0.99,
            'distribution': 'uniform',
        },
        'beta_2': {
            'min': 0.8,
            'max': 0.999,
            'distribution': 'uniform',
        },
        'epsilon': {
            'min': 1e-10,
            'max': 1e-7,
            'distribution': 'log_uniform_values',
        },
    }      
}

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
    sweep_id = wandb.sweep(sweep_configuration, project='ai-indent')
    wandb.agent(sweep_id, function=main)