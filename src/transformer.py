import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import base64
import hashlib
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from wandb.keras import WandbMetricsLogger

from features.build_features import MAX_INDENTATION
from features.tokenizer import Tokenizer
from models.transformers import Transformer

# Set a fixed seed for reproducibility, for the random module, numpy, and tensorflow
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

PROJECT_ROOT = Path(__file__).parent.absolute().parent


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
    def get_config(self):
        return {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps
        }


def train_transformer(
    train_ds,
    val_ds,
    epochs=5,
    max_indentation=MAX_INDENTATION,
    batch_size=32,
    num_layers=2,
    d_model=32,
    dff=128,
    num_heads=4,
    dropout_rate=0.1,
    warmup_steps=4000,
    beta_1=0.9,
    beta_2=0.98,
    epsilon=1e-9,
):
    model = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        vocab_size=Tokenizer.n_vocab(),
        dropout_rate=dropout_rate,
        max_indentation=max_indentation,
    )

    model_info = f'transformer_{max_indentation}_{epochs}_{batch_size}_{num_layers}_{d_model}_{dff}_{num_heads}_{dropout_rate}_{warmup_steps}_{beta_1}_{beta_2}_{epsilon}'
    model_info_hash = hashlib.sha1(model_info.encode('utf-8'))
    model_info_hash = base64.urlsafe_b64encode(model_info_hash.digest()[:32])

    model_path = PROJECT_ROOT / 'models' / f'transformer_2_{model_info_hash.decode("utf-8")}'

    learning_rate = CustomSchedule(d_model)
    adam = tf.keras.optimizers.Adam(learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    for x, _ in train_ds.take(1):
        model(x)

    # Create a TensorBoard callback
    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs, histogram_freq=1, profile_batch='500,520')

    model.summary()

    # model.fit(train_ds, batch_size=batch_size, epochs=epochs, validation_data=val_ds, verbose=1, callbacks=[tboard_callback])
    model.fit(train_ds, batch_size=batch_size, epochs=epochs, validation_data=val_ds, verbose=1)
    # model.fit(train_ds, batch_size=batch_size, epochs=epochs, validation_data=val_ds, verbose=1, callbacks=[WandbMetricsLogger()])
    model.save(model_path, include_optimizer=False)
    print(f"Model saved to: {model_path}")
