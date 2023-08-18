# %% Imports
import os
# import numpy as np
# from numpy import random
# import matplotlib.pyplot as plt
# from time import time
# from pickle import load

import tensorflow as tf
#from tensorflow import Module, convert_to_tensor, TensorArray, argmax, newaxis, transpose
from tensorflow.keras.optimizers import Adam

from transformer import Transformer
from lib.scheduler import LRScheduler
import utils as utils

# Set paths for data, logs, checkpoints
data_dir = "../data/SISO_three-masses/"
log_dir = "../logs/"
checkpoint_dir = "../checkpoints/"
data_path_weights_filename = os.path.join(checkpoint_dir, "model_weights")

#tf.debugging.experimental.enable_dump_debug_info(log_dir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

# create if they don't exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Load data and data specific parameters
train_test_ratio = [0.03, 0.97]
X, Y = utils.load_data(data_dir)
x_train, x_test, y_train, y_test = utils.split_data(X, Y, train_test_ratio)
y_train_shifted = utils.shift(y_train) # decoder input

encoder_seq_length = x_train.shape[1]
decoder_seq_length = y_train.shape[1]
encoder_dof = x_train.shape[2]

print(f"\nLoaded data from {data_dir}")
print(f"Encoder input shape: {x_train.shape}")
print(f"Decoder input and output shape: {y_train.shape}")
print("Shapes are in order (N_sim, N_timesteps, N_dof) and (N_batch, N_timesteps, d_model) internally.\n")

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
# Create and compile model

# Transformer architecture params
d_model = 8  # Dimensionality of the latent space - here the DOF of the model will be scaled to that  dim. Inner model dimensions are (batch_size/N_seq, seq_length, d_model)
N = 6  # Number of layers in the encoder and decoder stack
h = 2  # Number of self-attention heads
d_ff = 2023  # Dimensionality of the inner fully connected layer
dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers

# create model
model = Transformer(encoder_seq_length, decoder_seq_length, h, d_model, d_ff, N, dropout_rate)

# setup optimizer
beta_1 = 0.9
beta_2 = 0.98
epsilon = 1e-9
optimizer = Adam(LRScheduler(d_model), beta_1, beta_2, epsilon)

model.compile(optimizer=optimizer,
              loss="mse",
              metrics=["mse"],
              )
print("Compiled model, can start training...")

# %% Training
# training parameters
epochs = 1
batch_size = 64

# checkpoint object and manager
ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        data_path_weights_filename,
        monitor='loss',
        save_best_only=True,
        save_weights_only=True,
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1)
    ]

# metrics monitoring
#train_loss = tf.keras.metrics.Mean(name='train_loss')

# ensure returned values are TF tensors
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

train_data = (x_train, y_train_shifted, y_train)

history = model.fit(x=(x_train, y_train_shifted),
                    y=y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    #callbacks=callbacks,
                    # validation_spilit=0.2
                    )

print(history.history)
# %% Evaluate model
print("Evaluating model on test data...")
results = model.evaluate(x_test, y_test, batch_size=batch_size)
print(results)

# %%
