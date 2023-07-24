# %% Imports
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from time import time

import tensorflow as tf 
from tensorflow.keras.optimizers import Adam

from msd_transformer_tf import TransformerModel
from scheduler import LRScheduler

# %% Setup

# Transformer architecture
h = 8  # Number of self-attention heads
d_ff = 2048  # Dimensionality of the inner fully connected layer
d_model = 512  # Dimensionality of the latent space
N = 6  # Number of layers in the encoder and decoder stack
batch_size = 50  # Batch size from the training process
dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers

encoder_seq_length = 100 # length of input series
decoder_seq_length = 100
input_seq = random.random((batch_size, encoder_seq_length))
print(f"Input dimension: {input_seq.shape}")

# training parameters
epochs = 2
batch_size = 64
beta_1 = 0.9
beta_2 = 0.98
epsilon = 1e-9
dropout_rate = 0.1

# %% Create and Train model
# create model
model = TransformerModel(encoder_seq_length, decoder_seq_length, h, d_model, d_ff, N, dropout_rate)

# instantziate optimizer
optimizer = Adam(LRScheduler(d_model), beta_1, beta_2, epsilon)

model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

# %%
