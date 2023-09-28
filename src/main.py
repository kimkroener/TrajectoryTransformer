# %% Imports
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from transformer import Transformer
from lib.scheduler import LRScheduler
import utils as utils

# load config
config_file = "src/test_config.yaml"
config = utils.load_config(config_file)

# %%
# Set paths for data, logs, checkpoints
data_dir = config["data"]["data_dir"]
log_dir = config["data"]["logs_dir"]
checkpoint_dir = config["data"]["checkpoints_dir"]
data_path_weights_filename = os.path.join(checkpoint_dir, "model_weights")

tf.debugging.experimental.enable_dump_debug_info(log_dir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

# create if they don't exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Load data and data specific parameters
train_test_ratio = config["data"]["train_test_ratio"]
X, Y = utils.load_data(data_dir)
x_train, x_test, y_train, y_test = utils.split_data(X, Y, train_test_ratio)

y_train_shifted = utils.shift(y_train) # decoder input
y_test_shifted = utils.shift(y_test) 

# ensure returned values are TF tensors
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

encoder_seq_length = x_train.shape[1]
decoder_seq_length = y_train.shape[1]
d_input = x_train.shape[2]
d_output = y_train.shape[2] # output dimension in final layer

print(f"\nLoaded data from {data_dir}")
print(f"Encoder input shape: {x_train.shape}")
print(f"Decoder input and output shape: {y_train.shape}")
print("Shapes are in order (N_sim, N_timesteps, N_dof) and (N_batch, N_timesteps, d_model) internally.\n")

# Create and compile model

# Transformer architecture params
d_model = config["architecture"]["d_model"]  # Dimensionality of the latent space - here the DOF of the model will be scaled to that  dim. Inner model dimensions are (batch_size/N_seq, seq_length, d_model)
n_stacks = config["architecture"]["N_stacks"]  # Depth of encoder and decoder layers
h = config["architecture"]["h"]  # Number of self-attention heads
d_ff = config["architecture"]["d_ff"]  # Dimensionality of the inner fully connected layer
activation_ff = config["architecture"]["activation_ff"]  # Activation function of the feed-forward module
dropout_rate = config["architecture"]["dropout_rate"]  # Frequency of dropping the input units in the dropout layers

# create model
model = Transformer(encoder_seq_length, decoder_seq_length, h, d_model, d_ff, activation_ff, d_output, n_stacks, dropout_rate)

# setup optimizer
# for beta and eps use default values as no hyperparameter tuning was done
# beta_1 = config["training"]["beta_1"]
# beta_2 = config["training"]["beta_2"]
# epsilon = config["training"]["epsilon"]
optimizer = Adam(LRScheduler(d_model)) # , beta_1, beta_2, epsilon)

model.compile(optimizer=optimizer,
              loss="mse",
              metrics=["mse"],
              )
print("Compiled model, can start training...")

# %% Training
# training parameters
epochs = config["training"]["epochs"]
batch_size = config["training"]["batch_size"]

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


# train_data = (x_train, y_train_shifted, y_train)
# train_data = (x_train, y_train, y_train)


# create masks, propagate them through functional API
mask_value = -1000
masking_layer = tf.keras.layers.Masking(mask_value)

masked_encoder_input = masking_layer(x_train)
masked_decoder_input = masking_layer(y_train_shifted)
masked_decoder_output = masking_layer(y_train)



history = model.fit(x=[x_train, y_train_shifted],
                    y=y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    )

print(history.history)
# # %% Evaluate model
print("Evaluating model on test data...")
results = model.evaluate([x_test, y_test_shifted], y_test, batch_size=batch_size)
print(results)

# %%
# load data
# import pickle
# import os
# import numpy as np
# with open('../../data/occupant_data_transformer.pkl', 'rb') as f:
#     data = pickle.load(f)
#     displacements_train = data['displacements_train']
#     displacements_test = data['displacements_test']
#     displacements_red_train = data['displacements_red_train']
#     displacements_red_test = data['displacements_red_test']
#     params_train = data['params_train']
#     params_test = data['params_test']
#     times_train = data['times_train']
#     times_test = data['times_test']
#     reference_coordinates_train = data['reference_coordinates_train']
#     reference_coordinates_test = data['reference_coordinates_test']
#     projection_matrix = data['projection_matrix']
# %%
