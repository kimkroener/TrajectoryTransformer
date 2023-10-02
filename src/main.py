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

def setup_tf(config):
    # Set paths for data, logs, checkpoints
    log_dir = config["data"]["logs_dir"]
    
    #tf.debugging.experimental.enable_dump_debug_info(log_dir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()

    # create if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    
    return log_dir



def load_data(config):
    data_dir = config["data"]["data_dir"]
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")
    
    # Load data and data specific parameters
    # Load data and data specific parameters
    train_test_ratio = config["data"]["train_test_ratio"]
    X, Y = utils.get_spring_mass_damper_data(data_dir)
    x_train, x_test, y_train, y_test = utils.split_data(X, Y, train_test_ratio)

    y_train_shifted = utils.shift(y_train) # decoder input
    y_test_shifted = utils.shift(y_test) 

    # ensure returned values are TF tensors
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    # print info to console
    print(f"\nLoaded data from {data_dir}")
    print(f"Encoder input shape: {x_train.shape}")
    print(f"Decoder input and output shape: {y_train.shape}")
    print("Shapes are in order (N_sim, N_timesteps, N_dof) and (N_batch, N_timesteps, d_model) internally.\n")

    return x_train, y_train_shifted, y_train, x_test, y_test_shifted, y_test


def build_model(config, encoder_seq_length, decoder_seq_length, d_output):
    # Transformer architecture params
    d_model = config["architecture"]["d_model"]  # Dimensionality of the latent space
    n_stacks = config["architecture"]["N_stacks"]  # Depth of encoder and decoder layers
    n_heads = config["architecture"]["h"]  # Number of self-attention heads
    d_ff = config["architecture"]["d_ff"]  # Dimensionality of the inner fully connected layer
    activation_ff = config["architecture"]["activation_ff"]  # Activation function of the feed-forward module
    dropout_rate = config["architecture"]["dropout_rate"]  # Frequency of dropping the input units in the dropout layers

    # create model
    model = Transformer(encoder_seq_length, decoder_seq_length, n_heads, d_model, d_ff, activation_ff, d_output, n_stacks, dropout_rate)

    # setup optimizer
    # for beta and eps use default values as no hyperparameter tuning was done
    optimizer = Adam(LRScheduler(d_model)) # , beta_1, beta_2, epsilon)

    model.compile(optimizer=optimizer,
                loss="mse",
                metrics=["mse"],
                )
    
    print("Compiled model, can start training...")

    return model, optimizer


def train_model(config, optimizer):
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]  
    
   
    # setup callbacks - tensorboard, checkpoints, early stopping
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )

    checkpoint_dir = config["data"]["checkpoints_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    data_path_weights_filename = os.path.join(checkpoint_dir, "model_weights")
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

    model_ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
            data_path_weights_filename,
            monitor='loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )
    
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            verbose=1,
            restore_best_weights=True
        )

    callbacks = [model_ckpt_cb, tensorboard_cb, early_stopping_cb]

    # train model
    history = model.fit(x=[x_train, y_train_shifted],
                        y=y_train,
                        validation_data=([x_test, y_test_shifted], y_test),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        )

    print(history.history)

    # save model
    save_dir = config["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    tf.saved_model.save(model, save_dir)

    return model, history


def load_checkpoint(config, untrained_model):
    checkpoint_dir = config["data"]["checkpoints_dir"]
    
    model = tf.train.checkpoint.restore(checkpoint_dir)

    return model


#----

log_dir = setup_tf(config)
x_train, y_train_shifted, y_train, x_test, y_test_shifted, y_test = load_data(config)

n_seq, encoder_seq_length, d_input = x_train.shape
n_seq, decoder_seq_length, d_output = y_train.shape

model, optimizer = build_model(config, encoder_seq_length, decoder_seq_length, d_output)

model, history = train_model(config, optimizer)

# ----

# metrics monitoring
#train_loss = tf.keras.metrics.Mean(name='train_loss')


# train_data = (x_train, y_train_shifted, y_train)
# train_data = (x_train, y_train, y_train)


# create masks, propagate them through functional API
# mask_value = -1000
# masking_layer = tf.keras.layers.Masking(mask_value)

# masked_encoder_input = masking_layer(x_train)
# masked_decoder_input = masking_layer(y_train_shifted)
# masked_decoder_output = masking_layer(y_train)



# # %% Evaluate model
print("Evaluating model on test data...")
batch_size = config["training"]["batch_size"]  
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
