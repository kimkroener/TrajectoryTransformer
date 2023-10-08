import os
import pickle

import numpy as np
import tensorflow as tf
import yaml
from sklearn.model_selection import train_test_split

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from transformer import Transformer
from lib.scheduler import LRScheduler

# %%
def get_occupant_data(reduced=True):
    with open('../../data/occupant_data_transformer.pkl', 'rb') as f:
        data = pickle.load(f)
        displacements_train = data['displacements_train']
        displacements_test = data['displacements_test']
        displacements_red_train = data['displacements_red_train']
        displacements_red_test = data['displacements_red_test']
        params_train = data['params_train']
        params_test = data['params_test']
        times_train = data['times_train']
        times_test = data['times_test']
        reference_coordinates_train = data['reference_coordinates_train']
        reference_coordinates_test = data['reference_coordinates_test']
        projection_matrix = data['projection_matrix']

    if reduced:
        return displacements_red_train, displacements_red_test
    else:
        return displacements_train, displacements_test


def get_spring_mass_damper_data(dataset_dir: str):
    """Load train and test data of mass-spring-damper system

    Args:
        dataset_dir (str): e.g. "../data/SISO_three-masses"

    """
    u = np.load(f"{dataset_dir}u.npy")
    x = np.load(f"{dataset_dir}x.npy")

    if len(u.shape) == 2:
        u = u[:, :, tf.newaxis] # shape should be (N_seq, T, n_dof)
    if len(x.shape) == 2:
        x = x[:, :, tf.newaxis]

    # DEBUGGING - use all timesteps!!
    #print("\nOnly using first 5 timesteps and one dof for debugging\n")
    return u[:, :, :], x[:, :, :]


def split_data(X, Y, ratio):
    """Split data into training, test and if needed validation dataset.

    Args:
        X (_type_): Input
        Y (_type_): Output
        ratio (_type_): ratio to split. If len()==2 only split into train and test.

    Returns:
        _type_: x_train, x_test, x_val, y_train, y_test, y_val
    """
    assert np.sum(ratio) == 1

    if len(ratio) == 2: # split into train and test
        train, test = ratio
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1-train) # split in (train, test+val)
        return x_train, x_test, y_train, y_test

    elif len(ratio) == 3:
        train, test, val = ratio
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1-train) # split in (train, test+val)
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test/(test+val)) # split (test, val)
        return x_train, x_test, x_val, y_train, y_test, y_val


def shift(X):
    """shift input of the decoder
    here: shift each time series by one timestep and append a np.inf at the beginning as a mark for the padding mask
    """
    N, T, d_input = X.shape
    start = tf.zeros((N, 1, d_input))
    shifted_input = X[:, :-1, :]
    return tf.concat([start, shifted_input], axis=1)

def load_config(file_path):
    dir = os.getcwd()
    with open(file_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config



def load_data(config):
    data_dir = config["data"]["data_dir"]
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    # Load data and data specific parameters
    train_test_val_ratio = config["data"]["train_test_ratio"]
    X, Y = get_spring_mass_damper_data(data_dir)
    x_train, x_test, x_val, y_train, y_test, y_val = split_data(X, Y, train_test_val_ratio)

    y_train_shifted = shift(y_train) # decoder input
    y_test_shifted = shift(y_test)
    y_val_shifted = shift(y_val)

    # ensure returned values are TF tensors
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

    # print info to console
    print(f"\nLoaded data from {data_dir}")
    print(f"Encoder input shape: {x_train.shape}")
    print(f"Decoder input and output shape: {y_train.shape}")
    print("Shapes are in order (N_sim, N_timesteps, N_dof) and (N_batch, N_timesteps, d_model) internally.\n")

    return x_train, y_train_shifted, y_train, x_test, y_test_shifted, y_test, x_val, y_val_shifted, y_val



def build_transformer(config, encoder_seq_length, decoder_seq_length, d_output):
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
    optimizer = tf.keras.optimizers.Adam(LRScheduler(d_model)) # , beta_1, beta_2, epsilon)

    model.compile(optimizer=optimizer,
                loss="mse",
                metrics=["mse"],
                )

    print("Model compield.")

    return model, optimizer



def print_timedelta(start_time, end_time):
    """print elapsed time in format HH:MM:SS

    Args:
        start_time (datetime): start time
        end_time (datetime): end time
    """
    timedelta = end_time - start_time
    hours, remainder = divmod(timedelta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Elapsed time: {hours} hours, {minutes} minutes, {seconds} seconds")


def plot_train_and_val_loss(history, model_dir):
    """Plot training and validation loss

    Args:
        history (tf.keras.callbacks.History): history object from model.fit()
        model_dir (str): path to model directory to save plot
    """
    import matplotlib.pyplot as plt

    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.title('Model loss')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(f"{model_dir}/loss.png")
    plt.savefig(f"{model_dir}/loss.svg")
    plt.show()

    
def plot_train_val_loss_plotly(history: dict, model_dir: str):
    """plot training and validation loss with plotly

    Args:
        history (dict): history dict from model.fit() (history.history)
        model_dir (str): path to model directory to save plot
    """
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": False}]])

    # Add traces
    fig.add_trace(
        go.Scatter( y=history['val_loss'], name="val_loss"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter( y=history['loss'], name="loss"),
        secondary_y=False,
    )
    # Add figure title
    fig.update_layout(
        title_text="Loss of Transformer Model during Training"
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss (MSE)", secondary_y=False)

    fig.show()
