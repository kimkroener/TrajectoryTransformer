import os
import pickle

import numpy as np
import tensorflow as tf
import yaml
from sklearn.model_selection import train_test_split


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
