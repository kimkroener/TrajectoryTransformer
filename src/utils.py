import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import yaml
# %%
def load_data(dataset_dir: str):
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
    print("\nOnly using first 10 timesteps for debugging\n")
    return u[:, :10, :], x[:, :10, :1]


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
    N, T, dof = X.shape
    start = tf.ones((N, 1, dof))*np.inf
    shifted_input = X[:, :-1, :]
    return tf.concat([start, shifted_input], axis=1)

def load_config(file_path):
    with open(file_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config
    