# Trajectory-Transformer

Predict trajectories resulting from an exitation time series. 

## Directory Structure

This repo is structed as following: 
- `data/`: contains the data used for training and testing
- `models/`: contains the trained models
- `src/`: contains the source code for the time-series transformer implemented in Tensorflow 2.0 and Keras. 
    - `src/models/`: contains config scripts for the models - including hyperparameters defining the model architecture and training parameters
- `tests/`: contains small code snippets for testing modules of the implementation