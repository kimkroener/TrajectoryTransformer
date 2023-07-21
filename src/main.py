# %% Imports
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from time import time

import tensorflow as tf 
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam


from msd_transformer_tf import TransformerModel
from scheduler import LRScheduler

def loss_fcn(target, prediction):
    return  tf.keras.losses.mean_squared_error(target, prediction)


# %% TranSetupSetupsformer model parameters
# as in Attention is all you need
h = 8  # Number of self-attention heads
d_ff = 2048  # Dimensionality of the inner fully connected layer
d_model = 512  # Dimensionality of the model sub-layers' outputs
N = 6  # Number of layers in the encoder and decoder stack
 
batch_size = u_train.shape[0]  # Batch size from the training process
dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers

encoder_seq_length = u_train.shape[1] # length of input series
decoder_seq_length = x_train.shape[1]
input_seq = u_train
#input_seq = random.random((batch_size, input_seq_length))
print(f"Input dimension: {input_seq.shape}")

# training parameters
epochs = 2
batch_size = 64
beta_1 = 0.9
beta_2 = 0.98
epsilon = 1e-9
dropout_rate = 0.1

# Instantziate optimizer
optimizer = Adam(LRScheduler(d_model), beta_1, beta_2, epsilon)

# create model
training_model = TransformerModel(encoder_seq_length, decoder_seq_length, h, d_model, d_ff, N, dropout_rate)

train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))
train_dataset = train_dataset.batch(batch_size)


# Include metrics monitoring
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')


# Create a checkpoint object and manager to manage multiple checkpoints
ckpt = train.Checkpoint(model=training_model, optimizer=optimizer)
ckpt_manager = train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=3)

# Speeding up the training process
@function
def train_step(encoder_input, decoder_input, decoder_output):
    with tf.GradientTape() as tape:

        # Run the forward pass of the model to generate a prediction
        prediction = training_model(encoder_input, decoder_input, training=True)

        # Compute the training loss
        loss = loss_fcn(decoder_output, prediction)

        # Compute the training accuracy
        #accuracy = accuracy_fcn(decoder_output, prediction)

    # Retrieve gradients of the trainable variables with respect to the training loss
    gradients = tape.gradient(loss, training_model.trainable_weights)

    # Update the values of the trainable variables by gradient descent
    optimizer.apply_gradients(zip(gradients, training_model.trainable_weights))

    train_loss(loss)
    #train_accuracy(accuracy)


for epoch in range(epochs):

    train_loss.reset_states()
    train_accuracy.reset_states()

    print("\nStart of epoch %d" % (epoch + 1))

    start_time = time()

    # Iterate over the dataset batches
    for step, (train_batchX, train_batchY) in enumerate(train_dataset):

        # Define the encoder and decoder inputs, and the decoder output
        encoder_input = train_batchX[:, 1:]
        decoder_input = train_batchY[:, :-1]
        decoder_output = train_batchY[:, 1:]

        train_step(encoder_input, decoder_input, decoder_output)

        if step % 50 == 0:
            print(f'Epoch {epoch + 1} Step {step} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
            # print("Samples so far: %s" % ((step + 1) * batch_size))

    # Print epoch number and loss value at the end of every epoch
    print("Epoch %d: Training Loss %.4f" % (epoch + 1, train_loss.result()))#, Training Accuracy %.4f" % (epoch + 1, train_loss.result(), train_accuracy.result()))

    # Save a checkpoint after every five epochs
    if (epoch + 1) % 5 == 0:
        save_path = ckpt_manager.save()
        print("Saved checkpoint at epoch %d" % (epoch + 1))
