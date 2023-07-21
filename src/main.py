# %% Imports
from msd_transformer_tf import TransformerModel
from encoder import EncoderLayer
from decoder import DecoderLayer

import numpy as np
from numpy import random
import matplotlib.pyplot as plt

# %% load data 

u_train = np.load("../data/SISO_three-masses/u_train.npy")
u_test = np.load("../data/SISO_three-masses/u_test.npy")
x_train = np.load("../data/SISO_three-masses/x_train.npy")
x_test = np.load("../data/SISO_three-masses/x_test.npy")
print("Training data loaded...")
# %% Setup Transformer model 
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

# %% Show Parameters

# %%
training_model = TransformerModel(encoder_seq_length, decoder_seq_length, h, d_model, d_ff, N, dropout_rate)

# zero padding so each input has the length of max_seq, even though the zero padding will not be processed due to the padding mask

# %%
def loss_fcn(target, prediction):
    # Create mask so that the zero padding values are not included in the computation of loss
    padding_mask = math.logical_not(equal(target, 0))
    padding_mask = cast(padding_mask, float32)

    # Compute a sparse categorical cross-entropy loss on the unmasked values
    loss = sparse_categorical_crossentropy(target, prediction, from_logits=True) * padding_mask

    # Compute the mean loss over the unmasked values
    return reduce_sum(loss) / reduce_sum(padding_mask)

def accuracy_fcn(target, prediction):
    # Create mask so that the zero padding values are not included in the computation of accuracy
    padding_mask = math.logical_not(math.equal(target, 0))

    # Find equal prediction and target values, and apply the padding mask
    accuracy = equal(target, argmax(prediction, axis=2))
    accuracy = math.logical_and(padding_mask, accuracy)

    # Cast the True/False values to 32-bit-precision floating-point numbers
    padding_mask = cast(padding_mask, float32)
    accuracy = cast(accuracy, float32)

    # Compute the mean accuracy over the unmasked values
    return reduce_sum(accuracy) / reduce_sum(padding_mask)

class LRScheduler(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super(LRScheduler, self).__init__(**kwargs)

        self.d_model = cast(d_model, float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step_num):

        # Linearly increasing the learning rate for the first warmup_steps, and decreasing it thereafter
        arg1 = step_num ** -0.5
        arg2 = step_num * (self.warmup_steps ** -1.5)

        return (self.d_model ** -0.5) * math.minimum(arg1, arg2)
    
optimizer = Adam(LRScheduler(d_model), beta_1, beta_2, epsilon)

train_dataset = data.Dataset.from_tensor_slices((trainX, trainY))
train_dataset = train_dataset.batch(batch_size)

@function
def train_step(encoder_input, decoder_input, decoder_output):
    with GradientTape() as tape:

        # Run the forward pass of the model to generate a prediction
        prediction = training_model(encoder_input, decoder_input, training=True)

        # Compute the training loss
        loss = loss_fcn(decoder_output, prediction)

        # Compute the training accuracy
        accuracy = accuracy_fcn(decoder_output, prediction)

    # Retrieve gradients of the trainable variables with respect to the training loss
    gradients = tape.gradient(loss, training_model.trainable_weights)

    # Update the values of the trainable variables by gradient descent
    optimizer.apply_gradients(zip(gradients, training_model.trainable_weights))

    train_loss(loss)
    train_accuracy(accuracy)

train_loss = Mean(name='train_loss')
train_accuracy = Mean(name='train_accuracy')

# Create a checkpoint object and manager to manage multiple checkpoints
ckpt = train.Checkpoint(model=training_model, optimizer=optimizer)
ckpt_manager = train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=3)

for epoch in range(epochs):

    train_loss.reset_states()
    train_accuracy.reset_states()

    print("\nStart of epoch %d" % (epoch + 1))

    # Iterate over the dataset batches
    for step, (train_batchX, train_batchY) in enumerate(train_dataset):

        # Define the encoder and decoder inputs, and the decoder output
        encoder_input = train_batchX[:, 1:]
        decoder_input = train_batchY[:, :-1]
        decoder_output = train_batchY[:, 1:]

        train_step(encoder_input, decoder_input, decoder_output)

        if step % 50 == 0:
            print(f'Epoch {epoch + 1} Step {step} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
           
    # Print epoch number and loss value at the end of every epoch
    print("Epoch %d: Training Loss %.4f, Training Accuracy %.4f" % (epoch + 1, train_loss.result(), train_accuracy.result()))

    # Save a checkpoint after every five epochs
    if (epoch + 1) % 5 == 0:
        save_path = ckpt_manager.save()
        print("Saved checkpoint at epoch %d" % (epoch + 1))
