""""
Transformer model for a mass-spring-damper system

Implementation is roughly based on
https://machinelearningmastery.com/how-to-implement-multi-head-attention-from-scratch-in-tensorflow-and-keras
and Attention is All You Need
"""

# %% Imports
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from lib.encoder import Encoder
from lib.decoder import Decoder

class Transformer(Model):
    def __init__(self, encoder_seq_len, decoder_seq_len, h, d_model, d_ff, N, dropout_rate, **kwargs):
        super(Transformer, self).__init__(**kwargs)

        self.encoder = Encoder(encoder_seq_len, d_model, d_ff, h, N, dropout_rate, **kwargs)
        # self.encoder.encoder_layer.build_graph().summary()

        self.decoder = Decoder(decoder_seq_len, d_model, d_ff, h, N, dropout_rate, **kwargs)
        # self.decoder.encoder_layer.build_graph().summary()

        self.model_last_layer = Dense(decoder_seq_len)

    def padding_mask(self, input):
        # mask to filter out all invalid values resulting e.g. from shifting/padding the input
        # assume that if a datapoint within a sequence is invalid, its marked in [x, t, 0] as np.inf
        mask = tf.math.equal(input[:, :, 0], np.inf)
        mask = tf.cast(mask, tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :] #(batch_size, 1, 1, decoder_seq_len) - will be broadcased to axis=1,2

    def lookahead_mask(self, shape):
        # mask out subsequent data points in seq for autoregressive behaviour
        mask = 1 - tf.linalg.band_part(tf.ones((shape, shape)), -1, 0)
        return mask[tf.newaxis, tf.newaxis, :, : ]  # use broadcasting to scale to (batch_size, heads, seq_len, seq_len)

    def call(self, inputs, training=False):
        # unpack inputs
        encoder_input, decoder_input = inputs

        # create masks for attention
        enc_input_padding_mask = self.padding_mask(encoder_input)
        dec_input_padding_mask = self.padding_mask(decoder_input)

        dec_input_lookahead_mask = self.lookahead_mask(decoder_input.shape[1])
        dec_input_lookahead_mask = tf.maximum(dec_input_padding_mask, dec_input_lookahead_mask)

        # forward pass: encoder, decoder, dense output layer
        encoder_output = self.encoder(encoder_input, enc_input_padding_mask, training)
        decoder_output = self.decoder(decoder_input, encoder_output, dec_input_lookahead_mask, dec_input_padding_mask, training)
        model_output = self.model_last_layer(decoder_output)

        return model_output

    @tf.function
    def train_step(self, inputs):
        input, decoder_output = inputs
        encoder_input, decoder_input = input

        with tf.GradientTape() as tape:
            prediction = self([encoder_input, decoder_input], training=True) # Forward pass
            loss = self.compiled_loss(decoder_output, prediction) # loss

        gradients = tape.gradient(loss, self.trainable_weights)
        # Update the values of the trainable variables by gradient descent
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))  # noqa: B905
        return loss
