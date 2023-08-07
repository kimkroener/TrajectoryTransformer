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

# # %%
# def TransformerModel(Model):
#     #def __init__(self, encoder_seq_len, decoder_seq_len, h, d_model, d_ff, N, dropout_rate, **kwargs):
#     #    super(TransformerModel, self).__init__(**kwargs)

#     def __init__(self, encoder_seq_len, decoder_seq_len, h, d_model, d_ff, N, dropout_rate, **kwargs):
#         super(TransformerModel, self).__init__(**kwargs)

#         self.encoder = Encoder(encoder_seq_len, d_model, d_ff, h, N, dropout_rate, **kwargs)
#         self.decoder = Decoder(decoder_seq_len, d_model, d_ff, h, N, dropout_rate, **kwargs)

#         # final layer
#         self.model_last_layer = Dense(decoder_seq_len) # TODO Output size??

#     # def padding_mask(input):
#     #     """mark the zero padding values of the input with 1.
#     #     """
#     #     mask = tf.math.equal(input, 0)
#     #     mask = tf.cast(mask, tf.float32)
#     #     return mask

#     def lookahead_mask(shape):
#         # mask out subsequent entries by markinjg them with a 1.0
#         mask = 1 - tf.linalg.band_part(np.ones((shape, shape), dtype=tf.float32), -1, 0)
#         return mask


#     def call(self, encoder_input, decoder_input, training):
#         # setup mask for encoder
#         encoder_padding_mask = tf.zeros(tf.shape(encoder_input)) # self.padding_mask(encoder_input)

#         # combine masks for decoder
#         #decoder_input_padding_mask = padding_mask(decoder_input)
#         decoder_input_lookahead_mask = lookahead_mask(decoder_input.shape[1])
#         #decoder_input_lookahead_mask = tf.maximum(decoder_input_padding_mask, decoder_input_lookahead_mask)

#         # - feed input through the encoder and decoder, as well as final layer -
#         encoder_output = self.encoder(encoder_input, encoder_padding_mask, training)
#         decoder_output = self.decoder(decoder_input, encoder_output, decoder_input_lookahead_mask, encoder_padding_mask, training)
#         model_output = self.model_last_layer(decoder_output)

#         return model_output


# %%


class TransformerModel(Model):
    def __init__(self, encoder_seq_len, decoder_seq_len, h, d_model, d_ff, N, dropout_rate, **kwargs):
        super(TransformerModel, self).__init__(**kwargs)

        self.encoder = Encoder(encoder_seq_len, d_model, d_ff, h, N, dropout_rate, **kwargs)
        # self.encoder.encoder_layer.build_graph().summary()

        self.decoder = Decoder(decoder_seq_len, d_model, d_ff, h, N, dropout_rate, **kwargs)
        # self.decoder.encoder_layer.build_graph().summary()

        self.model_last_layer = Dense(decoder_seq_len)

    def padding_mask(self, input):
        # Create mask which marks the np.inf padding values in the input by a 1.0
        mask = tf.math.equal(input, np.inf)
        mask = tf.cast(mask, tf.float32)

        # The shape of the mask should be broadcastable to the shape
        # of the attention weights that it will be masking later on
        return mask[:, tf.newaxis, tf.newaxis, :]

    def lookahead_mask(self, shape):
        # Mask out future entries by marking them with a 1.0
        mask = 1 - tf.linalg.band_part(tf.ones((shape, shape)), -1, 0)

        return mask

    def call(self, inputs, training=False):
        encoder_input, decoder_input = inputs # unpack inputs

        # Create padding mask to mask the encoder inputs and the encoder outputs in the decoder
        enc_padding_mask = self.padding_mask(encoder_input)
        print(f"enc_padding_mask: {enc_padding_mask}")
        # Create and combine padding and look-ahead masks
        dec_in_padding_mask = self.padding_mask(decoder_input)
        print(f"dec input padding_mask: {dec_in_padding_mask}")
        dec_in_lookahead_mask = self.lookahead_mask(decoder_input.shape[1])
        print(f"Lookahead mask {dec_in_lookahead_mask}")
        #dec_in_lookahead_mask = tf.maximum(dec_in_padding_mask, dec_in_lookahead_mask)

        encoder_output = self.encoder(encoder_input, enc_padding_mask, training)

        decoder_output = self.decoder(decoder_input, encoder_output, dec_in_lookahead_mask, enc_padding_mask, training)
        
        model_output = self.model_last_layer(decoder_output)

        return model_output

    @tf.function
    def train_step(self, inputs): # encoder_input, decoder_input, decoder_output):
        
        encoder_input, decoder_input, decoder_output = inputs[0]
        
        print("Shapes in train_step:")
        print(encoder_input.shape)
        print(decoder_input.shape)
        print(decoder_output.shape)

        with tf.GradientTape() as tape:
            prediction = self([encoder_input, decoder_input], training=True) # Forward pass
            loss = self.compiled_loss(decoder_output, prediction) # loss

        gradients = tape.gradient(loss, self.trainable_weights)
        # Update the values of the trainable variables by gradient descent
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))  # noqa: B905
        return loss

# %%
