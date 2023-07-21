""""
Transformer model for a mass-spring-damper system

Implementation based on 
https://machinelearningmastery.com/how-to-implement-multi-head-attention-from-scratch-in-tensorflow-and-keras
and Attention is All You Need
"""

# %% Imports
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow import math, cast, float32, linalg, ones, maximum, newaxis
from encoder import Encoder
from decoder import Decoder

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
#         mask = 1 - tf.linalg.band_part(np.ones((shape, shape), dtype=np.float32), -1, 0)
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

        # Set up the encoder
        self.encoder = Encoder(encoder_seq_len, d_model, d_ff, h, N, dropout_rate, **kwargs)
        self.encoder.encoder_layer.build_graph().summary()

        # Set up the decoder
        self.decoder = Decoder(decoder_seq_len, d_model, d_ff, h, N, dropout_rate, **kwargs)
        self.decoder.encoder_layer.build_graph().summary()

        # Define the final dense layer
        self.model_last_layer = Dense(decoder_seq_len)

    def padding_mask(self, input):
        # Create mask which marks the zero padding values in the input by a 1.0
        mask = math.equal(input, 0)
        mask = cast(mask, float32)

        # The shape of the mask should be broadcastable to the shape
        # of the attention weights that it will be masking later on
        return mask[:, newaxis, newaxis, :]

    def lookahead_mask(self, shape):
        # Mask out future entries by marking them with a 1.0
        mask = 1 - linalg.band_part(ones((shape, shape)), -1, 0)

        return mask

    def call(self, encoder_input, decoder_input, training):

        # Create padding mask to mask the encoder inputs and the encoder outputs in the decoder
        enc_padding_mask = self.padding_mask(encoder_input)

        # Create and combine padding and look-ahead masks to be fed into the decoder
        dec_in_padding_mask = self.padding_mask(decoder_input)
        dec_in_lookahead_mask = self.lookahead_mask(decoder_input.shape[1])
        dec_in_lookahead_mask = maximum(dec_in_padding_mask, dec_in_lookahead_mask)

        # Feed the input into the encoder
        encoder_output = self.encoder(encoder_input, enc_padding_mask, training)

        # Feed the encoder output into the decoder
        decoder_output = self.decoder(decoder_input, encoder_output, dec_in_lookahead_mask, enc_padding_mask, training)

        # Pass the decoder output through a final dense layer
        model_output = self.model_last_layer(decoder_output)

        return model_output
# %%
