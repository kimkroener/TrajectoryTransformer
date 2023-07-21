
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dropout, Input
from transformer_modules import MultiHeadAttention, AddNormalization, FeedForward, PositionalEncoding

class DecoderLayer(Layer):
    def __init__(self, seq_len, h, d_model, d_ff, dropout, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.build(input_shape=[None, seq_len, d_model])
        self.d_model = d_model
        self.seq_len = seq_len
        self.multihead_attention = MultiHeadAttention(h, d_model)
        self.dropout = Dropout(dropout)
        self.add_norm = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
    
    def build_graph(self):
        input_layer = Input(shape=(self.seq_len, self.d_model))
        return Model(inputs=[input_layer], outputs=self.call(input_layer, None, True))

    def call(self, x, encoder_output, lookahead_mask, padding_mask, training):
        
        # normed masked self attention with dropout to masked out subsequent positions
        multihead_output1 = self.multihead_attention(x, x, x, lookahead_mask)
        multihead_output1 = self.dropout(multihead_output1, training=training)
        addnorm_output1 = self.add_norm(x, multihead_output1)
        
        # encoder-decoder attention with dropout
        multihead_output2 = self.multihead_attention(addnorm_output1, encoder_output, encoder_output, padding_mask)
        multihead_output2 = self.dropout(multihead_output2, training=training)
        addnorm_output2 = self.add_norm(addnorm_output1, multihead_output2)

        # feed forward
        feedforward_output = self.feed_forward(addnorm_output2)
        feedforward_output = self.dropout(feedforward_output, training=training)
        decoder_output = self.addnorm(addnorm_output2, feedforward_output)

        return decoder_output


class Decoder(Layer):
    def __init__(self, sequence_length, d_model, d_ff, h, N, dropout, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.pos_encoding = PositionalEncoding(sequence_length, d_model)
        self.droput = Dropout(dropout)
        self.decoder_layer = [DecoderLayer(sequence_length, h, d_model, d_ff, dropout) for _ in range(N)]

    def call(self, output_target, encoder_output, lookahead_mask, padding_mask, training):
        pos_encoding_output = self.pos_encoding(output_target)
        x = self.dropout(pos_encoding_output, training=training)

        for i, layer in enumerate(self.decoder_layer):
            x = layer(x, encoder_output, lookahead_mask, padding_mask, training)

        return x
    
