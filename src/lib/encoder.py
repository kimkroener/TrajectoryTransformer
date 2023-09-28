import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout, Dense
from lib.transformer_modules import AddNormalization, FeedForward, PositionalEncoding

class EncoderLayer(Layer):
    def __init__(self,
                 seq_len,
                 d_model, # dimension model
                 d_ff, # dimension hidden layer in FFN,
                 activation_ff,
                 dropout,
                 h, # n_heads in multi-head attention layer
                 **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        # self.build(input_shape=[None, seq_len, d_model])
        self.d_model = d_model
        self.seq_len = seq_len

        self.multihead_attention = tf.keras.layers.MultiHeadAttention(num_heads=h, key_dim=2,**kwargs) #  = MultiHeadAttention(h, d_model)
        self.dropout1 = Dropout(dropout)
        self.add_norm1 = AddNormalization()

        self.feed_forward = FeedForward(d_ff, d_model, activation_ff)
        self.dropout2 = Dropout(dropout)
        self.add_norm2 = AddNormalization()

    # def build_graph(self):
    #     input_layer = Input(shape=(self.seq_len, self.d_model))
    #     return Model(inputs=[input_layer], outputs=self.call(input_layer, None, True))

    def call(self, x, padding_mask, training):
        # latent shape between modules: (batch_size, sequence_length, d_model)

        # Global self attention layer with dropout
        multihead_output = self.multihead_attention(x, x, x) #, padding_mask)
        multihead_output = self.dropout1(multihead_output, training=training)
        addnorm_output = self.add_norm1(x, multihead_output)

        # position wise feed-forward with dropout
        feedforward_output = self.feed_forward(addnorm_output)
        feedforward_output = self.dropout2(feedforward_output, training=training)
        encoder_output = self.add_norm2(addnorm_output, feedforward_output)

        return encoder_output

# Encoder Stack
class Encoder(Layer):
    def __init__(self, sequence_length, d_model, d_ff, activation_ff, h,  n_encoderlayer, dropout, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.embedding = Dense(d_model)
        self.pos_encoding = PositionalEncoding(sequence_length, d_model)
        self.dropout = Dropout(dropout)
        self.encoder_layer = [EncoderLayer(sequence_length, d_model, d_ff, activation_ff, dropout, h) for _ in range( n_encoderlayer)]

    def call(self, input_sequence, padding_mask, training):
        # input embedding - scale from d_input to d_model
        embedded_input = self.embedding(input_sequence)

        # positional encoding with dropout
        pos_encoding_output = self.pos_encoding(embedded_input)
        x = self.dropout(pos_encoding_output, training=training)

        # Iterate over n_stack encoder layers
        print("Encoder layer ", end="")
        for i, layer in enumerate(self.encoder_layer):
            print(f"{i}..", end="")
            x = layer(x, padding_mask, training)

        print("")
        return x
