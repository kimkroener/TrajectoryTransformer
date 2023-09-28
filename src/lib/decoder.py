import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout, Dense
from lib.transformer_modules import AddNormalization, FeedForward, PositionalEncoding

class DecoderLayer(Layer):
    def __init__(self, seq_len, h, d_model, d_ff, activation_ff, dropout, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        # self.build(input_shape=[None, seq_len, d_model])
        self.d_model = d_model
        self.seq_len = seq_len
        self.multihead_attention = tf.keras.layers.MultiHeadAttention(num_heads=h, key_dim=2,**kwargs) #  = MultiHeadAttention(h, d_model)
        self.dropout = Dropout(dropout)
        self.add_norm = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model, activation_ff)

    # def build_graph(self):
    #     input_layer = Input(shape=(self.seq_len, self.d_model))
    #     return Model(inputs=[input_layer], outputs=self.call(input_layer, None, True))

    def call(self, x, encoder_output, training):
        print(f"Decoder input {x}")
        # normed masked self attention with dropout to masked out subsequent positions
        # causal self attention with lookahead mask for autorregressive behaviour
        multihead_output1 = self.multihead_attention(x, x, x)# query=x, key=x, value=x, use_causal_mask=True)  #self.multihead_attention(x, x, x, lookahead_mask)

        multihead_output1 = self.dropout(multihead_output1, training=training)
        addnorm_output1 = self.add_norm(x, multihead_output1)
        print(f"Masked self-attention {addnorm_output1}")
        # encoder-decoder cross attention with dropout
        multihead_output2 = self.multihead_attention(query = addnorm_output1, key=encoder_output, value=encoder_output) # , encoder_output, padding_mask)
        multihead_output2 = self.dropout(multihead_output2, training=training)
        addnorm_output2 = self.add_norm(addnorm_output1, multihead_output2)
      #  print(f"Encoder-decoder attention {addnorm_output2}")
        # feed forward
        feedforward_output = self.feed_forward(addnorm_output2)
        feedforward_output = self.dropout(feedforward_output, training=training)
        decoder_output = self.add_norm(addnorm_output2, feedforward_output)
       # print(f"Decoder output {decoder_output}")

        return decoder_output


class Decoder(Layer):
    def __init__(self, sequence_length, d_model, d_ff, activation_ff,  h, n_decoderlayer, dropout, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.embedding = Dense(d_model)
        self.pos_encoding = PositionalEncoding(sequence_length, d_model)
        self.dropout = Dropout(dropout)
        self.decoder_layer = [DecoderLayer(sequence_length, h, d_model, d_ff, activation_ff, dropout) for _ in range(n_decoderlayer)]

    def call(self, output_target, encoder_output, training):
        embedded_target = self.embedding(output_target)
        pos_encoding_output = self.pos_encoding(embedded_target)
        x = self.dropout(pos_encoding_output, training=training)

        print("Decoder layer ", end="")
        for i, layer in enumerate(self.decoder_layer):
            print(f"{i}..", end="")
            x = layer(x, encoder_output, training)
        print("")
        return x

