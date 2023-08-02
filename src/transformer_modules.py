"""
Tensorflow Implementation for the transformer modules
    - Add and Norm
    - Multihead Attention
    - Feed-Forward Dense NN
    - Positional Encoding
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Layer, Dense, ReLU

# Feedforward Module
class FeedForward(Layer):
    """Dense Feed-Forward NN with one hidden layer

    Args:
        d_ff (Int): dimensionality of the hidden layer
        d_model (Int): model/output dimensionality
    """
    def __init__(self,
                 d_ff, # inner-layer dimensionality
                 d_model, # model dimensionality
                 **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.fully_connected1 = Dense(d_ff)  # First fully connected layer
        self.fully_connected2 = Dense(d_model)  # Second fully connected layer
        self.activation = ReLU()  # ReLU activation layer

    def call(self, x):
        # FFN(x)= ReLU(x W1 + b1)W2 + b2
        x_fullyconnected1 = self.fully_connected1(x)

        return self.fully_connected2(self.activation(x_fullyconnected1))

# Add & Norm Layer
class AddNormalization(Layer):
    """Add input and output of a sublayer and apply layer normalization

    """
    def __init__(self, **kwargs):
        super(AddNormalization, self).__init__(**kwargs)
        self.layer_norm = LayerNormalization()

    def call(self, x, sublayer_x):
        add = x + sublayer_x

        # Apply layer normalization to the sum
        return self.layer_norm(add)

# Positional Encoding
class PositionalEncoding(Layer):
    """Compute positional encoding based on the input vector

    Args:
        seq_len (Int): length of the input sequence
        d_model (Int): dimensionality of the model
    """
    def __init__(self, seq_len, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)

        # create the position enconding matrix
        pe = np.zeros((seq_len, d_model), dtype=np.float32)
        for k in range(seq_len):
            for i in np.arange(int(d_model/2)):
                freq = 1./np.power(10000, 2*i/d_model)
                pe[k, 2*i] = np.sin(k*freq)
                pe[k, 2*i+1] = np.cos(k*freq)

        self.pe = tf.Variable(pe, trainable=False)

    def call(self, x):
        return x + self.pe[:x.shape[0], :]



# Multi-Head Attention Module
class ScaledDotProductAttention(Layer):
    """Compute the scaled dot product attention

    Args:
        queries:
        keys:
        values
        d_k: dimensionality of queries/keys
    """
    def __init__(self, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def call(self, queries, keys, values, d_k, mask=None):

        scores = tf.matmul(queries, keys, transpose_b=True) / tf.math.sqrt(tf.cast(d_k, tf.float32))

        # Apply mask to the attention scores
        if mask is not None:
            scores += -1e9 * mask

        weights = tf.softmax(scores)

        # Computing the attention by a weighted sum of the value vectors
        return tf.matmul(weights, values)

# Multi-Head Attention
class MultiHeadAttention(Layer):
    """Compute the multi-head attentionl layer

    Args:
        h (Int): number of heads
        d_model: dimensionality of the model
    """
    def __init__(self, h, d_model, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        print(f"Assert in multihead attention: {(d_model, h)}")
        assert d_model % h == 0 # infere d_k, d_v from model params

        self.attention = ScaledDotProductAttention()
        self.heads = h  # num attention heads
        self.d_model = d_model  # model dimensionality

        # projection matrices - queries, keys, values, output
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.W_q = Dense(self.d_k)
        self.W_k = Dense(self.d_k)
        self.W_v = Dense(self.d_v)
        self.W_o = Dense(d_model)

    def split_heads(self, x, heads, splitIntoHeads):
        if splitIntoHeads:
            # reshape to (batch_size, heads, seq_length, {d_v, d_k}/heads)
            x = tf.reshape(x, shape=(tf.shape(x)[0], tf.shape(x)[1], heads, -1))
            x = tf.transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping for concat: (batch_size, seq_length, d_k)
            x = tf.transpose(x, perm=(0, 2, 1, 3))
            x = tf.reshape(x, shape=(tf.shape(x)[0], tf.shape(x)[1], self.d_k))
        return x

    def call(self, queries, keys, values, mask=None):
        # Rearrange the queries to be able to compute all heads in parallel
        q_split = self.split_heads(self.W_q(queries), self.heads, True)
        k_split = self.split_heads(self.W_k(keys), self.heads, True)
        v_split = self.split_heads(self.W_v(values), self.heads, True)

        # Compute the multi-head attention output using the reshaped Q, K, V
        o_split = self.attention(q_split, k_split, v_split, self.d_k, mask)

        # Concat output
        output = self.split_heads(o_split, self.heads, False) # (batch_size, input_seq_length, d_v)

        # final linear projection
        return self.W_o(output)
