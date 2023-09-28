"""
Tensorflow Implementation for the transformer modules
    - Add and Norm layer
    - Multihead Attention
    - Feed-Forward Dense NN
    - Positional Encoding
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Add, LayerNormalization, Layer, Dense

# Feedforward Module
class FeedForward(Layer):
    """Dense Feed-Forward NN with one hidden layer

    Args:
        d_ff (Int): dimensionality of the hidden layer
        d_model (Int): model/output dimensionality
        activation_ff (str): activation function for the hidden layer "GELU" or "ReLU"
    """
    def __init__(self,
                 d_ff, # inner-layer dimensionality
                 d_model, # model dimensionality
                 activation_ff, # activation function "GELU" or "ReLU"
                 **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.fully_connected1 = Dense(d_ff)  # First fully connected layer
        self.fully_connected2 = Dense(d_model)  # Second fully connected layer
        self.activation = activation_ff
        

    def call(self, x):
        # FFN(x)= Activation(x W1 + b1)W2 + b2
        x = self.fully_connected1(x)

        if self.activation.lower() == "gelu": # Gaussian Error LU 
            x = tf.keras.activations.gelu(x, approximate=True)
        elif self.activation.lower() == "relu": 
            x = tf.keras.activations.relu(x)
        else:
            raise ValueError(f"Activation function {self.activation} not implemented. Use 'GELU' or 'ReLU'")

        return self.fully_connected2(x)

# Add & Norm Layer
class AddNormalization(Layer):
    """Add input and output of a sublayer and apply layer normalization

    """
    def __init__(self, **kwargs):
        super(AddNormalization, self).__init__(**kwargs)
        self.layer_norm = LayerNormalization()
        self.add = Add() # use keras layer instead of + operator to ensure kears masks are propagated

    def call(self, x, sublayer_x):
        added = self.add([x, sublayer_x])
        # Apply layer normalization to the sum
        return self.layer_norm(added)

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

        pe = pe[tf.newaxis, :, :] # expected dim (1, seq_len, d_model)
        self.pe = tf.Variable(pe, trainable=False)

    def call(self, x):
        # return x + tf.tile(self.pe, [batch_size, 1, 1])
        return x + self.pe  # shape (batch_size, seq_len, d_model) + (1, seq_len, d_model) -> element-wise broadcasting


# Multihead Attention
class BaseAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.multiheadattention = tf.keras.layers.MultiHeadAttention(**kwargs)

class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.multiheadattention(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True
        )
    
        # cache attention scores for plotting
        self.last_attention_scores = attn_scores
        return attn_output

class SelfAttention(BaseAttention):
    def call(self, x):
        attn_output, attn_scores = self.multiheadattention(
            query=x,
            key=x,
            value=x,
            return_attention_scores=True
        )
    
        # cache attention scores for plotting
        self.last_attention_scores = attn_scores
        return attn_output
    

# # Multi-Head Attention Module
# class ScaledDotProductAttention(Layer):
#     """Compute the scaled dot product attention

#     Args:
#         queries:
#         keys:
#         values
#         d_k: dimensionality of queries/keys
#     """
#     def __init__(self, **kwargs):
#         super(ScaledDotProductAttention, self).__init__(**kwargs)

#     def call(self, queries, keys, values, d_k, mask=None):

#         # expected shape (batch_size, h, d_q, d_k)
#         scores = tf.matmul(queries, keys, transpose_b=True) / tf.math.sqrt(tf.cast(d_k, tf.float32))
#         # Apply mask to the attention scores
#         if mask is not None: # expected shape of mask (batch_size, 1, 1, d_k)
#             print(f"Mask {mask}")

#             scores = tf.where(mask!=1, scores, -1e9) # where condition is zero, broacast -1e9 to scores
#             nanmask = tf.math.is_nan(scores)
#             scores = tf.where(~nanmask, scores, -1e9)
#         print(f"Attention scores with mask {scores}")
#         # expected shape (batch_size, h, d_k, d_k)
#         weights = tf.nn.softmax(scores, axis=-1)
#         print(f"Attention weights: {weights}")
#         # Computing the attention by a weighted sum of the value vectors
#         return tf.matmul(weights, values)

# # Multi-Head Attention
# class MultiHeadAttention(Layer):
#     """Compute the multi-head attentionl layer

#     Args:
#         h (Int): number of heads
#         d_model: dimensionality of the model
#     """
#     def __init__(self, h, d_model, **kwargs):
#         super(MultiHeadAttention, self).__init__(**kwargs)
#         #print(f"Assert in multihead attention: {(d_model, h)}")
#         assert d_model % h == 0 # infere d_k, d_v from model params

#         self.attention = ScaledDotProductAttention()
#         self.heads = h  # num attention heads
#         self.d_model = d_model  # model dimensionality

#         # dimensions
#         self.d_k = d_model // h
#         self.d_v = d_model // h

#         # learnable weight matrices
#         self.W_q = Dense(self.d_k)
#         self.W_k = Dense(self.d_k)
#         self.W_v = Dense(self.d_v)
#         self.W_o = Dense(d_model)

#     def split_heads(self, x, heads, splitIntoHeads):
#         # shape of x: (batch_size, seq_len, d_{v,k}) -> weight matrices applied
#         if splitIntoHeads:
#             # reshape to (batch_size, heads, seq_length, depth=d_{v,k}/heads)
#             # print(f"Before: batch_size, seq_len, d_model? {x.shape}")
#             x = tf.reshape(x, shape=(tf.shape(x)[0], tf.shape(x)[1], heads, -1))
#             x = tf.transpose(x, perm=(0, 2, 1, 3))
#             # print(f"Split into batch_size, heads, seq_length, d_v,k/heads)? {x.shape}")
#         else:
#             # Reverting the reshaping for concat: (batch_size, seq_length, d_model)
#             x = tf.transpose(x, perm=(0, 2, 1, 3))
#             x = tf.reshape(x, shape=(tf.shape(x)[0], tf.shape(x)[1], self.d_k))
#             #print(f"Concat heads (batch_size, seq_length, d_k)? {x.shape}")
#         return x

#     def call(self, queries, keys, values, mask=None):
#         # Shape of queries, keys, values: (batch_size, seq_len, d_model)
#         Q = self.split_heads(self.W_q(queries), self.heads, True)
#         K = self.split_heads(self.W_k(keys), self.heads, True)
#         V = self.split_heads(self.W_v(values), self.heads, True)

#         # Compute the multi-head attention output using the reshaped Q, K, V to scale it to d_model
#         output_split = self.attention(Q, K, V, self.d_k, mask)

#         # Concat output
#         output = self.split_heads(output_split, self.heads, False) # (batch_size, input_seq_length, d_v)
#         output_dmodel = self.W_o(output)

#         # final linear projection
#         return output_dmodel
