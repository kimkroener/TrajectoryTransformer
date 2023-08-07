# %%
from encoder import Encoder
from lib.decoder import Decoder
from lib.transformer_modules import PositionalEncoding, MultiHeadAttention
from lib.scheduler import LRScheduler

import matplotlib.pyplot as plt
import numpy.random as random
import numpy as np
#%matplotlib inline

# %%
"""
Test Positional Encoding
"""
def test_pe():
    n_T = 50 # timesteps
    d_model = 128 # model complexity
    PE = PositionalEncoding(seq_len=n_T, d_model=d_model)
    print(f"PE shape: {PE.pe.shape}")
    plt.figure()
    #plt.xlabel = "sequence position; timesteps"
    #plt.ylabel = "model dimension"
    plt.title("Positional Encoding Test")
    plt.imshow(PE.pe)
    plt.savefig("../tests/pe_test.svg")
    plt.show()

    print(PE.pe.dtype)

    print("Positional Encoding plot generated.")

"""
Test Multihead-Attention
"""
def test_attention():
    h = 5
    d_model=50
    attn = MultiHeadAttention(h=h, d_model=d_model)
    
    batch_size = 5
    n_T = 50

    queries = random.random((batch_size, n_T, attn.d_k))
    keys = random.random((batch_size, n_T, attn.d_k))
    values = random.random((batch_size, n_T, attn.d_v))

    a = attn(queries, keys, values)
    assert a.shape == (batch_size, n_T, d_model)

    print("Multihead Attention has expected output dimensions. ")
    

"""
Test encoder
"""
def test_encoder():
    n_T = 50
    h = 5
    d_ff = 16
    d_model = 50
    N = 2

    batch_size = 5
    dropout_rate = 0.1

    input_seq = random.random((batch_size, n_T))

    encoder = Encoder(n_T, d_model, d_ff, h, N, dropout_rate)
    print(encoder(input_seq, None, True))


# def test_masking():
#     input = np.array([0, 1, 2, 3, 1, 0, 0, 0])
#     print(padding_mask(input))
    
# %%
def test_lrscheduler():
    lr_scheduler = LRScheduler(d_model=64, warmup_steps=4000)

    n_points = 1000
    steps = np.linspace(0, 80000, n_points)
    lr_rate = np.empty(n_points, dtype=np.float32)

    for i in np.arange(n_points):
        lr_rate[i] = lr_scheduler(steps[i])

    plt.figure()
    #plt.xlabel = "sequence position; timesteps"
    #plt.ylabel = "model dimension"
    plt.title("Learning rate")
    plt.plot(steps, lr_rate)
    plt.savefig("../tests/lr_rate.svg")
    plt.show()
    print("LR Rate plot generated.")

    


# %%
test_pe()
#test_attention()
#test_encoder()
#test_lrscheduler()
# %%