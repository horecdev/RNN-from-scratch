import numpy as np

from core import RNN, SoftmaxCrossEntropy

t = np.linspace(0, 10, 50) # (50,)
data = np.sin(t) # (50,)

X = data[:-1].reshape(1, -1, 1) # (B, seq_len, input_dim) = (1, 49, 1)
Y = data[1:].reshape(1, -1, 1) # (B, seq_len, input_dim) = (1, 49, 1)

input_dim = 1
hidden_dim = 128
output_dim = 1

model = RNN(input_dim, hidden_dim, output_dim)

