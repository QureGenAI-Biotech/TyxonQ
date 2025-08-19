"""
pytorch is excellent to use together with tq, we will have unique features including:
1. turn OO paradigm to functional paradigm, i.e. reuse pytorch layer function in functional programming
2. batch on neural network weights
"""

import os

import torch
import torch.nn as nn
import numpy as np
import tyxonq as tq

K = tq.set_backend("pytorch")

batch = 8
n = 6
layer = nn.Linear(n, 1)
layer.eval()  # Set to evaluation mode for stateless operation

data_x = np.random.choice([0, 1], size=batch * n).reshape([batch, n])
# data_y = np.sum(data_x, axis=-1) % 2
data_y = data_x[:, 0]
data_y = data_y.reshape([batch, 1])
data_x = data_x.astype(np.float32)
data_y = data_y.astype(np.float32)


print("data", data_x, data_y)


def loss(xs, ys, params, weights):
    c = tq.Circuit(n)
    c.rx(range(n), theta=xs)
    c.cx(range(n - 1), range(1, n))
    c.rz(range(n), theta=params)
    outputs = K.stack([K.real(c.expectation_ps(z=[i])) for i in range(n)])
    
    # Use PyTorch functional approach
    with torch.no_grad():
        ypred = torch.sigmoid(torch.matmul(outputs, weights[0]) + weights[1])
    
    return torch.nn.functional.binary_cross_entropy(ypred, ys), ypred


# common data batch practice
# warning pytorch might be unable to do this exactly
vgf = K.jit(
    K.vectorized_value_and_grad(
        loss, argnums=(2, 3), vectorized_argnums=(0, 1), has_aux=True
    )
)

params = torch.nn.Parameter(torch.randn(n))
w = torch.nn.Parameter(torch.randn(n, 1))
b = torch.nn.Parameter(torch.randn(1))
optimizer = torch.optim.Adam([params, w, b], lr=1e-2)

for i in range(100):
    # Forward pass
    (v, yp), gs = vgf(data_x, data_y, params, [w, b])
    
    # Backward pass
    optimizer.zero_grad()
    v.backward()
    optimizer.step()
    
    if i % 10 == 0:
        print(torch.mean(v))

# Calculate accuracy
with torch.no_grad():
    yp_binary = (yp > 0.5).float()
    acc = (yp_binary == data_y).float().mean()
    print("acc", acc.item())


# data batch with batched and quantum neural weights

# warning pytorch might be unable to do this exactly
vgf2 = K.jit(
    K.vmap(
        K.vectorized_value_and_grad(
            loss, argnums=(2, 3), vectorized_argnums=(0, 1), has_aux=True
        ),
        vectorized_argnums=(2, 3),
    )
)

wbatch = 4
params = torch.nn.Parameter(torch.randn(wbatch, n))
w = torch.nn.Parameter(torch.randn(wbatch, n, 1))
b = torch.nn.Parameter(torch.randn(wbatch, 1))
optimizer = torch.optim.Adam([params, w, b], lr=1e-2)

for i in range(100):
    # Forward pass
    (v, yp), gs = vgf2(data_x, data_y, params, [w, b])
    
    # Backward pass
    optimizer.zero_grad()
    v.backward()
    optimizer.step()
    
    if i % 10 == 0:
        print(torch.mean(v, dim=-1))

for i in range(wbatch):
    with torch.no_grad():
        yp_binary = (yp[0] > 0.5).float()
        acc = (yp_binary == data_y).float().mean()
        print("acc", acc.item())
