"""
Quantum part in PyTorch, neural part in PyTorch, both on GPU
Hybrid quantum-classical pipeline demonstration
"""

import os
import time
import numpy as np
import torch
import torchvision
import tyxonq as tq

# Set PyTorch device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)

# Set quantum computing backend to PyTorch
K = tq.set_backend("pytorch")

# Prepare dataset using torchvision
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

# Convert to NumPy arrays to maintain original processing logic
x_train = train_dataset.data.numpy()
y_train = train_dataset.targets.numpy()
x_test = test_dataset.data.numpy()
y_test = test_dataset.targets.numpy()

x_train = x_train[..., np.newaxis] / 255.0

def filter_pair(x, y, a, b):
    """Filter dataset to only include two specified classes"""
    keep = (y == a) | (y == b)
    x, y = x[keep], y[keep]
    y = y == a
    return x, y

x_train, y_train = filter_pair(x_train, y_train, 1, 5)

# Use torch.nn.functional.interpolate
import torch.nn.functional as F
x_train_small = F.interpolate(
    torch.tensor(x_train).permute(0, 3, 1, 2), 
    size=(3, 3), 
    mode='bilinear',
    align_corners=False
).permute(0, 2, 3, 1).numpy()

x_train_bin = np.array(x_train_small > 0.5, dtype=np.float32)
x_train_bin = np.squeeze(x_train_bin).reshape([-1, 9])
y_train_torch = torch.tensor(y_train, dtype=torch.float32)
x_train_torch = torch.tensor(x_train_bin)
x_train_torch = x_train_torch.to(device=device)
y_train_torch = y_train_torch.to(device=device)

n = 9
nlayers = 3

# Define the quantum function (using PyTorch backend)
def qpreds(x, weights):
    """Quantum circuit for predictions"""
    c = tq.Circuit(n)
    for i in range(n):
        c.rx(i, theta=x[i])
    for j in range(nlayers):
        for i in range(n - 1):
            c.cnot(i, i + 1)
        for i in range(n):
            c.rx(i, theta=weights[2 * j, i])
            c.ry(i, theta=weights[2 * j + 1, i])

    return K.stack([K.real(c.expectation_ps(z=[i])) for i in range(n)])

# Create quantum neural network layer
quantumnet = tq.TorchLayer(
    qpreds,
    weights_shape=[2 * nlayers, n],
    use_vmap=True,
    use_interface=True,
    use_jit=True,
    enable_dlpack=True,  # Enable DLPack for efficient data transfer
)

model = torch.nn.Sequential(quantumnet, torch.nn.Linear(9, 1), torch.nn.Sigmoid())
model = model.to(device=device)

criterion = torch.nn.BCELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
nepochs = 300
nbatch = 32
times = []
for epoch in range(nepochs):
    index = np.random.randint(low=0, high=100, size=nbatch)
    inputs, labels = x_train_torch[index], y_train_torch[index]
    opt.zero_grad()

    with torch.set_grad_enabled(True):
        time0 = time.time()
        yps = model(inputs)
        loss = criterion(
            torch.reshape(yps, [nbatch, 1]), torch.reshape(labels, [nbatch, 1])
        )
        loss.backward()
        if epoch % 100 == 0:
            print(loss)
        opt.step()
        time1 = time.time()
        times.append(time1 - time0)
print("Training time per step: ", np.mean(times[1:]))