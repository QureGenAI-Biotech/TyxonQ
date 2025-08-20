import sys
sys.path.insert(0, '.')
import tyxonq as tq
import torch

# Set backend
tq.set_backend('pytorch')
K = tq.set_backend('pytorch')

# Test basic functionality
print("Testing basic functionality...")

# Test 1: Basic circuit creation
c = tq.Circuit(2)
c.h(0)
c.cnot(0, 1)
expectation = tq.backend.real(c.expectation((tq.gates.z(), [0])))
print(f"Basic circuit expectation: {expectation}")

# Test 2: JIT compilation
@K.jit
def simple_circuit(theta):
    c = tq.Circuit(2)
    c.rx(0, theta=theta)
    c.cnot(0, 1)
    return tq.backend.real(c.expectation((tq.gates.z(), [0])))

theta = torch.tensor(0.5)
result = simple_circuit(theta)
print(f"JIT circuit result: {result}")

# Test 3: Automatic differentiation
def loss_function(params):
    c = tq.Circuit(2)
    c.rx(0, theta=params[0])
    c.ry(1, theta=params[1])
    c.cnot(0, 1)
    return tq.backend.real(c.expectation((tq.gates.z(), [0])))

params = torch.nn.Parameter(torch.tensor([0.1, 0.2]))
optimizer = torch.optim.Adam([params], lr=0.1)

for i in range(5):
    optimizer.zero_grad()
    loss = loss_function(params)
    loss.backward()
    optimizer.step()
    print(f"Step {i}: loss = {loss.item()}")

print("Basic functionality tests passed!")
