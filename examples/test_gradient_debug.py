"""测试梯度计算是否正常"""
import numpy as np
import torch
import tyxonq as tq
from tyxonq.libs.quantum_library.kernels.pauli import pauli_string_sum_dense

tq.set_backend('pytorch')
N_QUBITS = 6
N_LAYERS = 4

# Build Hamiltonian
pauli_terms = []
weights = []
for i in range(N_QUBITS - 1):
    term = [0] * N_QUBITS
    term[i] = 3
    term[i+1] = 3
    pauli_terms.append(term)
    weights.append(-1.0)
for i in range(N_QUBITS):
    term = [0] * N_QUBITS
    term[i] = 1
    pauli_terms.append(term)
    weights.append(-1.0)

H_matrix = pauli_string_sum_dense(pauli_terms, weights)

def create_circuit_hea(params):
    c = tq.Circuit(N_QUBITS)
    for i in range(N_QUBITS):
        c.h(i)
    for layer in range(N_LAYERS):
        for i in range(N_QUBITS):
            c.ry(i, theta=params[layer, i, 0])
            c.rz(i, theta=params[layer, i, 1])
        for i in range(N_QUBITS - 1):
            c.cx(i, i + 1)
    return c

def tfim_energy(circuit, H):
    psi = circuit.state()
    H_tensor = H.to(psi.dtype) if isinstance(H, torch.Tensor) else torch.from_numpy(H).to(psi.dtype)
    Hpsi = H_tensor @ psi
    return torch.real(torch.dot(torch.conj(psi), Hpsi))

# 初始参数
np.random.seed(42)
initial_params = np.random.randn(N_LAYERS, N_QUBITS, 2) * 0.1

params = torch.nn.Parameter(torch.from_numpy(initial_params).float())
optimizer = torch.optim.Adam([params], lr=0.1)

print("初始参数 requires_grad:", params.requires_grad)
print("初始参数 dtype:", params.dtype)

energy = None
for step in range(5):
    optimizer.zero_grad()
    
    circuit = create_circuit_hea(params)
    energy = tfim_energy(circuit, H_matrix)
    
    print(f"\nStep {step}:")
    print(f"  Energy: {energy.item():.6f}")
    print(f"  Energy requires_grad: {energy.requires_grad}")
    
    energy.backward()
    
    if params.grad is not None:
        print(f"  Gradient norm: {params.grad.norm().item():.6f}")
        print(f"  Max gradient: {params.grad.abs().max().item():.6f}")
    else:
        print(f"  Gradient: None!")
    
    optimizer.step()

if energy is not None:
    print(f"\n最终能量: {energy.item():.6f}")
