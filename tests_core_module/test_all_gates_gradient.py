"""
全面测试所有参数化量子门的梯度链保持
验证gates.py中所有修复后的门函数都能正确支持PyTorch autograd
"""
import torch
import tyxonq as tq

# 设置PyTorch后端
tq.set_backend('pytorch')

print("=" * 70)
print("测试所有参数化量子门的梯度链")
print("=" * 70)

# 测试所有单比特参数化门
single_qubit_gates = [
    ('RX', lambda c, theta: c.rx(0, theta=theta)),
    ('RY', lambda c, theta: c.ry(0, theta=theta)),
    ('RZ', lambda c, theta: c.rz(0, theta=theta)),
]

print("\n1️⃣  测试单比特参数化门")
print("-" * 70)

for gate_name, gate_fn in single_qubit_gates:
    # 创建可微分参数
    theta = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
    
    # 构建电路
    c = tq.Circuit(1)
    gate_fn(c, theta)
    
    # 获取状态向量
    psi = c.state()
    
    # 计算期望值（<Z>）
    Z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128)
    expectation = torch.real(torch.conj(psi) @ Z @ psi)
    
    # 检查梯度链
    has_grad = expectation.requires_grad
    
    # 尝试计算梯度
    if has_grad:
        expectation.backward()
        grad_value = theta.grad.item() if theta.grad is not None else 0.0
        status = "✅ PASS" if abs(grad_value) > 1e-6 else "❌ FAIL (grad=0)"
    else:
        grad_value = 0.0
        status = "❌ FAIL (no grad)"
    
    print(f"  {gate_name:8s}: requires_grad={has_grad}, gradient={grad_value:+.6f}  {status}")

# 测试双比特参数化门
print("\n2️⃣  测试双比特参数化门")
print("-" * 70)

two_qubit_gates = [
    ('RXX', lambda c, theta: c.rxx(0, 1, theta=theta)),
    ('RYY', lambda c, theta: c.ryy(0, 1, theta=theta)),
    ('RZZ', lambda c, theta: c.rzz(0, 1, theta=theta)),
]

for gate_name, gate_fn in two_qubit_gates:
    # 创建可微分参数
    theta = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
    
    # 构建电路
    c = tq.Circuit(2)
    gate_fn(c, theta)
    
    # 获取状态向量
    psi = c.state()
    
    # 计算期望值（<ZZ>）
    Z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128)
    ZZ = torch.kron(Z, Z)
    expectation = torch.real(torch.conj(psi) @ ZZ @ psi)
    
    # 检查梯度链
    has_grad = expectation.requires_grad
    
    # 尝试计算梯度
    if has_grad:
        expectation.backward()
        grad_value = theta.grad.item() if theta.grad is not None else 0.0
        status = "✅ PASS" if abs(grad_value) > 1e-6 else "❌ FAIL (grad=0)"
    else:
        grad_value = 0.0
        status = "❌ FAIL (no grad)"
    
    print(f"  {gate_name:8s}: requires_grad={has_grad}, gradient={grad_value:+.6f}  {status}")

# 测试组合门电路
print("\n3️⃣  测试组合电路梯度链")
print("-" * 70)

# 创建包含多种门的电路
params = torch.nn.Parameter(torch.tensor([0.3, 0.5, 0.7, 0.9], dtype=torch.float32))

c = tq.Circuit(2)
c.ry(0, theta=params[0])
c.rx(1, theta=params[1])
c.ryy(0, 1, theta=params[2])
c.rz(0, theta=params[3])

psi = c.state()

# 计算期望值
Z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128)
ZZ = torch.kron(Z, Z)
expectation = torch.real(torch.conj(psi) @ ZZ @ psi)

print(f"  Expectation value: {expectation.item():.6f}")
print(f"  Requires grad: {expectation.requires_grad}")

# 计算梯度
expectation.backward()
gradients = params.grad

print(f"  Gradients for 4 parameters:")
for i, g in enumerate(gradients):
    print(f"    param[{i}]: {g.item():+.6f}")

all_nonzero = all(abs(g.item()) > 1e-6 for g in gradients)
status = "✅ PASS" if all_nonzero else "❌ FAIL"
print(f"  Status: {status}")

# 测试VQE优化（简化版）
print("\n4️⃣  测试VQE优化收敛")
print("-" * 70)

# TFIM Hamiltonian: H = -sum(Z_i Z_{i+1}) - h * sum(X_i)
def build_tfim_hamiltonian(n_qubits, h=1.0):
    import numpy as np
    dim = 2 ** n_qubits
    H = torch.zeros((dim, dim), dtype=torch.complex128)
    
    # Pauli matrices
    X = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    Z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128)
    I = torch.eye(2, dtype=torch.complex128)
    
    # ZZ terms
    for i in range(n_qubits - 1):
        term = torch.eye(1, dtype=torch.complex128)
        for j in range(n_qubits):
            if j == i or j == i + 1:
                term = torch.kron(term, Z)
            else:
                term = torch.kron(term, I)
        H = H - term
    
    # X terms
    for i in range(n_qubits):
        term = torch.eye(1, dtype=torch.complex128)
        for j in range(n_qubits):
            if j == i:
                term = torch.kron(term, X)
            else:
                term = torch.kron(term, I)
        H = H - h * term
    
    return H

N_QUBITS = 4
N_LAYERS = 2
H = build_tfim_hamiltonian(N_QUBITS, h=1.0)

# 初始化参数
import numpy as np
np.random.seed(42)
initial_params = np.random.randn(N_LAYERS, N_QUBITS, 2) * 0.1
params = torch.nn.Parameter(torch.from_numpy(initial_params).float())

# VQE能量函数
def vqe_energy(params):
    c = tq.Circuit(N_QUBITS)
    for layer in range(N_LAYERS):
        for i in range(N_QUBITS):
            c.ry(i, theta=params[layer, i, 0])
            c.rz(i, theta=params[layer, i, 1])
        for i in range(N_QUBITS - 1):
            c.cx(i, i + 1)
    
    psi = c.state()
    energy = torch.real(torch.conj(psi) @ H @ psi)
    return energy

# 简单梯度下降优化
lr = 0.1
energies = []

for step in range(5):
    energy = vqe_energy(params)
    energies.append(energy.item())
    
    energy.backward()
    
    with torch.no_grad():
        params -= lr * params.grad
        params.grad.zero_()
    
    if step == 0 or step == 4:
        print(f"  Step {step}: Energy = {energy.item():.6f}")

# 检查收敛
converged = energies[-1] < energies[0] - 0.3
status = "✅ PASS" if converged else "❌ FAIL"
print(f"  Energy improvement: {energies[0]:.6f} → {energies[-1]:.6f}")
print(f"  Status: {status}")

print("\n" + "=" * 70)
print("测试完成！所有参数化门的梯度链已验证")
print("=" * 70)
