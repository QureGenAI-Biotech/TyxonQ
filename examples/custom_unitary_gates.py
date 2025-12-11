"""
Custom Unitary Gates Example
=============================

This example demonstrates how to use the Circuit.unitary() method to apply
arbitrary unitary matrices to quantum circuits. This is useful for:

- Implementing custom gates not in the standard gate set
- Random circuit benchmarking
- Variational quantum algorithms with parameterized unitaries
- Clifford circuit optimization

Key Features:
- Single-qubit unitaries (2×2 matrices)
- Two-qubit unitaries (4×4 matrices)
- Integration with standard gates
- Chaining multiple custom gates

Author: TyxonQ Team
Date: 2024
"""

import numpy as np
import tyxonq as tq

print("=" * 70)
print("Custom Unitary Gates Example")
print("=" * 70)
print()

# ==============================================================================
# Example 1: Single-Qubit Custom Gate (√X gate)
# ==============================================================================

print("[Example 1] Single-Qubit Custom Gate: √X")
print("-" * 70)

# √X gate: applying it twice gives X gate
# Matrix: [[0.5+0.5i, 0.5-0.5i], [0.5-0.5i, 0.5+0.5i]]
sqrt_x = np.array([[0.5+0.5j, 0.5-0.5j],
                   [0.5-0.5j, 0.5+0.5j]], dtype=np.complex128)

c = tq.Circuit(1)
c.unitary(0, matrix=sqrt_x)
c.unitary(0, matrix=sqrt_x)  # Apply √X twice = X

state = c.state()
print(f"Initial state: |0⟩")
print(f"After √X × √X: {state}")
print(f"Expected |1⟩: [0, 1]")
print(f"Match: {np.allclose(state, [0, 1])}")
print()


# ==============================================================================
# Example 2: Two-Qubit Custom Gate (iSWAP gate)
# ==============================================================================

print("[Example 2] Two-Qubit Custom Gate: iSWAP")
print("-" * 70)

# iSWAP gate: swaps two qubits with a phase of i
# Matrix (4×4): [[1,0,0,0], [0,0,i,0], [0,i,0,0], [0,0,0,1]]
iswap = np.array([[1, 0, 0, 0],
                  [0, 0, 1j, 0],
                  [0, 1j, 0, 0],
                  [0, 0, 0, 1]], dtype=np.complex128)

c = tq.Circuit(2)
c.x(0)  # Prepare |10⟩
c.unitary(0, 1, matrix=iswap)

state = c.state()
print(f"Initial state: |10⟩")
print(f"After iSWAP: {state}")
print(f"Expected i|01⟩: [0, i, 0, 0]")
print(f"Match: {np.allclose(state, [0, 1j, 0, 0])}")
print()


# ==============================================================================
# Example 3: Parameterized Rotation Gates
# ==============================================================================

print("[Example 3] Parameterized Rotation Gates")
print("-" * 70)

from tyxonq.libs.quantum_library.kernels.gates import gate_rx, gate_ry, gate_rz

# Build variational circuit with custom rotations
theta1, theta2, theta3 = 0.5, 1.2, 0.8

c = tq.Circuit(2)
c.unitary(0, matrix=gate_ry(theta1))  # RY(θ₁) on qubit 0
c.unitary(1, matrix=gate_rx(theta2))  # RX(θ₂) on qubit 1
c.cx(0, 1)                             # CNOT entanglement
c.unitary(0, matrix=gate_rz(theta3))  # RZ(θ₃) on qubit 0

state = c.state()
print(f"Variational circuit with 3 rotation parameters")
print(f"θ₁={theta1:.2f}, θ₂={theta2:.2f}, θ₃={theta3:.2f}")
print(f"Final state shape: {state.shape}")
print(f"State norm: {np.linalg.norm(state):.6f}")
print()


# ==============================================================================
# Example 4: Random Unitary Benchmarking
# ==============================================================================

print("[Example 4] Random Unitary Benchmarking")
print("-" * 70)

def random_unitary(n):
    """Generate random n×n unitary matrix via QR decomposition."""
    # Generate random complex matrix
    z = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    # QR decomposition gives unitary matrix Q
    q, r = np.linalg.qr(z)
    # Ensure determinant is 1 (special unitary)
    d = np.diagonal(r)
    q = q @ np.diag(d / np.abs(d))
    return q

np.random.seed(42)
depth = 5

c = tq.Circuit(3)
for layer in range(depth):
    # Apply random single-qubit unitaries
    for qubit in range(3):
        u = random_unitary(2)
        c.unitary(qubit, matrix=u)
    
    # Apply random two-qubit unitaries
    for i in range(0, 2):
        u2 = random_unitary(4)
        c.unitary(i, i+1, matrix=u2)

state = c.state()
print(f"Random circuit with depth {depth}")
print(f"Total gates: {depth * 3 + depth * 2} unitaries")
print(f"Final state dimension: {len(state)}")
print(f"State norm (should be 1.0): {np.linalg.norm(state):.6f}")
print()


# ==============================================================================
# Example 5: Quantum Fourier Transform (QFT) Implementation
# ==============================================================================

print("[Example 5] Quantum Fourier Transform (QFT) via Custom Gates")
print("-" * 70)

def qft_gate(k):
    """Controlled phase rotation gate for QFT."""
    phase = 2 * np.pi / (2**k)
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, np.exp(1j * phase)]], dtype=np.complex128)

n_qubits = 3
c = tq.Circuit(n_qubits)

# Prepare initial state |3⟩ = |011⟩
c.x(0)
c.x(1)

# QFT implementation
for i in range(n_qubits):
    c.h(i)
    for j in range(i+1, n_qubits):
        k = j - i + 1
        c.unitary(i, j, matrix=qft_gate(k))

state = c.state()
print(f"QFT applied to |011⟩ (decimal 3)")
print(f"Output state amplitudes:")
for idx, amp in enumerate(state):
    if abs(amp) > 1e-10:
        print(f"  |{idx:03b}⟩: {amp:.4f}")
print()


# ==============================================================================
# Example 6: Integration with VQE-style Ansatz
# ==============================================================================

print("[Example 6] VQE Ansatz with Custom Gates")
print("-" * 70)

def hardware_efficient_layer(params):
    """Build one layer of hardware-efficient ansatz."""
    n = 2
    c = tq.Circuit(n)
    
    # Rotation layer
    for i in range(n):
        c.unitary(i, matrix=gate_ry(params[i]))
    
    # Entanglement layer
    c.cx(0, 1)
    
    return c

# Stack multiple layers
n_layers = 3
params = np.random.randn(n_layers, 2) * 0.5  # Small random initialization

c = tq.Circuit(2)
for layer in range(n_layers):
    layer_circuit = hardware_efficient_layer(params[layer])
    # Apply layer ops to main circuit
    for op in layer_circuit.ops:
        if op[0] == 'unitary':
            q = op[1]
            mat_key = op[2]
            matrix = layer_circuit._unitary_cache.get(mat_key)
            c.unitary(q, matrix=matrix)
        elif op[0] == 'cx':
            c.cx(op[1], op[2])

state = c.state()
print(f"VQE ansatz with {n_layers} layers")
print(f"Total parameters: {n_layers * 2}")
print(f"Final state: {state}")
print(f"State norm: {np.linalg.norm(state):.6f}")
print()


# ==============================================================================
# Summary
# ==============================================================================

print("=" * 70)
print("Summary")
print("=" * 70)
print()
print("This example demonstrated:")
print("  ✓ Single-qubit custom unitaries (√X gate)")
print("  ✓ Two-qubit custom unitaries (iSWAP gate)")
print("  ✓ Parameterized gates from library")
print("  ✓ Random unitary benchmarking")
print("  ✓ QFT implementation with custom gates")
print("  ✓ VQE ansatz integration")
print()
print("Use cases for Circuit.unitary():")
print("  • Implementing exotic gates not in standard set")
print("  • Random circuit generation for benchmarking")
print("  • Variational quantum algorithms")
print("  • Quantum compilation and optimization")
print("  • Research prototyping with novel gates")
print()
