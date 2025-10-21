"""
MERA VQE - Multi-scale Entanglement Renormalization Ansatz for Variational Quantum Eigensolver.

This example demonstrates:
1. Hierarchical MERA ansatz with logarithmic depth scaling
2. PyTorch-based optimization with automatic differentiation
3. Transverse-field Ising model (TFI) Hamiltonian expectation value
4. Hardware-efficient two-qubit gates (rxx, rzz)

MERA is particularly effective for systems with scale-invariant entanglement,
providing an efficient representation for many-body ground states.

References:
- Vidal, G. (2007). Phys. Rev. Lett. 99, 220405
- Evenbly, G., & Vidal, G. (2009). Phys. Rev. B 79, 144108
"""

import time
import numpy as np
import torch
import tyxonq as tq

# Configure backend
K = tq.set_backend("pytorch")
ctype, rtype = tq.set_dtype("complex64")


def mera_circuit(params, n_qubits, depth):
    """Construct MERA ansatz circuit with hierarchical entangling structure.
    
    MERA uses a tree-like architecture where qubits are progressively entangled
    at different length scales, mimicking the renormalization group flow.
    
    Args:
        params: Flattened parameter array
        n_qubits: Number of qubits (must be power of 2)
        depth: Number of entangling layers per scale
        
    Returns:
        Tuple of (circuit, param_count)
        
    Circuit structure:
        1. Initial single-qubit rotations (Rx, Rz)
        2. Hierarchical entangling layers:
           - Scale k: operates on 2^k qubit blocks
           - Alternating even/odd coupling pattern
           - Parametrized XX + ZZ interactions
        3. Mid-layer rotations after each scale
    """
    c = tq.Circuit(n_qubits)
    idx = 0

    # Initial layer: prepare product state with rotations
    for i in range(n_qubits):
        c.rx(i, theta=params[2 * i])
        c.rz(i, theta=params[2 * i + 1])
    idx += 2 * n_qubits

    # Hierarchical MERA structure: iterate over length scales
    for n_layer in range(1, int(np.log2(n_qubits)) + 1):
        n_qubit_block = 2 ** n_layer  # Current block size
        step = n_qubits // n_qubit_block  # Spacing between blocks

        # Multiple entangling layers per scale
        for _ in range(depth):
            # Even bonds: couple qubits at step, step+2*step, ...
            for i in range(step, n_qubits - step, 2 * step):
                c.rxx(i, i + step, theta=params[idx])
                c.rzz(i, i + step, theta=params[idx + 1])
                idx += 2

            # Odd bonds: couple qubits at 0, 2*step, 4*step, ...
            for i in range(0, n_qubits, 2 * step):
                if i + step < n_qubits:
                    c.rxx(i, i + step, theta=params[idx])
                    c.rzz(i, i + step, theta=params[idx + 1])
                    idx += 2

        # Mid-layer rotations after each scale
        for i in range(0, n_qubits, step):
            c.rx(i, theta=params[idx])
            c.rz(i, theta=params[idx + 1])
            idx += 2

    return c, idx


def tfi_hamiltonian_expectation(circuit, J, B):
    """Compute expectation value for transverse-field Ising Hamiltonian.
    
    H = -J Σᵢ ZᵢZᵢ₊₁ - B Σᵢ Xᵢ
    
    This implementation computes the expectation directly using local observables
    rather than full matrix representation, which is more efficient.
    
    Args:
        circuit: Quantum circuit in MERA ansatz
        J: Ising coupling strength (longitudinal field)
        B: Transverse field strength
        
    Returns:
        Energy expectation value ⟨ψ|H|ψ⟩
    """
    from tyxonq.libs.quantum_library.kernels.gates import gate_x, gate_z
    
    n = circuit.num_qubits
    energy = 0.0

    # ZZ interaction terms (nearest-neighbor with OBC)
    for i in range(n - 1):
        energy += -J * circuit.expectation((gate_z(), [i]), (gate_z(), [i + 1]))

    # X transverse field terms
    for i in range(n):
        energy += -B * circuit.expectation((gate_x(), [i]))

    return energy


def mera_vqe(params, n_qubits, depth, J, B):
    """MERA VQE objective function: energy expectation.
    
    Args:
        params: Variational parameters (torch.Tensor with requires_grad=True)
        n_qubits: Number of qubits
        depth: MERA depth per scale
        J: Ising coupling
        B: Transverse field
        
    Returns:
        Energy (scalar torch.Tensor for autograd)
    """
    circuit, _ = mera_circuit(params, n_qubits, depth)
    return tfi_hamiltonian_expectation(circuit, J, B)


def train_mera_vqe(n_qubits, depth, J, B, max_iter=100, lr=0.01, batch_size=1):
    """Train MERA ansatz using PyTorch optimizer.
    
    Args:
        n_qubits: System size (power of 2)
        depth: Entangling layers per scale
        J: Ising coupling strength
        B: Transverse field strength
        max_iter: Maximum optimization iterations
        lr: Learning rate
        batch_size: Mini-batch size (for future batched training)
        
    Returns:
        Optimized energy
        
    Training procedure:
        1. Initialize parameters randomly (small variance)
        2. Adam optimizer with exponential learning rate decay
        3. Track convergence and timing statistics
    """
    # Calculate total parameter count
    _, param_count = mera_circuit(torch.zeros(int(1e6)), n_qubits, depth)
    print(f"MERA VQE Configuration:")
    print(f"  Qubits: {n_qubits}")
    print(f"  Depth per scale: {depth}")
    print(f"  Total parameters: {param_count}")
    print(f"  Hamiltonian: J={J}, B={B}")
    print(f"  Optimizer: Adam(lr={lr})")
    print()

    # Initialize parameters
    params = torch.nn.Parameter(torch.randn(param_count, dtype=getattr(torch, K.dtypestr)) * 0.05)
    optimizer = torch.optim.Adam([params], lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    times = [time.time()]
    best_energy = float('inf')

    for iteration in range(max_iter):
        # Forward pass
        energy = mera_vqe(params, n_qubits, depth, J, B)

        # Backward pass
        optimizer.zero_grad()
        energy.backward()
        optimizer.step()

        # Learning rate decay every 500 iterations
        if iteration % 500 == 499:
            scheduler.step()

        times.append(time.time())

        # Track best energy
        current_energy = energy.detach().cpu().item()
        if current_energy < best_energy:
            best_energy = current_energy

        # Progress reporting
        if iteration % 10 == 9 or iteration == max_iter - 1:
            avg_time = (times[-1] - times[1]) / (len(times) - 1) if len(times) > 1 else 0
            print(f"Iteration {iteration + 1}/{max_iter}")
            print(f"  Energy: {current_energy:.6f}")
            print(f"  Best: {best_energy:.6f}")
            print(f"  Avg time/iter: {avg_time:.4f}s")
            print()

    return best_energy


if __name__ == "__main__":
    print("=" * 60)
    print("MERA VQE: Transverse-Field Ising Model Ground State")
    print("=" * 60)
    print()

    # Small system for quick demonstration
    # For production: increase to n_qubits=16, depth=2, max_iter=1000
    n_qubits = 8  # Power of 2 required
    depth = 1  # Layers per renormalization scale
    J = 1.0  # Ising coupling
    B = -1.0  # Transverse field (negative for ferromagnetic ground state)
    max_iter = 50  # Reduced for CI/quick test

    start_time = time.time()
    optimized_energy = train_mera_vqe(
        n_qubits=n_qubits,
        depth=depth,
        J=J,
        B=B,
        max_iter=max_iter,
        lr=0.02
    )
    total_time = time.time() - start_time

    print("=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Final optimized energy: {optimized_energy:.6f}")
    print(f"Total training time: {total_time:.2f}s")
    print()
    print("Note: For better convergence, increase max_iter and use")
    print("      larger systems (e.g., n_qubits=16, depth=2, max_iter=1000)")
    print("=" * 60)
