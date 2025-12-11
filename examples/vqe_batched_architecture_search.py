"""VQE Architecture Search with Batched Parameters.

This example demonstrates advanced variational quantum eigensolver (VQE) techniques:
1. **Batched parameter optimization**: Train multiple parameter initializations simultaneously
2. **Architecture search**: Optimize over different circuit structures (ansatz topologies)
3. **PyTorch autograd**: Leverage Circuit.state() gradient support for efficient training

Problem Setup:
- Target: Ground state of 1D Heisenberg chain (6 qubits)
- Hamiltonian: H = -∑⟨i,j⟩ Zᵢ Zⱼ - ∑ᵢ Xᵢ
- Ansatz: Parameterized circuit with structure selection

Key Features:
- Multiple circuit structures evaluated in parallel
- Batch training of different parameter initializations
- Automatic differentiation through quantum state computation
- Adam optimizer for gradient-based search

Performance:
- 2 circuit structures × 8 parameter sets = 16 parallel optimizations
- Depth 2 circuit with 6 qubits
- ~10 optimization steps demonstrate convergence
"""

import torch
import numpy as np
import tyxonq as tq
from tyxonq.libs.quantum_library.kernels.pauli import heisenberg_hamiltonian

# Set PyTorch backend for autograd support
tq.set_backend("pytorch")

# ============================================================================
# Problem Setup: 6-qubit 1D Heisenberg Chain
# ============================================================================

n_qubits = 6

# Build edges for 1D chain (periodic boundary conditions)
edges = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]

# Build Hamiltonian: H = -∑⟨i,j⟩ Zᵢ Zⱼ - ∑ᵢ Xᵢ
H_matrix = heisenberg_hamiltonian(
    n_qubits,
    edges,
    hzz=1.0,   # ZZ coupling strength
    hxx=0.0,   # XX coupling (disabled)
    hyy=0.0,   # YY coupling (disabled)
    hx=-1.0,   # X field strength
    hy=0.0,    # Y field (disabled)
    hz=0.0,    # Z field (disabled)
)

# Convert to PyTorch tensor for autograd
H_tensor = torch.tensor(H_matrix, dtype=torch.complex128)

print("=" * 70)
print("VQE Architecture Search with Batched Parameters")
print("=" * 70)
print(f"System size: {n_qubits} qubits")
print(f"Hamiltonian: 1D Heisenberg chain (ZZ + X field)")
print(f"Matrix dimension: {H_tensor.shape}")
print()


# ============================================================================
# Gate Set Definition
# ============================================================================

def build_gate_set(theta: torch.Tensor) -> list:
    """Build parameterized 2-qubit gate set.
    
    Gates applied as: Gate(qubit_i, qubit_{i+1})
    
    Returns:
        List of gate functions, indexed by structure parameter
    """
    # Convert theta to float for gate constructors (they expect scalar)
    # But we'll apply them to Circuit which preserves autograd
    
    gates = []
    
    # Gate 0: Identity (no operation)
    gates.append(lambda c, q0, q1, p: None)
    
    # Gate 1: X ⊗ I (X on first qubit)
    gates.append(lambda c, q0, q1, p: c.x(q0))
    
    # Gate 2: Y ⊗ I (Y on first qubit)
    gates.append(lambda c, q0, q1, p: c.y(q0))
    
    # Gate 3: Z ⊗ I (Z on first qubit)
    gates.append(lambda c, q0, q1, p: c.z(q0))
    
    # Gate 4: H ⊗ I (Hadamard on first qubit)
    gates.append(lambda c, q0, q1, p: c.h(q0))
    
    # Gate 5: Rx(θ) ⊗ I (Parameterized rotation on first qubit)
    gates.append(lambda c, q0, q1, p: c.rx(q0, p))
    
    # Gate 6: Ry(θ) ⊗ I (Parameterized rotation on first qubit)
    gates.append(lambda c, q0, q1, p: c.ry(q0, p))
    
    # Gate 7: Rz(θ) ⊗ I (Parameterized rotation on first qubit)
    gates.append(lambda c, q0, q1, p: c.rz(q0, p))
    
    # Gate 8: exp(-i θ/2 XX) (XX interaction)
    gates.append(lambda c, q0, q1, p: c.rxx(q0, q1, p))
    
    # Gate 9: exp(-i θ/2 YY) (YY interaction)
    gates.append(lambda c, q0, q1, p: c.ryy(q0, q1, p))
    
    # Gate 10: exp(-i θ/2 ZZ) (ZZ interaction)
    gates.append(lambda c, q0, q1, p: c.rzz(q0, q1, p))
    
    return gates


# ============================================================================
# Circuit Construction
# ============================================================================

def build_vqe_circuit(params: torch.Tensor, structure: torch.Tensor) -> tq.Circuit:
    """Build VQE ansatz circuit based on structure and parameters.
    
    Args:
        params: Parameter tensor of shape [depth, n_qubits]
        structure: Structure tensor of shape [depth, n_qubits], values 0-10 (gate indices)
    
    Returns:
        Circuit with ansatz applied
    """
    c = tq.Circuit(n_qubits)
    gate_set = build_gate_set(None)  # Gate set with parameter placeholders
    
    depth = structure.shape[0]
    
    for layer in range(depth):
        for qubit in range(n_qubits):
            gate_idx = int(structure[layer, qubit].item())
            param = params[layer, qubit]
            q_next = (qubit + 1) % n_qubits
            
            # Apply gate from gate set
            gate_set[gate_idx](c, qubit, q_next, param)
    
    return c


# ============================================================================
# Energy Computation
# ============================================================================

def compute_energy(params: torch.Tensor, structure: torch.Tensor) -> torch.Tensor:
    """Compute VQE energy ⟨ψ(params)|H|ψ(params)⟩.
    
    Args:
        params: Parameter tensor [depth, n_qubits]
        structure: Structure tensor [depth, n_qubits]
    
    Returns:
        Energy expectation value (real scalar)
    """
    # Build and execute circuit
    c = build_vqe_circuit(params, structure)
    psi = c.state()  # Returns torch.Tensor (preserves autograd)
    
    # Ensure dtype compatibility
    psi = psi.to(dtype=torch.complex128)
    
    # Compute ⟨ψ|H|ψ⟩
    energy = torch.real(torch.conj(psi) @ H_tensor @ psi)
    
    return energy


# ============================================================================
# Batched Optimization
# ============================================================================

def batched_loss(params_batch: torch.Tensor, structures: torch.Tensor) -> torch.Tensor:
    """Compute average energy over batched parameters and structures.
    
    Args:
        params_batch: [n_structures, n_params, depth, n_qubits]
        structures: [n_structures, depth, n_qubits]
    
    Returns:
        Average energy across all structure-parameter combinations
    """
    n_structures, n_params, depth, nq = params_batch.shape
    
    energies = []
    for s in range(n_structures):
        structure_s = structures[s]
        for w in range(n_params):
            params_w = params_batch[s, w]
            E = compute_energy(params_w, structure_s)
            energies.append(E)
    
    return torch.mean(torch.stack(energies))


# ============================================================================
# Main Optimization Loop
# ============================================================================

if __name__ == "__main__":
    # Hyperparameters
    n_structures = 2      # Number of circuit structures to search
    n_params_per_struct = 8  # Parameter initializations per structure
    depth = 2             # Circuit depth (layers)
    n_steps = 10          # Optimization steps
    learning_rate = 1e-2
    
    print("Hyperparameters:")
    print(f"  Structures: {n_structures}")
    print(f"  Parameter sets per structure: {n_params_per_struct}")
    print(f"  Circuit depth: {depth}")
    print(f"  Total parallel optimizations: {n_structures * n_params_per_struct}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Optimization steps: {n_steps}")
    print()
    
    # Define circuit structures (ansatz topologies)
    # Structure values: 0-10 corresponding to gate indices
    structure1 = torch.tensor([
        [0, 1, 0, 5, 0, 6],  # Layer 0: Identity, X, Identity, Rx, Identity, Ry
        [6, 0, 6, 0, 6, 0],  # Layer 1: Ry, Identity, Ry, Identity, Ry, Identity
    ], dtype=torch.int32)
    
    structure2 = torch.tensor([
        [0, 1, 0, 5, 0, 6],  # Layer 0: Same as structure1
        [9, 0, 8, 0, 3, 0],  # Layer 1: YY, Identity, XX, Identity, Z, Identity (different!)
    ], dtype=torch.int32)
    
    structures = torch.stack([structure1, structure2])
    
    # Initialize random parameters for all structure-parameter combinations
    params_batch = torch.randn(
        n_structures, n_params_per_struct, depth, n_qubits,
        dtype=torch.float64,
        requires_grad=True
    )
    
    # Setup optimizer
    optimizer = torch.optim.Adam([params_batch], lr=learning_rate)
    
    print("=" * 70)
    print("Starting Optimization")
    print("=" * 70)
    
    # Optimization loop
    for step in range(n_steps):
        optimizer.zero_grad()
        
        # Compute loss (average energy over all combinations)
        loss = batched_loss(params_batch, structures)
        
        # Backpropagate gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Report progress
        print(f"Step {step:2d}: Loss = {loss.item():.6f}")
    
    print()
    print("=" * 70)
    print("Optimization Complete")
    print("=" * 70)
    
    # Evaluate best result
    final_loss = batched_loss(params_batch, structures)
    print(f"Final average energy: {final_loss.item():.6f}")
    
    # Find best individual result
    with torch.no_grad():
        best_energy = float('inf')
        best_struct = None
        best_params_idx = None
        
        for s in range(n_structures):
            for w in range(n_params_per_struct):
                E = compute_energy(params_batch[s, w], structures[s])
                if E.item() < best_energy:
                    best_energy = E.item()
                    best_struct = s
                    best_params_idx = w
    
    print(f"Best individual energy: {best_energy:.6f}")
    print(f"  Structure index: {best_struct}")
    print(f"  Parameter set index: {best_params_idx}")
    print()
    
    # Display winning structure
    print("Winning circuit structure:")
    winning_structure = structures[best_struct]
    for layer in range(depth):
        gates = [int(winning_structure[layer, q].item()) for q in range(n_qubits)]
        print(f"  Layer {layer}: {gates}")
    
    print()
    print("✅ Batched architecture search completed successfully!")
    print("   Demonstrated PyTorch autograd support through Circuit.state()")
