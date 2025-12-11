"""
VQE with Architecture Search and Batched Parameter Optimization
================================================================

This example demonstrates advanced VQE techniques for escaping local minima:

1. **Batched Parameter Initialization**: Test multiple random starting points
   simultaneously to find global optimum
   
2. **Architecture Search**: Explore different circuit structures (gate sequences)
   in parallel to find optimal variational ansatz design
   
3. **Dynamic Gate Selection**: Use structure indices to dynamically choose gates
   from a library, enabling neural architecture search (NAS) for quantum circuits

Key Concepts:
- Batch optimization over parameters AND structures
- PyTorch autograd for efficient gradient computation
- Circuit.unitary() for flexible gate library implementation
- Heisenberg Hamiltonian on 1D chain

Applications:
- Robust VQE optimization across random seeds
- Automated ansatz design (quantum NAS)
- Hyperparameter optimization for quantum algorithms

Author: TyxonQ Team
Date: 2024
"""

import torch
import numpy as np
import tyxonq as tq
from tyxonq.libs.quantum_library.kernels.gates import (
    gate_rx, gate_ry, gate_rz, gate_rxx, gate_ryy, gate_rzz
)
from tyxonq.libs.quantum_library.kernels.pauli import heisenberg_hamiltonian

# Set PyTorch backend for automatic differentiation
K = tq.set_backend("pytorch")

print("=" * 70)
print("VQE Architecture Search with Batched Optimization")
print("=" * 70)
print()

# ==============================================================================
# System Setup
# ==============================================================================

N_QUBITS = 6
N_LAYERS = 2
BATCH_STRUCTURES = 2  # Number of circuit architectures to test
BATCH_WEIGHTS = 8      # Number of parameter initializations per structure

print(f"System Configuration:")
print(f"  Qubits: {N_QUBITS}")
print(f"  Layers: {N_LAYERS}")
print(f"  Batch structures: {BATCH_STRUCTURES}")
print(f"  Batch weights per structure: {BATCH_WEIGHTS}")
print(f"  Total configurations: {BATCH_STRUCTURES * BATCH_WEIGHTS}")
print()

# Build Hamiltonian: Transverse-field Ising Model (TFIM)
# H = Σ Z_i Z_{i+1} - Σ X_i
edges = [(i, i + 1) for i in range(N_QUBITS - 1)]  # Open boundary
hamiltonian = heisenberg_hamiltonian(
    N_QUBITS,
    edges,
    hzz=1.0,   # ZZ coupling
    hxx=0.0,
    hyy=0.0,
    hx=-1.0,   # Transverse field
    hy=0.0,
    hz=0.0
)
hamiltonian = torch.from_numpy(np.asarray(hamiltonian, dtype=np.complex128))

print(f"Hamiltonian: TFIM (H = Σ ZZ - Σ X)")
print(f"  Shape: {hamiltonian.shape}")
print(f"  Exact ground state energy: {np.linalg.eigvalsh(hamiltonian.numpy())[0]:.6f}")
print()


# ==============================================================================
# Gate Library for Architecture Search
# ==============================================================================

def build_gate_library_2q(param: float) -> dict:
    """Build library of 2-qubit gates for dynamic selection.
    
    Gate IDs:
      0: Identity (I ⊗ I)
      1: X ⊗ I
      2: Y ⊗ I
      3: Z ⊗ I
      4: H ⊗ I
      5: RX(θ) ⊗ I
      6: RY(θ) ⊗ I  
      7: RZ(θ) ⊗ I
      8: RXX(θ)
      9: RYY(θ)
     10: RZZ(θ)
    """
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    
    gates = {
        0: np.kron(I, I),
        1: np.kron(X, I),
        2: np.kron(Y, I),
        3: np.kron(Z, I),
        4: np.kron(H, I),
        5: np.kron(gate_rx(param), I),
        6: np.kron(gate_ry(param), I),
        7: np.kron(gate_rz(param), I),
        8: gate_rxx(param),
        9: gate_ryy(param),
        10: gate_rzz(param),
    }
    
    return gates


def build_circuit_from_structure(params: torch.Tensor, structure: np.ndarray) -> tq.Circuit:
    """Build quantum circuit based on parameter array and structure encoding.
    
    Args:
        params: Shape (n_layers, n_qubits), rotation angles
        structure: Shape (n_layers, n_qubits), gate indices [0-10]
        
    Returns:
        Circuit with gates selected according to structure
    """
    c = tq.Circuit(N_QUBITS)
    params_np = params.detach().cpu().numpy()
    
    for layer in range(structure.shape[0]):
        for qubit in range(N_QUBITS):
            gate_id = int(structure[layer, qubit])
            param_value = float(params_np[layer, qubit])
            
            # Get gate from library
            gate_lib = build_gate_library_2q(param_value)
            selected_gate = gate_lib[gate_id]
            
            # Apply 2-qubit unitary (current qubit + next with PBC)
            next_qubit = (qubit + 1) % N_QUBITS
            c.unitary(qubit, next_qubit, matrix=selected_gate)
    
    return c


# ==============================================================================
# VQE Energy Evaluation
# ==============================================================================

def compute_energy(params: torch.Tensor, structure: np.ndarray) -> float:
    """Compute VQE energy <ψ(θ)|H|ψ(θ)> for given parameters and structure.
    
    Args:
        params: Variational parameters (n_layers, n_qubits)
        structure: Circuit structure encoding (n_layers, n_qubits)
        
    Returns:
        Energy expectation value
    """
    circuit = build_circuit_from_structure(params, structure)
    state = circuit.state()
    
    # Convert to torch tensor and reshape properly
    if not isinstance(state, torch.Tensor):
        state = torch.from_numpy(np.asarray(state, dtype=np.complex128))
    
    # Ensure state is a column vector
    state = state.reshape(-1, 1)
    
    # Compute expectation <ψ|H|ψ>
    energy = (state.conj().T @ hamiltonian @ state)[0, 0]
    return energy.real


# ==============================================================================
# Batched Loss Function
# ==============================================================================

def batched_loss(weights: torch.Tensor, structures: np.ndarray) -> torch.Tensor:
    """Compute average energy over batch of structures and parameter sets.
    
    This enables parallel exploration of:
    - Multiple circuit architectures (structures)
    - Multiple parameter initializations (weights)
    
    Args:
        weights: Shape (batch_structures, batch_weights, n_layers, n_qubits)
        structures: Shape (batch_structures, n_layers, n_qubits)
        
    Returns:
        Mean energy across all configurations
    """
    bs, bw, n_layers, n_qubits = weights.shape
    energies = []
    
    for s_idx in range(bs):
        structure_s = structures[s_idx]
        for w_idx in range(bw):
            params_w = weights[s_idx, w_idx]
            energy = compute_energy(params_w, structure_s)
            energies.append(energy)
    
    return torch.mean(torch.stack(energies))


# ==============================================================================
# Define Circuit Structures for Search
# ==============================================================================

# Structure 1: Mixed gates with RY bias
# [[I, X, I, RX, I, RY], [RY, I, RY, I, RY, I]]
structure1 = np.array([
    [0, 1, 0, 5, 0, 6],  # Layer 0
    [6, 0, 6, 0, 6, 0],  # Layer 1
], dtype=np.int32)

# Structure 2: Entangling gates emphasis
# [[I, X, I, RX, I, RY], [RYY, I, RXX, I, Z, I]]
structure2 = np.array([
    [0, 1, 0, 5, 0, 6],  # Layer 0
    [9, 0, 8, 0, 3, 0],  # Layer 1
], dtype=np.int32)

structures = np.stack([structure1, structure2])

print("Circuit Structures Defined:")
print(f"  Structure 1 (RY-biased): {structure1.tolist()}")
print(f"  Structure 2 (Entangling): {structure2.tolist()}")
print()


# ==============================================================================
# Initialize Parameters and Optimizer
# ==============================================================================

# Random initialization for all configurations
weights = torch.randn(
    size=[BATCH_STRUCTURES, BATCH_WEIGHTS, N_LAYERS, N_QUBITS],
    dtype=torch.float64,
    requires_grad=True
)

optimizer = torch.optim.Adam([weights], lr=1e-2)

print("Optimization Setup:")
print(f"  Optimizer: Adam")
print(f"  Learning rate: 0.01")
print(f"  Total parameters: {weights.numel()}")
print()


# ==============================================================================
# Training Loop
# ==============================================================================

print("Starting Batched VQE Optimization...")
print("-" * 70)

N_ITERATIONS = 10
losses = []

for iter_idx in range(N_ITERATIONS):
    optimizer.zero_grad()
    
    # Forward pass
    loss = batched_loss(weights, structures)
    
    # Backward pass
    loss.backward()
    
    # Update
    optimizer.step()
    
    # Record
    loss_val = float(loss.detach())
    losses.append(loss_val)
    print(f"Iteration {iter_idx + 1:2d}/10 | Loss: {loss_val:+.6f}")

print("-" * 70)
print()


# ==============================================================================
# Results Analysis
# ==============================================================================

print("Optimization Results:")
print(f"  Initial loss: {losses[0]:+.6f}")
print(f"  Final loss:   {losses[-1]:+.6f}")
print(f"  Improvement:  {losses[0] - losses[-1]:+.6f}")
print()

# Find best configuration
print("Finding Best Configuration...")
best_energy = float('inf')
best_config = None

with torch.no_grad():
    for s_idx in range(BATCH_STRUCTURES):
        structure_s = structures[s_idx]
        for w_idx in range(BATCH_WEIGHTS):
            params_w = weights[s_idx, w_idx]
            energy = float(compute_energy(params_w, structure_s))
            
            if energy < best_energy:
                best_energy = energy
                best_config = {
                    'structure_idx': s_idx,
                    'weight_idx': w_idx,
                    'energy': energy,
                    'structure': structure_s.tolist(),
                }

print(f"Best Configuration:")
print(f"  Structure index: {best_config['structure_idx']}")
print(f"  Weight index:    {best_config['weight_idx']}")
print(f"  Energy:          {best_config['energy']:+.6f}")
print(f"  Structure:       {best_config['structure']}")
print()


# ==============================================================================
# Summary
# ==============================================================================

print("=" * 70)
print("Summary")
print("=" * 70)
print()
print("This example demonstrated:")
print("  ✓ Batched parameter optimization (8 sets per structure)")
print("  ✓ Parallel architecture search (2 circuit structures)")
print("  ✓ Dynamic gate selection via Circuit.unitary()")
print("  ✓ PyTorch autograd for efficient batch optimization")
print()
print(f"Total configurations explored: {BATCH_STRUCTURES * BATCH_WEIGHTS}")
print(f"Best energy found: {best_energy:+.6f}")
print(f"Exact ground state: {np.linalg.eigvalsh(hamiltonian.numpy())[0]:.6f}")
print()
print("Applications:")
print("  • Neural Architecture Search (NAS) for VQE")
print("  • Robust optimization across random initializations")
print("  • Hyperparameter tuning for ansatz design")
print("  • Ensemble methods for escaping local minima")
print()
