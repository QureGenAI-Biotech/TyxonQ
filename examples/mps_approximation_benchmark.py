"""Benchmark MPS simulator approximation power.

This example demonstrates:
1. How to use Circuit with MPS simulator backend via .device()
2. MPS approximation quality vs bond dimension
3. Comparing exact simulation with MPS approximation

Key insight: MPS can efficiently simulate low-entanglement states.
For highly entangled states, increasing bond dimension improves accuracy
but requires more memory.
"""

import tyxonq as tq
import numpy as np


def tfi_energy(c, n, j=1.0, h=-1.0):
    """Compute Transverse Field Ising (TFI) model energy.
    
    H = -h Σ_i X_i + j Σ_i Z_i Z_{i+1}
    
    Args:
        c: Circuit to measure (either exact or MPS)
        n: Number of qubits
        j: ZZ coupling strength
        h: Transverse field strength
        
    Returns:
        Energy expectation value
    """
    from tyxonq.libs.quantum_library.kernels.gates import gate_x, gate_z
    
    e = 0.0
    # Transverse field terms: -h * ⟨X_i⟩
    for i in range(n):
        e += h * c.expectation((gate_x(), [i]))
    # Coupling terms: j * ⟨Z_i Z_{i+1}⟩
    for i in range(n - 1):
        e += j * c.expectation((gate_z(), [i]), (gate_z(), [i + 1]))
    return e


def build_variational_circuit(param, n, nlayers, use_mps=False, max_bond=None):
    """Build a variational quantum circuit for TFI ground state.
    
    Args:
        param: Variational parameters [2*nlayers, n]
        n: Number of qubits
        nlayers: Number of ansatz layers
        use_mps: If True, configure for MPS simulation
        max_bond: MPS bond dimension (only used if use_mps=True)
        
    Returns:
        Configured Circuit object
    """
    c = tq.Circuit(n)
    
    # Configure MPS backend if requested
    if use_mps and max_bond is not None:
        c = c.device(
            provider="simulator",
            device="matrix_product_state",
            max_bond=int(max_bond)
        )
    
    # Initial layer: Hadamard on all qubits
    for i in range(n):
        c.h(i)
    
    # Variational layers
    for j in range(nlayers):
        # ZZ rotations (entangling layer)
        # exp(-i θ Z_i Z_{i+1}) via CX-RZ-CX decomposition
        for i in range(n - 1):
            theta = param[2 * j, i]
            c.cx(i, (i + 1) % n)
            c.rz((i + 1) % n, 2.0 * theta)
            c.cx(i, (i + 1) % n)
        
        # RX rotations (local layer)
        for i in range(n):
            c.rx(i, param[2 * j + 1, i])
    
    return c


def compute_entanglement_entropy(state, n, cut_position):
    """Compute von Neumann entropy across a bipartition.
    
    Args:
        state: Quantum state vector
        n: Total number of qubits
        cut_position: Position to cut (qubits 0:cut_position vs cut_position:n)
        
    Returns:
        Von Neumann entropy S = -Tr(ρ log ρ)
    """
    # Reshape state to matrix form
    state_matrix = state.reshape(2 ** cut_position, 2 ** (n - cut_position))
    # Compute reduced density matrix via partial trace
    rho = np.dot(state_matrix, state_matrix.conj().T)
    # Eigenvalue decomposition
    eigvals = np.linalg.eigvalsh(rho)
    # Remove numerical noise
    eigvals = eigvals[eigvals > 1e-12]
    # S = -Σ λ log λ
    entropy = -np.sum(eigvals * np.log(eigvals))
    return entropy


def main():
    print("=" * 60)
    print("MPS Approximation Power Benchmark")
    print("=" * 60)
    
    # Set backend to PyTorch with complex128
    tq.set_backend("numpy")
    
    # Problem parameters
    n = 10  # Number of qubits (reduced from 15 for faster demo)
    nlayers = 5  # Circuit depth (reduced from 20 for faster demo)
    
    print(f"Number of qubits: {n}")
    print(f"Number of layers: {nlayers}")
    print()
    
    # Random variational parameters
    np.random.seed(42)
    param = np.random.rand(2 * nlayers, n) * 2 * np.pi
    
    # NOTE: Using random parameters for demonstration.
    # For param = np.ones([2*nlayers, n]), MPS approximation degrades faster
    # because the circuit creates higher entanglement.
    
    # --- Exact simulation (baseline) ---
    print("Computing exact simulation (baseline)...")
    c_exact = build_variational_circuit(param, n, nlayers, use_mps=False)
    e_exact = tfi_energy(c_exact, n)
    
    # Get exact state for fidelity comparison
    state_exact = c_exact.state()
    
    # Compute entanglement entropy
    entropy = compute_entanglement_entropy(state_exact, n, cut_position=n // 2)
    
    print(f"Exact energy: {e_exact:.6f}")
    print(f"Entanglement entropy (cut at {n//2}): {entropy:.4f}")
    print()
    
    # --- MPS approximation with varying bond dimensions ---
    print("Testing MPS approximation with different bond dimensions:")
    print("-" * 60)
    
    bond_dims = [2, 5, 10, 20]  # Reduced bond dimensions for faster demo
    
    for mpsd in bond_dims:
        c_mps = build_variational_circuit(param, n, nlayers, use_mps=True, max_bond=mpsd)
        e_mps = tfi_energy(c_mps, n)
        state_mps = c_mps.state()
        
        # Compute fidelity: |⟨ψ_exact|ψ_mps⟩|
        fidelity = np.abs(np.vdot(state_exact, state_mps))
        
        # Energy relative error
        rel_error = np.abs((e_mps - e_exact) / e_exact) * 100
        
        print(f"Bond dimension: {mpsd:4d}")
        print(f"  Exact energy:    {e_exact:.8f}")
        print(f"  MPS energy:      {e_mps:.8f}")
        print(f"  Relative error:  {rel_error:.4f}%")
        print(f"  Fidelity:        {fidelity:.6f}")
        print()
    
    print("=" * 60)
    print("Key observations:")
    print("- Larger bond dimension → better approximation but more memory")
    print("- For low entanglement states, small bond dim is sufficient")
    print("- Fidelity is a good proxy for approximation quality")
    print("=" * 60)


if __name__ == "__main__":
    main()
