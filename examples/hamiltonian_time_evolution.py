"""Hamiltonian Time Evolution via Trotterization.

This example demonstrates:
1. Building Hamiltonians from Pauli strings
2. Trotterized time evolution circuits
3. Comparison between Trotter and exact evolution
4. Convergence analysis vs. number of Trotter steps

The Trotter-Suzuki decomposition approximates exp(-iHt) by breaking it into
small time steps: exp(-iHt) ≈ [exp(-iHδt)]^n where δt = t/n.

For H = Σⱼ wⱼ Pⱼ, each small step factorizes as:
exp(-iH δt) ≈ ∏ⱼ exp(-i wⱼ Pⱼ δt)

Author: TyxonQ Team
Date: 2025
"""

import numpy as np
import tyxonq as tq
from tyxonq.libs.circuits_library.trotter_circuit import build_trotter_circuit


def build_hamiltonian_pauli_strings(n_qubits, interaction_strength=1.0):
    """Build Hamiltonian as list of Pauli strings.
    
    Example: 2-qubit Heisenberg model
    H = J·(X₀X₁ + Y₀Y₁ + Z₀Z₁)
    
    Pauli encoding: 0=I, 1=X, 2=Y, 3=Z
    
    Args:
        n_qubits: Number of qubits
        interaction_strength: Coupling strength J
        
    Returns:
        (pauli_terms, weights) tuple
    """
    if n_qubits == 2:
        # H = J·(XX + YY + ZZ)
        pauli_terms = [
            [1, 1],  # XX
            [2, 2],  # YY
            [3, 3],  # ZZ
        ]
        weights = [interaction_strength] * 3
    else:
        # Nearest-neighbor Heisenberg chain
        pauli_terms = []
        weights = []
        for i in range(n_qubits - 1):
            # XX term
            ps_xx = [0] * n_qubits
            ps_xx[i] = 1
            ps_xx[i+1] = 1
            pauli_terms.append(ps_xx)
            weights.append(interaction_strength)
            
            # YY term
            ps_yy = [0] * n_qubits
            ps_yy[i] = 2
            ps_yy[i+1] = 2
            pauli_terms.append(ps_yy)
            weights.append(interaction_strength)
            
            # ZZ term
            ps_zz = [0] * n_qubits
            ps_zz[i] = 3
            ps_zz[i+1] = 3
            pauli_terms.append(ps_zz)
            weights.append(interaction_strength)
    
    return pauli_terms, weights


def build_dense_hamiltonian(pauli_terms, weights, n_qubits):
    """Build dense Hamiltonian matrix from Pauli strings.
    
    Args:
        pauli_terms: List of Pauli strings
        weights: Coefficients
        n_qubits: Number of qubits
        
    Returns:
        Dense Hamiltonian matrix (2^n × 2^n)
    """
    from tyxonq.numerics.api import get_backend
    K = get_backend(None)
    
    dim = 2 ** n_qubits
    H = K.zeros((dim, dim), dtype=K.complex128)
    
    # Pauli matrices
    I = K.array([[1, 0], [0, 1]], dtype=K.complex128)
    X = K.array([[0, 1], [1, 0]], dtype=K.complex128)
    Y = K.array([[0, -1j], [1j, 0]], dtype=K.complex128)
    Z = K.array([[1, 0], [0, -1]], dtype=K.complex128)
    pauli_map = [I, X, Y, Z]
    
    # Build each term
    for ps, w in zip(pauli_terms, weights):
        term = pauli_map[ps[0]]
        for p in ps[1:]:
            term = K.kron(term, pauli_map[p])
        H = H + w * term
    
    return H


def exact_time_evolution(H, psi0, time):
    """Compute exact time evolution via matrix exponential.
    
    |ψ(t)⟩ = exp(-iHt)|ψ₀⟩
    """
    from tyxonq.numerics.api import get_backend
    K = get_backend(None)
    
    # Compute exp(-iHt)
    U = K.expm(-1j * time * H)
    
    # Apply to initial state
    psi_t = U @ psi0.reshape((-1, 1))
    return psi_t.reshape(-1)


def demonstrate_trotter_evolution():
    """Demonstrate Trotterized time evolution."""
    print("\n" + "="*70)
    print("HAMILTONIAN TIME EVOLUTION VIA TROTTERIZATION")
    print("="*70)
    
    # System parameters
    n_qubits = 2
    J = 1.0  # Coupling strength
    time = 1.0  # Evolution time
    
    print(f"\nSystem Configuration:")
    print(f"  Qubits: {n_qubits}")
    print(f"  Hamiltonian: H = J·(XX + YY + ZZ), J = {J}")
    print(f"  Evolution time: t = {time}")
    
    # Build Hamiltonian
    pauli_terms, weights = build_hamiltonian_pauli_strings(n_qubits, J)
    print(f"\n  Pauli terms: {len(pauli_terms)}")
    for i, (ps, w) in enumerate(zip(pauli_terms, weights)):
        pauli_str = ''.join(['I', 'X', 'Y', 'Z'][p] for p in ps)
        print(f"    Term {i+1}: {w:.1f} × {pauli_str}")
    
    # Initial state: |10⟩ (qubit 0 in |1⟩, qubit 1 in |0⟩)
    print(f"\n  Initial state: |10⟩")
    
    # Trotter evolution with varying steps
    print(f"\nTrotter Evolution Results:")
    print(f"{'Steps':<8} {'⟨Z₀⟩':<12} {'⟨Z₁⟩':<12}")
    print("-" * 35)
    
    for n_steps in [1, 2, 4, 8, 16, 32]:
        # Prepare initial state |01⟩
        c_init = tq.Circuit(n_qubits)
        c_init.x(0)
        
        # Build Trotter evolution
        c_trot = build_trotter_circuit(
            pauli_terms,
            weights=weights,
            time=time,
            steps=n_steps,
            num_qubits=n_qubits
        )
        
        # Combine circuits
        c_init.ops.extend(c_trot.ops)
        
        # Run simulation
        result = c_init.device(provider="local", device="statevector", shots=0).run()
        
        # Extract Z expectations
        if isinstance(result, list):
            result = result[0] if result else {}
        exps = result.get("expectations", {})
        z0 = exps.get("Z0", 0.0)
        z1 = exps.get("Z1", 0.0)
        
        print(f"{n_steps:<8} {z0:<12.6f} {z1:<12.6f}")
    
    print("\n" + "-" * 35)
    print("Observation: Converges with increasing Trotter steps")


def demonstrate_trotter_accuracy():
    """Compare Trotter approximation with exact evolution."""
    print("\n" + "="*70)
    print("TROTTER APPROXIMATION ACCURACY")
    print("="*70)
    
    n_qubits = 2
    J = 1.0
    time = 1.0
    
    # Build Hamiltonian
    pauli_terms, weights = build_hamiltonian_pauli_strings(n_qubits, J)
    H = build_dense_hamiltonian(pauli_terms, weights, n_qubits)
    
    # Initial state |10⟩ (qubit 0 = 1, qubit 1 = 0)
    psi0 = np.zeros(4, dtype=np.complex128)
    psi0[2] = 1.0  # Binary: 10 = index 2
    
    # Exact evolution
    psi_exact = exact_time_evolution(H, psi0, time)
    
    print(f"\nAccuracy Analysis (Fidelity with exact evolution):")
    print(f"{'Steps':<8} {'Fidelity':<12} {'Error':<12}")
    print("-" * 35)
    
    for n_steps in [1, 2, 4, 8, 16, 32, 64]:
        # Trotter evolution
        c = tq.Circuit(n_qubits)
        c.x(0)  # Prepare |01⟩
        
        # Build Trotter circuit (without measurements)
        c_trotter = build_trotter_circuit(
            pauli_terms,
            weights=weights,
            time=time,
            steps=n_steps,
            num_qubits=n_qubits
        )
        
        # Remove measurement ops
        c_trotter.ops = [op for op in c_trotter.ops if op[0] != 'measure_z']
        
        # Combine circuits
        c.ops.extend(c_trotter.ops)
        
        # Get final state
        psi_trotter = c.state()
        
        # Compute fidelity
        fidelity = float(np.abs(np.vdot(psi_exact, psi_trotter))**2)
        error = 1.0 - fidelity
        
        print(f"{n_steps:<8} {fidelity:<12.9f} {error:<12.2e}")
    
    print("\n" + "-" * 35)
    print("Key insight: Error decreases exponentially with more steps")
    print("Rule of thumb: Use n_steps ≥ 10 × evolution_time for accuracy ~1e-4")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("HAMILTONIAN TIME EVOLUTION")
    print("="*70)
    print("\nThis example demonstrates:")
    print("  • Building Hamiltonians from Pauli strings")
    print("  • Trotterized time evolution circuits")
    print("  • Convergence with increasing Trotter steps")
    print("  • Accuracy comparison with exact evolution")
    
    demonstrate_trotter_evolution()
    demonstrate_trotter_accuracy()
    
    print("\n" + "="*70)
    print("KEY CONCEPTS")
    print("="*70)
    print("\n1. Trotter-Suzuki Decomposition:")
    print("   exp(-iHt) ≈ [∏ⱼ exp(-iwⱼPⱼδt)]^n")
    print("   where δt = t/n and n = number of steps")
    
    print("\n2. Error Scaling:")
    print("   • First-order Trotter: Error ∝ O(t²/n)")
    print("   • Higher-order methods: Better scaling but more gates")
    
    print("\n3. Practical Guidelines:")
    print("   • Small systems (<5 qubits): Use exact evolution")
    print("   • Large systems: Trotterization essential")
    print("   • Choose n_steps to balance accuracy vs. circuit depth")
    
    print("\n4. Supported Pauli Patterns:")
    print("   ✓ Single-qubit Z: RZ gate")
    print("   ✓ Single-qubit X: H-RZ-H")
    print("   ✓ Two-qubit ZZ: CX-RZ-CX")
    print("   ⚠️ Other patterns: Requires gate decomposition")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
