"""Deep Variational Quantum Structures with Tunable Gates.

This example demonstrates:
1. Deep circuit structures with tunable gate activation (structure learning)
2. Parameterized two-qubit gates (RZZ, RYY, RXX)
3. PyTorch-based optimization with automatic differentiation
4. Heisenberg Hamiltonian energy minimization

The circuit alternates between odd and even layers of entangling gates,
with each gate having:
- Rotation angle (trainable parameter θ)
- Activation weight (trainable structure w ∈ [0,1])

Gate form: w·I + (1-w)·exp(iθ·P⊗P) where P ∈ {X, Y, Z}

Author: TyxonQ Team
Date: 2025
"""

import time
import numpy as np
import tyxonq as tq


def build_heisenberg_hamiltonian_1d(n_qubits, Jx=1.0, Jy=1.0, Jz=1.0):
    """Build 1D Heisenberg Hamiltonian with nearest-neighbor interactions.
    
    H = Σᵢ (Jx·XᵢXᵢ₊₁ + Jy·YᵢYᵢ₊₁ + Jz·ZᵢZᵢ₊₁)
    
    Args:
        n_qubits: Number of qubits
        Jx, Jy, Jz: Coupling strengths
        
    Returns:
        Hamiltonian matrix (2^n × 2^n) as backend tensor
    """
    from tyxonq.numerics.api import get_backend
    K = get_backend(None)
    
    dim = 2 ** n_qubits
    H = K.zeros((dim, dim), dtype=K.complex128)
    
    # Pauli matrices
    X = K.array([[0, 1], [1, 0]], dtype=K.complex128)
    Y = K.array([[0, -1j], [1j, 0]], dtype=K.complex128)
    Z = K.array([[1, 0], [0, -1]], dtype=K.complex128)
    I = K.eye(2, dtype=K.complex128)
    
    # Nearest-neighbor interactions
    for bond in range(n_qubits - 1):
        # Build operators for bond (bond, bond+1)
        ops_x = []
        ops_y = []
        ops_z = []
        
        for q in range(n_qubits):
            if q == bond or q == bond + 1:
                ops_x.append(X)
                ops_y.append(Y)
                ops_z.append(Z)
            else:
                ops_x.append(I)
                ops_y.append(I)
                ops_z.append(I)
        
        # Build full operators via Kronecker product
        XX_term = ops_x[0]
        YY_term = ops_y[0]
        ZZ_term = ops_z[0]
        for q in range(1, n_qubits):
            XX_term = K.kron(XX_term, ops_x[q])
            YY_term = K.kron(YY_term, ops_y[q])
            ZZ_term = K.kron(ZZ_term, ops_z[q])
        
        H = H + Jx * XX_term + Jy * YY_term + Jz * ZZ_term
    
    return H


def build_tunable_layer(c, params, structures, layer_idx, n_qubits, parity='odd'):
    """Build one layer of tunable two-qubit gates.
    
    Args:
        c: Circuit object
        params: Rotation angles [3, n_qubits]
        structures: Gate activation weights [3, n_qubits]
        layer_idx: Layer index (for offset calculation)
        n_qubits: Number of qubits
        parity: 'odd' for bonds (1,2), (3,4),... or 'even' for (0,1), (2,3),...
    """
    from tyxonq.numerics.api import get_backend
    K = get_backend(None)
    
    # Pauli pair matrices
    II = K.eye(4, dtype=K.complex128)
    XX = K.kron(K.array([[0, 1], [1, 0]], dtype=K.complex128), 
                 K.array([[0, 1], [1, 0]], dtype=K.complex128))
    YY = K.kron(K.array([[0, -1j], [1j, 0]], dtype=K.complex128),
                 K.array([[0, -1j], [1j, 0]], dtype=K.complex128))
    ZZ = K.kron(K.array([[1, 0], [0, -1]], dtype=K.complex128),
                 K.array([[1, 0], [0, -1]], dtype=K.complex128))
    
    start_qubit = 1 if parity == 'odd' else 0
    
    for gate_type, pauli_pair in enumerate([ZZ, YY, XX]):
        for i in range(start_qubit, n_qubits - 1, 2):
            # Get parameters
            w = structures[gate_type, i]  # Activation weight [0, 1]
            theta = params[gate_type, i]  # Rotation angle
            
            # Tunable gate: w·I + (1-w)·exp(iθ·P⊗P)
            # Simplified: w·I + (1-w)·[cos(θ)·I + i·sin(θ)·P⊗P]
            gate = w * II + (1 - w) * (K.cos(theta) * II + 1j * K.sin(theta) * pauli_pair)
            
            # Apply unitary gate
            c.unitary(i, i + 1, matrix=gate)
    
    return c


def compute_energy(params, structures, n_qubits, n_layers, hamiltonian):
    """Compute ground state energy expectation value.
    
    Args:
        params: Rotation angles [n_layers, 3, n_qubits]
        structures: Gate weights [n_layers, 3, n_qubits]
        n_qubits: System size
        n_layers: Circuit depth
        hamiltonian: Hamiltonian matrix
        
    Returns:
        Energy expectation value <ψ|H|ψ>
    """
    from tyxonq.numerics.api import get_backend
    K = get_backend(None)
    
    # Normalize structures to [0, 1]
    structures = (K.sign(structures) + 1) / 2
    
    # Initial state: |1010...⟩ (Néel state)
    c = tq.Circuit(n_qubits)
    for i in range(n_qubits):
        c.x(i)
    for i in range(0, n_qubits, 2):
        c.h(i)
        if i + 1 < n_qubits:
            c.cx(i, i + 1)
    
    # Build variational circuit
    for layer in range(n_layers):
        # Odd bonds: (1,2), (3,4), ...
        build_tunable_layer(c, params[layer], structures[layer], layer, n_qubits, 'odd')
        # Even bonds: (0,1), (2,3), ...
        build_tunable_layer(c, params[layer], structures[layer], layer, n_qubits, 'even')
    
    # Get final state
    psi = c.wavefunction()
    psi = psi.reshape((-1, 1))
    
    # Compute <ψ|H|ψ>
    from tyxonq.numerics import NumericBackend as NB
    expval = (NB.adjoint(psi) @ (hamiltonian @ psi))[0, 0]
    
    return K.real(expval)


def demonstrate_structure_learning():
    """Train deep variational circuit with structure learning."""
    print("\n" + "="*70)
    print("DEEP VARIATIONAL STRUCTURES WITH TUNABLE GATES")
    print("="*70)
    
    try:
        import torch
        tq.set_backend('pytorch')
        from tyxonq.numerics.api import get_backend
        K = get_backend(None)
        
        # System parameters
        n_qubits = 6
        n_layers = 3
        
        print(f"\nSystem Configuration:")
        print(f"  Qubits: {n_qubits}")
        print(f"  Layers: {n_layers}")
        print(f"  Total trainable params: {3 * n_layers * n_qubits * 2} (angles + structures)")
        
        # Build Heisenberg Hamiltonian
        print(f"\nBuilding Heisenberg Hamiltonian (Jx=Jy=Jz=1.0)...")
        H = build_heisenberg_hamiltonian_1d(n_qubits, Jx=1.0, Jy=1.0, Jz=1.0)
        print(f"  Hamiltonian shape: {H.shape}")
        
        # Initialize parameters
        params = torch.nn.Parameter(torch.randn(n_layers, 3, n_qubits, dtype=torch.float32) * 0.1)
        structures = torch.nn.Parameter(torch.randn(n_layers, 3, n_qubits, dtype=torch.float32))
        
        # Optimizer
        optimizer = torch.optim.Adam([params, structures], lr=0.01)
        
        print(f"\nStarting optimization (Adam, lr=0.01)...")
        print(f"{'Iter':<6} {'Energy':<12} {'Time (s)':<10} {'Improvement':<12}")
        print("-" * 50)
        
        prev_energy = None
        for iteration in range(5):  # Reduced for demo
            t0 = time.time()
            
            # Forward pass
            optimizer.zero_grad()
            energy = compute_energy(params, structures, n_qubits, n_layers, H)
            
            # Backward pass
            energy.backward()
            
            # Update
            optimizer.step()
            
            t1 = time.time()
            
            # Report
            e_val = float(energy.detach())
            improvement = f"{prev_energy - e_val:+.6f}" if prev_energy is not None else "N/A"
            print(f"{iteration:<6} {e_val:<12.6f} {t1-t0:<10.3f} {improvement:<12}")
            prev_energy = e_val
        
        print("\n" + "-" * 50)
        print(f"Final energy: {float(energy):.6f}")
        print(f"\nStructure statistics:")
        struct_normalized = (torch.sign(structures.detach()) + 1) / 2
        print(f"  Active gates (w > 0.5): {(struct_normalized > 0.5).sum().item()}/{structures.numel()}")
        print(f"  Mean activation: {struct_normalized.mean().item():.3f}")
        
        print("\n✓ Structure learning demonstration complete!")
        
        # Reset backend
        tq.set_backend('numpy')
        
    except ImportError:
        print("\n⚠️  PyTorch not available - skipping demonstration")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("DEEP VARIATIONAL QUANTUM STRUCTURES")
    print("="*70)
    print("\nThis example demonstrates:")
    print("  • Deep ansatz with tunable gate structures")
    print("  • Joint optimization of angles and gate activations")
    print("  • Heisenberg model ground state preparation")
    print("  • PyTorch automatic differentiation")
    
    demonstrate_structure_learning()
    
    print("\n" + "="*70)
    print("KEY CONCEPTS")
    print("="*70)
    print("\n1. Tunable Gates:")
    print("   Gate = w·I + (1-w)·exp(iθ·P⊗P)")
    print("   - w ∈ [0,1]: activation weight (structure parameter)")
    print("   - θ: rotation angle (variational parameter)")
    print("\n2. Structure Learning:")
    print("   - Jointly optimize θ and w")
    print("   - Sparse circuits emerge naturally")
    print("   - Reduces gate count for fixed accuracy")
    print("\n3. Gradient-Based Optimization:")
    print("   - PyTorch autograd for both params and structures")
    print("   - Adam optimizer for efficient convergence")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
