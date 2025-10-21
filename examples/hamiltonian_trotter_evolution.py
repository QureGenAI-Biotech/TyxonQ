"""
Hamiltonian Time Evolution via Trotterization
==============================================

This example demonstrates:
- Building Hamiltonians from Pauli strings
- Trotterized time evolution circuits
- Comparing Trotter approximation with exact numerical evolution
- Expectation value calculations

Author: TyxonQ Development Team
Date: 2024
"""

from __future__ import annotations

import numpy as np
import tyxonq as tq
from tyxonq.libs.circuits_library.trotter_circuit import build_trotter_circuit
from tyxonq.libs.quantum_library.dynamics import (
    PauliSumCOO,
    evolve_state,
    expectation,
)


def build_demo_hamiltonian() -> tuple[list[list[int]], list[float]]:
    """Build a simple two-qubit Hamiltonian.
    
    H = 1.0 * Z₀Z₁ + 0.5 * X₀
    
    Pauli encoding: 0=I, 1=X, 2=Y, 3=Z
    - Z₀Z₁ → [3, 3]
    - X₀ → [1, 0]
    
    Returns:
        Tuple of (terms, weights) for Pauli strings
    """
    terms = [[3, 3], [1, 0]]  # Z0Z1, X0
    weights = [1.0, 0.5]
    return terms, weights


def demonstrate_trotter_evolution(time: float = 1.0, steps: int = 8) -> None:
    """Demonstrate Trotterized time evolution.
    
    Args:
        time: Total evolution time
        steps: Number of Trotter steps
    """
    print("=" * 70)
    print("Trotter Evolution Demo")
    print("=" * 70)
    
    terms, weights = build_demo_hamiltonian()
    
    print(f"\nHamiltonian:")
    print(f"  H = {weights[0]:.1f} * Z₀Z₁ + {weights[1]:.1f} * X₀")
    print(f"\nEvolution parameters:")
    print(f"  Time: {time:.2f}")
    print(f"  Trotter steps: {steps}")
    
    # Build Trotterized circuit
    circuit = build_trotter_circuit(
        terms, 
        weights=weights, 
        time=time, 
        steps=steps, 
        num_qubits=2
    )
    
    print(f"\nCircuit depth: {len(circuit.ops)} operations")
    
    # Execute on local simulator
    results = (
        circuit.compile()
        .device(provider="local", device="statevector", shots=0)
        .postprocessing(method=None)
        .run()
    )
    
    for idx, result in enumerate(results):
        state = result.get("state")
        if state is not None:
            print(f"\nFinal state vector (first 4 elements):")
            state_np = np.asarray(state)
            print(f"  {state_np[:4]}")
            
            # Calculate expectation value <H>
            H_dense = PauliSumCOO(terms, weights).to_dense()
            expval = expectation(state_np, H_dense)
            print(f"\nExpectation value ⟨H⟩ = {expval:.6f}")


def compare_with_exact_evolution(
    time: float = 1.0, 
    trotter_steps: int = 8,
    exact_steps: int = 256
) -> None:
    """Compare Trotter approximation with exact numerical evolution.
    
    Args:
        time: Evolution time
        trotter_steps: Trotter discretization steps
        exact_steps: Steps for exact numerical integration
    """
    print("\n" + "=" * 70)
    print("Comparison: Trotter vs. Exact Evolution")
    print("=" * 70)
    
    terms, weights = build_demo_hamiltonian()
    H_dense = PauliSumCOO(terms, weights).to_dense()
    
    # Initial state |11⟩
    psi0 = np.zeros(4, dtype=np.complex128)
    psi0[-1] = 1.0
    
    print(f"\nInitial state: |11⟩")
    print(f"Evolution time: {time:.2f}")
    
    # Exact evolution
    psi_exact = evolve_state(H_dense, psi0, time, steps=exact_steps)
    expval_exact = expectation(psi_exact, H_dense)
    
    print(f"\n[Exact Evolution]")
    print(f"  Integration steps: {exact_steps}")
    print(f"  ⟨H⟩ = {expval_exact:.8f}")
    
    # Trotter evolution with initial state |11⟩
    circuit = build_trotter_circuit(
        terms, 
        weights=weights, 
        time=time, 
        steps=trotter_steps, 
        num_qubits=2
    )
    
    # Create a new circuit with initial state and append Trotter operations
    circuit_with_init = tq.Circuit(2, inputs=psi0)
    # Copy the Trotter gates to the new circuit
    for op in circuit.ops:
        circuit_with_init.ops.append(op)
    
    results = (
        circuit_with_init.compile()
        .device(provider="local", device="statevector", shots=0)
        .postprocessing(method=None)
        .run()
    )
    
    for result in results:
        state = result.get("state")
        if state is not None:
            psi_trotter = np.asarray(state)
            expval_trotter = expectation(psi_trotter, H_dense)
            
            print(f"\n[Trotter Evolution]")
            print(f"  Trotter steps: {trotter_steps}")
            print(f"  ⟨H⟩ = {expval_trotter:.8f}")
            
            # Calculate fidelity
            fidelity = np.abs(np.vdot(psi_exact, psi_trotter)) ** 2
            error = np.abs(expval_trotter - expval_exact)
            
            print(f"\n[Comparison]")
            print(f"  Fidelity: {fidelity:.8f}")
            print(f"  Energy error: {error:.8e}")
            print(f"  Relative error: {error / np.abs(expval_exact) * 100:.4f}%")


def demonstrate_trotter_convergence() -> None:
    """Show convergence of Trotter approximation with increasing steps."""
    print("\n" + "=" * 70)
    print("Trotter Convergence Analysis")
    print("=" * 70)
    
    terms, weights = build_demo_hamiltonian()
    H_dense = PauliSumCOO(terms, weights).to_dense()
    
    time = 1.0
    psi0 = np.zeros(4, dtype=np.complex128)
    psi0[-1] = 1.0
    
    # Exact reference
    psi_exact = evolve_state(H_dense, psi0, time, steps=1000)
    expval_exact = expectation(psi_exact, H_dense)
    
    print(f"\nExact ⟨H⟩ = {expval_exact:.8f}\n")
    print(f"{'Steps':<10} {'⟨H⟩':<15} {'Error':<15} {'Fidelity':<15}")
    print("-" * 55)
    
    for steps in [2, 4, 8, 16, 32, 64]:
        circuit = build_trotter_circuit(
            terms, weights=weights, time=time, 
            steps=steps, num_qubits=2
        )
        
        # Create circuit with initial state
        circuit_with_init = tq.Circuit(2, inputs=psi0)
        for op in circuit.ops:
            circuit_with_init.ops.append(op)
        
        results = (
            circuit_with_init.compile()
            .device(provider="local", device="statevector", shots=0)
            .postprocessing(method=None)
            .run()
        )
        
        for result in results:
            state = result.get("state")
            if state is not None:
                psi_trotter = np.asarray(state)
                expval = expectation(psi_trotter, H_dense)
                fidelity = np.abs(np.vdot(psi_exact, psi_trotter)) ** 2
                error = np.abs(expval - expval_exact)
                
                print(f"{steps:<10} {expval:<15.8f} {error:<15.8e} {fidelity:<15.8f}")


def main():
    """Run all demonstrations."""
    # Set backend - PyTorch支持自动微分
    K = tq.set_backend("pytorch")  # or "numpy"
    print(f"Using backend: {K.name}")
    print(f"Note: Using PyTorch backend for automatic differentiation support\n")
    
    # Demo 1: Basic Trotter evolution
    demonstrate_trotter_evolution(time=1.0, steps=8)
    
    # Demo 2: Compare with exact evolution
    compare_with_exact_evolution(time=1.0, trotter_steps=8, exact_steps=256)
    
    # Demo 3: Convergence analysis
    demonstrate_trotter_convergence()
    
    print("\n" + "=" * 70)
    print("All demonstrations completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
