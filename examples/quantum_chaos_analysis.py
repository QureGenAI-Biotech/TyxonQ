"""
Quantum Chaos and Complexity Analysis
量子混沌与复杂性分析

This example demonstrates various quantum chaos indicators and complexity measures:
- Frame potential (t-design measure)
- Jacobian matrix (sensitivity analysis)
- Hessian matrix (curvature analysis)  
- Entanglement entropy
- Energy landscape optimization

本示例展示各种量子混沌指标和复杂性度量：
- Frame potential（t-设计度量）
- Jacobian矩阵（灵敏度分析）
- Hessian矩阵（曲率分析）
- 纠缠熵
- 能量景观优化

Reference:
- Frame potential: arXiv:2002.08539
- Quantum chaos: arXiv:1804.08655

Note:
    This example uses numerical simulation features (state vectors, custom initial states).
    On real quantum hardware, these would require measurement-based protocols.
"""

from functools import partial
import numpy as np
import torch
import tyxonq as tq
from tyxonq.libs.quantum_library.kernels.quantum_info import reduced_density_matrix, entropy
from tyxonq.libs.quantum_library.kernels.gates import gate_z

# Set PyTorch backend for automatic differentiation support
K = tq.set_backend("pytorch")

# Circuit parameters
N_QUBITS = 6
N_LAYERS = 3


def build_parameterized_circuit(params, n_qubits, n_layers, inputs=None):
    """Build a parameterized quantum circuit.
    
    Architecture: RY rotation + CNOT ring
    
    Args:
        params: Shape [n_layers, n_qubits], rotation angles
        n_qubits: Number of qubits
        n_layers: Number of layers
        inputs: Initial state (optional). For numerical simulation only.
                If None, starts from |00...0⟩ (physically realizable).
    
    Returns:
        Wavefunction state
    
    Note:
        On real quantum hardware, inputs must be None (always starts from |00...0⟩).
        Custom initial states are only supported in numerical simulation.
    """
    c = tq.Circuit(n_qubits, inputs=inputs)
    
    for layer in range(n_layers):
        # Rotation layer
        for i in range(n_qubits):
            c.ry(i, theta=params[layer, i])
        
        # Entanglement layer (ring topology)
        for i in range(n_qubits):
            c.cx(i, (i + 1) % n_qubits)
    
    return c.state()


def demonstrate_entanglement():
    """Calculate entanglement entropy of the quantum state.
    
    Uses reduced density matrix and von Neumann entropy.
    
    Note: Entanglement entropy calculation is simulation-only.
    Real quantum hardware cannot directly measure it.
    """
    print("=" * 60)
    print("Entanglement Entropy Calculation")
    print("=" * 60)
    
    params = torch.randn([N_LAYERS, N_QUBITS], requires_grad=True)
    state = build_parameterized_circuit(params, N_QUBITS, N_LAYERS)
    
    # Calculate reduced density matrix (trace out half of qubits)
    cut_position = N_QUBITS // 2
    rho_reduced = reduced_density_matrix(state.detach().cpu().numpy(), cut=cut_position)
    
    # Von Neumann entropy
    ent = entropy(rho_reduced)
    
    print(f"System: {N_QUBITS} qubits, {N_LAYERS} layers")
    print(f"Subsystem A: first {cut_position} qubits")
    print(f"Entanglement entropy S(ρ_A) = {ent:.6f}")
    print(f"Maximum possible entropy = {np.log(2**cut_position):.6f}")
    print(f"Entanglement ratio = {ent / np.log(2**cut_position):.2%}")
    print()


def frame_potential(param1, param2, t, n_qubits, n_layers):
    """Calculate t-th frame potential.
    
    Frame potential measures how close the ensemble of unitaries
    is to a t-design (uniformly distributed over Haar measure).
    
    F_t = E[|⟨ψ_1|ψ_2⟩|^(2t)]
    
    Args:
        param1, param2: Two sets of parameters
        t: Design order
        n_qubits, n_layers: Circuit architecture
    
    Returns:
        Frame potential value
    
    Note: Simulation-only. Real hardware would require shadow tomography.
    """
    state1 = build_parameterized_circuit(param1, n_qubits, n_layers)
    state2 = build_parameterized_circuit(param2, n_qubits, n_layers)
    
    # Inner product ⟨ψ_1|ψ_2⟩
    inner = K.tensordot(K.conj(state1), state2, 1)
    
    # |⟨ψ_1|ψ_2⟩|^(2t)
    return K.abs(inner) ** (2 * t)


def demonstrate_frame_potential():
    """Demonstrate frame potential calculation.
    
    Frame potential is used to:
    - Verify expressibility of quantum circuits
    - Measure randomness/scrambling
    - Assess quantum advantage
    """
    print("=" * 60)
    print("Frame Potential (t-Design Measure)")
    print("=" * 60)
    
    # Generate random parameter pairs
    batch_size = 10
    params1 = torch.randn([batch_size, N_LAYERS, N_QUBITS])
    params2 = torch.randn([batch_size, N_LAYERS, N_QUBITS])
    
    # Calculate frame potential for different t
    for t in [1, 2, 3]:
        potentials = []
        for i in range(batch_size):
            fp = frame_potential(params1[i], params2[i], t, N_QUBITS, N_LAYERS)
            potentials.append(fp.item())
        
        mean_fp = np.mean(potentials)
        std_fp = np.std(potentials)
        
        print(f"t={t} Frame Potential:")
        print(f"  Mean: {mean_fp:.6f} ± {std_fp:.6f}")
        print(f"  Haar value (ideal): {1/(2**N_QUBITS + 1)**t:.6f}")
        print()


def demonstrate_jacobian():
    """Calculate Jacobian matrix ∂ψ_i/∂θ_j.
    
    Jacobian measures sensitivity of output state to parameter changes.
    Large Jacobian entries indicate important parameters.
    
    Note: Simulation-only. Real hardware uses parameter shift rule.
    """
    print("=" * 60)
    print("Jacobian Matrix (Sensitivity Analysis)")
    print("=" * 60)
    
    params = torch.randn([N_LAYERS, N_QUBITS], requires_grad=True)
    
    try:
        # Forward-mode automatic differentiation
        jac_func = K.jacfwd(partial(build_parameterized_circuit, 
                                    n_qubits=N_QUBITS, 
                                    n_layers=N_LAYERS))
        jacobian = jac_func(params)
        
        # Jacobian shape: [2^N_QUBITS, N_LAYERS, N_QUBITS]
        print(f"Jacobian shape: {jacobian.shape}")
        print(f"Interpretation: [state_dim, n_layers, n_qubits]")
        
        # Calculate sensitivity metrics
        jac_norm = torch.norm(jacobian, dim=0)  # [N_LAYERS, N_QUBITS]
        
        print(f"\nParameter sensitivity (Frobenius norm):")
        for layer in range(N_LAYERS):
            print(f"  Layer {layer}: {jac_norm[layer].detach().numpy()}")
        
        most_sensitive = torch.argmax(jac_norm)
        layer_idx = most_sensitive // N_QUBITS
        qubit_idx = most_sensitive % N_QUBITS
        print(f"\nMost sensitive parameter: Layer {layer_idx}, Qubit {qubit_idx}")
        print()
        
    except Exception as e:
        print(f"Jacobian calculation skipped: {str(e)[:100]}")
        print("Note: PyTorch jacfwd may have limitations with complex operations")
        print()


def correlation_function(params, n_qubits, n_layers):
    """Calculate ⟨Z_1 Z_2⟩ correlation.
    
    This observable is used for Hessian and optimization demonstrations.
    """
    state = build_parameterized_circuit(params, n_qubits, n_layers)
    c = tq.Circuit(n_qubits, inputs=state)
    
    # Measure ⟨Z_1 Z_2⟩
    z1z2 = c.expectation([gate_z(), [1]], [gate_z(), [2]])
    
    return K.real(z1z2)


def demonstrate_hessian():
    """Calculate Hessian matrix ∂²E/∂θ_i∂θ_j.
    
    Hessian characterizes the energy landscape:
    - Positive eigenvalues: local minimum
    - Negative eigenvalues: local maximum/saddle point
    - Zero eigenvalues: flat direction (barren plateau)
    """
    print("=" * 60)
    print("Hessian Matrix (Landscape Curvature)")
    print("=" * 60)
    
    params = torch.randn([N_LAYERS, N_QUBITS], requires_grad=True)
    
    try:
        hessian_func = K.hessian(partial(correlation_function, 
                                        n_qubits=N_QUBITS, 
                                        n_layers=N_LAYERS))
        hess = hessian_func(params)
        
        print(f"Hessian shape: {hess.shape}")
        print(f"Interpretation: [n_layers, n_qubits, n_layers, n_qubits]")
        
        # Flatten to 2D matrix for eigenvalue analysis
        hess_2d = hess.reshape(N_LAYERS * N_QUBITS, N_LAYERS * N_QUBITS)
        eigenvalues = torch.linalg.eigvalsh(hess_2d)
        
        print(f"\nEigenvalue statistics:")
        print(f"  Min: {eigenvalues.min().item():.6f}")
        print(f"  Max: {eigenvalues.max().item():.6f}")
        print(f"  Mean: {eigenvalues.mean().item():.6f}")
        print(f"  Std: {eigenvalues.std().item():.6f}")
        
        # Check for barren plateau (many near-zero eigenvalues)
        near_zero = (torch.abs(eigenvalues) < 1e-3).sum().item()
        print(f"  Near-zero eigenvalues: {near_zero}/{len(eigenvalues)}")
        print()
        
    except Exception as e:
        print(f"Hessian calculation skipped: {str(e)[:100]}")
        print("Note: Hessian computation is expensive and may have limitations")
        print()


def demonstrate_optimization():
    """Demonstrate gradient-based optimization.
    
    Task: Minimize ⟨Z_1 Z_2⟩ correlation using Adam optimizer.
    """
    print("=" * 60)
    print("Gradient-Based Optimization")
    print("=" * 60)
    
    # Initialize parameters
    params = torch.randn([N_LAYERS, N_QUBITS], requires_grad=True)
    
    # Create value_and_grad function
    vg_func = K.value_and_grad(correlation_function)
    
    # PyTorch Adam optimizer
    optimizer = torch.optim.Adam([params], lr=0.05)
    
    print(f"Objective: Minimize ⟨Z_1 Z_2⟩")
    print(f"Optimizer: Adam (lr=0.05)")
    print(f"Iterations: 50\n")
    
    energies = []
    for iteration in range(50):
        optimizer.zero_grad()
        
        # Compute energy and gradient
        energy, grads = vg_func(params, N_QUBITS, N_LAYERS)
        
        # Backward pass
        energy.backward()
        
        # Update parameters
        optimizer.step()
        
        energies.append(energy.item())
        
        if iteration % 10 == 0:
            grad_norm = torch.norm(grads).item()
            print(f"Iter {iteration:3d}: Energy = {energy.item():+.6f}, "
                  f"||∇E|| = {grad_norm:.6f}")
    
    print(f"\nOptimization complete!")
    print(f"Initial energy: {energies[0]:+.6f}")
    print(f"Final energy: {energies[-1]:+.6f}")
    print(f"Improvement: {energies[0] - energies[-1]:.6f}")
    print()


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("Quantum Chaos and Complexity Analysis")
    print("量子混沌与复杂性分析")
    print("=" * 60)
    print()
    
    # 1. Entanglement entropy
    demonstrate_entanglement()
    
    # 2. Frame potential (t-design)
    demonstrate_frame_potential()
    
    # 3. Jacobian matrix (sensitivity)
    demonstrate_jacobian()
    
    # 4. Hessian matrix (curvature)
    demonstrate_hessian()
    
    # 5. Gradient optimization
    demonstrate_optimization()
    
    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
