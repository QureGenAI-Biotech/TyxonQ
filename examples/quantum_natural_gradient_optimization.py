"""
Quantum Natural Gradient (QNG) Optimization Demonstration

This example showcases quantum natural gradient descent, a more efficient
optimization method for variational quantum algorithms compared to standard
gradient descent. QNG uses the quantum Fisher information matrix (QFIM) to
precondition gradients, accounting for the geometry of the quantum state space.

Key Concepts:
- Quantum Fisher Information Matrix (QFIM): Metric tensor of quantum state space
- Natural Gradient: Fisher-preconditioned gradient direction  
- Fubini-Study Metric: Geometric structure of quantum states
- Superior convergence for VQAs compared to Adam/SGD

Mathematical Background:
  Standard GD:  θ_{t+1} = θ_t - η ∇L(θ)
  Natural GD:   θ_{t+1} = θ_t - η F^{-1}(θ) ∇L(θ)
  where F is the quantum Fisher information matrix

Performance Benefits:
- Faster convergence (fewer iterations)
- Better optimization landscape navigation
- Reduced sensitivity to parametrization

References:
- Stokes et al., "Quantum Natural Gradient", Quantum 4, 269 (2020)
- Yamamoto, "On the Natural Gradient for Variational Quantum Eigensolver

", arXiv:1909.05074

Migrated from: examples-ng/quantumng.py
Utilizes: src/tyxonq/compiler/stages/gradients/qng.py
"""

import time
import numpy as np
import torch

import tyxonq as tq
from tyxonq.compiler.stages.gradients.qng import qng_metric


# ==================== Problem Setup ====================

N_QUBITS = 6  # System size
N_LAYERS = 2  # Ansatz depth
LEARNING_RATE = 0.02
N_EPOCHS = 30
REGULARIZATION = 1e-4  # QFI matrix regularization


# ==================== Hamiltonian: Transverse Field Ising Model ====================

def build_tfim_hamiltonian(n: int, J: float = 1.0, h: float = -1.0):
    """Construct TFIM Hamiltonian: H = -J Σ Z_i Z_{i+1} - h Σ X_i
    
    Args:
        n: Number of qubits
        J: ZZ coupling strength
        h: Transverse field strength
    
    Returns:
        Hamiltonian matrix (2^n × 2^n)
    """
    from tyxonq.libs.quantum_library.kernels.pauli import heisenberg_hamiltonian
    
    # Build edges for periodic boundary conditions
    edges = [(i, (i + 1) % n) for i in range(n)]
    
    # TFIM: J*ZZ coupling + h*X field
    H = heisenberg_hamiltonian(
        n, 
        edges,
        hzz=J,  # ZZ interaction
        hxx=0.0, hyy=0.0,  # No XX, YY
        hx=h, hy=0.0, hz=0.0  # Only X field
    )
    
    return H


# ==================== Variational Ansatz ====================

def hardware_efficient_ansatz(n: int, nlayers: int, params):
    """Hardware-efficient ansatz with RY rotations and CNOT entanglement
    
    Args:
        n: Number of qubits
        nlayers: Circuit depth
        params: Parameters shaped [nlayers, n, 2] for [RY, RZ] rotations
    
    Returns:
        Circuit instance
    """
    K = tq.get_backend()
    params = K.reshape(params, [nlayers, n, 2])
    
    c = tq.Circuit(n)
    
    # Initial layer: Hadamards for superposition
    for i in range(n):
        c.h(i)
    
    # Variational layers
    for layer in range(nlayers):
        # Entangling layer: CNOT ladder
        for i in range(n - 1):
            c.cnot(i, i + 1)
        c.cnot(n - 1, 0)  # Periodic boundary
        
        # Rotation layer
        for i in range(n):
            c.ry(i, theta=params[layer, i, 0])
            c.rz(i, theta=params[layer, i, 1])
    
    return c


def state_function(params):
    """Return quantum state for given parameters
    
    This function is used by QNG to compute Fisher information.
    Uses Circuit.state() method for convenient statevector access.
    """
    c = hardware_efficient_ansatz(N_QUBITS, N_LAYERS, params)
    return c.state()


# ==================== Energy Evaluation ====================

def compute_energy(params, hamiltonian):
    """Compute expectation value <ψ(θ)|H|ψ(θ)>
    
    Args:
        params: Variational parameters
        hamiltonian: Hamiltonian matrix
    
    Returns:
        Real-valued energy
    """
    K = tq.get_backend()
    
    # Get state vector
    psi = state_function(params)
    
    # Compute expectation: <ψ|H|ψ>
    # Use einsum to avoid conjugate bit issue in PyTorch
    if hasattr(K, 'name') and K.name == 'pytorch':
        # PyTorch path: use einsum to avoid conjugate() issues
        import torch
        psi_t = torch.as_tensor(psi, dtype=torch.complex64)
        H_t = torch.as_tensor(hamiltonian, dtype=torch.complex64)
        # <ψ|H|ψ> = Σ_ij ψ*_i H_ij ψ_j
        energy = torch.einsum('i,ij,j->', psi_t.conj(), H_t, psi_t)
        return torch.real(energy)
    else:
        # NumPy path: standard matrix multiplication
        psi_col = K.reshape(psi, [-1, 1])
        energy = K.reshape(psi_col.conj().T @ hamiltonian @ psi_col, [])
        return K.real(energy)


# ==================== Optimization Methods ====================

def standard_gradient_descent(params_init, hamiltonian, lr, n_steps):
    """Standard gradient descent using PyTorch autograd
    
    Args:
        params_init: Initial parameters
        hamiltonian: Hamiltonian matrix
        lr: Learning rate
        n_steps: Number of optimization steps
    
    Returns:
        Tuple of (final_energy, parameter_history, energy_history, time_per_step)
    """
    print("\n" + "=" * 70)
    print("Standard Gradient Descent (PyTorch Autograd)")
    print("=" * 70)
    
    # Ensure PyTorch backend
    tq.set_backend("pytorch")
    K = tq.get_backend()
    
    # Convert to PyTorch tensor with gradients
    params = torch.tensor(params_init, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.SGD([params], lr=lr)
    
    energy_history = []
    param_history = []
    times = []
    
    print(f"{'Step':<8} {'Energy':<15} {'Time (s)':<10}")
    print("-" * 70)
    
    for step in range(n_steps):
        t0 = time.time()
        
        optimizer.zero_grad()
        
        # Compute energy
        energy = compute_energy(params, hamiltonian)
        
        # Backward pass
        energy.backward()
        
        # Gradient descent step
        optimizer.step()
        
        t1 = time.time()
        times.append(t1 - t0)
        
        # Record
        energy_val = float(energy.detach())
        energy_history.append(energy_val)
        param_history.append(params.detach().clone().numpy())
        
        if step % 5 == 0 or step == n_steps - 1:
            print(f"{step:<8} {energy_val:<15.8f} {times[-1]:<10.4f}")
    
    avg_time = np.mean(times[1:])  # Exclude first (compilation)
    print(f"\nAverage time per step: {avg_time:.4f}s")
    
    return energy_val, param_history, energy_history, avg_time


def quantum_natural_gradient(params_init, hamiltonian, lr, n_steps, reg=1e-4):
    """Quantum natural gradient descent using QFI
    
    Args:
        params_init: Initial parameters  
        hamiltonian: Hamiltonian matrix
        lr: Learning rate
        n_steps: Number of optimization steps
        reg: Regularization for QFI matrix inversion
    
    Returns:
        Tuple of (final_energy, parameter_history, energy_history, time_per_step)
    """
    print("\n" + "=" * 70)
    print("Quantum Natural Gradient Descent (QFI-based)")
    print("=" * 70)
    
    # Use NumPy for QNG computation (qng_metric requires numpy-compatible)
    tq.set_backend("numpy")
    K = tq.get_backend()
    
    params = np.array(params_init, dtype=np.float32)
    
    energy_history = []
    param_history = []
    times = []
    
    print(f"{'Step':<8} {'Energy':<15} {'Grad Norm':<12} {'Nat Grad Norm':<15} {'Time (s)':<10}")
    print("-" * 70)
    
    for step in range(n_steps):
        t0 = time.time()
        
        # Compute energy and gradient
        def energy_fn(p):
            return float(compute_energy(p, hamiltonian))
        
        # Numerical gradient via finite differences
        eps_grad = 1e-5
        flat_params = params.flatten()
        grad = np.zeros_like(flat_params)
        
        for i in range(len(flat_params)):
            params_plus = flat_params.copy()
            params_minus = flat_params.copy()
            params_plus[i] += eps_grad
            params_minus[i] -= eps_grad
            
            grad[i] = (energy_fn(params_plus.reshape(params.shape)) - 
                      energy_fn(params_minus.reshape(params.shape))) / (2 * eps_grad)
        
        grad = grad.reshape(params.shape)
        
        # Compute QFI matrix
        qfi = qng_metric(state_function, params, eps=1e-5, kernel="qng")
        
        # Regularize QFI for numerical stability
        qfi_reg = qfi + reg * np.eye(qfi.shape[0])
        
        # Compute natural gradient: F^{-1} grad
        try:
            nat_grad = np.linalg.solve(qfi_reg, grad.flatten())
            nat_grad = nat_grad.reshape(params.shape)
        except np.linalg.LinAlgError:
            print(f"Warning: QFI singular at step {step}, using standard gradient")
            nat_grad = grad
        
        # Update parameters
        params = params - lr * nat_grad
        
        t1 = time.time()
        times.append(t1 - t0)
        
        # Compute current energy
        energy_val = energy_fn(params)
        energy_history.append(energy_val)
        param_history.append(params.copy())
        
        if step % 5 == 0 or step == n_steps - 1:
            grad_norm = np.linalg.norm(grad)
            nat_grad_norm = np.linalg.norm(nat_grad)
            print(f"{step:<8} {energy_val:<15.8f} {grad_norm:<12.6f} {nat_grad_norm:<15.6f} {times[-1]:<10.4f}")
    
    avg_time = np.mean(times[1:])
    print(f"\nAverage time per step: {avg_time:.4f}s")
    
    return energy_val, param_history, energy_history, avg_time


# ==================== Comparison & Visualization ====================

def compare_optimizers():
    """Run and compare standard GD vs QNG"""
    print("\n" + "=" * 70)
    print("TyxonQ Quantum Natural Gradient Optimization Demo")
    print("Problem: Transverse Field Ising Model Ground State")
    print("=" * 70)
    
    print(f"\nConfiguration:")
    print(f"  Qubits: {N_QUBITS}")
    print(f"  Layers: {N_LAYERS}")
    print(f"  Parameters: {N_LAYERS * N_QUBITS * 2}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {N_EPOCHS}")
    
    # Build Hamiltonian
    print("\nBuilding TFIM Hamiltonian...")
    H = build_tfim_hamiltonian(N_QUBITS, J=1.0, h=-1.0)
    
    # Exact ground state energy (for reference)
    eigenvalues = np.linalg.eigvalsh(H)
    exact_gs_energy = eigenvalues[0]
    print(f"Exact ground state energy: {exact_gs_energy:.8f}")
    
    # Initialize parameters (same for both methods)
    np.random.seed(42)
    params_init = np.random.randn(N_LAYERS, N_QUBITS, 2) * 0.1
    
    # Run standard gradient descent
    e_sgd, params_sgd, energies_sgd, time_sgd = standard_gradient_descent(
        params_init, H, LEARNING_RATE, N_EPOCHS
    )
    
    # Run quantum natural gradient
    e_qng, params_qng, energies_qng, time_qng = quantum_natural_gradient(
        params_init, H, LEARNING_RATE, N_EPOCHS, reg=REGULARIZATION
    )
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("Optimization Comparison Summary")
    print("=" * 70)
    print(f"{'Method':<25} {'Final Energy':<20} {'Error':<15} {'Time/Step (s)':<15}")
    print("-" * 70)
    
    error_sgd = abs(e_sgd - exact_gs_energy)
    error_qng = abs(e_qng - exact_gs_energy)
    
    print(f"{'Standard GD':<25} {e_sgd:<20.8f} {error_sgd:<15.8f} {time_sgd:<15.4f}")
    print(f"{'Quantum Natural GD':<25} {e_qng:<20.8f} {error_qng:<15.8f} {time_qng:<15.4f}")
    print(f"{'Exact (Diagonalization)':<25} {exact_gs_energy:<20.8f} {0.0:<15.8f} {'N/A':<15}")
    
    print("\n" + "=" * 70)
    print("Key Insights")
    print("=" * 70)
    print("1. QNG accounts for quantum state space geometry via Fisher metric")
    print("2. Natural gradient is invariant to parameter reparametrization")  
    print("3. Typically achieves lower energy in same number of steps")
    print("4. Computational cost: O(P^3) for QFI inversion (P = #parameters)")
    print("5. Best suited for small-medium parameter counts (<100)")
    
    print("\nFuture Enhancements:")
    print("- Block-diagonal QFI approximation for scaling")
    print("- Stochastic QNG with parameter subsampling")
    print("- Integration with TyxonQ optimizer library")
    
    return {
        'energies_sgd': energies_sgd,
        'energies_qng': energies_qng,
        'exact': exact_gs_energy
    }


if __name__ == "__main__":
    results = compare_optimizers()
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
    
    print("\nNext Steps:")
    print("1. Experiment with different learning rates")
    print("2. Try larger systems (8-10 qubits)")
    print("3. Compare with Adam optimizer")
    print("4. Test on other Hamiltonians (e.g., molecular systems)")
    print("5. Implement block-diagonal QFI for scalability")
