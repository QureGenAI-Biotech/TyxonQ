"""
Quantum Optimal Control: Time Minimization
===========================================

This example demonstrates:
- Time-dependent Hamiltonian evolution
- Quantum optimal control with PyTorch autograd
- Joint optimization of control parameters and evolution time
- Physics-informed loss with regularization

This is an advanced example showing how to:
1. Define a control Hamiltonian H(t, θ) = sin(θ) * X
2. Minimize evolution time while achieving target expectation value
3. Use PyTorch automatic differentiation for gradient-based optimization

Author: TyxonQ Development Team
Date: 2024
"""

from __future__ import annotations

import torch
import tyxonq as tq
from tyxonq.libs.quantum_library.dynamics import PauliSumCOO


def build_control_hamiltonian() -> PauliSumCOO:
    """Build control Hamiltonian: H_c = X (single-qubit Pauli-X).
    
    Returns:
        PauliSumCOO representation of control Hamiltonian
    """
    # X gate on qubit 0: Pauli encoding [1] means X
    terms = [[1]]
    weights = [1.0]
    return PauliSumCOO(terms, weights)


def time_dependent_evolution(
    initial_circuit: tq.Circuit,
    h_func,
    time: torch.Tensor,
    control_param: torch.Tensor,
    steps: int = 50
) -> tq.Circuit:
    """Evolve quantum state under time-dependent Hamiltonian.
    
    Uses first-order time integration: |ψ(t+dt)⟩ = (1 - i H(t) dt) |ψ(t)⟩
    
    Args:
        initial_circuit: Initial quantum circuit with state
        h_func: Function h(t, θ) returning Hamiltonian as dense matrix
        time: Total evolution time (PyTorch tensor)
        control_param: Control parameters θ (PyTorch tensor)
        steps: Number of discretization steps
        
    Returns:
        Final quantum circuit after evolution
    """
    K = tq.get_backend()
    
    # Get initial state
    current_state = initial_circuit.state()
    n_qubits = initial_circuit.num_qubits
    
    # Time step
    dt = time / steps
    
    # Evolve step by step
    for i in range(steps):
        current_time = i * dt
        
        # Get Hamiltonian at current time H(t, θ)
        H = h_func(current_time, control_param)
        
        # Convert sparse to dense if needed
        # Note: Dense matrices are needed for matrix-vector multiplication
        if hasattr(H, 'todense'):
            H = H.todense()
        elif hasattr(H, 'to_dense'):
            H = H.to_dense()
        
        # First-order evolution: |ψ⟩ ← |ψ⟩ - i H |ψ⟩ dt
        H_times_dt = -1.0j * H * dt
        state_column = K.reshape(current_state, [-1, 1])
        delta_state = H_times_dt @ state_column
        current_state = current_state + K.reshape(delta_state, [-1])
    
    # Return new circuit with evolved state
    return tq.Circuit(n_qubits, inputs=current_state)


def objective_function(
    time: torch.Tensor,
    control_param: torch.Tensor,
    *,
    target_z_expval: float = -1.0,
    time_penalty: float = 0.08
) -> tuple[torch.Tensor, torch.Tensor]:
    """Objective function for optimal control.
    
    Goal: Minimize evolution time while achieving target Z expectation value.
    
    Loss = |⟨Z⟩ - target|² + λ * t²
    
    where:
    - Physics loss: ensures target state is reached
    - Time penalty: encourages shorter evolution time
    
    Args:
        time: Evolution time to optimize
        control_param: Control parameter θ in H(t,θ) = sin(θ) * X
        target_z_expval: Target expectation value ⟨Z⟩ (default: -1 for |1⟩ state)
        time_penalty: Regularization weight λ for time minimization
        
    Returns:
        Tuple of (total_loss, physics_loss)
    """
    K = tq.get_backend()
    
    # Build control Hamiltonian H_c = X
    H_pauli = build_control_hamiltonian()
    H_dense = H_pauli.to_dense()
    
    # Time-dependent Hamiltonian: H(t, θ) = sin(θ) * X
    def h_func(t, theta):
        return K.sin(theta) * H_dense
    
    # Initial state |0⟩
    initial_circuit = tq.Circuit(1)  # Defaults to |0⟩
    
    # Evolve under control Hamiltonian
    final_circuit = time_dependent_evolution(
        initial_circuit, 
        h_func, 
        time, 
        control_param, 
        steps=50
    )
    
    # Measure Z expectation value
    from tyxonq.libs.quantum_library.kernels.gates import gate_z
    z_gate = gate_z()
    z_expval = K.real(final_circuit.expectation([z_gate, [0]]))
    
    # Physics loss: how far from target state
    physics_loss = (z_expval - target_z_expval) ** 2
    
    # Time penalty: encourage shorter evolution time
    time_loss = time_penalty * (time ** 2)
    
    # Total loss
    total_loss = physics_loss + time_loss
    
    return total_loss, physics_loss


def optimize_control(
    initial_time: float = 1.0,
    initial_param: float = 0.5,
    learning_rate: float = 0.05,
    num_iterations: int = 100,
    print_every: int = 20
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run optimization to find optimal control parameters and time.
    
    Args:
        initial_time: Initial guess for evolution time
        initial_param: Initial guess for control parameter
        learning_rate: Adam optimizer learning rate
        num_iterations: Number of optimization iterations
        print_every: Print progress every N iterations
        
    Returns:
        Tuple of (optimized_time, optimized_param)
    """
    K = tq.get_backend()
    
    # Initialize optimization variables
    time = torch.tensor(
        initial_time, 
        dtype=torch.float64, 
        requires_grad=True
    )
    control_param = torch.tensor(
        initial_param, 
        dtype=torch.float64, 
        requires_grad=True
    )
    
    # Adam optimizer
    optimizer = torch.optim.Adam([time, control_param], lr=learning_rate)
    
    print("=" * 70)
    print("Quantum Optimal Control Optimization")
    print("=" * 70)
    print(f"\nOptimization settings:")
    print(f"  Initial time: {initial_time:.4f}")
    print(f"  Initial param: {initial_param:.4f}")
    print(f"  Learning rate: {learning_rate:.4f}")
    print(f"  Iterations: {num_iterations}")
    print(f"\nTarget: ⟨Z⟩ = -1.0 (flip |0⟩ → |1⟩ with minimal time)\n")
    print(f"{'Iter':<8} {'Total Loss':<15} {'Physics Loss':<15} {'Param θ':<12} {'Time t':<12}")
    print("-" * 70)
    
    # Optimization loop
    total_loss = None
    physics_loss = None
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Compute loss
        total_loss, physics_loss = objective_function(time, control_param)
        
        # Backpropagation
        total_loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Print progress
        if iteration % print_every == 0:
            print(f"{iteration:<8} {total_loss.item():<15.6f} {physics_loss.item():<15.6f} "
                  f"{control_param.item():<12.6f} {time.item():<12.6f}")
    
    print("-" * 70)
    print(f"\nOptimization completed!")
    print(f"  Final time: {time.item():.6f}")
    print(f"  Final param: {control_param.item():.6f}")
    if total_loss is not None and physics_loss is not None:
        print(f"  Final total loss: {total_loss.item():.6f}")
        print(f"  Final physics loss: {physics_loss.item():.6f}")
    
    return time, control_param


def verify_solution(time: torch.Tensor, control_param: torch.Tensor) -> None:
    """Verify the optimized solution by computing final state.
    
    Args:
        time: Optimized evolution time
        control_param: Optimized control parameter
    """
    K = tq.get_backend()
    
    print("\n" + "=" * 70)
    print("Solution Verification")
    print("=" * 70)
    
    # Build control Hamiltonian
    H_pauli = build_control_hamiltonian()
    H_dense = H_pauli.to_dense()
    
    def h_func(t, theta):
        return K.sin(theta) * H_dense
    
    # Evolve with optimized parameters
    initial_circuit = tq.Circuit(1)
    final_circuit = time_dependent_evolution(
        initial_circuit, h_func, time, control_param, steps=100
    )
    
    # Get final state
    final_state = final_circuit.state()
    
    # Calculate Z expectation
    from tyxonq.libs.quantum_library.kernels.gates import gate_z
    z_gate = gate_z()
    z_expval = K.real(final_circuit.expectation([z_gate, [0]]))
    
    print(f"\nFinal state (amplitude):")
    print(f"  |0⟩: {K.abs(final_state[0]).item():.6f}")
    print(f"  |1⟩: {K.abs(final_state[1]).item():.6f}")
    print(f"\nExpectation value:")
    print(f"  ⟨Z⟩ = {z_expval.item():.6f} (target: -1.0)")
    print(f"\nControl Hamiltonian:")
    print(f"  H(t, θ) = sin({control_param.item():.6f}) * X")
    print(f"  H(t, θ) ≈ {K.sin(control_param).item():.6f} * X")


def main():
    """Run the complete optimal control demonstration."""
    # Set PyTorch backend (required for autograd)
    K = tq.set_backend("pytorch")
    print(f"Using backend: {K.name}")
    print(f"Note: PyTorch backend is required for automatic differentiation\n")
    
    # Run optimization
    optimized_time, optimized_param = optimize_control(
        initial_time=1.0,
        initial_param=0.5,
        learning_rate=0.05,
        num_iterations=100,
        print_every=20
    )
    
    # Verify solution
    verify_solution(optimized_time, optimized_param)
    
    print("\n" + "=" * 70)
    print("Demonstration completed!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  • Time-dependent Hamiltonians enable flexible quantum control")
    print("  • PyTorch autograd simplifies gradient-based optimization")
    print("  • Physics-informed loss + regularization balance objectives")
    print("  • Joint optimization finds minimal-time control sequences")


if __name__ == "__main__":
    main()
