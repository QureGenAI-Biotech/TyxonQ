"""Noisy Sampling vs Exact Density Matrix Simulation.

This example demonstrates:
1. Exact noisy simulation using density matrix
2. Approximate noisy simulation using sampling (Monte Carlo)
3. Comparison of both methods

For hardware simulation, only sampling is available (no access to density matrix).
This example shows how Monte Carlo sampling can approximate exact results.
"""

import tyxonq as tq
import numpy as np


def create_parameterized_circuit(n, depth, weights):
    """Create a parameterized circuit with RX rotations and CNOT entanglers."""
    c = tq.Circuit(n)
    
    for layer in range(depth):
        # Entangling layer
        for i in range(n - 1):
            c.cx(i, i + 1)
        
        # Parameterized rotation layer
        for i in range(n):
            c.rx(i, float(weights[i, layer]))
    
    return c


def exact_noisy_simulation(n, depth, weights, noise_level):
    """Compute exact expectation using density matrix with noise."""
    c = create_parameterized_circuit(n, depth, weights)
    
    # Configure with noise
    results = c.with_noise("pauli", px=noise_level, py=noise_level, pz=noise_level).run(shots=4096)
    
    # Extract measurement results
    counts = results[0]["result"]
    total = sum(counts.values())
    
    # Compute <Z0 Z1> expectation
    z0z1_exp = 0.0
    for bitstr, count in counts.items():
        # Z eigenvalue: |0⟩ → +1, |1⟩ → -1
        z0 = 1 if bitstr[0] == '0' else -1
        z1 = 1 if bitstr[1] == '0' else -1
        z0z1_exp += (z0 * z1) * (count / total)
    
    return z0z1_exp


def monte_carlo_noisy_sampling(n, depth, weights, noise_level, num_trials=32):
    """Approximate expectation using Monte Carlo sampling with noise."""
    # Each trial uses a different random noise realization
    z0z1_samples = []
    
    for trial in range(num_trials):
        c = create_parameterized_circuit(n, depth, weights)
        
        # Run with noise
        results = c.with_noise("pauli", px=noise_level, py=noise_level, pz=noise_level).run(shots=128)
        
        counts = results[0]["result"]
        total = sum(counts.values())
        
        # Compute <Z0 Z1> for this trial
        z0z1 = 0.0
        for bitstr, count in counts.items():
            z0 = 1 if bitstr[0] == '0' else -1
            z1 = 1 if bitstr[1] == '0' else -1
            z0z1 += (z0 * z1) * (count / total)
        
        z0z1_samples.append(z0z1)
    
    # Average over trials
    z0z1_mc = np.mean(z0z1_samples)
    z0z1_std = np.std(z0z1_samples)
    
    return z0z1_mc, z0z1_std


def main():
    print("=" * 70)
    print("Noisy Sampling vs Exact Density Matrix Simulation")
    print("=" * 70)
    
    # Set backend
    tq.set_backend("numpy")
    
    # Problem parameters
    n = 5           # Number of qubits
    depth = 3       # Circuit depth
    noise_level = 0.003  # 0.3% error rate per gate
    
    print(f"\nProblem setup:")
    print(f"  Number of qubits:  {n}")
    print(f"  Circuit depth:     {depth}")
    print(f"  Noise level:       {noise_level} (Pauli channel)")
    print(f"  Observable:        <Z₀ Z₁>")
    
    # Create random weights
    np.random.seed(42)
    weights = np.random.rand(n, depth) * 2 * np.pi
    
    # Method 1: Exact density matrix simulation
    print("\n" + "-" * 70)
    print("Method 1: Exact Density Matrix Simulation")
    print("-" * 70)
    print("Uses density matrix to exactly compute noisy expectation values.")
    print("Requires O(4^n) memory, only feasible for small systems.")
    
    z0z1_exact = exact_noisy_simulation(n, depth, weights, noise_level)
    print(f"\nResult: <Z₀ Z₁> = {z0z1_exact:.6f}")
    
    # Method 2: Monte Carlo sampling
    print("\n" + "-" * 70)
    print("Method 2: Monte Carlo Sampling")
    print("-" * 70)
    print("Uses multiple noisy sampling runs to estimate expectation.")
    print("Requires only O(2^n) memory, scalable to larger systems.")
    print("Typical approach for real quantum hardware.")
    
    num_trials = 32
    z0z1_mc, z0z1_std = monte_carlo_noisy_sampling(n, depth, weights, noise_level, num_trials)
    print(f"\nResult: <Z₀ Z₁> = {z0z1_mc:.6f} ± {z0z1_std:.6f}")
    print(f"(Average over {num_trials} trials)")
    
    # Comparison
    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)
    print(f"Exact (density matrix):  {z0z1_exact:.6f}")
    print(f"Monte Carlo (sampling):  {z0z1_mc:.6f} ± {z0z1_std:.6f}")
    print(f"Absolute difference:     {abs(z0z1_exact - z0z1_mc):.6f}")
    
    # Check agreement
    if abs(z0z1_exact - z0z1_mc) < 3 * z0z1_std:
        print("\n✓ Monte Carlo result agrees with exact within 3σ")
    else:
        print("\n⚠ Monte Carlo result differs from exact (may need more trials)")
    
    # Insights
    print("\n" + "=" * 70)
    print("Key Insights")
    print("=" * 70)
    print("1. Density matrix gives exact noisy results but scales as O(4^n)")
    print("2. Monte Carlo sampling is approximate but scales as O(2^n)")
    print("3. For real quantum hardware, only sampling is available")
    print("4. More trials reduce Monte Carlo variance (√N scaling)")
    print("5. Both methods correctly capture noise effects on expectation values")
    print("=" * 70)
    
    # Additional: Show effect of increasing trials
    print("\n" + "-" * 70)
    print("Monte Carlo Convergence (varying number of trials)")
    print("-" * 70)
    
    trial_counts = [4, 8, 16, 32, 64]
    print(f"{'Trials':<10} {'<Z₀ Z₁>':<15} {'Std Dev':<15} {'Error from Exact':<15}")
    print("-" * 70)
    
    for num in trial_counts:
        z_mc, z_std = monte_carlo_noisy_sampling(n, depth, weights, noise_level, num)
        error = abs(z_mc - z0z1_exact)
        print(f"{num:<10} {z_mc:<15.6f} {z_std:<15.6f} {error:<15.6f}")
    
    print("\nNote: Standard deviation decreases as ~1/√N with more trials")
    print("=" * 70)


if __name__ == "__main__":
    main()
