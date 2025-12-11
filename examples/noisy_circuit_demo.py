"""Noisy Circuit Simulation Demo.

This example demonstrates:
1. How to simulate quantum circuits with realistic noise
2. Comparing noiseless vs noisy results
3. Different noise models: depolarizing, amplitude damping, phase damping
4. Using density matrix simulator for mixed states

The density matrix simulator in TyxonQ natively supports noise through Kraus operators,
making it ideal for studying realistic quantum hardware imperfections.
"""

import tyxonq as tq
import numpy as np


def create_bell_circuit(n_qubits=2):
    """Create a Bell state circuit |Φ+⟩ = (|00⟩ + |11⟩)/√2."""
    c = tq.Circuit(n_qubits)
    c.h(0)
    c.cx(0, 1)
    return c


def create_ghz_circuit(n_qubits=3):
    """Create a GHZ state circuit |GHZ⟩ = (|000⟩ + |111⟩)/√2."""
    c = tq.Circuit(n_qubits)
    c.h(0)
    for i in range(n_qubits - 1):
        c.cx(i, i + 1)
    return c


def run_noiseless_simulation():
    """Run noiseless simulation using density matrix simulator."""
    print("=" * 60)
    print("Noiseless Simulation (Ideal Quantum Computer)")
    print("=" * 60)
    
    # Configure to use density matrix simulator
    tq.set_backend("numpy")
    
    # Create Bell state circuit
    c = create_bell_circuit(2)
    
    # Run with density matrix simulator, shots=0 for exact result
    results = c.device(provider="simulator", device="density_matrix").run(shots=1024)
    
    print("\nBell State Circuit (|00⟩ + |11⟩)/√2:")
    print("Expected: 50% |00⟩, 50% |11⟩")
    print("Results:", results[0]["result"])
    
    # Check purity (should be 1.0 for pure states)
    counts = results[0]["result"]
    total = sum(counts.values())
    probs = {k: v/total for k, v in counts.items()}
    print(f"\nProbabilities: {probs}")
    
    return results


def run_depolarizing_noise():
    """Run simulation with depolarizing noise."""
    print("\n" + "=" * 60)
    print("Depolarizing Noise Simulation")
    print("=" * 60)
    print("\nDepolarizing noise: Each qubit has probability p of being")
    print("replaced by a random Pauli operator (I, X, Y, or Z).")
    
    tq.set_backend("numpy")
    
    # Noise parameters
    noise_levels = [0.01, 0.05, 0.1]
    
    for p in noise_levels:
        print(f"\n--- Noise level p = {p} ---")
        
        c = create_bell_circuit(2)
        
        # Configure noise through device options
        noise_config = {
            "type": "depolarizing",
            "p": p
        }
        
        results = c.device(
            provider="simulator",
            device="density_matrix",
            use_noise=True,
            noise=noise_config
        ).run(shots=2048)
        
        counts = results[0]["result"]
        total = sum(counts.values())
        
        # Calculate fidelity to ideal Bell state
        ideal_prob = 0.5
        measured_00 = counts.get("00", 0) / total
        measured_11 = counts.get("11", 0) / total
        fidelity = measured_00 + measured_11
        
        print(f"Results: {dict(sorted(counts.items()))}")
        print(f"Fidelity to ideal Bell state: {fidelity:.4f}")
        print(f"Expected degradation: ~{1 - (1-4*p/3)**2:.4f}")


def run_amplitude_damping():
    """Run simulation with amplitude damping (T1 decay)."""
    print("\n" + "=" * 60)
    print("Amplitude Damping (T1 Decay) Simulation")
    print("=" * 60)
    print("\nAmplitude damping: Models energy loss, |1⟩ → |0⟩ transition.")
    print("Common in superconducting qubits due to T1 relaxation.")
    
    tq.set_backend("numpy")
    
    # Start with |+⟩ state (superposition)
    c = tq.Circuit(1)
    c.h(0)
    c.h(0)  # Apply H twice to test gate application
    
    gamma_values = [0.0, 0.1, 0.3, 0.5]
    
    for gamma in gamma_values:
        print(f"\n--- Damping rate γ = {gamma} ---")
        
        # Reset circuit
        c = tq.Circuit(1)
        c.h(0)  # |+⟩ = (|0⟩ + |1⟩)/√2
        c.x(0)   # Now |1⟩
        
        noise_config = {
            "type": "amplitude_damping",
            "gamma": gamma
        }
        
        results = c.device(
            provider="simulator",
            device="density_matrix",
            use_noise=True,
            noise=noise_config
        ).run(shots=2048)
        
        counts = results[0]["result"]
        total = sum(counts.values())
        prob_0 = counts.get("0", 0) / total
        prob_1 = counts.get("1", 0) / total
        
        print(f"P(|0⟩) = {prob_0:.4f}, P(|1⟩) = {prob_1:.4f}")
        print(f"Expected P(|0⟩) ≈ {gamma:.4f} (for single gate)")


def run_phase_damping():
    """Run simulation with phase damping (T2 dephasing)."""
    print("\n" + "=" * 60)
    print("Phase Damping (T2 Dephasing) Simulation")
    print("=" * 60)
    print("\nPhase damping: Models loss of quantum coherence without energy loss.")
    print("Destroys superposition but preserves population in computational basis.")
    
    tq.set_backend("numpy")
    
    # Create superposition |+⟩
    c = tq.Circuit(1)
    c.h(0)
    
    lambda_values = [0.0, 0.2, 0.5, 0.8]
    
    for lam in lambda_values:
        print(f"\n--- Dephasing rate λ = {lam} ---")
        
        c = tq.Circuit(1)
        c.h(0)  # |+⟩ state
        
        noise_config = {
            "type": "phase_damping",
            "lambda": lam
        }
        
        results = c.device(
            provider="simulator",
            device="density_matrix",
            use_noise=True,
            noise=noise_config
        ).run(shots=2048)
        
        counts = results[0]["result"]
        total = sum(counts.values())
        prob_0 = counts.get("0", 0) / total
        
        print(f"Results: {counts}")
        print(f"P(|0⟩) = {prob_0:.4f}")
        print(f"Note: Phase damping preserves |+⟩ populations (50/50)")
        print(f"      but destroys off-diagonal coherence")


def run_pauli_channel():
    """Run simulation with general Pauli channel."""
    print("\n" + "=" * 60)
    print("Pauli Channel Simulation")
    print("=" * 60)
    print("\nPauli channel: Applies X, Y, or Z errors with specified probabilities.")
    print("More flexible than depolarizing noise.")
    
    tq.set_backend("numpy")
    
    c = create_bell_circuit(2)
    
    # Asymmetric Pauli noise: more Z errors than X/Y
    noise_config = {
        "type": "pauli",
        "px": 0.01,
        "py": 0.01,
        "pz": 0.05  # Dominant dephasing
    }
    
    print(f"\nNoise: px={noise_config['px']}, py={noise_config['py']}, pz={noise_config['pz']}")
    
    results = c.device(
        provider="simulator",
        device="density_matrix",
        use_noise=True,
        noise=noise_config
    ).run(shots=2048)
    
    counts = results[0]["result"]
    print(f"Results: {dict(sorted(counts.items()))}")
    
    total = sum(counts.values())
    bell_fidelity = (counts.get("00", 0) + counts.get("11", 0)) / total
    print(f"Bell state fidelity: {bell_fidelity:.4f}")


def compare_noise_models():
    """Compare different noise models side-by-side."""
    print("\n" + "=" * 60)
    print("Noise Model Comparison: GHZ State (3 qubits)")
    print("=" * 60)
    
    tq.set_backend("numpy")
    
    c = create_ghz_circuit(3)
    
    noise_configs = [
        {"type": None, "label": "Noiseless"},
        {"type": "depolarizing", "p": 0.05, "label": "Depolarizing (p=0.05)"},
        {"type": "amplitude_damping", "gamma": 0.1, "label": "Amplitude Damping (γ=0.1)"},
        {"type": "phase_damping", "lambda": 0.1, "label": "Phase Damping (λ=0.1)"},
    ]
    
    print("\nGHZ state: (|000⟩ + |111⟩)/√2")
    print("Ideal: 50% |000⟩, 50% |111⟩\n")
    
    for config in noise_configs:
        print(f"--- {config['label']} ---")
        
        c_test = create_ghz_circuit(3)
        
        if config["type"] is None:
            results = c_test.device(provider="simulator", device="density_matrix").run(shots=2048)
        else:
            noise_params = {k: v for k, v in config.items() if k not in ["label"]}
            results = c_test.device(
                provider="simulator",
                device="density_matrix",
                use_noise=True,
                noise=noise_params
            ).run(shots=2048)
        
        counts = results[0]["result"]
        total = sum(counts.values())
        
        # Show top 4 outcomes
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:4]
        for bitstr, count in sorted_counts:
            print(f"  {bitstr}: {count/total*100:.2f}%")
        
        ghz_fidelity = (counts.get("000", 0) + counts.get("111", 0)) / total
        print(f"  GHZ fidelity: {ghz_fidelity:.4f}\n")


def main():
    """Run all noise simulation demos."""
    print("\n" + "=" * 60)
    print("TyxonQ Noisy Circuit Simulation Demo")
    print("=" * 60)
    print("\nThis demo showcases realistic quantum noise simulation")
    print("using the density matrix simulator with Kraus operators.\n")
    
    # Run demonstrations
    run_noiseless_simulation()
    run_depolarizing_noise()
    run_amplitude_damping()
    run_phase_damping()
    run_pauli_channel()
    compare_noise_models()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. Density matrix simulator supports 4 noise types")
    print("  2. Noise degrades quantum state fidelity")
    print("  3. Different noise models have different effects:")
    print("     - Depolarizing: uniform random errors")
    print("     - Amplitude damping: energy loss (T1)")
    print("     - Phase damping: decoherence (T2)")
    print("     - Pauli channel: custom error rates")
    print("  4. Configure via device(..., use_noise=True, noise={...})")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
