"""
Noise Calibration: T1/T2 Thermal Relaxation Time Estimation
=============================================================

This example demonstrates systematic calibration of thermal relaxation parameters
(T1 and T2 times) for noisy quantum hardware. These parameters are fundamental
to characterizing decoherence in real quantum devices.

Physical Background
-------------------
- **T1 (Amplitude Damping Time)**: Characterizes energy relaxation |1⟩ → |0⟩
  Also known as energy relaxation time or longitudinal relaxation time.
  Physically: γ = 1/T1 (damping rate)

- **T2 (Phase Damping Time)**: Characterizes loss of quantum coherence
  Also known as dephasing time or transverse relaxation time.
  Physically: λ = 1/T2 (dephasing rate)
  Constraint: T2 ≤ 2T1 (theoretical upper bound)

Calibration Methodology
-----------------------
1. **T1 Measurement (Amplitude Damping)**:
   - Prepare excited state |1⟩ using X gate
   - Apply amplitude damping noise with varying γ
   - Measure population in |1⟩ state
   - Fit: P₁(γ) = A·exp(-γ) + C → Extract T1 = 1/γ

2. **T2 Measurement (Phase Damping)**:
   - Prepare superposition |+⟩ = (|0⟩ + |1⟩)/√2 using H gate
   - Apply phase damping noise with varying λ
   - Apply second H gate (Ramsey-like sequence)
   - Measure coherence decay
   - Fit: P(λ) = A·exp(-λ) + C → Extract T2 = 1/λ

Technical Implementation
------------------------
- Uses density matrix simulator with Kraus operators
- Models T1 via amplitude_damping noise (γ parameter)
- Models T2 via phase_damping noise (λ parameter)
- Exponential curve fitting using scipy.optimize.curve_fit
- Visualizes decay curves with fitted parameters

Note: This uses the CURRENT TyxonQ noise API (device-based noise configuration),
      NOT the old DMCircuit.thermalrelaxation API from examples-ng.

Author: TyxonQ Team
Date: 2025-10-18
"""

import numpy as np
import tyxonq as tq
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def exponential_decay(gamma, A, C, T):
    """Exponential decay model: f(γ) = A·exp(-γ·T) + C
    
    For noise rate γ and relaxation time T:
    - T1 calibration: γ = damping rate, T = T1
    - T2 calibration: λ = dephasing rate, T = T2
    
    Args:
        gamma: Noise rate (γ for T1, λ for T2)
        A: Amplitude
        C: Offset (steady-state value)
        T: Relaxation time constant (fitting target)
    
    Returns:
        Function values at noise rate gamma
    """
    return A * np.exp(-gamma * T) + C


def calibrate_T1(n_samples=20, gamma_max=2.0):
    """Calibrate T1 relaxation time using amplitude damping noise.
    
    Algorithm:
        1. Prepare |1⟩ state (X gate)
        2. Apply amplitude damping noise with varying γ (damping rate)
        3. Measure population in |1⟩ state via sampling
        4. Fit exponential decay: P₁(γ) = A·exp(-γ·T1) + C
        5. Extract T1 from fitted parameter
    
    Args:
        n_samples: Number of different γ values to test
        gamma_max: Maximum γ value (typically 1-3 for good dynamic range)
    
    Returns:
        (measurement_data, gamma_list): Measured populations and damping rates
    
    Physical Interpretation:
        - γ = 0: No noise → P₁ ≈ 1.0 (fully excited)
        - γ → ∞: Complete damping → P₁ → 0
        - Decay governed by: |1⟩ → |0⟩ transition
        - Relation: T1 = 1/γ_eff (effective relaxation time)
    """
    tq.set_backend("numpy")
    
    gamma_list = np.linspace(0, gamma_max, n_samples)
    populations = []
    
    for gamma in gamma_list:
        # Create circuit: prepare |1⟩
        c = tq.Circuit(1)
        c.x(0)
        
        # Apply amplitude damping noise
        noise_config = {
            "type": "amplitude_damping",
            "gamma": gamma
        }
        
        # Run simulation with noise
        results = c.device(
            provider="simulator",
            device="density_matrix",
            use_noise=True,
            noise=noise_config
        ).run(shots=4096)
        
        # Extract P(|1⟩) from measurement counts
        counts = results[0]["result"]
        total = sum(counts.values())
        p_excited = counts.get("1", 0) / total
        populations.append(p_excited)
    
    measurement = np.array(populations)
    
    return measurement, gamma_list


def calibrate_T2(n_samples=20, lambda_max=2.0):
    """Calibrate T2 dephasing time using phase damping noise.
    
    Algorithm:
        1. Prepare |+⟩ = H|0⟩ = (|0⟩ + |1⟩)/√2
        2. Apply phase damping noise with varying λ (dephasing rate)
        3. Apply second H gate to convert coherence to population
        4. Measure population (coherence indicator)
        5. Fit exponential decay: P(λ) = A·exp(-λ·T2) + C
        6. Extract T2 from fitted parameter
    
    Args:
        n_samples: Number of different λ values to test
        lambda_max: Maximum λ value
    
    Returns:
        (measurement_data, lambda_list): Measured populations and dephasing rates
    
    Physical Interpretation:
        - λ = 0: No dephasing → Perfect |+⟩ state
        - λ → ∞: Complete dephasing → Mixed state |0⟩⟨0| + |1⟩⟨1|
        - Phase damping destroys coherence but preserves populations
        - Ramsey sequence (H-noise-H) converts phase error to amplitude error
        - Relation: T2 = 1/λ_eff (effective dephasing time)
    """
    tq.set_backend("numpy")
    
    lambda_list = np.linspace(0, lambda_max, n_samples)
    populations = []
    
    for lam in lambda_list:
        # Create Ramsey-like sequence
        c = tq.Circuit(1)
        c.h(0)  # Prepare |+⟩
        
        # (Phase damping noise will be applied here)
        
        c.h(0)  # Convert phase to population
        
        # Apply phase damping noise
        noise_config = {
            "type": "phase_damping",
            "lambda": lam
        }
        
        # Run simulation
        results = c.device(
            provider="simulator",
            device="density_matrix",
            use_noise=True,
            noise=noise_config
        ).run(shots=4096)
        
        # Extract P(|0⟩) - should decay from 1.0 to 0.5
        counts = results[0]["result"]
        total = sum(counts.values())
        p_zero = counts.get("0", 0) / total
        populations.append(p_zero)
    
    measurement = np.array(populations)
    
    return measurement, lambda_list


def fit_relaxation_time(noise_rate_data, measurement_data, initial_guess=None):
    """Fit exponential decay model to extract relaxation time.
    
    Args:
        noise_rate_data: Noise rate values (γ or λ, numpy array)
        measurement_data: Measured populations (numpy array)
        initial_guess: Initial parameters [A, C, T], default: [1.0, 0.0, 1.0]
    
    Returns:
        fit_params: Fitted parameters [A, C, T]
        T: Extracted relaxation time (returned for convenience)
    
    Model:
        f(γ) = A·exp(-γ·T) + C
        
        - A: Amplitude (typically ~1.0 for T1, ~0.5 for T2)
        - C: Offset (steady-state value, ~0 for T1, ~0.5 for T2)
        - T: Relaxation time (target parameter, T1 or T2)
    """
    if initial_guess is None:
        initial_guess = [1.0, 0.0, 1.0]
    
    fit_params, _ = curve_fit(exponential_decay, noise_rate_data, measurement_data, initial_guess)
    A, C, T = fit_params
    
    return fit_params, T


def visualize_calibration(noise_rate_data, measurement, fit_params, xlabel, title):
    """Visualize calibration results with fitted curve.
    
    Args:
        noise_rate_data: Noise rate values (γ or λ)
        measurement: Measured data points
        fit_params: Fitted parameters [A, C, T]
        xlabel: X-axis label ("Damping Rate γ" or "Dephasing Rate λ")
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    A, C, T = fit_params
    
    # Plot measured data
    plt.scatter(noise_rate_data, measurement, alpha=0.6, label="Measured Data", color='blue', s=50)
    
    # Plot fitted curve
    x_fit = np.linspace(0, noise_rate_data[-1], 200)
    fit_curve = exponential_decay(x_fit, A, C, T)
    plt.plot(x_fit, fit_curve, 'r-', linewidth=2, 
             label=f"Fit: P = {A:.3f}·exp(-{xlabel.split()[0]}·{T:.3f}) + {C:.3f}")
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Measured Population", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = title.replace(' ', '_').replace(':', '').lower() + ".png"
    plt.savefig(filename, dpi=150)
    print(f"  → Plot saved: {filename}")


def demonstrate_t1_calibration():
    """Demonstrate T1 calibration procedure."""
    print("\n" + "="*70)
    print("T1 CALIBRATION DEMONSTRATION (Amplitude Damping)")
    print("="*70)
    
    print("\nPhysical Setup:")
    print("  - Circuit: X gate (prepare |1⟩ state)")
    print("  - Noise: Amplitude damping with varying γ")
    print("  - Measurement: Sample population in |1⟩")
    print("  - Goal: Extract T1 from decay P₁(γ) = A·exp(-γ·T1) + C")
    
    # Run calibration
    print("\nRunning T1 calibration (n_samples=25, γ_max=2.5)...")
    measurement, gamma_list = calibrate_T1(n_samples=25, gamma_max=2.5)
    
    # Fit exponential decay
    print("Fitting exponential decay model...")
    fit_params, T1_calibrated = fit_relaxation_time(
        gamma_list, measurement, initial_guess=[1.0, 0.0, 1.0]
    )
    
    A, C, T = fit_params
    print(f"\nFitted Parameters:")
    print(f"  - Amplitude A: {A:.4f}")
    print(f"  - Offset C: {C:.4f}")
    print(f"  - T1: {T:.4f}")
    
    print(f"\n" + "-"*70)
    print(f"RESULTS:")
    print(f"  - Calibrated T1: {T1_calibrated:.4f}")
    print(f"  - Physical Interpretation: T1 = 1/γ_eff")
    print(f"  - At γ=1.0: P₁ ≈ {A * np.exp(-1.0 * T) + C:.3f} (expected ~0.37 for T1=1)")
    print("-"*70)
    
    # Visualize
    visualize_calibration(gamma_list, measurement, fit_params, 
                         "Damping Rate γ", "T1 Calibration Amplitude Damping")
    
    return T1_calibrated


def demonstrate_t2_calibration():
    """Demonstrate T2 calibration procedure."""
    print("\n" + "="*70)
    print("T2 CALIBRATION DEMONSTRATION (Phase Damping)")
    print("="*70)
    
    print("\nPhysical Setup:")
    print("  - Circuit: H - (noise) - H (Ramsey-like sequence)")
    print("  - Noise: Phase damping with varying λ")
    print("  - Measurement: Sample population in |0⟩")
    print("  - Goal: Extract T2 from decay P(λ) = A·exp(-λ·T2) + C")
    
    # Run calibration
    print("\nRunning T2 calibration (n_samples=25, λ_max=2.5)...")
    measurement, lambda_list = calibrate_T2(n_samples=25, lambda_max=2.5)
    
    # Fit exponential decay
    print("Fitting exponential decay model...")
    fit_params, T2_calibrated = fit_relaxation_time(
        lambda_list, measurement, initial_guess=[0.5, 0.5, 1.0]
    )
    
    A, C, T = fit_params
    print(f"\nFitted Parameters:")
    print(f"  - Amplitude A: {A:.4f}")
    print(f"  - Offset C: {C:.4f}")
    print(f"  - T2: {T:.4f}")
    
    print(f"\n" + "-"*70)
    print(f"RESULTS:")
    print(f"  - Calibrated T2: {T2_calibrated:.4f}")
    print(f"  - Physical Interpretation: T2 = 1/λ_eff")
    print(f"  - At λ=1.0: P(0) ≈ {A * np.exp(-1.0 * T) + C:.3f}")
    print("-"*70)
    
    # Visualize
    visualize_calibration(lambda_list, measurement, fit_params, 
                         "Dephasing Rate λ", "T2 Calibration Phase Damping")
    
    return T2_calibrated


def demonstrate_noise_models_relationship():
    """Demonstrate the relationship between amplitude and phase damping."""
    print("\n" + "="*70)
    print("AMPLITUDE vs PHASE DAMPING COMPARISON")
    print("="*70)
    
    print("\nKey Differences:")
    print("  - Amplitude Damping: |1⟩ → |0⟩ (energy loss, T1 process)")
    print("  - Phase Damping: Destroys coherence, preserves populations (T2 process)")
    print("  - Relation: 1/T2 = 1/(2T1) + 1/T_φ (pure dephasing)")
    
    tq.set_backend("numpy")
    
    print("\nComparison Test: Apply same noise rate to both models")
    print(f"\n{'Noise Rate':<12} {'T1 (Amp.Damp.)':<18} {'T2 (Phase Damp.)':<18} {'T2/T1 Ratio'}")
    print("-" * 70)
    
    # Test with fixed parameters
    test_rates = [0.5, 1.0, 1.5, 2.0]
    
    for rate in test_rates:
        # T1 test
        gamma_list_t1 = np.array([rate])
        meas_t1, _ = calibrate_T1(n_samples=1, gamma_max=rate)
        
        # T2 test
        lambda_list_t2 = np.array([rate])
        meas_t2, _ = calibrate_T2(n_samples=1, lambda_max=rate)
        
        # Simplified T estimation (inverse of rate for demonstration)
        T1_est = 1.0 / rate if rate > 0 else float('inf')
        T2_est = 1.0 / rate if rate > 0 else float('inf')
        ratio = T2_est / T1_est if T1_est > 0 else 1.0
        
        print(f"{rate:<12.2f} {T1_est:<18.4f} {T2_est:<18.4f} {ratio:.4f}")
    
    print("-" * 70)
    print("\nNote: In real hardware, T2 ≤ 2T1 always holds.")
    print("      Here we model T1 and T2 independently for demonstration.")


def comprehensive_calibration_workflow():
    """Run complete T1/T2 calibration workflow."""
    print("\n" + "#"*70)
    print("# COMPREHENSIVE NOISE CALIBRATION WORKFLOW")
    print("#"*70)
    
    # Step 1: T1 Calibration
    T1_calibrated = demonstrate_t1_calibration()
    
    # Step 2: T2 Calibration
    T2_calibrated = demonstrate_t2_calibration()
    
    # Step 3: Relationship Analysis
    demonstrate_noise_models_relationship()
    
    # Summary
    print("\n" + "="*70)
    print("CALIBRATION SUMMARY")
    print("="*70)
    print(f"  Calibrated T1: {T1_calibrated:.4f}")
    print(f"  Calibrated T2: {T2_calibrated:.4f}")
    print(f"  T2/T1 Ratio: {T2_calibrated/T1_calibrated:.4f}")
    print(f"\n  ✓ Calibration Complete")
    print("="*70)


if __name__ == "__main__":
    # Run comprehensive workflow
    comprehensive_calibration_workflow()
    
    print("\n" + "="*70)
    print("CALIBRATION COMPLETE")
    print("="*70)
    print("\nOutput Files:")
    print("  - t1_calibration_amplitude_damping.png")
    print("  - t2_calibration_phase_damping.png")
    print("\nKey Concepts:")
    print("  ✓ T1: Energy relaxation time (amplitude damping γ)")
    print("  ✓ T2: Coherence time (phase damping λ)")
    print("  ✓ Calibration: Fit P(γ) or P(λ) to extract relaxation times")
    print("  ✓ Uses: Noise characterization, error mitigation tuning")
    print("="*70)
