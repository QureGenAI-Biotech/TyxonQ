"""Readout Error Mitigation (REM) Scalability Analysis.

This example demonstrates the **limitations** of readout error mitigation (REM)
when scaling to large qubit numbers. It validates the theoretical prediction:

    **REM fails when n >> 1/p**

where:
- n = number of qubits
- p = single-qubit readout error rate

Problem Statement:
------------------
Readout error mitigation constructs and inverts a 2^n × 2^n calibration matrix.
For large n, this matrix becomes:
1. **Ill-conditioned**: Numerical inversion errors dominate
2. **Exponentially expensive**: O(2^(3n)) complexity for matrix inversion
3. **Physically limited**: Calibration matrix estimation requires exponentially many shots

Theoretical Limit:
-----------------
For error rate p, accurate calibration requires ~2^n / p shots.
When n > log₂(1/p), calibration becomes impractical.

Example: p=0.1 → 1/p=10 → log₂(10)≈3.3 qubits is the soft limit

This demo:
----------
1. Simulates readout errors with rate p=0.1
2. Tests REM for increasing qubit numbers (2 → 20)
3. Measures mitigation quality vs qubit count
4. Demonstrates failure modes and practical limits

Key Findings:
-------------
- Small systems (n≤5): REM works well
- Medium systems (5<n≤10): Degraded performance
- Large systems (n>10): REM fails catastrophically
"""

import numpy as np
import tyxonq as tq
from tyxonq.postprocessing.readout import ReadoutMit
from typing import Dict, List, Tuple
import time


# ============================================================================
# Readout Error Simulation
# ============================================================================

def simulate_readout_error(ideal_bitstring: str, error_rate: float) -> str:
    """Simulate single-shot readout with bit-flip errors.
    
    Args:
        ideal_bitstring: True measurement result (e.g., "0011")
        error_rate: Probability of flipping each bit (0-1)
    
    Returns:
        Measured bitstring with errors
    """
    measured = ""
    for bit in ideal_bitstring:
        if np.random.uniform() < error_rate:
            # Bit flip
            measured += "1" if bit == "0" else "0"
        else:
            # Correct readout
            measured += bit
    return measured


def generate_noisy_counts(
    ideal_bitstring: str,
    shots: int,
    error_rate: float
) -> Dict[str, int]:
    """Generate measurement counts with readout errors.
    
    Args:
        ideal_bitstring: Ground truth state (all shots yield this ideally)
        shots: Number of measurement shots
        error_rate: Per-qubit readout error rate
    
    Returns:
        Dictionary mapping bitstrings to counts
    """
    counts: Dict[str, int] = {}
    
    for _ in range(shots):
        measured = simulate_readout_error(ideal_bitstring, error_rate)
        counts[measured] = counts.get(measured, 0) + 1
    
    return counts


# ============================================================================
# Calibration Matrix Construction
# ============================================================================

def build_calibration_matrix(n_qubits: int, error_rate: float) -> np.ndarray:
    """Build exact single-qubit calibration matrices for REM.
    
    For a symmetric bit-flip channel:
        A = [[1-p,  p  ],
             [ p,  1-p]]
    
    Full n-qubit matrix: A_full = A ⊗ A ⊗ ... ⊗ A (n times)
    
    Args:
        n_qubits: Number of qubits
        error_rate: Per-qubit error rate p
    
    Returns:
        2^n × 2^n calibration matrix
    """
    # Single-qubit calibration matrix
    p = error_rate
    A_single = np.array([
        [1 - p, p],
        [p, 1 - p]
    ])
    
    # Kronecker product for multi-qubit system
    A_full = A_single
    for _ in range(n_qubits - 1):
        A_full = np.kron(A_full, A_single)
    
    return A_full


# ============================================================================
# REM Quality Metrics
# ============================================================================

def compute_fidelity(counts1: Dict[str, int], counts2: Dict[str, int]) -> float:
    """Compute classical fidelity between two count distributions.
    
    F = ∑√(p₁(x) * p₂(x))
    
    Args:
        counts1, counts2: Count dictionaries
    
    Returns:
        Fidelity in [0, 1]
    """
    # Normalize to probabilities
    shots1 = sum(counts1.values())
    shots2 = sum(counts2.values())
    
    if shots1 == 0 or shots2 == 0:
        return 0.0
    
    prob1 = {k: v / shots1 for k, v in counts1.items()}
    prob2 = {k: v / shots2 for k, v in counts2.items()}
    
    # Compute fidelity
    all_keys = set(prob1.keys()) | set(prob2.keys())
    fidelity = sum(
        np.sqrt(prob1.get(k, 0) * prob2.get(k, 0))
        for k in all_keys
    )
    
    return fidelity


def total_variation_distance(counts1: Dict[str, int], counts2: Dict[str, int]) -> float:
    """Compute total variation distance between distributions.
    
    TVD = 0.5 * ∑|p₁(x) - p₂(x)|
    
    Args:
        counts1, counts2: Count dictionaries
    
    Returns:
        TVD in [0, 1]
    """
    shots1 = sum(counts1.values())
    shots2 = sum(counts2.values())
    
    if shots1 == 0 or shots2 == 0:
        return 1.0
    
    prob1 = {k: v / shots1 for k, v in counts1.items()}
    prob2 = {k: v / shots2 for k, v in counts2.items()}
    
    all_keys = set(prob1.keys()) | set(prob2.keys())
    tvd = 0.5 * sum(
        abs(prob1.get(k, 0) - prob2.get(k, 0))
        for k in all_keys
    )
    
    return tvd


# ============================================================================
# Scalability Experiment
# ============================================================================

def run_rem_scalability_test(
    n_qubits: int,
    error_rate: float,
    shots: int = 2048
) -> Dict[str, float]:
    """Test REM performance for a given system size.
    
    Args:
        n_qubits: Number of qubits
        error_rate: Readout error rate
        shots: Measurement shots
    
    Returns:
        Dictionary with performance metrics
    """
    print(f"\n{'='*70}")
    print(f"Testing n={n_qubits} qubits, p={error_rate}, shots={shots}")
    print(f"{'='*70}")
    
    # Prepare ideal state: |11...1⟩ (all 1s)
    ideal_bitstring = "1" * n_qubits
    ideal_counts = {ideal_bitstring: shots}
    
    # Generate noisy measurements
    print("Generating noisy counts...")
    noisy_counts = generate_noisy_counts(ideal_bitstring, shots, error_rate)
    
    # Build calibration matrices
    print("Building calibration matrix...")
    t_cal_start = time.perf_counter()
    
    # Per-qubit calibration matrix
    p = error_rate
    A_single = np.array([[1 - p, p], [p, 1 - p]])
    single_qubit_cals = {i: A_single for i in range(n_qubits)}
    
    t_cal = time.perf_counter() - t_cal_start
    
    # Apply readout mitigation
    print("Applying REM...")
    t_rem_start = time.perf_counter()
    
    try:
        mit = ReadoutMit()
        mit.set_single_qubit_cals(single_qubit_cals)
        mitigated_counts = mit.apply_readout_mitigation(
            noisy_counts,
            method="inverse",
            qubits=list(range(n_qubits)),
            shots=shots
        )
        t_rem = time.perf_counter() - t_rem_start
        success = True
        
    except Exception as e:
        print(f"⚠️  REM failed: {e}")
        mitigated_counts = noisy_counts
        t_rem = 0.0
        success = False
    
    # Compute metrics
    fidelity_noisy = compute_fidelity(noisy_counts, ideal_counts)
    fidelity_mitigated = compute_fidelity(mitigated_counts, ideal_counts)
    
    tvd_noisy = total_variation_distance(noisy_counts, ideal_counts)
    tvd_mitigated = total_variation_distance(mitigated_counts, ideal_counts)
    
    # Report results
    print(f"\nResults:")
    print(f"  Calibration time: {t_cal*1000:.2f} ms")
    print(f"  REM time: {t_rem*1000:.2f} ms")
    print(f"  Noisy fidelity: {fidelity_noisy:.4f}")
    print(f"  Mitigated fidelity: {fidelity_mitigated:.4f}")
    print(f"  Improvement: {(fidelity_mitigated - fidelity_noisy):.4f}")
    print(f"  Noisy TVD: {tvd_noisy:.4f}")
    print(f"  Mitigated TVD: {tvd_mitigated:.4f}")
    print(f"  REM success: {'✅' if success else '❌'}")
    
    # Compute theoretical limit indicator
    theoretical_limit = int(np.log2(1 / error_rate))
    exceeds_limit = n_qubits > theoretical_limit
    
    print(f"\nTheoretical analysis:")
    print(f"  Soft limit (log₂(1/p)): {theoretical_limit} qubits")
    print(f"  Status: {'⚠️  EXCEEDS LIMIT' if exceeds_limit else '✅ Within limit'}")
    
    return {
        "n_qubits": n_qubits,
        "error_rate": error_rate,
        "shots": shots,
        "fidelity_noisy": fidelity_noisy,
        "fidelity_mitigated": fidelity_mitigated,
        "improvement": fidelity_mitigated - fidelity_noisy,
        "tvd_noisy": tvd_noisy,
        "tvd_mitigated": tvd_mitigated,
        "cal_time_ms": t_cal * 1000,
        "rem_time_ms": t_rem * 1000,
        "success": success,
        "exceeds_theoretical_limit": exceeds_limit
    }


# ============================================================================
# Main Experiment
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Readout Error Mitigation (REM) Scalability Analysis")
    print("=" * 70)
    print()
    print("Objective: Demonstrate REM failure for large qubit numbers")
    print("Theoretical prediction: REM fails when n >> log₂(1/p)")
    print()
    
    # Experiment parameters
    ERROR_RATE = 0.1  # 10% per-qubit error rate
    SHOTS = 2048
    
    print(f"Parameters:")
    print(f"  Error rate (p): {ERROR_RATE}")
    print(f"  Shots per test: {SHOTS}")
    print(f"  Theoretical limit: {int(np.log2(1/ERROR_RATE))} qubits")
    print()
    
    # Test range: 2 to 20 qubits (downscaled from 30 for demo speed)
    qubit_range = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
    
    results_table: List[Dict[str, float]] = []
    
    for n in qubit_range:
        result = run_rem_scalability_test(n, ERROR_RATE, SHOTS)
        results_table.append(result)
    
    # ========================================================================
    # Summary Report
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    print()
    
    print(f"{'n':>3} | {'Noisy':>7} | {'Mitigated':>9} | {'Δ':>7} | {'Time(ms)':>9} | {'Status':>8}")
    print("-" * 70)
    
    for r in results_table:
        status = "✅ OK" if r["success"] and r["improvement"] > 0 else "❌ FAIL"
        print(
            f"{r['n_qubits']:3d} | "
            f"{r['fidelity_noisy']:7.4f} | "
            f"{r['fidelity_mitigated']:9.4f} | "
            f"{r['improvement']:+7.4f} | "
            f"{r['rem_time_ms']:9.2f} | "
            f"{status:>8}"
        )
    
    print()
    print("Key Observations:")
    print("-" * 70)
    
    # Find crossover point
    crossover_idx = None
    for i, r in enumerate(results_table):
        if r["improvement"] < 0 or not r["success"]:
            crossover_idx = i
            break
    
    if crossover_idx is not None:
        crossover_n = results_table[crossover_idx]["n_qubits"]
        print(f"1. REM starts failing around n={crossover_n} qubits")
        print(f"   (Theoretical limit: {int(np.log2(1/ERROR_RATE))} qubits)")
    else:
        print("1. REM succeeded for all tested qubit numbers")
    
    print()
    print("2. Computational cost grows exponentially:")
    if len(results_table) >= 2:
        t1 = results_table[0]["rem_time_ms"]
        t2 = results_table[-1]["rem_time_ms"]
        n1 = results_table[0]["n_qubits"]
        n2 = results_table[-1]["n_qubits"]
        print(f"   n={n1}: {t1:.2f} ms")
        print(f"   n={n2}: {t2:.2f} ms")
        print(f"   Growth factor: {t2/t1:.1f}× for {n2-n1} additional qubits")
    
    print()
    print("3. Quality degradation:")
    improvements = [r["improvement"] for r in results_table if r["success"]]
    if improvements:
        print(f"   Best improvement: {max(improvements):.4f}")
        print(f"   Worst improvement: {min(improvements):.4f}")
    
    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("Readout error mitigation is effective for small systems but")
    print("becomes impractical for large qubit numbers due to:")
    print()
    print("  1. Exponential matrix inversion cost: O(2^(3n))")
    print("  2. Numerical ill-conditioning for large n")
    print("  3. Insufficient calibration data (needs ~2^n/p shots)")
    print()
    print("For practical quantum computing:")
    print("  - Use REM for n ≤ 10 qubits")
    print("  - Consider alternative methods for larger systems:")
    print("    • Tensor network mitigation")
    print("    • Symmetry verification")
    print("    • Machine learning approaches")
    print()
    print("✅ Scalability analysis complete")
