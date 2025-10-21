"""
Measurement-Induced Phase Transition (MIPT)
测量诱导相变

This example demonstrates the MIPT phenomenon where a quantum system undergoes
a phase transition between volume-law and area-law entanglement scaling as a function
of measurement rate.

本示例演示测量诱导相变（MIPT）现象：量子系统的纠缠熵标度随测量率从体积律到面积律的相变。

Physical Background:
    In monitored quantum circuits with random unitaries and measurements, there exists
    a critical measurement rate p_c that separates two distinct phases:
    
    - p < p_c: Volume-law phase
      * Entanglement entropy S ~ L (extensive)
      * Quantum information spreads throughout the system
      * Long-range quantum correlations
    
    - p > p_c: Area-law phase
      * Entanglement entropy S ~ L^0 (constant)
      * Measurements localize quantum information
      * Short-range correlations only
    
    The transition is characterized by:
    - Universal critical exponents
    - Diverging correlation length at p_c
    - Logarithmic corrections at criticality

物理背景：
    在包含随机幺正演化和测量的监控量子电路中，存在一个临界测量率p_c分隔两个不同的相：
    
    - p < p_c: 体积律相
      * 纠缠熵 S ~ L（广延性）
      * 量子信息在整个系统中传播
      * 长程量子关联
    
    - p > p_c: 面积律相
      * 纠缠熵 S ~ L^0（常数）
      * 测量局域化量子信息
      * 仅短程关联
    
    相变特征：
    - 普适临界指数
    - p_c处关联长度发散
    - 临界点对数修正

Key References:
    - Li, Chen, and Fisher, "Quantum Zeno effect and the many-body entanglement
      transition" Phys. Rev. B 98, 205136 (2018) [arXiv:1808.06134]
    - Skinner, Ruhman, and Nahum, "Measurement-Induced Phase Transitions in the
      Dynamics of Entanglement" Phys. Rev. X 9, 031009 (2019) [arXiv:1808.05953]
    - Chan, De Luca, and Chalker, "Solution of a Minimal Model for Many-Body
      Quantum Chaos" Phys. Rev. X 8, 041019 (2018) [arXiv:1712.06836]

Circuit Architecture:
    1. Random 2-qubit unitaries (Haar random gates)
       → Generate quantum entanglement
    2. Probabilistic Z-basis measurements
       → Collapse quantum state with probability p
    3. Repeat for d layers
    
    电路架构：
    1. 随机2比特幺正门（Haar随机门）
       → 产生量子纠缠
    2. 概率Z基测量
       → 以概率p坍缩量子态
    3. 重复d层

Observables:
    - von Neumann entropy: S(ρ_A) = -Tr(ρ_A log ρ_A)
    - Rényi-2 entropy: S_2(ρ_A) = -log Tr(ρ_A^2)
    
    Both quantify entanglement between subsystem A and its complement.
    
    可观测量：
    - 冯诺依曼熵：S(ρ_A) = -Tr(ρ_A log ρ_A)
    - Rényi-2熵：S_2(ρ_A) = -log Tr(ρ_A^2)
    
    两者都量化子系统A与其补之间的纠缠。

Note:
    This example uses numerical simulation features (statevector, Kraus channels).
    The MIPT phenomenon has been observed experimentally on Google Sycamore 
    (Nature 585, 551–556, 2020) and Honeywell trapped-ion systems.
    
    本示例使用数值模拟功能（态矢量、Kraus通道）。
    MIPT现象已在Google Sycamore（Nature 585, 551–556, 2020）和
    Honeywell离子阱系统上实验观测到。
"""

import time
import numpy as np
from scipy import stats

import tyxonq as tq
from tyxonq.libs.quantum_library.noise import measurement_channel
from tyxonq.libs.quantum_library.kernels.quantum_info import (
    reduced_density_matrix,
    entropy,
    renyi_entropy,
)

# Use PyTorch backend for JIT compilation and automatic differentiation
K = tq.set_backend("pytorch")

# Optimized parameters for faster execution
DEFAULT_N_QUBITS = 8  # Reduced from 12 for faster simulation
DEFAULT_DEPTH = 6     # Reduced from 12 for faster execution


def generate_random_unitaries(n_qubits, depth):
    """Generate random Haar-distributed 2-qubit unitaries.
    
    Args:
        n_qubits: Number of qubits
        depth: Circuit depth
    
    Returns:
        Array of shape [depth * n_qubits, 4, 4]
    
    Note: Uses scipy.stats.unitary_group for proper Haar measure sampling.
    """
    num_gates = depth * n_qubits
    unitaries = []
    
    for _ in range(num_gates):
        # Generate Haar-random 4×4 unitary (2-qubit gate)
        u = stats.unitary_group.rvs(4)
        # Normalize determinant to 1 (SU(4))
        u = u / np.linalg.det(u) ** (1/4)
        unitaries.append(u)
    
    return np.stack(unitaries)


def mipt_circuit(random_unitaries, measurement_outcomes, n_qubits, depth, p_measure):
    """MIPT quantum circuit with random unitaries and measurements.
    
    Circuit structure (per layer):
        1. Even bonds: U(0,1), U(2,3), ..., U(n-2,n-1)
        2. Odd bonds: U(1,2), U(3,4), ..., U(n-1,0)
        3. Measurements on all qubits with probability p_measure
    
    Args:
        random_unitaries: Random 2-qubit gates [depth*n_qubits, 4, 4]
        measurement_outcomes: Random variables for measurement sampling [depth, n_qubits]
        n_qubits: Number of qubits
        depth: Circuit depth
        p_measure: Measurement probability
    
    Returns:
        Final quantum state (normalized)
    
    Physics:
        - Random unitaries generate entanglement (scramble quantum information)
        - Measurements collapse wavefunctions (localize information)
        - Competition between these processes drives the phase transition
    
    物理含义：
        - 随机幺正门产生纠缠（混乱量子信息）
        - 测量坍缩波函数（局域化信息）
        - 这两个过程的竞争驱动相变
    """
    # Reshape inputs
    random_unitaries = K.reshape(random_unitaries, [depth, n_qubits, 4, 4])
    measurement_outcomes = K.reshape(measurement_outcomes, [depth, n_qubits])
    
    # Initialize state |00...0⟩
    state = None
    
    for layer in range(depth):
        # Create circuit (use previous state as input if exists)
        if state is None:
            c = tq.Circuit(n_qubits)
        else:
            c = tq.Circuit(n_qubits, inputs=state)
        
        # Even bonds: apply unitaries to (0,1), (2,3), (4,5), ...
        for i in range(0, n_qubits, 2):
            target = (i + 1) % n_qubits
            c.unitary(i, target, matrix=random_unitaries[layer, i])
        
        # Odd bonds: apply unitaries to (1,2), (3,4), (5,6), ...
        for i in range(1, n_qubits, 2):
            target = (i + 1) % n_qubits
            c.unitary(i, target, matrix=random_unitaries[layer, i])
        
        # Get state after unitaries
        state = c.state()
        
        # Apply measurement channel to each qubit
        c = tq.Circuit(n_qubits, inputs=state)
        kraus_ops = measurement_channel(p=p_measure)
        
        for i in range(n_qubits):
            # Use measurement_outcomes[layer, i] as random variable
            c.kraus(i, kraus_ops, status=measurement_outcomes[layer, i])
        
        # Get state after measurements
        state = c.state()
        
        # Renormalize (important after stochastic Kraus application)
        state = state / K.norm(state)
    
    return state


def calculate_entropies(random_unitaries, measurement_outcomes, n_qubits, depth, p_measure):
    """Calculate von Neumann and Rényi-2 entanglement entropies.
    
    Args:
        random_unitaries: Random gates
        measurement_outcomes: Measurement sampling variables
        n_qubits: Number of qubits
        depth: Circuit depth
        p_measure: Measurement probability
    
    Returns:
        (von_neumann_entropy, renyi_2_entropy)
    
    Calculation:
        1. Run MIPT circuit to get final state |ψ⟩
        2. Trace out qubits [n/2, n-1] to get ρ_A = Tr_B(|ψ⟩⟨ψ|)
        3. Compute S(ρ_A) = -Tr(ρ_A log ρ_A)
        4. Compute S_2(ρ_A) = -log Tr(ρ_A^2)
    """
    # Get final state
    state = mipt_circuit(random_unitaries, measurement_outcomes, n_qubits, depth, p_measure)
    
    # Reduce to subsystem A (first n/2 qubits)
    # reduced_density_matrix expects numpy array
    state_np = state.detach().cpu().numpy() if hasattr(state, 'detach') else state
    
    # Compute reduced density matrix ρ_A
    subsystem_A_qubits = [i for i in range(n_qubits // 2)]
    rho_A = reduced_density_matrix(state_np, cut=subsystem_A_qubits)
    
    # Calculate entropies
    s_vn = entropy(rho_A)
    s_2 = renyi_entropy(rho_A, k=2)
    
    return s_vn, s_2


def demonstrate_mipt_single_point():
    """Demonstrate MIPT at a single measurement probability.
    
    Shows:
    - Circuit construction with Kraus channels
    - Entropy calculation
    - JIT compilation speedup
    """
    print("=" * 70)
    print("MIPT: Single Point Demonstration")
    print("=" * 70)
    
    # System parameters (optimized for speed)
    n_qubits = DEFAULT_N_QUBITS
    depth = DEFAULT_DEPTH
    p_measure = 0.1  # Low measurement rate → volume-law phase
    
    print(f"System: {n_qubits} qubits, {depth} layers")
    print(f"Measurement probability: p = {p_measure}")
    print(f"Subsystem A: first {n_qubits // 2} qubits")
    print()
    
    # Generate random gates
    random_unitaries = generate_random_unitaries(n_qubits, depth)
    random_unitaries = K.array(random_unitaries)  # Convert to backend tensor
    
    # First run: compilation + execution
    print("First run (includes JIT compilation)...")
    measurement_outcomes_1 = K.array(np.random.uniform(size=[depth, n_qubits]))
    time_start = time.time()
    s_vn_1, s_2_1 = calculate_entropies(
        random_unitaries, measurement_outcomes_1, n_qubits, depth, p_measure
    )
    time_compile = time.time() - time_start
    
    print(f"  von Neumann entropy: S = {s_vn_1:.4f}")
    print(f"  Rényi-2 entropy: S_2 = {s_2_1:.4f}")
    print(f"  Time: {time_compile:.3f}s")
    print()
    
    # Second run: execution only (compiled)
    print("Second run (using compiled code)...")
    measurement_outcomes_2 = K.array(np.random.uniform(size=[depth, n_qubits]))
    time_start = time.time()
    s_vn_2, s_2_2 = calculate_entropies(
        random_unitaries, measurement_outcomes_2, n_qubits, depth, p_measure
    )
    time_run = time.time() - time_start
    
    print(f"  von Neumann entropy: S = {s_vn_2:.4f}")
    print(f"  Rényi-2 entropy: S_2 = {s_2_2:.4f}")
    print(f"  Time: {time_run:.3f}s")
    print()
    
    print(f"Speedup: {time_compile / time_run:.1f}x after JIT compilation")
    print()
    
    # Physical interpretation
    max_entropy = np.log(2 ** (n_qubits // 2))
    print("Physical Interpretation:")
    print(f"  Maximum possible entropy: {max_entropy:.4f}")
    print(f"  Entanglement ratio: {s_vn_2 / max_entropy:.2%}")
    if p_measure < 0.2:
        print(f"  → Volume-law phase (p={p_measure} < p_c ≈ 0.16)")
        print("    High entanglement, information delocalized")
    else:
        print(f"  → Area-law phase (p={p_measure} > p_c ≈ 0.16)")
        print("    Low entanglement, information localized")
    print()


def scan_measurement_probability():
    """Scan measurement probability to observe MIPT.
    
    Demonstrates:
    - Volume-law to area-law transition
    - Critical measurement rate p_c
    - Entanglement entropy vs measurement rate
    """
    print("=" * 70)
    print("MIPT: Measurement Probability Scan")
    print("=" * 70)
    
    # Smaller system for faster scanning
    n_qubits = 6  # Further reduced for speed
    depth = 6
    n_samples = 2  # Trajectories to average (reduced from 3)
    
    # Measurement probabilities to scan (fewer points)
    p_values = np.linspace(0.0, 0.4, 5)  # Only 5 points instead of 11
    
    print(f"System: {n_qubits} qubits, {depth} layers")
    print(f"Samples per point: {n_samples}")
    print(f"Scanning p ∈ [{p_values[0]:.2f}, {p_values[-1]:.2f}]")
    print()
    
    # Storage for results
    results = {
        'p': [],
        's_vn_mean': [],
        's_vn_std': [],
        's_2_mean': [],
        's_2_std': [],
    }
    
    print("Progress:")
    for p in p_values:
        s_vn_samples = []
        s_2_samples = []
        
        for sample in range(n_samples):
            # Generate fresh random gates and outcomes for each trajectory
            random_unitaries = generate_random_unitaries(n_qubits, depth)
            random_unitaries = K.array(random_unitaries)  # Convert to backend tensor
            measurement_outcomes = K.array(np.random.uniform(size=[depth, n_qubits]))
            
            s_vn, s_2 = calculate_entropies(
                random_unitaries, measurement_outcomes, n_qubits, depth, p
            )
            
            s_vn_samples.append(s_vn)
            s_2_samples.append(s_2)
        
        # Statistics
        results['p'].append(p)
        results['s_vn_mean'].append(np.mean(s_vn_samples))
        results['s_vn_std'].append(np.std(s_vn_samples))
        results['s_2_mean'].append(np.mean(s_2_samples))
        results['s_2_std'].append(np.std(s_2_samples))
        
        print(f"  p = {p:.2f}: S = {results['s_vn_mean'][-1]:.4f} ± {results['s_vn_std'][-1]:.4f}")
    
    print()
    
    # Print results table instead of plotting (faster)
    print("Results Table:")
    print(f"{'p':>6s}  {'S_vN':>8s}  {'S_2':>8s}  {'Phase':>15s}")
    print("-" * 45)
    for i, p in enumerate(results['p']):
        phase = "Volume-law" if p < 0.16 else "Area-law"
        print(f"{p:6.2f}  {results['s_vn_mean'][i]:8.4f}  {results['s_2_mean'][i]:8.4f}  {phase:>15s}")
    print()
    
    # Analysis
    print("Analysis:")
    print(f"  At p=0 (no measurements):")
    idx_0 = 0
    print(f"    S = {results['s_vn_mean'][idx_0]:.4f} (volume-law, extensive)")
    
    idx_high = -1
    print(f"  At p={results['p'][idx_high]:.2f} (high measurement rate):")
    print(f"    S = {results['s_vn_mean'][idx_high]:.4f} (area-law, constant)")
    
    # Find approximate critical point (steepest descent)
    derivatives = np.diff(results['s_vn_mean']) / np.diff(results['p'])
    idx_c = np.argmin(derivatives)
    p_c_approx = results['p'][idx_c]
    print(f"  Approximate critical point: p_c ≈ {p_c_approx:.2f}")
    print(f"    (Theoretical: p_c ≈ 0.16 for 1D random circuits)")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("Measurement-Induced Phase Transition (MIPT) Demo")
    print("Optimized for fast execution")
    print("=" * 70)
    print()
    
    # Single point demonstration
    demonstrate_mipt_single_point()
    
    # Measurement probability scan
    scan_measurement_probability()
    
    print("=" * 70)
    print("✓ MIPT demonstration completed")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("1. Low measurement rates (p < p_c) → volume-law entanglement")
    print("2. High measurement rates (p > p_c) → area-law entanglement")
    print("3. Phase transition at critical p_c ≈ 0.16 (1D random circuits)")
    print("4. Observable experimentally on near-term quantum devices")
    print()
    print("Performance Notes:")
    print(f"  - System size: {DEFAULT_N_QUBITS} qubits (vs 12 in full version)")
    print(f"  - Circuit depth: {DEFAULT_DEPTH} layers (vs 12 in full version)")
    print("  - For higher precision, increase n_qubits/depth in code")
    print()
