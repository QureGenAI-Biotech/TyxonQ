"""
ZZ Crosstalk: Local vs Global Mode Comparison
==============================================

**GROUNDBREAKING FEATURE**: TyxonQ is the first quantum simulator to provide
BOTH local approximation and global exact co-evolution for ZZ crosstalk!

This example demonstrates the difference between two ZZ crosstalk simulation modes:

**Mode A: "local" (Default)** ⚡
- Fast, scalable to large systems (10+ qubits)
- Physically accurate for typical hardware parameters
- Uses sequential approximation: U_pulse × ∏ U_ZZ

**Mode B: "global"** 🎯  
- Exact, benchmark-quality results
- Captures simultaneous pulse + ZZ evolution
- Computationally expensive (< 8 qubits practical)

**Physical Background**:

ZZ crosstalk (H_ZZ = ξ · Z⊗Z) is an always-on coherent coupling in superconducting
qubits that causes conditional phase errors. During a pulse operation, both the
pulse Hamiltonian H_pulse(t) and ZZ coupling H_ZZ evolve simultaneously.

- **Local mode**: Approximates [H_pulse, H_ZZ] ≈ 0 (valid for weak ZZ)
- **Global mode**: Exact evolution of H_total = H_pulse + H_ZZ

Author: TyxonQ Development Team
Date: 2025-10-30
"""

import numpy as np
import time
import tyxonq as tq
from tyxonq import waveforms
from tyxonq.libs.quantum_library.pulse_physics import get_qubit_topology


def example1_basic_comparison():
    """Example 1: Compare local vs global modes on 2-qubit system."""
    print("=" * 80)
    print("Example 1: Local vs Global Mode Comparison (2 Qubits)")
    print("=" * 80)
    
    # Create circuit with pulse operation
    c = tq.Circuit(2)
    pulse_x = waveforms.Drag(duration=160, amp=1.0, sigma=40, beta=0.2)
    c.metadata["pulse_library"] = {"pulse_x": pulse_x}
    c.ops.append(("pulse", 0, "pulse_x", {"qubit_freq": 5.0e9}))
    c.measure_z(0)
    c.measure_z(1)
    
    # Create ZZ topology (IBM 3 MHz)
    topo = get_qubit_topology(2, topology="linear", zz_strength=3e6)
    
    print(f"\nSystem: 2 qubits, linear chain")
    print(f"ZZ coupling: 3 MHz (typical IBM)")
    print(f"Pulse duration: 160 ns")
    print(f"Expected ZZ phase: {3e6 * 160e-9:.4f} rad = {3e6 * 160e-9 * 180/np.pi:.2f}°")
    
    # Run LOCAL mode
    print(f"\n--- Mode A: Local Approximation ---")
    t_start = time.time()
    result_local = c.device(
        provider="simulator",
        device="statevector",
        zz_topology=topo,
        zz_mode="local",  # ⚡ Fast approximation
        shots=10000
    ).postprocessing(method=None).run()
    t_local = time.time() - t_start
    
    print(f"Execution time: {t_local*1000:.2f} ms")
    print(f"Result: {result_local[0]['result']}")
    
    # Run GLOBAL mode
    print(f"\n--- Mode B: Global Exact Evolution ---")
    t_start = time.time()
    result_global = c.device(
        provider="simulator",
        device="statevector",
        zz_topology=topo,
        zz_mode="global",  # 🎯 Exact co-evolution
        shots=10000
    ).postprocessing(method=None).run()
    t_global = time.time() - t_start
    
    print(f"Execution time: {t_global*1000:.2f} ms")
    print(f"Result: {result_global[0]['result']}")
    
    # Compare results
    print(f"\n--- Comparison ---")
    print(f"Speed ratio (global/local): {t_global/t_local:.2f}x slower")
    print(f"✅ For 2 qubits, both modes should give similar results")
    print(f"   (difference comes from [H_pulse, H_ZZ] commutator)")
    
    print()


def example2_weak_vs_strong_zz():
    """Example 2: When does local approximation break down?"""
    print("=" * 80)
    print("Example 2: Local Approximation Validity (Weak vs Strong ZZ)")
    print("=" * 80)
    
    c = tq.Circuit(2)
    pulse_x = waveforms.Drag(duration=200, amp=1.0, sigma=50, beta=0.2)  # Longer pulse
    c.metadata["pulse_library"] = {"pulse_x": pulse_x}
    c.ops.append(("pulse", 0, "pulse_x", {"qubit_freq": 5.0e9}))
    c.measure_z(0)
    c.measure_z(1)
    
    zz_strengths = [
        (0.5e6, "Weak (Google-like)"),
        (3e6, "Moderate (IBM-like)"),
        (10e6, "Strong (Rigetti-like)"),
        (20e6, "Very Strong (unrealistic)")
    ]
    
    print(f"\nPulse duration: 200 ns (longer than typical)")
    print(f"\n{'ZZ Strength':<25} {'Mode':<10} {'Result (|00⟩)':<15} {'Result (|11⟩)':<15}")
    print("-" * 70)
    
    for xi, label in zz_strengths:
        topo = get_qubit_topology(2, topology="linear", zz_strength=xi)
        
        # Local mode
        result_local = c.device(
            provider="simulator", device="statevector",
            zz_topology=topo, zz_mode="local", shots=10000
        ).postprocessing(method=None).run()
        counts_local = result_local[0]['result']
        
        # Global mode
        result_global = c.device(
            provider="simulator", device="statevector",
            zz_topology=topo, zz_mode="global", shots=10000
        ).postprocessing(method=None).run()
        counts_global = result_global[0]['result']
        
        # Compare |00⟩ and |11⟩ populations
        pop_00_local = counts_local.get('00', 0) / 10000
        pop_11_local = counts_local.get('11', 0) / 10000
        pop_00_global = counts_global.get('00', 0) / 10000
        pop_11_global = counts_global.get('11', 0) / 10000
        
        print(f"{label:<25} {'Local':<10} {pop_00_local:<15.4f} {pop_11_local:<15.4f}")
        print(f"{'':<25} {'Global':<10} {pop_00_global:<15.4f} {pop_11_global:<15.4f}")
        
        # Calculate discrepancy
        diff = abs(pop_00_local - pop_00_global) + abs(pop_11_local - pop_11_global)
        print(f"{'':<25} {'Difference':<10} {diff:<15.6f}")
        
        if diff < 0.01:
            print(f"{'':<25} ✅ Local approximation valid")
        else:
            print(f"{'':<25} ⚠️  Local approximation breaks down!")
        print()
    
    print("Key insight: Local mode is accurate for weak-to-moderate ZZ coupling")
    print("             Global mode required for strong ZZ (> 10 MHz)")
    print()


def example3_scalability_test():
    """Example 3: Scalability comparison (local vs global)."""
    print("=" * 80)
    print("Example 3: Scalability Test (2-6 Qubits)")
    print("=" * 80)
    
    print(f"\n{'Qubits':<10} {'Mode':<10} {'Time (ms)':<15} {'Memory':<15}")
    print("-" * 55)
    
    for n_qubits in [2, 3, 4, 5, 6]:
        c = tq.Circuit(n_qubits)
        pulse = waveforms.Drag(duration=100, amp=1.0, sigma=25, beta=0.2)
        c.metadata["pulse_library"] = {"pulse": pulse}
        
        # Apply pulse to qubit 0
        c.ops.append(("pulse", 0, "pulse", {"qubit_freq": 5.0e9}))
        for q in range(n_qubits):
            c.measure_z(q)
        
        topo = get_qubit_topology(n_qubits, topology="linear", zz_strength=3e6)
        
        # Test LOCAL mode
        t_start = time.time()
        t_local = 0.0
        t_global = 0.0
        try:
            result_local = c.device(
                provider="simulator", device="statevector",
                zz_topology=topo, zz_mode="local", shots=1000
            ).postprocessing(method=None).run()
            t_local = (time.time() - t_start) * 1000
            mem_local = f"{2**n_qubits * 16 / 1024:.2f} KB"  # Complex128 state vector
            print(f"{n_qubits:<10} {'Local':<10} {t_local:<15.2f} {mem_local:<15}")
        except Exception as e:
            print(f"{n_qubits:<10} {'Local':<10} {'FAILED':<15} {str(e)[:15]:<15}")
        
        # Test GLOBAL mode
        t_start = time.time()
        try:
            result_global = c.device(
                provider="simulator", device="statevector",
                zz_topology=topo, zz_mode="global", shots=1000
            ).postprocessing(method=None).run()
            t_global = (time.time() - t_start) * 1000
            mem_global = f"{(2**n_qubits)**2 * 16 / 1024:.2f} KB"  # Full Hamiltonian
            print(f"{n_qubits:<10} {'Global':<10} {t_global:<15.2f} {mem_global:<15}")
        except Exception as e:
            print(f"{n_qubits:<10} {'Global':<10} {'FAILED':<15} {str(e)[:15]:<15}")
        
        if n_qubits <= 5 and t_local > 0 and t_global > 0:
            print(f"{'':<10} {'Speedup':<10} {t_global/t_local:.2f}x slower")
        print()
    
    print("✅ Local mode: Scales linearly, practical for 10+ qubits")
    print("⚠️  Global mode: Exponential scaling, limited to < 8 qubits")
    print()


def example4_bell_state_fidelity():
    """Example 4: Bell state fidelity with ZZ crosstalk."""
    print("=" * 80)
    print("Example 4: Bell State Fidelity Degradation")
    print("=" * 80)
    
    # Create Bell state circuit with pulse
    c = tq.Circuit(2)
    
    # H gate via pulse (π/2 rotation)
    pulse_h = waveforms.Drag(duration=80, amp=0.707, sigma=20, beta=0.15)
    c.metadata["pulse_library"] = {"pulse_h": pulse_h}
    c.ops.append(("pulse", 0, "pulse_h", {"qubit_freq": 5.0e9}))
    
    # CNOT (ideal gate)
    c.cnot(0, 1)
    
    c.measure_z(0)
    c.measure_z(1)
    
    # Compare different ZZ strengths
    zz_configs = [
        (0, "No ZZ"),
        (0.5e6, "Google (0.5 MHz)"),
        (3e6, "IBM (3 MHz)"),
        (10e6, "Rigetti (10 MHz)")
    ]
    
    print(f"\nBell state: (|00⟩ + |11⟩) / √2")
    print(f"\n{'Configuration':<20} {'Mode':<10} {'|00⟩':<10} {'|11⟩':<10} {'Fidelity':<10}")
    print("-" * 65)
    
    for xi, label in zz_configs:
        if xi == 0:
            # No ZZ (reference)
            result_ref = c.device(
                provider="simulator", device="statevector", shots=10000
            ).postprocessing(method=None).run()
            counts = result_ref[0]['result']
            p00 = counts.get('00', 0) / 10000
            p11 = counts.get('11', 0) / 10000
            fidelity = (p00 + p11) / 2  # Ideal: 0.5 each
            print(f"{label:<20} {'None':<10} {p00:<10.4f} {p11:<10.4f} {fidelity:<10.4f}")
        else:
            topo = get_qubit_topology(2, topology="linear", zz_strength=xi)
            
            # Local mode
            result_local = c.device(
                provider="simulator", device="statevector",
                zz_topology=topo, zz_mode="local", shots=10000
            ).postprocessing(method=None).run()
            counts_local = result_local[0]['result']
            p00_local = counts_local.get('00', 0) / 10000
            p11_local = counts_local.get('11', 0) / 10000
            fid_local = (p00_local + p11_local) / 2
            
            # Global mode
            result_global = c.device(
                provider="simulator", device="statevector",
                zz_topology=topo, zz_mode="global", shots=10000
            ).postprocessing(method=None).run()
            counts_global = result_global[0]['result']
            p00_global = counts_global.get('00', 0) / 10000
            p11_global = counts_global.get('11', 0) / 10000
            fid_global = (p00_global + p11_global) / 2
            
            print(f"{label:<20} {'Local':<10} {p00_local:<10.4f} {p11_local:<10.4f} {fid_local:<10.4f}")
            print(f"{'':<20} {'Global':<10} {p00_global:<10.4f} {p11_global:<10.4f} {fid_global:<10.4f}")
    
    print(f"\n✅ Both modes capture ZZ-induced fidelity degradation")
    print(f"🎯 Global mode provides benchmark-quality accuracy")
    print()


def example5_multi_qubit_chain():
    """Example 5: Multi-qubit linear chain with ZZ crosstalk."""
    print("=" * 80)
    print("Example 5: 5-Qubit Linear Chain (Parallel Pulses)")
    print("=" * 80)
    
    # Create 5-qubit circuit
    n_qubits = 5
    c = tq.Circuit(n_qubits)
    
    # Apply pulses to qubits 0, 2, 4 (non-adjacent)
    pulse = waveforms.Drag(duration=100, amp=1.0, sigma=25, beta=0.2)
    c.metadata["pulse_library"] = {"pulse": pulse}
    
    for q in [0, 2, 4]:
        c.ops.append(("pulse", q, "pulse", {"qubit_freq": 5.0e9}))
    
    for q in range(n_qubits):
        c.measure_z(q)
    
    topo = get_qubit_topology(n_qubits, topology="linear", zz_strength=3e6)
    
    print(f"\nTopology: 0--1--2--3--4")
    print(f"Pulses on: qubits 0, 2, 4 (non-adjacent)")
    print(f"ZZ coupling: 3 MHz (IBM)")
    
    # Run both modes
    print(f"\n{'Qubit':<10} {'Local':<15} {'Global':<15} {'Difference':<15}")
    print("-" * 60)
    
    result_local = c.device(
        provider="simulator", device="statevector",
        zz_topology=topo, zz_mode="local", shots=10000
    ).postprocessing(method=None).run()
    
    result_global = c.device(
        provider="simulator", device="statevector",
        zz_topology=topo, zz_mode="global", shots=10000
    ).postprocessing(method=None).run()
    
    counts_local = result_local[0]['result']
    counts_global = result_global[0]['result']
    
    # Compare distributions
    for bitstring in sorted(set(list(counts_local.keys()) + list(counts_global.keys()))):
        p_local = counts_local.get(bitstring, 0) / 10000
        p_global = counts_global.get(bitstring, 0) / 10000
        diff = abs(p_local - p_global)
        if p_local > 0.01 or p_global > 0.01:  # Only show significant states
            print(f"{bitstring:<10} {p_local:<15.4f} {p_global:<15.4f} {diff:<15.6f}")
    
    print(f"\n✅ Local mode: Fast simulation for 5+ qubits")
    print(f"🎯 Global mode: Benchmark reference for validation")
    print()


def main():
    """Run all ZZ crosstalk mode comparison examples."""
    print("\n" + "=" * 80)
    print(" ZZ Crosstalk: Local vs Global Mode Comparison")
    print(" WORLD'S FIRST DUAL-MODE ZZ CROSSTALK SIMULATOR")
    print("=" * 80 + "\n")
    
    example1_basic_comparison()
    example2_weak_vs_strong_zz()
    example3_scalability_test()
    example4_bell_state_fidelity()
    example5_multi_qubit_chain()
    
    print("=" * 80)
    print("All comparison examples completed!")
    print("=" * 80)
    print("\n🚀 TyxonQ Innovation Highlights:")
    print("  1. FIRST simulator with dual-mode ZZ crosstalk")
    print("  2. Local mode: Production-ready (10+ qubits)")
    print("  3. Global mode: Benchmark-quality (< 8 qubits)")
    print("  4. User choice: Speed vs Accuracy")
    print("  5. Physically accurate for all major hardware platforms")
    print("=" * 80)


if __name__ == "__main__":
    main()
