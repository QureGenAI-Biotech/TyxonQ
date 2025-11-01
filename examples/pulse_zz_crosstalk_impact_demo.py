"""
ZZ Crosstalk Impact Demonstration
==================================

This example clearly shows the DIFFERENCE between simulations with and without
ZZ crosstalk, demonstrating the real-world impact of this always-on coupling.

**Key Questions Answered**:
1. What happens when you enable ZZ crosstalk?
2. How much does it degrade gate fidelity?
3. Which qubits are affected (spectator errors)?
4. How does it vary with hardware platforms?

Author: TyxonQ Development Team
"""

import numpy as np
import tyxonq as tq
from tyxonq import waveforms
from tyxonq.libs.quantum_library.pulse_physics import get_qubit_topology


def example1_single_qubit_gate_spectator_error():
    """Example 1: Spectator qubit affected by ZZ crosstalk."""
    print("=" * 80)
    print("Example 1: Single-Qubit Gate with Spectator Qubit")
    print("ZZ Crosstalk causes errors on IDLE qubits!")
    print("=" * 80)
    
    # Initial state: |01⟩ (qubit 0 in |0⟩, qubit 1 in |1⟩)
    c = tq.Circuit(2)
    c.x(1)  # Prepare qubit 1 in |1⟩
    
    # Apply X pulse ONLY to qubit 0
    pulse_x = waveforms.Drag(duration=200, amp=1.0, sigma=50, beta=0.2)
    c.metadata["pulse_library"] = {"pulse_x": pulse_x}
    c.ops.append(("pulse", 0, "pulse_x", {"qubit_freq": 5.0e9}))
    
    # Measure both qubits
    c.measure_z(0)
    c.measure_z(1)
    
    # Create ZZ topology
    topo = get_qubit_topology(2, topology="linear", zz_strength=3e6)  # IBM 3 MHz
    
    print(f"\nSetup:")
    print(f"  Initial state: |01⟩")
    print(f"  Operation: X pulse on qubit 0 (200 ns)")
    print(f"  Qubit 1: IDLE (spectator)")
    print(f"  ZZ coupling: 3 MHz (IBM typical)")
    
    # Without ZZ crosstalk
    print(f"\n--- WITHOUT ZZ Crosstalk ---")
    result_no_zz = c.device(
        provider="simulator",
        device="statevector",
        shots=10000
    ).postprocessing(method=None).run()
    
    counts_no_zz = result_no_zz[0]["result"]
    print(f"Measurement results:")
    for state, count in sorted(counts_no_zz.items(), key=lambda x: -x[1]):
        prob = count / 10000
        print(f"  |{state}⟩: {count:5d} ({prob*100:5.2f}%)")
    
    # With ZZ crosstalk (local mode)
    print(f"\n--- WITH ZZ Crosstalk (Local Mode) ---")
    result_with_zz = c.device(
        provider="simulator",
        device="statevector",
        zz_topology=topo,
        zz_mode="local",
        shots=10000
    ).postprocessing(method=None).run()
    
    counts_with_zz = result_with_zz[0]["result"]
    print(f"Measurement results:")
    for state, count in sorted(counts_with_zz.items(), key=lambda x: -x[1]):
        prob = count / 10000
        print(f"  |{state}⟩: {count:5d} ({prob*100:5.2f}%)")
    
    # Analysis
    print(f"\n--- Impact Analysis ---")
    
    # Expected: |01⟩ → |11⟩ (X flips qubit 0)
    ideal_final = "11"
    prob_ideal_no_zz = counts_no_zz.get(ideal_final, 0) / 10000
    prob_ideal_with_zz = counts_with_zz.get(ideal_final, 0) / 10000
    
    print(f"Expected final state: |{ideal_final}⟩")
    print(f"  Probability (no ZZ):   {prob_ideal_no_zz*100:.2f}%")
    print(f"  Probability (with ZZ): {prob_ideal_with_zz*100:.2f}%")
    
    # ZZ phase
    xi = 3e6
    t = 200e-9
    phi_zz = xi * t
    print(f"\nZZ phase accumulated: {phi_zz:.4f} rad = {phi_zz*180/np.pi:.2f}°")
    
    # Fidelity estimate
    fidelity_no_zz = prob_ideal_no_zz
    fidelity_with_zz = prob_ideal_with_zz
    fidelity_loss = (fidelity_no_zz - fidelity_with_zz) * 100
    
    print(f"\n⚠️  ZZ Crosstalk Effect:")
    print(f"  Gate fidelity degradation: {fidelity_loss:.2f}%")
    print(f"  Spectator qubit 1 picks up conditional phase!")
    
    print()


def example2_bell_state_with_without_zz():
    """Example 2: Bell state fidelity with/without ZZ."""
    print("=" * 80)
    print("Example 2: Bell State Fidelity Degradation")
    print("=" * 80)
    
    # Create Bell state: H(0) → CNOT(0,1)
    c = tq.Circuit(2)
    
    # H gate via pulse
    pulse_h = waveforms.Drag(duration=80, amp=0.707, sigma=20, beta=0.15)
    c.metadata["pulse_library"] = {"pulse_h": pulse_h}
    c.ops.append(("pulse", 0, "pulse_h", {"qubit_freq": 5.0e9}))
    
    # CNOT (standard gate)
    c.cnot(0, 1)
    
    c.measure_z(0)
    c.measure_z(1)
    
    print(f"\nBell state preparation: H(0) → CNOT(0,1)")
    print(f"Ideal state: (|00⟩ + |11⟩) / √2")
    
    # Test different ZZ strengths
    zz_configs = [
        (None, "No ZZ"),
        (0.5e6, "Google (0.5 MHz)"),
        (3e6, "IBM (3 MHz)"),
        (10e6, "Rigetti (10 MHz)")
    ]
    
    print(f"\n{'Platform':<20} {'|00⟩':<10} {'|11⟩':<10} {'|01⟩+|10⟩':<12} {'Fidelity':<10}")
    print("-" * 67)
    
    for config in zz_configs:
        if config[0] is None:
            # No ZZ
            result = c.device(
                provider="simulator",
                device="statevector",
                shots=10000
            ).postprocessing(method=None).run()
            label = config[1]
        else:
            xi, label = config
            topo = get_qubit_topology(2, topology="linear", zz_strength=xi)
            result = c.device(
                provider="simulator",
                device="statevector",
                zz_topology=topo,
                zz_mode="local",
                shots=10000
            ).postprocessing(method=None).run()
        
        counts = result[0]["result"]
        
        # Bell state should have 50% |00⟩ and 50% |11⟩
        p00 = counts.get("00", 0) / 10000
        p11 = counts.get("11", 0) / 10000
        p01 = counts.get("01", 0) / 10000
        p10 = counts.get("10", 0) / 10000
        
        # Fidelity to ideal Bell state
        fidelity = (p00 + p11) / 2  # Should be ~0.5 each for ideal
        
        print(f"{label:<20} {p00:<10.4f} {p11:<10.4f} {p01+p10:<12.4f} {fidelity:<10.4f}")
    
    print(f"\n📊 Observation:")
    print(f"  • No ZZ: Perfect Bell state (|00⟩ + |11⟩)")
    print(f"  • With ZZ: Phase errors accumulate")
    print(f"  • Stronger ZZ → Lower fidelity")
    print(f"  • Google's tunable couplers minimize ZZ impact")
    
    print()


def example3_parallel_gates_crosstalk():
    """Example 3: Parallel gates on neighboring qubits."""
    print("=" * 80)
    print("Example 3: Parallel Gates on Neighboring Qubits")
    print("ZZ crosstalk during simultaneous operations!")
    print("=" * 80)
    
    # 3-qubit chain: 0--1--2
    c = tq.Circuit(3)
    
    # Apply X pulses to qubits 0 and 1 SIMULTANEOUSLY
    pulse_x = waveforms.Drag(duration=100, amp=1.0, sigma=25, beta=0.2)
    c.metadata["pulse_library"] = {"pulse_x": pulse_x}
    
    # Both pulses happen at the same time
    c.ops.append(("pulse", 0, "pulse_x", {"qubit_freq": 5.0e9}))
    c.ops.append(("pulse", 1, "pulse_x", {"qubit_freq": 5.0e9}))
    
    for q in range(3):
        c.measure_z(q)
    
    topo = get_qubit_topology(3, topology="linear", zz_strength=3e6)
    
    print(f"\nSetup:")
    print(f"  Topology: 0--1--2 (linear chain)")
    print(f"  Operation: X pulse on qubits 0 AND 1 (parallel)")
    print(f"  Duration: 100 ns each")
    print(f"  ZZ coupling: 3 MHz")
    
    # Without ZZ
    print(f"\n--- WITHOUT ZZ Crosstalk ---")
    result_no_zz = c.device(
        provider="simulator",
        device="statevector",
        shots=10000
    ).postprocessing(method=None).run()
    
    counts_no_zz = result_no_zz[0]["result"]
    print(f"Top 5 measurement outcomes:")
    for state, count in sorted(counts_no_zz.items(), key=lambda x: -x[1])[:5]:
        prob = count / 10000
        print(f"  |{state}⟩: {count:5d} ({prob*100:5.2f}%)")
    
    # With ZZ
    print(f"\n--- WITH ZZ Crosstalk ---")
    result_with_zz = c.device(
        provider="simulator",
        device="statevector",
        zz_topology=topo,
        zz_mode="local",
        shots=10000
    ).postprocessing(method=None).run()
    
    counts_with_zz = result_with_zz[0]["result"]
    print(f"Top 5 measurement outcomes:")
    for state, count in sorted(counts_with_zz.items(), key=lambda x: -x[1])[:5]:
        prob = count / 10000
        print(f"  |{state}⟩: {count:5d} ({prob*100:5.2f}%)")
    
    # Analysis
    print(f"\n--- Analysis ---")
    
    # Expected outcome: |110⟩ (X on qubits 0 and 1)
    expected = "110"
    prob_no_zz = counts_no_zz.get(expected, 0) / 10000
    prob_with_zz = counts_with_zz.get(expected, 0) / 10000
    
    print(f"Expected outcome: |{expected}⟩")
    print(f"  Probability (no ZZ):   {prob_no_zz*100:.2f}%")
    print(f"  Probability (with ZZ): {prob_with_zz*100:.2f}%")
    
    xi = 3e6
    t = 100e-9
    phi_zz = xi * t
    print(f"\nZZ phase between qubits 0-1: {phi_zz:.4f} rad = {phi_zz*180/np.pi:.2f}°")
    print(f"\n⚠️  Parallel gates on neighbors accumulate ZZ phase!")
    print(f"    This is why gate scheduling is critical in real hardware.")
    
    print()


def example4_idle_time_crosstalk():
    """Example 4: ZZ crosstalk during idle time."""
    print("=" * 80)
    print("Example 4: Idle Time Crosstalk")
    print("ZZ coupling ALWAYS active, even during waiting!")
    print("=" * 80)
    
    # 2-qubit system
    # Qubit 0: Long pulse (200 ns)
    # Qubit 1: Idle entire time
    
    c = tq.Circuit(2)
    
    # Prepare qubit 1 in |1⟩
    c.x(1)
    
    # Long pulse on qubit 0
    pulse_long = waveforms.Drag(duration=200, amp=1.0, sigma=50, beta=0.2)
    c.metadata["pulse_library"] = {"pulse_long": pulse_long}
    c.ops.append(("pulse", 0, "pulse_long", {"qubit_freq": 5.0e9}))
    
    c.measure_z(0)
    c.measure_z(1)
    
    topo = get_qubit_topology(2, topology="linear", zz_strength=3e6)
    
    print(f"\nSetup:")
    print(f"  Qubit 0: 200 ns pulse (active)")
    print(f"  Qubit 1: IDLE (but still coupled via ZZ!)")
    print(f"  Initial state: |01⟩")
    
    # Without ZZ
    print(f"\n--- WITHOUT ZZ Crosstalk ---")
    result_no_zz = c.device(
        provider="simulator",
        device="statevector",
        shots=10000
    ).postprocessing(method=None).run()
    
    counts_no_zz = result_no_zz[0]["result"]
    for state, count in sorted(counts_no_zz.items(), key=lambda x: -x[1]):
        prob = count / 10000
        print(f"  |{state}⟩: {count:5d} ({prob*100:5.2f}%)")
    
    # With ZZ
    print(f"\n--- WITH ZZ Crosstalk ---")
    result_with_zz = c.device(
        provider="simulator",
        device="statevector",
        zz_topology=topo,
        zz_mode="local",
        shots=10000
    ).postprocessing(method=None).run()
    
    counts_with_zz = result_with_zz[0]["result"]
    for state, count in sorted(counts_with_zz.items(), key=lambda x: -x[1]):
        prob = count / 10000
        print(f"  |{state}⟩: {count:5d} ({prob*100:5.2f}%)")
    
    # Analysis
    print(f"\n--- Idle Qubit Analysis ---")
    
    # Qubit 1 should stay in |1⟩, but ZZ causes phase errors
    xi = 3e6
    t = 200e-9
    phi_zz = xi * t
    
    print(f"Idle time: 200 ns")
    print(f"ZZ phase accumulated: {phi_zz:.4f} rad = {phi_zz*180/np.pi:.2f}°")
    print(f"\n⚠️  Key Insight:")
    print(f"    Even IDLE qubits are affected by ZZ crosstalk!")
    print(f"    This is called 'spectator error' in real hardware.")
    print(f"    Longer idle times → More ZZ phase → Lower fidelity")
    
    print()


def example5_hardware_comparison_real_impact():
    """Example 5: Real-world impact across hardware platforms."""
    print("=" * 80)
    print("Example 5: Hardware Platform Comparison")
    print("How much does ZZ matter for different vendors?")
    print("=" * 80)
    
    # Simple circuit: X pulse on qubit 0
    c = tq.Circuit(2)
    c.x(1)  # Prepare |01⟩
    
    pulse_x = waveforms.Drag(duration=100, amp=1.0, sigma=25, beta=0.2)
    c.metadata["pulse_library"] = {"pulse_x": pulse_x}
    c.ops.append(("pulse", 0, "pulse_x", {"qubit_freq": 5.0e9}))
    
    c.measure_z(0)
    c.measure_z(1)
    
    # Get reference (no ZZ)
    result_ref = c.device(
        provider="simulator",
        device="statevector",
        shots=10000
    ).postprocessing(method=None).run()
    counts_ref = result_ref[0]["result"]
    
    platforms = [
        (0.5e6, "Google Sycamore"),
        (3e6, "IBM Eagle"),
        (10e6, "Rigetti Aspen")
    ]
    
    print(f"\nPulse: 100 ns X gate on qubit 0")
    print(f"Initial state: |01⟩, Expected final: |11⟩")
    print(f"\n{'Platform':<20} {'ZZ (MHz)':<12} {'Fidelity Loss':<15} {'Impact':<15}")
    print("-" * 67)
    
    print(f"{'Ideal (no ZZ)':<20} {0:<12.1f} {0:<15.2f} {'Reference':<15}")
    
    for xi, label in platforms:
        topo = get_qubit_topology(2, topology="linear", zz_strength=xi)
        
        result = c.device(
            provider="simulator",
            device="statevector",
            zz_topology=topo,
            zz_mode="local",
            shots=10000
        ).postprocessing(method=None).run()
        
        counts = result[0]["result"]
        
        # Calculate fidelity to expected outcome |11⟩
        fid_ref = counts_ref.get("11", 0) / 10000
        fid_zz = counts.get("11", 0) / 10000
        fid_loss = (fid_ref - fid_zz) * 100
        
        # ZZ phase
        t = 100e-9
        phi_zz = xi * t * 180 / np.pi
        
        if fid_loss < 1:
            impact = "✅ Minimal"
        elif fid_loss < 5:
            impact = "⚠️  Moderate"
        else:
            impact = "❌ Severe"
        
        print(f"{label:<20} {xi/1e6:<12.1f} {fid_loss:<15.2f}% {impact:<15}")
    
    print(f"\n📊 Conclusion:")
    print(f"  • Google's tunable couplers reduce ZZ to < 1 MHz → minimal impact")
    print(f"  • IBM's 3 MHz ZZ requires echo sequences for mitigation")
    print(f"  • Rigetti's 10 MHz ZZ is severe → limits gate fidelity")
    print(f"  • ZZ crosstalk is a MAJOR error source in NISQ devices!")
    
    print()


def main():
    """Run all ZZ crosstalk impact demonstrations."""
    print("\n" + "=" * 80)
    print(" ZZ Crosstalk Impact Demonstration")
    print(" See the REAL difference with and without ZZ crosstalk!")
    print("=" * 80 + "\n")
    
    example1_single_qubit_gate_spectator_error()
    example2_bell_state_with_without_zz()
    example3_parallel_gates_crosstalk()
    example4_idle_time_crosstalk()
    example5_hardware_comparison_real_impact()
    
    print("=" * 80)
    print("Summary: Why ZZ Crosstalk Matters")
    print("=" * 80)
    print("\n1️⃣  SPECTATOR ERRORS: Idle qubits pick up phase errors")
    print("2️⃣  FIDELITY DEGRADATION: Gate quality degrades over time")
    print("3️⃣  PARALLEL GATE LIMITS: Can't run gates on neighbors simultaneously")
    print("4️⃣  HARDWARE DEPENDENT: 0.5 MHz (Google) vs 10 MHz (Rigetti)")
    print("5️⃣  MITIGATION NEEDED: Echo sequences, ZZ-free gates, tunable couplers")
    print("\n🎯 TyxonQ provides ACCURATE ZZ crosstalk simulation for realistic NISQ modeling!")
    print("=" * 80)


if __name__ == "__main__":
    main()
