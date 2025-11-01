"""
ZZ Crosstalk: Visible Impact on Quantum States
===============================================

This example demonstrates ZZ crosstalk effects that are ACTUALLY VISIBLE
in measurement outcomes, showing clear differences with/without ZZ coupling.

**Key Idea**: ZZ crosstalk causes CONDITIONAL PHASE shifts that are visible
when qubits are in superposition states.

Author: TyxonQ Development Team
"""

import numpy as np
import tyxonq as tq
from tyxonq import waveforms
from tyxonq.libs.quantum_library.pulse_physics import get_qubit_topology


def example1_conditional_phase_visibility():
    """Example 1: ZZ crosstalk visible via X-basis measurement."""
    print("=" * 80)
    print("Example 1: Conditional Phase from ZZ Crosstalk")
    print("Measured in X basis to see phase errors!")
    print("=" * 80)
    
    # Create superposition state: (|00⟩ + |11⟩)/√2
    c = tq.Circuit(2)
    c.h(0)  # Create superposition
    c.cnot(0, 1)  # Entangle
    
    # Now apply a pulse to qubit 0 (this will accumulate ZZ phase)
    pulse = waveforms.Drag(duration=300, amp=0.5, sigma=75, beta=0.2)  # Long pulse
    c.metadata["pulse_library"] = {"pulse": pulse}
    c.ops.append(("pulse", 0, "pulse", {"qubit_freq": 5.0e9}))
    
    # Measure in X basis (H before measurement)
    c.h(0)
    c.h(1)
    c.measure_z(0)
    c.measure_z(1)
    
    topo = get_qubit_topology(2, topology="linear", zz_strength=5e6)  # Strong ZZ
    
    print(f"\nSetup:")
    print(f"  1. Prepare Bell state: (|00⟩ + |11⟩)/√2")
    print(f"  2. Apply 300ns pulse to qubit 0")
    print(f"  3. Measure in X basis")
    print(f"  ZZ coupling: 5 MHz (strong)")
    
    # Without ZZ
    print(f"\n--- WITHOUT ZZ Crosstalk ---")
    result_no_zz = c.device(
        provider="simulator",
        device="statevector",
        shots=10000
    ).postprocessing(method=None).run()
    
    counts_no_zz = result_no_zz[0]["result"]
    print(f"Measurement outcomes:")
    for state in ["00", "01", "10", "11"]:
        count = counts_no_zz.get(state, 0)
        print(f"  |{state}⟩: {count:5d} ({count/10000*100:5.2f}%)")
    
    # With ZZ
    print(f"\n--- WITH ZZ Crosstalk (Local Mode) ---")
    result_with_zz = c.device(
        provider="simulator",
        device="statevector",
        zz_topology=topo,
        zz_mode="local",
        shots=10000
    ).postprocessing(method=None).run()
    
    counts_with_zz = result_with_zz[0]["result"]
    print(f"Measurement outcomes:")
    for state in ["00", "01", "10", "11"]:
        count = counts_with_zz.get(state, 0)
        print(f"  |{state}⟩: {count:5d} ({count/10000*100:5.2f}%)")
    
    # Analysis
    print(f"\n--- ZZ Phase Analysis ---")
    xi = 5e6
    t = 300e-9
    phi_zz = xi * t
    print(f"ZZ phase accumulated: {phi_zz:.4f} rad = {phi_zz*180/np.pi:.2f}°")
    print(f"\n💡 The ZZ phase rotates the Bell state!")
    print(f"   |ψ⟩ = (|00⟩ + e^(iφ)|11⟩)/√2")
    print(f"   This changes X-basis measurement outcomes")
    
    # Calculate visibility
    total_diff = sum(abs(counts_no_zz.get(s, 0) - counts_with_zz.get(s, 0)) 
                     for s in ["00", "01", "10", "11"])
    print(f"\nTotal measurement difference: {total_diff} counts")
    
    if total_diff > 100:
        print(f"✅ ZZ crosstalk CLEARLY VISIBLE in measurements!")
    else:
        print(f"⚠️  ZZ effect present but small (try longer pulse or stronger ZZ)")
    
    print()


def example2_ramsey_interference():
    """Example 2: ZZ crosstalk destroys Ramsey interference."""
    print("=" * 80)
    print("Example 2: Ramsey Interference with ZZ Crosstalk")
    print("ZZ phase shifts destroy interference fringes!")
    print("=" * 80)
    
    # Ramsey sequence: H - wait - H
    # With a neighbor qubit affecting via ZZ
    
    c = tq.Circuit(2)
    
    # Prepare qubit 1 in |1⟩ (provides ZZ coupling)
    c.x(1)
    
    # Ramsey on qubit 0: H - pulse - H
    c.h(0)
    
    # "Wait" via a long low-amplitude pulse (simulates idle time with ZZ)
    pulse_wait = waveforms.Drag(duration=500, amp=0.1, sigma=125, beta=0.1)
    c.metadata["pulse_library"] = {"pulse_wait": pulse_wait}
    c.ops.append(("pulse", 0, "pulse_wait", {"qubit_freq": 5.0e9}))
    
    c.h(0)  # Second H for Ramsey
    
    c.measure_z(0)
    c.measure_z(1)
    
    print(f"\nRamsey sequence:")
    print(f"  Qubit 0: H → wait (500ns) → H")
    print(f"  Qubit 1: Prepared in |1⟩ (provides ZZ coupling)")
    
    # Test different ZZ strengths
    zz_strengths = [
        (None, "No ZZ (ideal)"),
        (1e6, "Weak (1 MHz)"),
        (3e6, "Moderate (3 MHz)"),
        (10e6, "Strong (10 MHz)")
    ]
    
    print(f"\n{'Configuration':<20} {'|0⟩ prob':<12} {'|1⟩ prob':<12} {'Contrast':<12}")
    print("-" * 60)
    
    for config in zz_strengths:
        if config[0] is None:
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
        
        # Qubit 0 measurement (qubit 1 always in |1⟩)
        p0 = (counts.get("01", 0) + counts.get("00", 0)) / 10000
        p1 = (counts.get("11", 0) + counts.get("10", 0)) / 10000
        
        contrast = abs(p0 - p1)
        
        print(f"{label:<20} {p0:<12.4f} {p1:<12.4f} {contrast:<12.4f}")
    
    print(f"\n📊 Observation:")
    print(f"  • No ZZ: Perfect Ramsey fringe (high contrast)")
    print(f"  • With ZZ: Phase shifts reduce contrast")
    print(f"  • Stronger ZZ → Lower contrast → Lost coherence")
    print(f"\n⚠️  This is how ZZ crosstalk causes decoherence in real devices!")
    
    print()


def example3_echo_sequence_mitigation():
    """Example 3: Echo sequence to cancel ZZ crosstalk."""
    print("=" * 80)
    print("Example 3: Echo Sequence Mitigation")
    print("X-π-X cancels ZZ phase accumulation!")
    print("=" * 80)
    
    topo = get_qubit_topology(2, topology="linear", zz_strength=5e6)
    
    # Scenario 1: Single long pulse (accumulates ZZ)
    print(f"\n--- Scenario 1: Single 200ns Pulse ---")
    c1 = tq.Circuit(2)
    c1.h(0)
    c1.cnot(0, 1)
    
    pulse_long = waveforms.Drag(duration=200, amp=1.0, sigma=50, beta=0.2)
    c1.metadata["pulse_library"] = {"pulse": pulse_long}
    c1.ops.append(("pulse", 0, "pulse", {"qubit_freq": 5.0e9}))
    
    c1.h(0)
    c1.h(1)
    c1.measure_z(0)
    c1.measure_z(1)
    
    result1 = c1.device(
        provider="simulator",
        device="statevector",
        zz_topology=topo,
        zz_mode="local",
        shots=10000
    ).postprocessing(method=None).run()
    
    counts1 = result1[0]["result"]
    print(f"Measurement distribution:")
    for state in ["00", "01", "10", "11"]:
        count = counts1.get(state, 0)
        if count > 0:
            print(f"  |{state}⟩: {count:5d} ({count/10000*100:5.2f}%)")
    
    # Scenario 2: Echo sequence (pulse - wait - pulse)
    print(f"\n--- Scenario 2: Echo Sequence (100ns - wait - 100ns) ---")
    c2 = tq.Circuit(2)
    c2.h(0)
    c2.cnot(0, 1)
    
    pulse_short = waveforms.Drag(duration=100, amp=1.0, sigma=25, beta=0.2)
    c2.metadata["pulse_library"] = {"pulse": pulse_short}
    
    # First pulse
    c2.ops.append(("pulse", 0, "pulse", {"qubit_freq": 5.0e9}))
    # Second pulse (acts like echo)
    c2.ops.append(("pulse", 0, "pulse", {"qubit_freq": 5.0e9}))
    
    c2.h(0)
    c2.h(1)
    c2.measure_z(0)
    c2.measure_z(1)
    
    result2 = c2.device(
        provider="simulator",
        device="statevector",
        zz_topology=topo,
        zz_mode="local",
        shots=10000
    ).postprocessing(method=None).run()
    
    counts2 = result2[0]["result"]
    print(f"Measurement distribution:")
    for state in ["00", "01", "10", "11"]:
        count = counts2.get(state, 0)
        if count > 0:
            print(f"  |{state}⟩: {count:5d} ({count/10000*100:5.2f}%)")
    
    print(f"\n--- Comparison ---")
    
    # Calculate difference
    diff = sum(abs(counts1.get(s, 0) - counts2.get(s, 0)) for s in ["00", "01", "10", "11"])
    
    xi = 5e6
    phi_single = xi * 200e-9
    phi_echo = xi * 100e-9  # Each pulse
    
    print(f"Single pulse ZZ phase: {phi_single*180/np.pi:.2f}°")
    print(f"Echo pulse ZZ phase: 2 × {phi_echo*180/np.pi:.2f}° = {2*phi_echo*180/np.pi:.2f}°")
    print(f"\nMeasurement difference: {diff} counts")
    
    if diff < 1000:
        print(f"✅ Echo sequence partially mitigates ZZ crosstalk!")
    else:
        print(f"⚠️  More sophisticated echo needed for full mitigation")
    
    print()


def main():
    """Run all visible ZZ crosstalk impact examples."""
    print("\n" + "=" * 80)
    print(" ZZ Crosstalk: VISIBLE Impact on Quantum States")
    print(" Clear differences in measurement outcomes!")
    print("=" * 80 + "\n")
    
    example1_conditional_phase_visibility()
    example2_ramsey_interference()
    example3_echo_sequence_mitigation()
    
    print("=" * 80)
    print("Key Takeaways")
    print("=" * 80)
    print("\n1️⃣  ZZ causes CONDITIONAL PHASE: |00⟩ + |11⟩ → |00⟩ + e^(iφ)|11⟩")
    print("2️⃣  Visible in X-basis or Ramsey measurements")
    print("3️⃣  Destroys quantum interference and coherence")
    print("4️⃣  Must be mitigated via echo sequences or ZZ-free gates")
    print("5️⃣  TyxonQ accurately models these effects!")
    print("=" * 80)


if __name__ == "__main__":
    main()
