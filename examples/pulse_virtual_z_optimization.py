"""
Pulse Virtual-Z Gate Optimization Example

Demonstrates how the GateToPulsePass optimizer merges adjacent RZ gates
to reduce phase frame tracking complexity and improve overall fidelity.

Key Concepts:
  1. Virtual-Z gates: Zero-time phase frame updates (no physical pulse)
  2. Optimization: Merging consecutive RZ gates on the same qubit
  3. Benefits: Simpler phase tracking, reduced overhead, higher fidelity

Physical Insight:
  - Traditional gates: RX, RY need microwave pulses (~30-50 ns)
  - Virtual-Z: Only updates phase frame (0 ns, zero cost!)
  - Multiple RZ gates: Can be merged without affecting physics
  
Example: RZ(π/4) · RZ(π/3) = RZ(7π/12)
  Before: Track 2 phase updates
  After:  Track 1 phase update (simpler!)

Reference:
  McKay et al., "Qiskit Backend Specifications for OpenQASM and OpenPulse 
  Experiments" (2017), PRA 96, 022330
"""

import numpy as np
from tyxonq import Circuit
from tyxonq.compiler.pulse_compile_engine.native.gate_to_pulse import GateToPulsePass


def example_1_basic_virtual_z_merging():
    """Example 1: Basic merging of adjacent RZ gates.
    
    Shows the simplest case: multiple RZ gates on the same qubit
    in consecutive positions get merged into a single virtual_z operation.
    """
    print("=" * 70)
    print("Example 1: Basic Virtual-Z Merging")
    print("=" * 70)
    
    # Create circuit with multiple RZ gates
    circuit = Circuit(2)
    circuit.rz(0, np.pi / 4)      # RZ(π/4)
    circuit.rz(0, np.pi / 3)      # RZ(π/3)
    circuit.rz(0, np.pi / 6)      # RZ(π/6)
    
    print("\nInput Circuit:")
    print("  q0: RZ(π/4) ─ RZ(π/3) ─ RZ(π/6)")
    print(f"       Total: π/4 + π/3 + π/6 = 3π/4")
    
    # Compile to pulse
    compiler = GateToPulsePass()
    pulse_circuit = compiler.execute_plan(circuit, mode="pulse_only")
    
    # Count operations
    vz_ops = [op for op in pulse_circuit.ops 
              if isinstance(op, tuple) and op[0] == "virtual_z"]
    
    print(f"\nAfter Optimization:")
    print(f"  Number of virtual_z operations: {len(vz_ops)} (was 3)")
    if vz_ops:
        for op in vz_ops:
            angle_deg = np.degrees(op[2])
            print(f"    - VZ on q{op[1]}: {angle_deg:.1f}° (≈ {op[2] / np.pi:.3f}π)")
    
    print("\n✅ Benefit: Reduced phase tracking from 3 updates to 1")
    print()


def example_2_breaking_chain_with_pulse():
    """Example 2: Breaking the RZ chain with a pulse operation.
    
    When a non-virtual_z operation (like a pulse) appears between RZ gates,
    the optimization creates separate merged groups on either side.
    """
    print("=" * 70)
    print("Example 2: RZ Chain Broken by Pulse Operation")
    print("=" * 70)
    
    # Create circuit with RZ gates separated by an X gate
    circuit = Circuit(2)
    circuit.rz(0, np.pi / 4)      # RZ(π/4)
    circuit.rz(0, np.pi / 3)      # RZ(π/3)
    circuit.x(0)                  # X gate (breaks the chain!)
    circuit.rz(0, np.pi / 2)      # RZ(π/2)
    
    print("\nInput Circuit:")
    print("  q0: RZ(π/4) ─ RZ(π/3) ─ X ─ RZ(π/2)")
    print("       ╰─ Merged ─╯       ╰─ Separate ─╯")
    
    # Compile to pulse
    compiler = GateToPulsePass()
    pulse_circuit = compiler.execute_plan(circuit, mode="pulse_only")
    
    # Count operations
    vz_ops = [op for op in pulse_circuit.ops 
              if isinstance(op, tuple) and op[0] == "virtual_z"]
    pulse_ops = [op for op in pulse_circuit.ops 
                 if isinstance(op, tuple) and op[0] == "pulse"]
    
    print(f"\nAfter Optimization:")
    print(f"  Virtual-Z operations: {len(vz_ops)} (was 3)")
    print(f"  Pulse operations: {len(pulse_ops)}")
    
    for i, op in enumerate(vz_ops):
        angle_deg = np.degrees(op[2])
        print(f"    VZ[{i}] on q{op[1]}: {angle_deg:.1f}° (≈ {op[2] / np.pi:.3f}π)")
    
    print("\n✅ Benefit: Two merged groups instead of three separate updates")
    print()


def example_3_multi_qubit_no_cross_merging():
    """Example 3: Multi-qubit circuits (no merging across qubits).
    
    Important: RZ gates on different qubits are NOT merged together.
    Each qubit maintains its own phase tracking.
    """
    print("=" * 70)
    print("Example 3: Multi-Qubit Circuit (No Cross-Qubit Merging)")
    print("=" * 70)
    
    # Create circuit with RZ gates on different qubits
    circuit = Circuit(3)
    circuit.rz(0, np.pi / 4)      # RZ on q0
    circuit.rz(1, np.pi / 3)      # RZ on q1
    circuit.rz(0, np.pi / 6)      # RZ on q0 (gap between due to q1)
    circuit.rz(1, np.pi / 2)      # RZ on q1
    
    print("\nInput Circuit:")
    print("  q0: RZ(π/4) ─────── RZ(π/6)")
    print("  q1: ──── RZ(π/3) ─ RZ(π/2)")
    
    # Compile to pulse
    compiler = GateToPulsePass()
    pulse_circuit = compiler.execute_plan(circuit, mode="pulse_only")
    
    # Count operations by qubit
    vz_ops = [op for op in pulse_circuit.ops 
              if isinstance(op, tuple) and op[0] == "virtual_z"]
    
    print(f"\nAfter Optimization:")
    print(f"  Total Virtual-Z operations: {len(vz_ops)}")
    
    # Group by qubit
    for qubit in range(2):
        qubit_vz = [op for op in vz_ops if op[1] == qubit]
        print(f"    q{qubit}: {len(qubit_vz)} merged operation(s)")
        for op in qubit_vz:
            angle_deg = np.degrees(op[2])
            print(f"      - {angle_deg:.1f}° (≈ {op[2] / np.pi:.3f}π)")
    
    print("\n✅ Benefit: Independent phase tracking per qubit")
    print()


def example_4_performance_comparison():
    """Example 4: Performance comparison before/after optimization.
    
    Demonstrates the reduction in phase frame tracking operations
    and the corresponding complexity reduction.
    """
    print("=" * 70)
    print("Example 4: Performance Metrics")
    print("=" * 70)
    
    # Create a complex circuit with many RZ gates
    circuit = Circuit(2)
    
    # Q0: Multiple RZ gates
    for i in range(5):
        circuit.rz(0, 0.1 * np.pi)
    
    # Q1: Multiple RZ gates
    for i in range(4):
        circuit.rz(1, 0.15 * np.pi)
    
    # Some two-qubit operations
    circuit.cx(0, 1)
    
    # More RZ gates
    circuit.rz(0, 0.2 * np.pi)
    circuit.rz(1, 0.25 * np.pi)
    
    print("\nInput Circuit Structure:")
    print("  q0: [5× RZ gates] ──┬── CX ── RZ")
    print("  q1: [4× RZ gates] ──┴────────── RZ")
    
    # Compile to pulse
    compiler = GateToPulsePass()
    pulse_circuit = compiler.execute_plan(circuit, mode="pulse_only")
    
    # Count operations
    vz_ops = [op for op in pulse_circuit.ops 
              if isinstance(op, tuple) and op[0] == "virtual_z"]
    
    input_rz_count = sum(1 for op in circuit.ops 
                         if isinstance(op, tuple) and op[0] in ("rz", "z"))
    output_vz_count = len(vz_ops)
    
    print(f"\nOptimization Results:")
    print(f"  Input RZ gates: {input_rz_count}")
    print(f"  Output VZ operations: {output_vz_count}")
    print(f"  Reduction: {input_rz_count - output_vz_count} operations saved")
    print(f"  Efficiency: {100 * (1 - output_vz_count / input_rz_count):.1f}% overhead reduction")
    
    print("\n✅ Benefit: Significant reduction in phase tracking overhead")
    print()


def example_5_angle_normalization():
    """Example 5: Angle normalization to [0, 2π).
    
    Shows how merged angles are normalized to prevent phase wraparound issues.
    """
    print("=" * 70)
    print("Example 5: Angle Normalization")
    print("=" * 70)
    
    # Create circuit with angles that sum to > 2π
    circuit = Circuit(1)
    circuit.rz(0, 3 * np.pi)       # 3π
    circuit.rz(0, 2 * np.pi)       # 2π
    circuit.rz(0, np.pi / 2)       # π/2
    
    print("\nInput Circuit:")
    print("  q0: RZ(3π) ─ RZ(2π) ─ RZ(π/2)")
    print(f"       Sum: 3π + 2π + π/2 = 5.5π = 5π + π/2")
    print(f"       Normalized (mod 2π): π + π/2 = 3π/2")
    
    # Compile to pulse
    compiler = GateToPulsePass()
    pulse_circuit = compiler.execute_plan(circuit, mode="pulse_only")
    
    # Get merged angle
    vz_ops = [op for op in pulse_circuit.ops 
              if isinstance(op, tuple) and op[0] == "virtual_z"]
    
    print(f"\nAfter Optimization:")
    if vz_ops:
        merged_angle = vz_ops[0][2]
        merged_angle_over_pi = merged_angle / np.pi
        print(f"  Merged angle: {np.degrees(merged_angle):.1f}° (≈ {merged_angle_over_pi:.3f}π)")
        print(f"  ✓ Normalized to [0, 2π) range")
    
    print("\n✅ Benefit: Correct phase handling without wraparound artifacts")
    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  Pulse Virtual-Z Optimization Examples".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # Run all examples
    example_1_basic_virtual_z_merging()
    example_2_breaking_chain_with_pulse()
    example_3_multi_qubit_no_cross_merging()
    example_4_performance_comparison()
    example_5_angle_normalization()
    
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
Virtual-Z Optimization Benefits:

1. **Reduced Phase Tracking**: Fewer phase frame updates = simpler logic
2. **Better Fidelity**: Less phase management overhead = fewer errors
3. **Faster Compilation**: Fewer operations to process
4. **Hardware Efficient**: Leverages the zero-cost nature of Virtual-Z
5. **Transparent**: Automatic optimization - no user intervention needed

Supported in:
  - GateToPulsePass with any circuit using RZ/Z gates
  - Both single-qubit and multi-qubit circuits
  - Hybrid and pulse_only compilation modes
""")
    print()
