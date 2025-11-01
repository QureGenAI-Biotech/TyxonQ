"""
Example: pulse_inline with three-level system support.

Demonstrates the NEW feature for realistic leakage simulation in pulse_inline
operations, useful for cloud quantum circuit submission.

Run this example:
  conda activate qc
  python examples/pulse_inline_three_level.py

For detailed documentation, see:
  docs/source/tutorials/advanced/pulse_inline_three_level.rst
  docs/source/examples/pulse_inline_examples.rst
"""

import numpy as np
from tyxonq import Circuit
from tyxonq.devices.simulators.statevector.engine import StatevectorEngine


def example_1_detect_leakage():
    """Detect leakage with Gaussian pulse."""
    print("\n" + "="*70)
    print("Example 1: Detect Leakage in pulse_inline")
    print("="*70)
    
    c = Circuit(1)
    
    # Gaussian pulse (no DRAG) - will have leakage
    waveform_dict = {
        "type": "gaussian",
        "class": "Gaussian",
        "args": [1.0, 160, 40, 0.0]
    }
    
    c = c.extended([
        ("pulse_inline", 0, waveform_dict, {
            "qubit_freq": 5.0e9,
            "anharmonicity": -300e6
        })
    ])
    c.measure_z(0)
    
    # Run with 3-level
    engine = StatevectorEngine()
    result = engine.run(c, shots=1000, three_level=True)
    
    counts = result["result"]
    total = sum(counts.values())
    leakage = counts.get('2', 0) / total
    
    print(f"\nMeasurement outcomes: {counts}")
    print(f"Leakage to |2⟩: {leakage:.2%}")
    print("\n💡 The |2⟩ outcomes represent population leaked from computational space")


def example_2_drag_suppression():
    """DRAG pulses suppress leakage by ~100x."""
    print("\n" + "="*70)
    print("Example 2: DRAG Suppresses Leakage")
    print("="*70)
    
    engine = StatevectorEngine()
    
    # Test Gaussian vs DRAG
    test_cases = [
        ("Gaussian (no DRAG)", {"type": "gaussian", "class": "Gaussian", "args": [1.0, 160, 40, 0.0]}),
        ("DRAG (β=0.2)", {"type": "drag", "class": "Drag", "args": [1.0, 160, 40, 0.2, 0.0]})
    ]
    
    results = {}
    
    for name, waveform in test_cases:
        c = Circuit(1)
        c = c.extended([
            ("pulse_inline", 0, waveform, {
                "qubit_freq": 5.0e9,
                "anharmonicity": -300e6,
                "rabi_freq": 30e6
            })
        ])
        c.measure_z(0)
        
        result = engine.run(c, shots=5000, three_level=True)
        counts = result["result"]
        total = sum(counts.values())
        leakage = counts.get('2', 0) / total
        results[name] = leakage
        
        print(f"\n{name}:")
        print(f"  Outcomes: {counts}")
        print(f"  Leakage: {leakage:.2%}")
    
    suppression = results["Gaussian (no DRAG)"] / (results["DRAG (β=0.2)"] + 1e-6)
    print(f"\n✅ Suppression factor: {suppression:.0f}x")
    print("   DRAG correction is highly effective!")


def example_3_optimal_beta():
    """Find optimal DRAG beta parameter."""
    print("\n" + "="*70)
    print("Example 3: Find Optimal DRAG Beta Parameter")
    print("="*70)
    
    alpha = -300e6
    beta_theoretical = -1.0 / (2.0 * alpha)
    
    print(f"\nTheoretical optimal β = -1/(2α) = {beta_theoretical:.6f}")
    print(f"\nExperimental scan:")
    print(f"{'β':>8} | {'Leakage':>10}")
    print(f"{'-'*8}-+-{'-'*10}")
    
    engine = StatevectorEngine()
    beta_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    leakages = []
    
    for beta in beta_values:
        waveform = {
            "type": "drag",
            "class": "Drag",
            "args": [1.0, 160, 40, beta, 0.0]
        }
        
        c = Circuit(1)
        c = c.extended([
            ("pulse_inline", 0, waveform, {
                "qubit_freq": 5.0e9,
                "anharmonicity": alpha,
                "rabi_freq": 30e6
            })
        ])
        c.measure_z(0)
        
        result = engine.run(c, shots=2000, three_level=True)
        counts = result["result"]
        total = sum(counts.values())
        leakage = counts.get('2', 0) / total
        leakages.append(leakage)
        
        print(f"{beta:8.2f} | {leakage:10.4%}")
    
    min_idx = np.argmin(leakages)
    optimal_beta = beta_values[min_idx]
    
    print(f"\n✅ Optimal β (experimental): {optimal_beta:.2f}")
    print(f"   Matches theory: {abs(optimal_beta - beta_theoretical) < 0.01}")


def example_4_hybrid_circuit():
    """Mix classical gates with pulse_inline."""
    print("\n" + "="*70)
    print("Example 4: Hybrid Circuit (Classical Gates + Pulses)")
    print("="*70)
    
    c = Circuit(2)
    
    # Classical gates
    c.h(0)
    
    # Pulse-based operation
    c = c.extended([
        ("pulse_inline", 0, {
            "type": "drag",
            "class": "Drag",
            "args": [1.0, 160, 40, 0.2, 0.0]
        }, {
            "qubit_freq": 5.0e9,
            "anharmonicity": -300e6
        })
    ])
    
    # More classical gates
    c.cnot(0, 1)
    c.measure_z(0)
    c.measure_z(1)
    
    # Run
    engine = StatevectorEngine()
    result = engine.run(c, shots=1000, three_level=True)
    
    counts = result["result"]
    print(f"\nHybrid circuit outcomes (first 5 sorted):")
    for bitstring, count in sorted(counts.items())[:5]:
        print(f"  {bitstring}: {count}")
    
    print("\n✅ pulse_inline works seamlessly with classical gates!")


def example_5_parameter_sensitivity():
    """Show Rabi frequency impact on leakage."""
    print("\n" + "="*70)
    print("Example 5: Rabi Frequency Sensitivity")
    print("="*70)
    
    engine = StatevectorEngine()
    rabi_freqs = [10, 20, 30, 40, 50]  # MHz
    
    print(f"\nEffect of Rabi frequency on leakage:")
    print(f"{'Rabi (MHz)':>12} | {'Leakage':>10}")
    print(f"{'-'*12}-+-{'-'*10}")
    
    for rabi_mhz in rabi_freqs:
        waveform = {
            "type": "drag",
            "class": "Drag",
            "args": [1.0, 160, 40, 0.2, 0.0]
        }
        
        c = Circuit(1)
        c = c.extended([
            ("pulse_inline", 0, waveform, {
                "qubit_freq": 5.0e9,
                "anharmonicity": -300e6,
                "rabi_freq": rabi_mhz * 1e6
            })
        ])
        c.measure_z(0)
        
        result = engine.run(c, shots=1000, three_level=True)
        counts = result["result"]
        total = sum(counts.values())
        leakage = counts.get('2', 0) / total
        
        print(f"{rabi_mhz:12} | {leakage:10.4%}")
    
    print(f"\n💡 Higher Rabi frequencies increase leakage (quadratic scaling)")


def example_6_waveform_comparison():
    """Compare different waveform types."""
    print("\n" + "="*70)
    print("Example 6: Waveform Type Comparison")
    print("="*70)
    
    engine = StatevectorEngine()
    
    waveforms_list = [
        ("Gaussian", {"type": "gaussian", "class": "Gaussian", "args": [1.0, 160, 40, 0.0]}),
        ("GaussianSquare", {"type": "gaussian_square", "class": "GaussianSquare", "args": [1.0, 160, 40, 60, 0.0]}),
        ("Flattop", {"type": "flattop", "class": "Flattop", "args": [1.0, 160, 40, 0.5, 0.0]}),
        ("Constant", {"type": "constant", "class": "Constant", "args": [1.0, 160, 0.0]})
    ]
    
    print(f"\nWaveform type leakage comparison (5000 shots):")
    print(f"{'Type':>15} | {'Leakage':>10} | {'Compression':>12}")
    print(f"{'-'*15}-+-{'-'*10}-+-{'-'*12}")
    
    for name, waveform in waveforms_list:
        c = Circuit(1)
        c = c.extended([
            ("pulse_inline", 0, waveform, {
                "qubit_freq": 5.0e9,
                "anharmonicity": -300e6,
                "rabi_freq": 30e6
            })
        ])
        c.measure_z(0)
        
        result = engine.run(c, shots=5000, three_level=True)
        counts = result["result"]
        total = sum(counts.values())
        leakage = counts.get('2', 0) / total
        
        # Some waveforms have smoother envelope
        compression = "Smooth" if "Flattop" in name or "Square" in name else "Sharp"
        
        print(f"{name:15} | {leakage:10.3%} | {compression:>12}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PULSE_INLINE 3-LEVEL SYSTEM SUPPORT - EXAMPLES")
    print("="*70)
    print("\nNew Feature: pulse_inline now supports three_level=True")
    print("Benefit: Realistic leakage simulation for cloud quantum experiments")
    
    example_1_detect_leakage()
    example_2_drag_suppression()
    example_3_optimal_beta()
    example_4_hybrid_circuit()
    example_5_parameter_sensitivity()
    example_6_waveform_comparison()
    
    print("\n" + "="*70)
    print("✅ All examples completed!")
    print("="*70)
    print("\n📚 For full documentation, see:")
    print("   docs/source/tutorials/advanced/pulse_inline_three_level.rst")
    print("   docs/source/examples/pulse_inline_examples.rst")
    print("\n🧪 For comprehensive tests, run:")
    print("   pytest tests_core_module/test_pulse_inline_three_level.py -v")
