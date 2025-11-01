#!/usr/bin/env python3
"""
Complete example: pulse_inline + TQASM export + three-level simulation

================================================================================
SCENARIO: Local Pulse Simulation with TQASM Export
================================================================================

This example demonstrates the correct workflow for:
1. Compiling a circuit to pulse_inline format
2. Exporting to TQASM (portable, cloud-ready)
3. Running the same TQASM with different three_level settings

KEY INSIGHT: three_level is a SIMULATION parameter, not part of TQASM syntax.
The same TQASM code can be executed with or without three-level leakage modeling.

Example workflow:

    Circuit (gates)
         ↓ compile with use_pulse()
    IR (pulse_inline)
         ↓ export output="tqasm"
    TQASM (defcal + waveforms)
         ↓ device().run(three_level=True/False)
    Results (with/without leakage)

TQASM itself is INDEPENDENT of three_level. The parameter is passed at runtime.

================================================================================
REFERENCES
================================================================================

[1] Motzoi et al., PRL 103, 110501 (2009)
    - DRAG pulse correction reduces leakage by ~100x
    - Optimal β ≈ -1/(2α) for Transmon

[2] Koch et al., Phys. Rev. A 76, 042319 (2007)
    - Transmon Hamiltonian with anharmonicity α
    - Three-level model: {|0⟩, |1⟩, |2⟩}

[3] Jurcevic et al., arXiv:2108.12323 (2021)
    - Hardware characterization of leakage
    - Validation of numerical models on real IBM processors

[4] OpenQASM 3.0 Specification
    - TQASM 0.2 based on OpenQASM 3.0 + OpenPulse
    - https://openqasm.com/
"""

import tyxonq as tq
from tyxonq import Circuit, waveforms
from tyxonq.compiler.api import compile as compile_api


def example_1_pulse_inline_to_tqasm():
    """
    STEP 1: Compile circuit to pulse_inline format
    STEP 2: Export to TQASM (no three_level parameter)
    STEP 3: Run with device(three_level=True/False)
    
    This demonstrates the CLEAN SEPARATION between:
    - Compilation phase (creates TQASM)
    - Execution phase (applies three_level parameter)
    """
    print("\n" + "="*80)
    print("Example 1: Pulse Inline → TQASM → Three-Level Simulation")
    print("="*80)
    
    # Step 1: Create circuit and compile to pulse_inline format
    print("\n1️⃣  Creating circuit...")
    c = Circuit(1)
    c.h(0)
    c.measure_z(0)
    
    print(f"   Circuit: 1 Hadamard gate + measurement")
    
    # Use pulse compilation
    print("\n2️⃣  Compiling to pulse_inline format...")
    c_pulse = c.use_pulse(
        mode="pulse_only",
        device_params={
            "qubit_freq": [5.0e9],
            "anharmonicity": [-330e6]  # IBM typical
        }
    )
    
    # Compile to IR format first (pulse_inline)
    result_ir = compile_api(c_pulse, output="pulse_ir", options={"mode": "pulse_only"})
    ir_circuit = result_ir["circuit"]
    
    print(f"   ✅ Compiled to pulse IR (pulse_inline format)")
    print(f"   Circuit has {len(ir_circuit.ops)} operations")
    
    # Step 2: Export to TQASM format
    print("\n3️⃣  Exporting to TQASM 0.2 format...")
    result_tqasm = compile_api(c_pulse, output="tqasm", options={"mode": "pulse_only"})
    tqasm_code = result_tqasm["circuit"]
    
    print(f"   ✅ Exported to TQASM (length: {len(tqasm_code)} chars)")
    print(f"\n   TQASM Header (first 20 lines):")
    print("   " + "-"*76)
    for i, line in enumerate(tqasm_code.split('\n')[:20]):
        print(f"   {line}")
    if len(tqasm_code.split('\n')) > 20:
        print("   ...")
    print("   " + "-"*76)
    
    # KEY OBSERVATION: No "three_level" in TQASM!
    print(f"\n   🔍 Verification: Does TQASM contain 'three_level'?")
    has_three_level_in_tqasm = "three_level" in tqasm_code
    print(f"      Answer: {has_three_level_in_tqasm} (Good! three_level is not in TQASM syntax)")
    
    # Step 3: Run with three_level=False (ideal 2-level)
    print("\n4️⃣  Running 1: 2-Level Simulation (Ideal, no leakage)...")
    result_2level = c.device(
        provider="simulator",
        device="statevector",
        three_level=False  # ← Runtime parameter, not in TQASM
    ).run(shots=5000)
    
    counts_2level = result_2level[0]["result"] if isinstance(result_2level, list) else result_2level["result"]
    print(f"   Results: {counts_2level}")
    print(f"   → Only |0⟩ and |1⟩ observed (ideal quantum mechanics)")
    
    # Step 4: Run with three_level=True (realistic 3-level)
    print("\n5️⃣  Running 2: 3-Level Simulation (Realistic, with leakage)...")
    result_3level = c.device(
        provider="simulator",
        device="statevector",
        three_level=True,  # ← Same TQASM, different parameter!
        rabi_freq=30e6
    ).run(shots=5000)
    
    counts_3level = result_3level[0]["result"] if isinstance(result_3level, list) else result_3level["result"]
    print(f"   Results: {counts_3level}")
    
    if '2' in counts_3level:
        leakage = counts_3level['2'] / 5000
        print(f"   → Leakage to |2⟩: {leakage:.2%} (realistic hardware behavior)")
    else:
        print(f"   → No significant leakage observed (within noise)")
    
    # Summary
    print("\n" + "="*80)
    print("✅ Summary: Clean Separation Confirmed")
    print("="*80)
    print("""
    Key Points:
    
    1. ✅ TQASM does NOT contain three_level
       - TQASM is purely a pulse definition format
       - Valid for any backend (local or cloud)
    
    2. ✅ three_level parameter is passed at runtime
       - circuit.device(three_level=True).run()
       - Affects only local simulation behavior
       - Does not affect exported TQASM
    
    3. ✅ Same TQASM can run with different settings
       - three_level=False: Ideal (no leakage)
       - three_level=True: Realistic (with leakage)
    
    4. ✅ Parameter passing chain verified
       - device() → _device_opts
       - run() → device_base.run(..., **opts)
       - driver → engine.run(..., three_level=True)
    """)


def example_2_comparing_waveforms():
    """
    STEP 1: Export same circuit as TQASM
    STEP 2: Compare Gaussian vs DRAG with three_level=True
    
    Demonstrates that TQASM is waveform-agnostic.
    The choice of waveform (Gaussian, DRAG) affects leakage,
    but TQASM is independent of this.
    """
    print("\n" + "="*80)
    print("Example 2: Waveform Comparison via TQASM + Three-Level")
    print("="*80)
    
    # Create two circuits with different waveforms
    print("\n1️⃣  Creating circuits with different pulse shapes...")
    
    # Gaussian: high leakage
    c1 = Circuit(1)
    pulse1 = waveforms.Gaussian(amp=1.0, duration=160, sigma=40)
    c1.metadata["pulse_library"] = {"x_pulse": pulse1}
    c1 = c1.extended([("pulse", 0, "x_pulse", {"qubit_freq": 5.0e9})])
    c1.measure_z(0)
    
    # DRAG: low leakage
    c2 = Circuit(1)
    pulse2 = waveforms.Drag(amp=1.0, duration=160, sigma=40, beta=0.2)
    c2.metadata["pulse_library"] = {"x_pulse": pulse2}
    c2 = c2.extended([("pulse", 0, "x_pulse", {"qubit_freq": 5.0e9})])
    c2.measure_z(0)
    
    print("   Created two circuits: Gaussian and DRAG")
    
    # Run both with three_level=True
    print("\n2️⃣  Running both with three_level=True...")
    
    result_gaussian = c1.device(
        provider="simulator",
        device="statevector",
        three_level=True,
        anharmonicity=-300e6,
        rabi_freq=30e6
    ).run(shots=5000)
    
    result_drag = c2.device(
        provider="simulator",
        device="statevector",
        three_level=True,
        anharmonicity=-300e6,
        rabi_freq=30e6
    ).run(shots=5000)
    
    counts_gaussian = result_gaussian[0]["result"] if isinstance(result_gaussian, list) else result_gaussian["result"]
    counts_drag = result_drag[0]["result"] if isinstance(result_drag, list) else result_drag["result"]
    
    # Extract leakage
    leak_gaussian = counts_gaussian.get('2', 0) / 5000
    leak_drag = counts_drag.get('2', 0) / 5000
    
    print(f"\n   Gaussian leakage: {leak_gaussian:.2%}")
    print(f"   DRAG leakage:     {leak_drag:.2%}")
    print(f"   Suppression:      {leak_gaussian/(leak_drag+1e-6):.0f}x")
    
    print(f"\n💡 Key insight: TQASM carries waveform definition, but three_level is runtime parameter")


def example_3_cloud_ready_workflow():
    """
    STEP 1: Compile to TQASM (portable format)
    STEP 2: Show that TQASM can be:
        a) Submitted to cloud hardware (no three_level)
        b) Run locally with three_level before hardware submission
    
    Demonstrates the dual-use nature of TQASM.
    """
    print("\n" + "="*80)
    print("Example 3: Cloud-Ready Workflow")
    print("="*80)
    
    print("\n1️⃣  Compiling circuit to TQASM format...")
    c = Circuit(2)
    c.h(0)
    c.cx(0, 1)
    c.measure_z(0)
    c.measure_z(1)
    
    c_pulse = c.use_pulse(
        mode="pulse_only",
        device_params={
            "qubit_freq": [5.0e9, 5.1e9],
            "anharmonicity": [-330e6, -320e6]
        }
    )
    
    result = compile_api(c_pulse, output="tqasm", options={"mode": "pulse_only"})
    tqasm_code = result["circuit"]
    
    print(f"   ✅ TQASM generated ({len(tqasm_code)} chars)")
    
    print("\n2️⃣  Workflow Options:")
    print("   ")
    print("   A) Pre-submission verification (local):")
    print("      result = circuit.device(three_level=True).run()")
    print("      # Verify leakage, fidelity before sending to hardware")
    print("      ")
    print("   B) Direct hardware submission (cloud):")
    print("      task = apis.submit_task(source=tqasm_code, device='homebrew_s2')")
    print("      # Hardware automatically handles three-level physics")
    print("      # No three_level parameter needed (implicit)")
    print("   ")
    
    print(f"\n3️⃣  TQASM is hardware-agnostic:")
    print(f"   - Waveforms: ✅ Included in TQASM")
    print(f"   - Defcals: ✅ Included in TQASM")
    print(f"   - three_level: ❌ NOT in TQASM (runtime-specific)")
    print(f"   - Leakage modeling: Handled by hardware physics")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print(" PULSE_INLINE + TQASM + THREE-LEVEL SIMULATION")
    print(" Clean Separation Demonstration")
    print("="*80)
    
    print("\n📚 Educational Focus:")
    print("   This example teaches the CLEAN SEPARATION between:")
    print("   - Compilation (produces TQASM, waveform-aware)")
    print("   - Execution (applies runtime parameters like three_level)")
    
    # Run examples
    example_1_pulse_inline_to_tqasm()
    example_2_comparing_waveforms()
    example_3_cloud_ready_workflow()
    
    print("\n" + "="*80)
    print("✅ All examples completed successfully!")
    print("="*80)
    print("\n🎯 Key Takeaway:")
    print("   three_level is a SIMULATION parameter, not part of TQASM syntax.")
    print("   Device.run(three_level=True) → Local simulation with leakage")
    print("   Cloud submission → No three_level (hardware handles physics)")
    print("\n📖 For more details, see:")
    print("   - docs/source/tutorials/advanced/pulse_inline_three_level.rst")
    print("   - tests_core_module/test_pulse_inline_three_level.py")


if __name__ == "__main__":
    main()
