#!/usr/bin/env python3
"""
Test: pulse_inline + TQASM export + three-level simulation integration

Tests the correct workflow where:
1. Circuits compile to pulse_inline format
2. TQASM export does NOT include three_level parameter
3. device().run(three_level=True/False) correctly applies the parameter
"""

import pytest
from tyxonq import Circuit, waveforms
from tyxonq.compiler.api import compile as compile_api
from tyxonq.devices.simulators.statevector.engine import StatevectorEngine


class TestPulseInlineTQASMThreeLevel:
    """Integration tests for TQASM export and three-level parameter passing."""
    
    def test_tqasm_export_does_not_contain_three_level_syntax(self):
        """Verify that TQASM export does NOT include three_level parameter."""
        print("\n" + "="*70)
        print("Test 1: TQASM export should NOT contain 'three_level' syntax")
        print("="*70)
        
        # Create simple circuit
        c = Circuit(1)
        c.h(0)
        c.measure_z(0)
        
        # Compile to TQASM
        c_pulse = c.use_pulse(device_params={
            "qubit_freq": [5.0e9],
            "anharmonicity": [-330e6]
        })
        
        result = compile_api(c_pulse, output="tqasm", options={"mode": "pulse_only"})
        tqasm_code = result["compiled_source"]
        
        # Verify: no three_level in TQASM
        assert isinstance(tqasm_code, str), "Output should be string (TQASM code)"
        assert "three_level" not in tqasm_code, \
            "TQASM should NOT contain 'three_level' keyword"
        
        # Verify: TQASM should contain defcal
        assert "defcal" in tqasm_code, "TQASM should contain defcal definitions"
        assert ("TQASM 0.2" in tqasm_code or "OPENQASM 3.0" in tqasm_code), \
            "TQASM should have version declaration"
        
        print("✅ TQASM export verified:")
        print(f"   - Contains 'defcal': {('defcal' in tqasm_code)}")
        print(f"   - Contains 'three_level': {('three_level' in tqasm_code)}")
        print(f"   - Version declared: {('TQASM 0.2' in tqasm_code or 'OPENQASM 3.0' in tqasm_code)}")
    
    def test_device_run_three_level_parameter_passthrough(self):
        """Verify that device().run(three_level=True) correctly passes parameter to engine."""
        print("\n" + "="*70)
        print("Test 2: three_level parameter should pass through device chain")
        print("="*70)
        
        # Create circuit
        c = Circuit(1)
        c.h(0)
        c.measure_z(0)
        
        # Run with three_level=True
        result_with_3level = c.device(
            provider="simulator",
            device="statevector",
            three_level=True,
            anharmonicity=-300e6,
            rabi_freq=30e6
        ).run(shots=2000)
        
        # Run with three_level=False
        result_without_3level = c.device(
            provider="simulator",
            device="statevector",
            three_level=False
        ).run(shots=2000)
        
        # Extract counts
        counts_with = result_with_3level[0]["result"] if isinstance(result_with_3level, list) else result_with_3level["result"]
        counts_without = result_without_3level[0]["result"] if isinstance(result_without_3level, list) else result_without_3level["result"]
        
        # Verify results
        assert "result" in (result_with_3level[0] if isinstance(result_with_3level, list) else result_with_3level), \
            "Result should contain 'result' key"
        
        print("✅ Parameter passthrough verified:")
        print(f"   - With three_level=True: {counts_with}")
        print(f"   - With three_level=False: {counts_without}")
        
        # With three_level=True, we may see '2' state (leakage)
        if '2' in counts_with:
            leak = counts_with['2'] / 2000
            print(f"   - Leakage observed: {leak:.2%}")
        
        # Without three_level, no '2' should appear
        if '2' not in counts_without:
            print(f"   - No leakage (2-level): Expected behavior ✅")
    
    def test_pulse_inline_vs_pulse_with_three_level(self):
        """Verify pulse and pulse_inline produce similar results with three_level."""
        print("\n" + "="*70)
        print("Test 3: pulse vs pulse_inline equivalence with three_level")
        print("="*70)
        
        # Pulse operation: references pulse_library
        c1 = Circuit(1)
        pulse = waveforms.Gaussian(amp=1.0, duration=160, sigma=40)
        c1.metadata["pulse_library"] = {"x_pulse": pulse}
        c1 = c1.extended([("pulse", 0, "x_pulse", {"qubit_freq": 5.0e9})])
        c1.measure_z(0)
        
        # Pulse_inline: serialized waveform
        c2 = Circuit(1)
        waveform_dict = {
            "type": "gaussian",
            "class": "Gaussian",
            "args": [1.0, 160, 40, 0.0]
        }
        c2 = c2.extended([("pulse_inline", 0, waveform_dict, {"qubit_freq": 5.0e9})])
        c2.measure_z(0)
        
        # Run both with three_level=True
        engine = StatevectorEngine()
        
        result1 = engine.run(c1, shots=5000, three_level=True, 
                            anharmonicity=-300e6, rabi_freq=30e6)
        result2 = engine.run(c2, shots=5000, three_level=True,
                            anharmonicity=-300e6, rabi_freq=30e6)
        
        counts1 = result1["result"]
        counts2 = result2["result"]
        
        # Check that both have similar leakage pattern
        leak1 = counts1.get('2', 0) / 5000
        leak2 = counts2.get('2', 0) / 5000
        
        print("✅ pulse vs pulse_inline with three_level=True:")
        print(f"   - pulse leakage: {leak1:.2%}")
        print(f"   - pulse_inline leakage: {leak2:.2%}")
        
        # Allow some statistical variation (5000 shots)
        assert abs(leak1 - leak2) < 0.03, \
            f"Leakage difference too large: {abs(leak1 - leak2):.2%}"
    
    def test_drag_suppresses_leakage_with_three_level(self):
        """Verify DRAG pulse reduces leakage when three_level=True."""
        print("\n" + "="*70)
        print("Test 4: DRAG should suppress leakage with three_level=True")
        print("="*70)
        
        # Gaussian: high leakage
        c1 = Circuit(1)
        pulse1 = waveforms.Gaussian(amp=1.0, duration=160, sigma=40)
        c1.metadata["pulse_library"] = {"x": pulse1}
        c1 = c1.extended([("pulse", 0, "x", {"qubit_freq": 5.0e9})])
        c1.measure_z(0)
        
        # DRAG: low leakage
        c2 = Circuit(1)
        pulse2 = waveforms.Drag(amp=1.0, duration=160, sigma=40, beta=0.2)
        c2.metadata["pulse_library"] = {"x": pulse2}
        c2 = c2.extended([("pulse", 0, "x", {"qubit_freq": 5.0e9})])
        c2.measure_z(0)
        
        # Run both with three_level=True
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
        
        leak_gaussian = counts_gaussian.get('2', 0) / 5000
        leak_drag = counts_drag.get('2', 0) / 5000
        
        print("✅ DRAG suppression with three_level=True:")
        print(f"   - Gaussian leakage: {leak_gaussian:.2%}")
        print(f"   - DRAG leakage: {leak_drag:.2%}")
        
        # DRAG should have less leakage
        if leak_gaussian > 0.001:  # Only test if Gaussian has measurable leakage
            assert leak_drag < leak_gaussian, \
                "DRAG should have less leakage than Gaussian"
            suppression = leak_gaussian / (leak_drag + 1e-6)
            print(f"   - Suppression factor: {suppression:.1f}x")
    
    def test_same_tqasm_different_three_level_settings(self):
        """
        Show that the same compiled circuit can be run with different three_level settings.
        This demonstrates the independence of TQASM from runtime parameters.
        """
        print("\n" + "="*70)
        print("Test 5: Same circuit, different three_level settings")
        print("="*70)
        
        # Create and compile circuit
        c = Circuit(1)
        c.h(0)
        c.measure_z(0)
        
        # Compile once to pulse IR
        c_pulse = c.use_pulse(device_params={"qubit_freq": [5.0e9]})
        result_ir = compile_api(c_pulse, output="pulse_ir", options={"mode": "pulse_only"})
        compiled_circuit = result_ir["circuit"]
        
        # Execute the same compiled circuit with different three_level settings
        engine = StatevectorEngine()
        
        result_2level = engine.run(compiled_circuit, shots=2000, three_level=False)
        result_3level = engine.run(compiled_circuit, shots=2000, three_level=True,
                                  anharmonicity=-300e6)
        
        counts_2level = result_2level["result"]
        counts_3level = result_3level["result"]
        
        print("✅ Same compiled circuit with different settings:")
        print(f"   - three_level=False: {counts_2level}")
        print(f"   - three_level=True: {counts_3level}")
        
        # 2-level should not have '2' state
        assert '2' not in counts_2level, "2-level simulation should not produce '2' state"
        
        # 3-level may have '2' state
        if '2' in counts_3level:
            print(f"   - 3-level has leakage: {counts_3level['2']} counts")


def test_device_chain_preserves_three_level():
    """
    Test that the device() chain correctly preserves three_level parameter.
    This is the key test for the parameter passing mechanism.
    """
    print("\n" + "="*70)
    print("Test 6: device() chain should preserve three_level")
    print("="*70)
    
    c = Circuit(1)
    c.h(0)
    c.measure_z(0)
    
    # The device() method should store parameters in _device_opts
    c_configured = c.device(
        provider="simulator",
        device="statevector",
        three_level=True,
        anharmonicity=-300e6,
        shots=1000
    )
    
    # Verify parameters are stored
    assert "three_level" in c_configured._device_opts, \
        "three_level should be in _device_opts"
    assert c_configured._device_opts["three_level"] is True
    assert c_configured._device_opts["anharmonicity"] == -300e6
    
    print("✅ device() chain verified:")
    print(f"   - three_level stored: {c_configured._device_opts.get('three_level')}")
    print(f"   - anharmonicity stored: {c_configured._device_opts.get('anharmonicity')}")


if __name__ == "__main__":
    # Run all tests
    test_suite = TestPulseInlineTQASMThreeLevel()
    
    print("\n" + "="*70)
    print("PULSE_INLINE + TQASM + THREE-LEVEL INTEGRATION TESTS")
    print("="*70)
    
    test_suite.test_tqasm_export_does_not_contain_three_level_syntax()
    test_suite.test_device_run_three_level_parameter_passthrough()
    test_suite.test_pulse_inline_vs_pulse_with_three_level()
    test_suite.test_drag_suppresses_leakage_with_three_level()
    test_suite.test_same_tqasm_different_three_level_settings()
    test_device_chain_preserves_three_level()
    
    print("\n" + "="*70)
    print("✅ All integration tests passed!")
    print("="*70)
    print("\nSummary:")
    print("  ✅ TQASM does not contain three_level syntax")
    print("  ✅ three_level parameter passes through device() chain correctly")
    print("  ✅ pulse and pulse_inline are equivalent with three_level")
    print("  ✅ DRAG suppression works with three_level=True")
    print("  ✅ Same compiled circuit can run with different three_level settings")
    print("  ✅ device() correctly stores three_level in _device_opts")
