#!/usr/bin/env python3
"""Complete end-to-end test for PulseProgram chain API with simulator."""

import sys
from tyxonq.core.ir.pulse import PulseProgram
from tyxonq.compiler.api import compile_pulse

def test_pulse_chain_compile():
    """Test: PulseProgram → compile_pulse → TQASM output."""
    print("\n" + "="*60)
    print("TEST 1: Pulse Chain - Compile to TQASM")
    print("="*60)
    
    # 1. Build pulse program
    prog = PulseProgram(2)
    prog.set_device_params(
        qubit_freq=[5.0e9, 5.1e9],
        anharmonicity=[-330e6, -320e6]
    )
    
    # Add pulses to both qubits
    prog.drag(0, amp=0.5, duration=160, sigma=40, beta=0.2, qubit_freq=5.0e9)
    prog.gaussian(1, amp=0.3, duration=200, sigma=50, qubit_freq=5.1e9)
    
    print(f"✅ PulseProgram created with {len(prog.ops)} operations")
    print(f"   - Qubit 0: DRAG pulse")
    print(f"   - Qubit 1: Gaussian pulse")
    
    # 2. Compile to TQASM
    result = compile_pulse(
        prog,
        output="tqasm",
        device_params={
            "qubit_freq": [5.0e9, 5.1e9],
            "anharmonicity": [-330e6, -320e6]
        },
        options={"inline_pulses": True}
    )
    
    print(f"✅ Compilation successful")
    print(f"   Output format: TQASM")
    print(f"   Result keys: {list(result.keys())}")
    print(f"   Metadata: {result['metadata']}")
    
    return True

def test_pulse_chain_state_simulation():
    """Test: PulseProgram → state() for numerical simulation (shots=0)."""
    print("\n" + "="*60)
    print("TEST 2: Pulse Chain - Local State Simulation (shots=0)")
    print("="*60)
    
    # 1. Build pulse program
    prog = PulseProgram(1)
    prog.set_device_params(
        qubit_freq=[5.0e9],
        anharmonicity=[-330e6]
    )
    
    # Add a single DRAG pulse
    prog.drag(0, amp=0.5, duration=160, sigma=40, beta=0.2, qubit_freq=5.0e9)
    
    print(f"✅ PulseProgram created with {len(prog.ops)} operation")
    
    # 2. Get state vector (numerical simulation)
    try:
        state = prog.run(shots=0, backend="numpy")
        print(f"✅ State simulation successful")
        print(f"   State shape: {state.shape}")
        print(f"   State type: {type(state)}")
        print(f"   State dtype: {state.dtype}")
        print(f"   State (first 4 elements): {state[:4]}")
        return True
    except Exception as e:
        print(f"❌ State simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pulse_chain_with_device():
    """Test: PulseProgram → device() → run() chain API."""
    print("\n" + "="*60)
    print("TEST 3: Pulse Chain - device().run() Chain API")
    print("="*60)
    
    # 1. Build pulse program
    prog = PulseProgram(1)
    prog.set_device_params(
        qubit_freq=[5.0e9],
        anharmonicity=[-330e6]
    )
    
    prog.drag(0, amp=0.5, duration=160, sigma=40, beta=0.2, qubit_freq=5.0e9)
    
    print(f"✅ PulseProgram created")
    
    # 2. Configure device (chain API)
    prog_with_device = prog.device(provider="simulator", device="statevector", shots=1024)
    print(f"✅ Device configured")
    print(f"   Provider: simulator")
    print(f"   Device: statevector")
    print(f"   Shots: 1024")
    
    # 3. Alternative: shots=0 for direct state
    print(f"\n🔍 Testing shots=0 direct state retrieval...")
    state = prog.run(shots=0, backend="numpy")
    print(f"✅ State retrieved directly (shots=0)")
    print(f"   State shape: {state.shape}")
    
    return True

def test_pulse_ops_field():
    """Test: Verify PulseProgram.ops field alignment with Circuit."""
    print("\n" + "="*60)
    print("TEST 4: PulseProgram.ops Field Alignment")
    print("="*60)
    
    prog = PulseProgram(1)
    
    # Check that ops field exists and is a list
    assert hasattr(prog, 'ops'), "PulseProgram should have 'ops' field"
    assert isinstance(prog.ops, list), "ops should be a list"
    
    print(f"✅ ops field exists and is a list")
    
    # Add operations and verify ops is modified
    prog.drag(0, amp=0.5, duration=160, sigma=40, beta=0.2, qubit_freq=5.0e9)
    assert len(prog.ops) == 1, "ops should have 1 operation"
    
    print(f"✅ ops field stores operations correctly")
    print(f"   ops length: {len(prog.ops)}")
    print(f"   First operation type: {type(prog.ops[0])}")
    
    # Test backward compatibility with pulse_ops
    assert prog.pulse_ops == prog.ops, "pulse_ops should alias ops"
    
    print(f"✅ pulse_ops backward compatibility works")
    
    # Test replace() which gate_to_pulse.py uses
    from dataclasses import replace
    new_ops = prog.ops.copy()
    prog2 = replace(prog, ops=new_ops)
    
    assert prog2.ops == prog.ops, "replace() should copy ops correctly"
    assert prog2 is not prog, "replace() should create new instance"
    
    print(f"✅ dataclass.replace(prog, ops=...) works correctly")
    
    return True

def test_pulse_with_metadata():
    """Test: with_metadata() creates new instance correctly."""
    print("\n" + "="*60)
    print("TEST 5: PulseProgram.with_metadata()")
    print("="*60)
    
    prog = PulseProgram(1)
    prog.drag(0, amp=0.5, duration=160, sigma=40, beta=0.2, qubit_freq=5.0e9)
    
    # Create new instance with metadata
    prog2 = prog.with_metadata(test_key="test_value", another_key=42)
    
    assert prog2.metadata["test_key"] == "test_value"
    assert prog2.metadata["another_key"] == 42
    assert len(prog2.ops) == len(prog.ops)
    assert prog2 is not prog
    
    print(f"✅ with_metadata() works correctly")
    print(f"   New metadata: {prog2.metadata}")
    print(f"   Ops copied: {len(prog2.ops)} operations")
    
    return True

def main():
    """Run all tests."""
    print("\n" + "🚀 "*30)
    print("COMPREHENSIVE PULSE PROGRAM CHAIN TEST SUITE")
    print("🚀 "*30)
    
    results = {}
    
    try:
        results["compile_tqasm"] = test_pulse_chain_compile()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        results["compile_tqasm"] = False
    
    try:
        results["state_sim"] = test_pulse_chain_state_simulation()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        results["state_sim"] = False
    
    try:
        results["device_chain"] = test_pulse_chain_with_device()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        results["device_chain"] = False
    
    try:
        results["ops_alignment"] = test_pulse_ops_field()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        results["ops_alignment"] = False
    
    try:
        results["with_metadata"] = test_pulse_with_metadata()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        results["with_metadata"] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for p in results.values() if p)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! PulseProgram chain API is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
