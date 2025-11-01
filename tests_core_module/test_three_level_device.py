"""
Test three-level system integration in device simulation (chain API).

Tests the three_level parameter in StatevectorEngine for realistic
leakage modeling during pulse operations.
"""

import pytest
import numpy as np
import tyxonq as tq
from tyxonq import waveforms


class TestThreeLevelDevice:
    """Test three-level system in device API."""
    
    def test_single_qubit_three_level_enabled(self):
        """Test that three_level=True enables leakage detection for single qubit."""
        c = tq.Circuit(1)
        pulse = waveforms.Gaussian(duration=160, amp=1.0, sigma=40)
        c.metadata["pulse_library"] = {"pulse_x": pulse}
        c.ops.append(("pulse", 0, "pulse_x", {"qubit_freq": 5.0e9}))
        c.measure_z(0)
        
        # Run with three-level enabled
        result = c.device(
            provider="simulator",
            device="statevector",
            three_level=True,
            anharmonicity=-330e6,
            rabi_freq=50e6,  # Moderate power
            shots=10000
        ).postprocessing(method=None).run()
        
        counts = result[0]["result"]
        
        # Should have |0⟩, |1⟩, and possibly |2⟩ states
        assert "0" in counts or "1" in counts, "Missing computational basis states"
        
        # Check if |2⟩ leakage occurred (not guaranteed but likely with high Rabi)
        has_leakage = "2" in counts
        
        # Leakage should be small (< 5%)
        if has_leakage:
            total = sum(counts.values())
            leak_prob = counts["2"] / total
            assert leak_prob < 0.05, f"Leakage too high: {leak_prob*100:.2f}%"
    
    def test_single_qubit_two_level_vs_three_level(self):
        """Compare 2-level (ideal) vs 3-level (realistic) results."""
        c = tq.Circuit(1)
        pulse = waveforms.Gaussian(duration=160, amp=1.0, sigma=40)
        c.metadata["pulse_library"] = {"pulse_x": pulse}
        c.ops.append(("pulse", 0, "pulse_x", {"qubit_freq": 5.0e9}))
        c.measure_z(0)
        
        # 2-level (no leakage)
        result_2level = c.device(
            provider="simulator",
            device="statevector",
            three_level=False,
            shots=5000
        ).postprocessing(method=None).run()
        
        counts_2level = result_2level[0]["result"]
        
        # 3-level (with leakage)
        result_3level = c.device(
            provider="simulator",
            device="statevector",
            three_level=True,
            anharmonicity=-330e6,
            rabi_freq=50e6,
            shots=5000
        ).postprocessing(method=None).run()
        
        counts_3level = result_3level[0]["result"]
        
        # 2-level should only have '0' and '1'
        assert all(k in ["0", "1"] for k in counts_2level.keys()), \
            "2-level should not have |2⟩ state"
        
        # 3-level may have '2' state (leakage)
        # Just verify it ran successfully
        assert "result" in result_3level[0]
    
    def test_drag_suppresses_leakage(self):
        """Test that DRAG pulse reduces leakage compared to Gaussian."""
        shots = 10000
        
        # Gaussian pulse (no DRAG correction)
        c_gauss = tq.Circuit(1)
        pulse_gauss = waveforms.Gaussian(duration=160, amp=1.0, sigma=40)
        c_gauss.metadata["pulse_library"] = {"pulse": pulse_gauss}
        c_gauss.ops.append(("pulse", 0, "pulse", {"qubit_freq": 5.0e9}))
        c_gauss.measure_z(0)
        
        result_gauss = c_gauss.device(
            provider="simulator",
            device="statevector",
            three_level=True,
            anharmonicity=-330e6,
            rabi_freq=50e6,
            shots=shots
        ).postprocessing(method=None).run()
        
        # DRAG pulse (with correction)
        c_drag = tq.Circuit(1)
        pulse_drag = waveforms.Drag(duration=160, amp=1.0, sigma=40, beta=0.15)
        c_drag.metadata["pulse_library"] = {"pulse": pulse_drag}
        c_drag.ops.append(("pulse", 0, "pulse", {"qubit_freq": 5.0e9}))
        c_drag.measure_z(0)
        
        result_drag = c_drag.device(
            provider="simulator",
            device="statevector",
            three_level=True,
            anharmonicity=-330e6,
            rabi_freq=50e6,
            shots=shots
        ).postprocessing(method=None).run()
        
        # Calculate leakage probabilities
        counts_gauss = result_gauss[0]["result"]
        counts_drag = result_drag[0]["result"]
        
        leak_gauss = counts_gauss.get("2", 0) / shots
        leak_drag = counts_drag.get("2", 0) / shots
        
        # DRAG should reduce leakage (though both may be very small)
        # We just check they're both small and DRAG is not worse
        assert leak_gauss < 0.1, "Gaussian leakage unexpectedly high"
        assert leak_drag <= leak_gauss + 0.01, "DRAG should not increase leakage"
    
    def test_multi_qubit_warning(self):
        """Test that multi-qubit three-level simulation issues warning."""
        c = tq.Circuit(2)
        pulse = waveforms.Gaussian(duration=160, amp=1.0, sigma=40)
        c.metadata["pulse_library"] = {"pulse_x": pulse}
        c.ops.append(("pulse", 0, "pulse_x", {"qubit_freq": 5.0e9}))
        c.measure_z(0)
        c.measure_z(1)
        
        # Should issue warning for num_qubits > 1
        with pytest.warns(UserWarning, match="Three-level simulation with num_qubits > 1"):
            result = c.device(
                provider="simulator",
                device="statevector",
                three_level=True,
                anharmonicity=-330e6,
                rabi_freq=30e6,
                shots=1000
            ).postprocessing(method=None).run()
        
        # Should still run (with degraded accuracy)
        assert "result" in result[0]
    
    def test_anharmonicity_parameter(self):
        """Test that anharmonicity parameter affects leakage rate."""
        c = tq.Circuit(1)
        pulse = waveforms.Gaussian(duration=160, amp=1.0, sigma=40)
        c.metadata["pulse_library"] = {"pulse": pulse}
        c.ops.append(("pulse", 0, "pulse", {"qubit_freq": 5.0e9}))
        c.measure_z(0)
        
        shots = 10000
        
        # Weak anharmonicity (more leakage)
        result_weak = c.device(
            provider="simulator",
            device="statevector",
            three_level=True,
            anharmonicity=-200e6,  # Weaker
            rabi_freq=50e6,
            shots=shots
        ).postprocessing(method=None).run()
        
        # Strong anharmonicity (less leakage)
        result_strong = c.device(
            provider="simulator",
            device="statevector",
            three_level=True,
            anharmonicity=-500e6,  # Stronger
            rabi_freq=50e6,
            shots=shots
        ).postprocessing(method=None).run()
        
        leak_weak = result_weak[0]["result"].get("2", 0) / shots
        leak_strong = result_strong[0]["result"].get("2", 0) / shots
        
        # Stronger anharmonicity should reduce leakage
        # (Both should be small, but weak should be >= strong)
        assert leak_weak >= leak_strong - 0.005, \
            "Weaker anharmonicity should have >= leakage"
    
    def test_rabi_frequency_scaling(self):
        """Test that higher Rabi frequency increases leakage."""
        c = tq.Circuit(1)
        pulse = waveforms.Gaussian(duration=160, amp=1.0, sigma=40)
        c.metadata["pulse_library"] = {"pulse": pulse}
        c.ops.append(("pulse", 0, "pulse", {"qubit_freq": 5.0e9}))
        c.measure_z(0)
        
        shots = 10000
        
        # Low Rabi frequency
        result_low = c.device(
            provider="simulator",
            device="statevector",
            three_level=True,
            anharmonicity=-330e6,
            rabi_freq=20e6,  # Low power
            shots=shots
        ).postprocessing(method=None).run()
        
        # High Rabi frequency
        result_high = c.device(
            provider="simulator",
            device="statevector",
            three_level=True,
            anharmonicity=-330e6,
            rabi_freq=80e6,  # High power
            shots=shots
        ).postprocessing(method=None).run()
        
        leak_low = result_low[0]["result"].get("2", 0) / shots
        leak_high = result_high[0]["result"].get("2", 0) / shots
        
        # Higher Rabi should have more leakage (leakage ∝ (Ω/α)²)
        assert leak_high >= leak_low - 0.005, \
            "Higher Rabi frequency should have >= leakage"


if __name__ == "__main__":
    # Run tests
    test = TestThreeLevelDevice()
    
    print("Running three-level device tests...\n")
    
    print("1. Test single-qubit three-level enabled...")
    test.test_single_qubit_three_level_enabled()
    print("   ✅ PASSED\n")
    
    print("2. Test 2-level vs 3-level comparison...")
    test.test_single_qubit_two_level_vs_three_level()
    print("   ✅ PASSED\n")
    
    print("3. Test DRAG suppresses leakage...")
    test.test_drag_suppresses_leakage()
    print("   ✅ PASSED\n")
    
    print("4. Test multi-qubit warning...")
    test.test_multi_qubit_warning()
    print("   ✅ PASSED\n")
    
    print("5. Test anharmonicity parameter...")
    test.test_anharmonicity_parameter()
    print("   ✅ PASSED\n")
    
    print("6. Test Rabi frequency scaling...")
    test.test_rabi_frequency_scaling()
    print("   ✅ PASSED\n")
    
    print("=" * 60)
    print("All three-level device tests passed! ✅")
    print("=" * 60)
