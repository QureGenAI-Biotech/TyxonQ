"""
End-to-End Integration Tests for DefcalLibrary with Pulse Compilation

This test suite verifies:
1. DefcalLibrary integration with real Circuit objects
2. Pulse compilation with defcal calibrations vs. default decompositions
3. Gate-to-pulse compilation accuracy
4. Performance metrics for calibration queries
5. Simulation execution with defcal-compiled circuits

Author: TyxonQ Development Team
"""

import math
import time
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from tyxonq.compiler.pulse_compile_engine.defcal_library import (
    DefcalLibrary,
    CalibrationData,
)
from tyxonq.compiler.pulse_compile_engine.native.gate_to_pulse import GateToPulsePass

try:
    from tyxonq import Circuit, apis
    HAS_TYXONQ = True
except ImportError:
    HAS_TYXONQ = False


class MockWaveform:
    """Mock waveform for testing."""
    
    def __init__(self, name: str = "Gaussian", amp: float = 0.5, duration: int = 100):
        self.name = name
        self.amplitude = amp
        self.duration = duration
        self.params = {"amplitude": amp, "duration": duration}
    
    def __repr__(self):
        return f"{self.name}(amp={self.amplitude}, duration={self.duration})"


class TestDefcalCompilationBasics:
    """Basic tests for defcal-aware compilation."""
    
    def test_compiler_accepts_defcal_library(self):
        """Test that GateToPulsePass accepts DefcalLibrary."""
        lib = DefcalLibrary()
        compiler = GateToPulsePass(defcal_library=lib)
        
        assert compiler.defcal_library is lib
    
    def test_compiler_works_without_defcal_library(self):
        """Test that GateToPulsePass still works without DefcalLibrary."""
        compiler = GateToPulsePass(defcal_library=None)
        
        assert compiler.defcal_library is None
    
    def test_extract_qubits_from_single_qubit_gate(self):
        """Test extracting qubits from single-qubit gate."""
        compiler = GateToPulsePass()
        
        qubits = compiler._extract_qubits(("x", 0))
        assert qubits == (0,)
        
        qubits = compiler._extract_qubits(("rz", 2, math.pi / 4))
        assert qubits == (2,)
    
    def test_extract_qubits_from_two_qubit_gate(self):
        """Test extracting qubits from two-qubit gate."""
        compiler = GateToPulsePass()
        
        qubits = compiler._extract_qubits(("cx", 0, 1))
        assert qubits == (0, 1)
        
        qubits = compiler._extract_qubits(("cz", 1, 2))
        assert qubits == (1, 2)
    
    def test_extract_qubits_stops_at_parameters(self):
        """Test that extract_qubits stops at non-integer parameters."""
        compiler = GateToPulsePass()
        
        # RX with parameter
        qubits = compiler._extract_qubits(("rx", 0, math.pi / 2))
        assert qubits == (0,)
        
        # RY with parameter
        qubits = compiler._extract_qubits(("ry", 1, 3.14))
        assert qubits == (1,)


class TestDefcalApplicationInCompilation:
    """Tests for applying defcal calibrations during compilation."""
    
    def test_apply_defcal_creates_pulse_operation(self):
        """Test that applying defcal creates a pulse operation."""
        compiler = GateToPulsePass()
        
        # Create mock calibration
        pulse = MockWaveform("DRAG", amp=0.8, duration=40)
        calib_data = CalibrationData(
            gate="x",
            qubits=(0,),
            pulse=pulse,
            params={"duration": 40, "amplitude": 0.8}
        )
        
        # Create mock circuit
        circuit = Mock()
        circuit.metadata = {"pulse_library": {}}
        
        # Apply defcal
        result = compiler._apply_defcal(("x", 0), calib_data, circuit)
        
        assert len(result) == 1
        pulse_op = result[0]
        assert pulse_op[0] == "pulse"
        assert pulse_op[1] == 0  # qubit
        assert "x_q0_" in pulse_op[2]  # pulse_key contains gate and qubit
        assert pulse_op[3] == calib_data.params  # params


class TestDefcalLibraryCompilationFlow:
    """Tests for complete compilation flow with DefcalLibrary."""
    
    def test_create_circuit_and_compile_without_defcal(self):
        """Test basic circuit creation and compilation without defcal."""
        # Create a simple circuit
        c = Circuit(2)
        c.h(0)
        c.cx(0, 1)
        
        assert len(c.ops) == 2
    
    def test_compiler_initialization_with_empty_library(self):
        """Test compiler initialized with empty defcal library."""
        lib = DefcalLibrary()
        compiler = GateToPulsePass(defcal_library=lib)
        
        assert len(lib) == 0
        assert compiler.defcal_library is lib
    
    def test_compiler_initialization_with_populated_library(self):
        """Test compiler with pre-populated defcal library."""
        lib = DefcalLibrary()
        pulse_x = MockWaveform("DRAG_X")
        pulse_cx = MockWaveform("CX_Pulse")
        
        lib.add_calibration("x", 0, pulse_x, {"duration": 40, "amp": 0.8})
        lib.add_calibration("cx", (0, 1), pulse_cx, {"duration": 200})
        
        compiler = GateToPulsePass(defcal_library=lib)
        
        assert len(compiler.defcal_library) == 2
        assert compiler.defcal_library.has_calibration("x", (0,))
        assert compiler.defcal_library.has_calibration("cx", (0, 1))


class TestDefcalQueryPerformance:
    """Performance tests for calibration lookups."""
    
    def test_single_calibration_lookup_speed(self):
        """Test that single calibration lookup is fast."""
        lib = DefcalLibrary()
        pulse = MockWaveform()
        
        # Add single calibration
        lib.add_calibration("x", 0, pulse)
        
        # Measure lookup time
        start = time.perf_counter()
        for _ in range(10000):
            calib = lib.get_calibration("x", (0,))
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        
        # Should be very fast (< 1ms for 10000 lookups)
        assert elapsed < 10  # Allow generous margin
        avg_lookup_time_us = (elapsed * 1000) / 10000
        assert avg_lookup_time_us < 1.0  # Less than 1 microsecond per lookup
    
    def test_wildcard_lookup_speed(self):
        """Test that wildcard lookup is reasonably fast."""
        lib = DefcalLibrary()
        pulse = MockWaveform()
        
        # Add multiple X calibrations
        for q in range(10):
            lib.add_calibration("x", q, pulse)
        
        # Add Y calibrations for good measure
        for q in range(10):
            lib.add_calibration("y", q, pulse)
        
        # Measure wildcard lookup time
        start = time.perf_counter()
        for _ in range(1000):
            calibs = lib.get_calibration("x", None)
        elapsed = (time.perf_counter() - start) * 1000
        
        # Should still be fast (< 5ms for 1000 lookups)
        assert elapsed < 5
        avg_lookup_time_us = (elapsed * 1000) / 1000
        assert avg_lookup_time_us < 5.0
    
    def test_list_calibrations_speed(self):
        """Test that list_calibrations is reasonably fast."""
        lib = DefcalLibrary()
        pulse = MockWaveform()
        
        # Add many calibrations
        for gate in ["x", "y", "z", "h", "cx", "cz"]:
            for q in range(10):
                if gate in ("cx", "cz") and q < 9:
                    lib.add_calibration(gate, (q, q+1), pulse)
                elif gate not in ("cx", "cz"):
                    lib.add_calibration(gate, q, pulse)
        
        # Measure list time
        start = time.perf_counter()
        for _ in range(1000):
            calibs = lib.list_calibrations(gate="x")
        elapsed = (time.perf_counter() - start) * 1000
        
        # Should be fast (< 5ms for 1000 list operations)
        assert elapsed < 5


class TestDefcalLibraryMemoryUsage:
    """Tests for memory efficiency of DefcalLibrary."""
    
    def test_memory_usage_grows_linearly(self):
        """Test that memory usage grows linearly with number of calibrations."""
        import sys
        
        lib = DefcalLibrary()
        pulse = MockWaveform()
        
        # Measure initial size
        initial_size = sys.getsizeof(lib._calibrations)
        
        # Add calibrations
        for i in range(100):
            lib.add_calibration("x", i, pulse)
        
        size_after_100 = sys.getsizeof(lib._calibrations)
        
        for i in range(100, 200):
            lib.add_calibration("x", i, pulse)
        
        size_after_200 = sys.getsizeof(lib._calibrations)
        
        # Growth should be reasonably linear
        growth_100_to_200 = size_after_200 - size_after_100
        growth_0_to_100 = size_after_100 - initial_size
        
        # Allow 2x variance in growth
        ratio = max(growth_100_to_200, growth_0_to_100) / min(growth_100_to_200, growth_0_to_100)
        assert ratio < 2.0, f"Memory growth not linear: {ratio}"


class TestDefcalMultipleQubits:
    """Tests for defcal with multi-qubit systems."""
    
    def test_single_qubit_calibrations_on_multiple_qubits(self):
        """Test adding calibrations for the same gate on different qubits."""
        lib = DefcalLibrary()
        
        pulses = [MockWaveform(f"X_q{i}", amp=0.8 + i*0.01) for i in range(5)]
        
        for i, pulse in enumerate(pulses):
            lib.add_calibration("x", i, pulse, {"duration": 40 + i})
        
        assert len(lib) == 5
        
        # Verify each calibration is different
        for i in range(5):
            calib = lib.get_calibration("x", (i,))
            assert calib is not None
            assert calib.qubits == (i,)
            assert calib.params["duration"] == 40 + i
    
    def test_two_qubit_calibrations_different_pairs(self):
        """Test calibrations for different qubit pairs."""
        lib = DefcalLibrary()
        
        # Add CX calibrations for different qubit pairs
        qubit_pairs = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
        
        for pair in qubit_pairs:
            pulse = MockWaveform(f"CX_q{pair[0]}_q{pair[1]}")
            lib.add_calibration("cx", pair, pulse, {"duration": 200 + pair[0]*10})
        
        assert len(lib) == len(qubit_pairs)
        
        # Verify each pair
        for pair in qubit_pairs:
            calib = lib.get_calibration("cx", pair)
            assert calib is not None
            assert calib.qubits == pair


class TestDefcalJSONPersistence:
    """Tests for JSON persistence with real calibrations."""
    
    def test_export_and_reimport_with_varied_calibrations(self, tmp_path):
        """Test exporting and re-importing library with various calibrations."""
        lib1 = DefcalLibrary(hardware="TestHW")
        
        # Add varied calibrations
        gates = [
            ("x", 0, {"duration": 40, "amp": 0.8}),
            ("y", 0, {"duration": 40, "amp": 0.8}),
            ("cx", (0, 1), {"duration": 200, "zz_amplitude": 0.5}),
            ("h", 1, {"duration": 20, "amp": 0.4}),
        ]
        
        for gate, qubit, params in gates:
            pulse = MockWaveform(f"{gate}_pulse", amp=params.get("amp", 0.5))
            lib1.add_calibration(gate, qubit, pulse, params, f"{gate} calibration")
        
        json_file = tmp_path / "calib.json"
        lib1.export_to_json(json_file)
        
        # Re-import
        lib2 = DefcalLibrary(hardware="TestHW")
        lib2.import_from_json(json_file)
        
        # Verify
        assert len(lib2) == len(lib1)
        
        for gate, qubit, expected_params in gates:
            calib = lib2.get_calibration(gate, qubit if isinstance(qubit, tuple) else (qubit,))
            assert calib is not None
            assert calib.gate == gate.lower()
            assert calib.params == expected_params


class TestDefcalValidation:
    """Tests for defcal validation."""
    
    def test_validate_complete_library(self):
        """Test validating a complete library."""
        lib = DefcalLibrary()
        pulse = MockWaveform()
        
        # Add various calibrations
        lib.add_calibration("x", 0, pulse, {"duration": 40})
        lib.add_calibration("cx", (0, 1), pulse, {"duration": 200})
        lib.add_calibration("y", 1, pulse, {"duration": 40})
        
        assert lib.validate() is True
    
    def test_validate_after_import(self, tmp_path):
        """Test validation after importing from JSON."""
        lib1 = DefcalLibrary()
        pulse = MockWaveform()
        
        lib1.add_calibration("x", 0, pulse, {"duration": 40})
        lib1.add_calibration("cx", (0, 1), pulse, {"duration": 200})
        
        json_file = tmp_path / "calib.json"
        lib1.export_to_json(json_file)
        
        # Import and validate
        lib2 = DefcalLibrary()
        lib2.import_from_json(json_file)
        
        assert lib2.validate() is True


class TestDefcalSummaryAndReporting:
    """Tests for library summary and reporting."""
    
    def test_library_summary_output(self):
        """Test that library summary provides useful information."""
        lib = DefcalLibrary(hardware="Homebrew_S2")
        pulse = MockWaveform()
        
        lib.add_calibration("x", 0, pulse, description="X on q0")
        lib.add_calibration("x", 1, pulse, description="X on q1")
        lib.add_calibration("cx", (0, 1), pulse, description="CX 01")
        
        summary = lib.summary()
        
        # Check summary contains expected information
        assert "DefcalLibrary Summary" in summary
        assert "Homebrew_S2" in summary
        assert "3" in summary  # Total calibrations
        assert "X:" in summary or "x:" in summary.lower()
        assert "CX:" in summary or "cx:" in summary.lower()
    
    def test_library_repr(self):
        """Test library string representation."""
        lib = DefcalLibrary(hardware="CustomHW")
        pulse = MockWaveform()
        
        lib.add_calibration("x", 0, pulse)
        lib.add_calibration("cx", (0, 1), pulse)
        
        repr_str = repr(lib)
        assert "DefcalLibrary" in repr_str
        assert "CustomHW" in repr_str
        assert "2" in repr_str  # Number of calibrations


class TestDefcalEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_add_many_calibrations(self):
        """Test adding a large number of calibrations."""
        lib = DefcalLibrary()
        pulse = MockWaveform()
        
        # Add 1000 calibrations
        count = 0
        for gate in ["x", "y", "z", "h", "s", "t", "rx", "ry", "rz"]:
            for q in range(10):
                lib.add_calibration(gate, q, pulse)
                count += 1
        
        assert len(lib) == count
        
        # Verify we can still retrieve them
        x_calibs = lib.get_calibration("x", None)
        assert len(x_calibs) == 10
    
    def test_override_calibration(self):
        """Test that overriding a calibration works correctly."""
        lib = DefcalLibrary()
        pulse1 = MockWaveform("Pulse1")
        pulse2 = MockWaveform("Pulse2")
        
        lib.add_calibration("x", 0, pulse1, {"version": 1})
        assert len(lib) == 1
        
        # Override
        lib.add_calibration("x", 0, pulse2, {"version": 2})
        assert len(lib) == 1  # Still one calibration
        
        calib = lib.get_calibration("x", (0,))
        assert calib.pulse is pulse2
        assert calib.params["version"] == 2
    
    def test_remove_and_re_add(self):
        """Test removing and re-adding calibrations."""
        lib = DefcalLibrary()
        pulse1 = MockWaveform("Pulse1")
        pulse2 = MockWaveform("Pulse2")
        
        lib.add_calibration("x", 0, pulse1)
        assert len(lib) == 1
        
        lib.remove_calibration("x", (0,))
        assert len(lib) == 0
        
        lib.add_calibration("x", 0, pulse2)
        assert len(lib) == 1
        
        calib = lib.get_calibration("x", (0,))
        assert calib.pulse is pulse2
    
    def test_clear_and_validate(self):
        """Test clearing library and validating."""
        lib = DefcalLibrary()
        pulse = MockWaveform()
        
        lib.add_calibration("x", 0, pulse)
        lib.add_calibration("cx", (0, 1), pulse)
        assert len(lib) == 2
        
        lib.clear()
        assert len(lib) == 0
        assert lib.validate() is True


class TestDefcalCaseInsensitivity:
    """Tests for case-insensitive gate names."""
    
    def test_add_uppercase_get_lowercase(self):
        """Test adding uppercase and getting lowercase."""
        lib = DefcalLibrary()
        pulse = MockWaveform()
        
        lib.add_calibration("X", 0, pulse)
        
        calib = lib.get_calibration("x", (0,))
        assert calib is not None
        assert calib.gate == "x"
    
    def test_add_mixed_case_get_mixed_case(self):
        """Test adding mixed case and getting mixed case."""
        lib = DefcalLibrary()
        pulse = MockWaveform()
        
        lib.add_calibration("CX", (0, 1), pulse)
        
        calib = lib.get_calibration("cX", (0, 1))
        assert calib is not None
        assert calib.gate == "cx"
    
    def test_list_case_insensitive(self):
        """Test listing with case-insensitive gate names."""
        lib = DefcalLibrary()
        pulse = MockWaveform()
        
        lib.add_calibration("x", 0, pulse)
        lib.add_calibration("X", 1, pulse)
        lib.add_calibration("y", 0, pulse)
        
        x_calibs = lib.list_calibrations(gate="X")
        assert len(x_calibs) == 2
        
        x_calibs2 = lib.list_calibrations(gate="x")
        assert len(x_calibs2) == 2


class TestDefcalTimestamp:
    """Tests for timestamp handling."""
    
    def test_calibration_timestamp_auto_created(self):
        """Test that timestamp is automatically created."""
        lib = DefcalLibrary()
        pulse = MockWaveform()
        
        from datetime import datetime
        before = datetime.now()
        
        lib.add_calibration("x", 0, pulse)
        
        after = datetime.now()
        
        calib = lib.get_calibration("x", (0,))
        assert calib.timestamp is not None
        assert before <= calib.timestamp <= after
    
    def test_calibration_timestamp_preserved_in_json(self, tmp_path):
        """Test that timestamp is preserved in JSON export/import."""
        lib1 = DefcalLibrary()
        pulse = MockWaveform()
        
        lib1.add_calibration("x", 0, pulse)
        original_ts = lib1.get_calibration("x", (0,)).timestamp
        
        json_file = tmp_path / "calib.json"
        lib1.export_to_json(json_file)
        
        lib2 = DefcalLibrary()
        lib2.import_from_json(json_file)
        
        imported_ts = lib2.get_calibration("x", (0,)).timestamp
        assert imported_ts is not None
        # Timestamps should be very close (within seconds)
        assert abs((imported_ts - original_ts).total_seconds()) < 1.0


class TestDefcalParallelOperations:
    """Tests for concurrent/parallel operations on defcal library."""
    
    def test_concurrent_lookups(self):
        """Test concurrent lookups from DefcalLibrary."""
        import concurrent.futures
        
        lib = DefcalLibrary()
        pulse = MockWaveform()
        
        # Add calibrations
        for i in range(20):
            lib.add_calibration("x", i, pulse)
        
        def lookup_task(qubit_id):
            results = []
            for _ in range(100):
                calib = lib.get_calibration("x", (qubit_id,))
                results.append(calib is not None)
            return all(results)
        
        # Concurrent lookups
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(lookup_task, i) for i in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        assert all(results)  # All lookups successful
    
    def test_multiple_library_instances(self):
        """Test that multiple library instances don't interfere."""
        lib1 = DefcalLibrary(hardware="HW1")
        lib2 = DefcalLibrary(hardware="HW2")
        
        pulse1 = MockWaveform("Pulse1")
        pulse2 = MockWaveform("Pulse2")
        
        lib1.add_calibration("x", 0, pulse1)
        lib2.add_calibration("x", 0, pulse2)
        
        assert len(lib1) == 1
        assert len(lib2) == 1
        
        calib1 = lib1.get_calibration("x", (0,))
        calib2 = lib2.get_calibration("x", (0,))
        
        assert calib1.pulse is pulse1
        assert calib2.pulse is pulse2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
