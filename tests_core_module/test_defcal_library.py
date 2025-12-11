"""
Unit tests for DefcalLibrary and CalibrationData classes.

This test suite verifies:
1. Basic calibration operations (add, get, remove)
2. Query interfaces (exact, wildcard, list)
3. JSON serialization/deserialization
4. Data validation
5. Integration with gate_to_pulse compiler

Author: TyxonQ Development Team
"""

import json
import math
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from tyxonq.compiler.pulse_compile_engine.defcal_library import (
    DefcalLibrary,
    CalibrationData,
)


class MockPulse:
    """Mock pulse waveform for testing."""
    
    def __init__(self, name: str = "Gaussian"):
        self.name = name
        self.amplitude = 0.5
        self.duration = 100
    
    def __repr__(self):
        return f"MockPulse({self.name})"


class TestCalibrationData:
    """Tests for CalibrationData data class."""
    
    def test_create_single_qubit_calibration(self):
        """Test creating a single-qubit calibration."""
        pulse = MockPulse("DRAG")
        params = {"duration": 40, "amplitude": 0.8, "beta": 0.2}
        
        calib = CalibrationData(
            gate="x",
            qubits=(0,),
            pulse=pulse,
            params=params,
            description="X gate on q0"
        )
        
        assert calib.gate == "x"
        assert calib.qubits == (0,)
        assert calib.pulse is pulse
        assert calib.params == params
        assert calib.description == "X gate on q0"
        assert calib.is_single_qubit()
        assert not calib.is_two_qubit()
        assert calib.get_qubit_count() == 1
    
    def test_create_two_qubit_calibration(self):
        """Test creating a two-qubit calibration."""
        pulse = MockPulse("CXPulse")
        params = {"duration": 200, "amplitude": 0.9}
        
        calib = CalibrationData(
            gate="cx",
            qubits=(0, 1),
            pulse=pulse,
            params=params,
        )
        
        assert calib.gate == "cx"
        assert calib.qubits == (0, 1)
        assert calib.is_two_qubit()
        assert not calib.is_single_qubit()
        assert calib.get_qubit_count() == 2
    
    def test_normalize_qubits_int(self):
        """Test that single qubit int is normalized to tuple."""
        pulse = MockPulse()
        calib = CalibrationData(gate="x", qubits=0, pulse=pulse)
        
        assert calib.qubits == (0,)
    
    def test_normalize_qubits_list(self):
        """Test that qubit list is normalized to tuple."""
        pulse = MockPulse()
        calib = CalibrationData(gate="cx", qubits=[0, 1], pulse=pulse)
        
        assert calib.qubits == (0, 1)
    
    def test_timestamp_auto_creation(self):
        """Test that timestamp is auto-created if not provided."""
        pulse = MockPulse()
        calib = CalibrationData(gate="x", qubits=0, pulse=pulse)
        
        assert calib.timestamp is not None
        assert isinstance(calib.timestamp, datetime)
    
    def test_to_dict_serialization(self):
        """Test conversion to dictionary for JSON serialization."""
        pulse = MockPulse("DRAG")
        params = {"duration": 40, "amplitude": 0.8}
        
        calib = CalibrationData(
            gate="x",
            qubits=(0,),
            pulse=pulse,
            params=params,
            description="Test calibration"
        )
        
        data = calib.to_dict()
        
        assert data["gate"] == "x"
        assert data["qubits"] == [0]
        assert data["pulse_type"] == "MockPulse"
        assert data["params"] == params
        assert data["description"] == "Test calibration"
        assert "timestamp" in data
    
    def test_from_dict_deserialization(self):
        """Test creation from dictionary."""
        data = {
            "gate": "x",
            "qubits": [0],
            "pulse_type": "DRAG",
            "params": {"duration": 40, "amplitude": 0.8},
            "description": "Test",
            "timestamp": datetime.now().isoformat(),
            "hardware": "Homebrew_S2"
        }
        
        calib = CalibrationData.from_dict(data)
        
        assert calib.gate == "x"
        assert calib.qubits == (0,)
        assert calib.params == data["params"]
        assert calib.description == "Test"
    
    def test_calibration_equality(self):
        """Test calibration equality based on gate and qubits."""
        pulse1 = MockPulse()
        pulse2 = MockPulse()
        
        calib1 = CalibrationData(gate="x", qubits=(0,), pulse=pulse1)
        calib2 = CalibrationData(gate="x", qubits=(0,), pulse=pulse2)
        calib3 = CalibrationData(gate="y", qubits=(0,), pulse=pulse1)
        
        assert calib1 == calib2  # Same gate and qubits
        assert calib1 != calib3  # Different gate
    
    def test_calibration_hashable(self):
        """Test that calibrations can be hashed."""
        pulse = MockPulse()
        calib = CalibrationData(gate="x", qubits=(0,), pulse=pulse)
        
        # Should be hashable
        calibs = {calib}
        assert len(calibs) == 1


class TestDefcalLibraryBasics:
    """Tests for basic DefcalLibrary operations."""
    
    def test_create_library(self):
        """Test creating a DefcalLibrary."""
        lib = DefcalLibrary(hardware="Homebrew_S2")
        
        assert lib.hardware == "Homebrew_S2"
        assert len(lib) == 0
    
    def test_add_single_qubit_calibration(self):
        """Test adding a single-qubit calibration."""
        lib = DefcalLibrary()
        pulse = MockPulse()
        params = {"duration": 40, "amplitude": 0.8}
        
        lib.add_calibration("x", 0, pulse, params, "X gate on q0")
        
        assert len(lib) == 1
        calib = lib.get_calibration("x", (0,))
        assert calib is not None
        assert calib.gate == "x"
        assert calib.qubits == (0,)
    
    def test_add_two_qubit_calibration(self):
        """Test adding a two-qubit calibration."""
        lib = DefcalLibrary()
        pulse = MockPulse()
        
        lib.add_calibration("cx", (0, 1), pulse)
        
        assert len(lib) == 1
        calib = lib.get_calibration("cx", (0, 1))
        assert calib is not None
        assert calib.qubits == (0, 1)
    
    def test_add_calibration_invalid_gate(self):
        """Test that adding calibration with invalid gate raises error."""
        lib = DefcalLibrary()
        pulse = MockPulse()
        
        with pytest.raises(ValueError):
            lib.add_calibration("", 0, pulse)
        
        with pytest.raises(ValueError):
            lib.add_calibration(None, 0, pulse)
    
    def test_add_calibration_invalid_qubits(self):
        """Test that adding calibration with invalid qubits raises error."""
        lib = DefcalLibrary()
        pulse = MockPulse()
        
        with pytest.raises(ValueError):
            lib.add_calibration("x", None, pulse)
        
        with pytest.raises(ValueError):
            lib.add_calibration("x", (-1,), pulse)
    
    def test_add_calibration_no_pulse(self):
        """Test that adding calibration without pulse raises error."""
        lib = DefcalLibrary()
        
        with pytest.raises(ValueError):
            lib.add_calibration("x", 0, None)
    
    def test_get_calibration_exact_match(self):
        """Test retrieving exact calibration match."""
        lib = DefcalLibrary()
        pulse = MockPulse()
        
        lib.add_calibration("x", 0, pulse)
        
        calib = lib.get_calibration("x", (0,))
        assert calib is not None
        assert calib.gate == "x"
        assert calib.qubits == (0,)
    
    def test_get_calibration_not_found(self):
        """Test getting non-existent calibration."""
        lib = DefcalLibrary()
        
        calib = lib.get_calibration("x", (0,))
        assert calib is None
    
    def test_get_calibration_wildcard(self):
        """Test retrieving all calibrations for a gate."""
        lib = DefcalLibrary()
        pulse1 = MockPulse()
        pulse2 = MockPulse()
        
        lib.add_calibration("x", 0, pulse1)
        lib.add_calibration("x", 1, pulse2)
        lib.add_calibration("y", 0, pulse1)
        
        # Get all X calibrations
        x_calibs = lib.get_calibration("x", None)
        assert x_calibs is not None
        assert len(x_calibs) == 2
        
        # Get all Y calibrations
        y_calibs = lib.get_calibration("y", None)
        assert y_calibs is not None
        assert len(y_calibs) == 1
    
    def test_get_calibration_case_insensitive(self):
        """Test that gate names are case-insensitive."""
        lib = DefcalLibrary()
        pulse = MockPulse()
        
        lib.add_calibration("X", 0, pulse)
        
        # Should find calibration with lowercase query
        calib = lib.get_calibration("x", (0,))
        assert calib is not None
    
    def test_remove_calibration(self):
        """Test removing a calibration."""
        lib = DefcalLibrary()
        pulse = MockPulse()
        
        lib.add_calibration("x", 0, pulse)
        assert len(lib) == 1
        
        removed = lib.remove_calibration("x", (0,))
        assert removed is True
        assert len(lib) == 0
        
        # Try removing again
        removed = lib.remove_calibration("x", (0,))
        assert removed is False
    
    def test_has_calibration(self):
        """Test checking if calibration exists."""
        lib = DefcalLibrary()
        pulse = MockPulse()
        
        lib.add_calibration("x", 0, pulse)
        
        assert lib.has_calibration("x", (0,))
        assert not lib.has_calibration("y", (0,))
        assert not lib.has_calibration("x", (1,))
    
    def test_list_calibrations_all(self):
        """Test listing all calibrations."""
        lib = DefcalLibrary()
        pulse1 = MockPulse()
        pulse2 = MockPulse()
        
        lib.add_calibration("x", 0, pulse1)
        lib.add_calibration("x", 1, pulse1)
        lib.add_calibration("y", 0, pulse2)
        
        all_calibs = lib.list_calibrations()
        assert len(all_calibs) == 3
    
    def test_list_calibrations_by_gate(self):
        """Test listing calibrations filtered by gate."""
        lib = DefcalLibrary()
        pulse1 = MockPulse()
        pulse2 = MockPulse()
        
        lib.add_calibration("x", 0, pulse1)
        lib.add_calibration("x", 1, pulse1)
        lib.add_calibration("y", 0, pulse2)
        
        x_calibs = lib.list_calibrations(gate="x")
        assert len(x_calibs) == 2
        assert all(c.gate == "x" for c in x_calibs)
    
    def test_list_calibrations_by_qubit(self):
        """Test listing calibrations filtered by qubit."""
        lib = DefcalLibrary()
        pulse1 = MockPulse()
        pulse2 = MockPulse()
        
        lib.add_calibration("x", 0, pulse1)
        lib.add_calibration("y", 0, pulse1)
        lib.add_calibration("x", 1, pulse2)
        
        q0_calibs = lib.list_calibrations(qubit=0)
        assert len(q0_calibs) == 2
        assert all(0 in c.qubits for c in q0_calibs)


class TestDefcalLibraryAdvanced:
    """Tests for advanced DefcalLibrary features."""
    
    def test_calibration_override(self):
        """Test that adding calibration with same gate/qubits overrides."""
        lib = DefcalLibrary()
        pulse1 = MockPulse("DRAG")
        pulse2 = MockPulse("Gaussian")
        
        lib.add_calibration("x", 0, pulse1)
        assert lib.get_calibration("x", (0,)).pulse is pulse1
        
        lib.add_calibration("x", 0, pulse2)
        assert len(lib) == 1  # Still just one calibration
        assert lib.get_calibration("x", (0,)).pulse is pulse2
    
    def test_validate_library(self):
        """Test validating library."""
        lib = DefcalLibrary()
        pulse = MockPulse()
        
        lib.add_calibration("x", 0, pulse)
        assert lib.validate() is True
    
    def test_validate_empty_library(self):
        """Test validating empty library."""
        lib = DefcalLibrary()
        assert lib.validate() is True
    
    def test_clear_library(self):
        """Test clearing all calibrations."""
        lib = DefcalLibrary()
        pulse1 = MockPulse()
        pulse2 = MockPulse()
        
        lib.add_calibration("x", 0, pulse1)
        lib.add_calibration("y", 0, pulse2)
        assert len(lib) == 2
        
        lib.clear()
        assert len(lib) == 0
    
    def test_library_repr(self):
        """Test library string representation."""
        lib = DefcalLibrary(hardware="CustomHW")
        pulse = MockPulse()
        
        lib.add_calibration("x", 0, pulse)
        
        repr_str = repr(lib)
        assert "CustomHW" in repr_str
        assert "1" in repr_str  # Number of calibrations
    
    def test_library_summary(self):
        """Test library summary output."""
        lib = DefcalLibrary()
        pulse1 = MockPulse()
        pulse2 = MockPulse()
        
        lib.add_calibration("x", 0, pulse1, description="X on q0")
        lib.add_calibration("cx", (0, 1), pulse2)
        
        summary = lib.summary()
        assert "DefcalLibrary Summary" in summary
        assert "X:" in summary
        assert "CX:" in summary


class TestDefcalLibraryJsonSerialization:
    """Tests for JSON export/import functionality."""
    
    def test_export_to_json(self, tmp_path):
        """Test exporting library to JSON file."""
        lib = DefcalLibrary(hardware="Homebrew_S2")
        pulse = MockPulse()
        
        lib.add_calibration("x", 0, pulse, {"duration": 40}, "X on q0")
        lib.add_calibration("cx", (0, 1), pulse)
        
        json_file = tmp_path / "calibrations.json"
        lib.export_to_json(json_file)
        
        assert json_file.exists()
        
        # Verify file contents
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        assert data["version"] == "1.0"
        assert data["hardware"] == "Homebrew_S2"
        assert data["calibration_count"] == 2
        assert len(data["calibrations"]) == 2
    
    def test_import_from_json(self, tmp_path):
        """Test importing library from JSON file."""
        # Create and export library
        lib1 = DefcalLibrary(hardware="Homebrew_S2")
        pulse = MockPulse()
        
        lib1.add_calibration("x", 0, pulse, {"duration": 40}, "X on q0")
        lib1.add_calibration("y", 0, pulse, {"duration": 40}, "Y on q0")
        
        json_file = tmp_path / "calibrations.json"
        lib1.export_to_json(json_file)
        
        # Import into new library
        lib2 = DefcalLibrary(hardware="Homebrew_S2")
        lib2.import_from_json(json_file)
        
        # Verify imported data
        assert len(lib2) == 2
        
        x_calib = lib2.get_calibration("x", (0,))
        assert x_calib is not None
        assert x_calib.description == "X on q0"
        assert x_calib.params == {"duration": 40}
        
        y_calib = lib2.get_calibration("y", (0,))
        assert y_calib is not None
    
    def test_round_trip_json(self, tmp_path):
        """Test that library survives export/import round trip."""
        lib1 = DefcalLibrary(hardware="Homebrew_S2")
        pulse = MockPulse()
        
        lib1.add_calibration("x", 0, pulse, {"duration": 40, "amp": 0.8})
        lib1.add_calibration("cx", (0, 1), pulse, {"duration": 200})
        lib1.add_calibration("z", 2, pulse, {"phase": math.pi / 4})
        
        json_file = tmp_path / "round_trip.json"
        lib1.export_to_json(json_file)
        
        lib2 = DefcalLibrary(hardware="Homebrew_S2")
        lib2.import_from_json(json_file)
        
        # Compare
        assert len(lib2) == len(lib1)
        
        for gate, qubits in [("x", (0,)), ("cx", (0, 1)), ("z", (2,))]:
            calib1 = lib1.get_calibration(gate, qubits)
            calib2 = lib2.get_calibration(gate, qubits)
            
            assert calib1 is not None and calib2 is not None
            assert calib1.gate == calib2.gate
            assert calib1.qubits == calib2.qubits
            assert calib1.params == calib2.params
    
    def test_import_nonexistent_file(self):
        """Test importing from non-existent file."""
        lib = DefcalLibrary()
        
        with pytest.raises(FileNotFoundError):
            lib.import_from_json("/nonexistent/path/calibrations.json")
    
    def test_import_invalid_json(self, tmp_path):
        """Test importing invalid JSON file."""
        lib = DefcalLibrary()
        json_file = tmp_path / "invalid.json"
        
        with open(json_file, 'w') as f:
            f.write("{ invalid json")
        
        with pytest.raises(json.JSONDecodeError):
            lib.import_from_json(json_file)
    
    def test_import_hardware_mismatch_warning(self, tmp_path):
        """Test warning when importing from different hardware."""
        lib1 = DefcalLibrary(hardware="Hardware_A")
        pulse = MockPulse()
        lib1.add_calibration("x", 0, pulse)
        
        json_file = tmp_path / "calib_a.json"
        lib1.export_to_json(json_file)
        
        # Import into library with different hardware
        lib2 = DefcalLibrary(hardware="Hardware_B")
        
        # Should not raise, but log warning
        with patch('tyxonq.compiler.pulse_compile_engine.defcal_library.logger') as mock_logger:
            lib2.import_from_json(json_file)
            # Verify warning was logged
            assert mock_logger.warning.called


class TestGateToPulseIntegration:
    """Integration tests with gate_to_pulse compiler."""
    
    def test_gate_to_pulse_with_defcal_library(self):
        """Test that GateToPulsePass uses DefcalLibrary when provided."""
        from tyxonq.compiler.pulse_compile_engine.native.gate_to_pulse import GateToPulsePass
        
        # Create library with calibration
        lib = DefcalLibrary()
        pulse = MockPulse("DRAG")
        lib.add_calibration("x", 0, pulse, {"duration": 40, "amplitude": 0.8})
        
        # Create compiler with defcal library
        compiler = GateToPulsePass(defcal_library=lib)
        
        assert compiler.defcal_library is lib
    
    def test_gate_to_pulse_without_defcal_library(self):
        """Test that GateToPulsePass works without DefcalLibrary."""
        from tyxonq.compiler.pulse_compile_engine.native.gate_to_pulse import GateToPulsePass
        
        compiler = GateToPulsePass()
        
        assert compiler.defcal_library is None
    
    def test_extract_qubits(self):
        """Test qubit extraction from gate operation."""
        from tyxonq.compiler.pulse_compile_engine.native.gate_to_pulse import GateToPulsePass
        
        compiler = GateToPulsePass()
        
        # Single qubit
        qubits = compiler._extract_qubits(("x", 0))
        assert qubits == (0,)
        
        # Two qubit
        qubits = compiler._extract_qubits(("cx", 0, 1))
        assert qubits == (0, 1)
        
        # With parameter
        qubits = compiler._extract_qubits(("rx", 0, math.pi / 2))
        assert qubits == (0,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
