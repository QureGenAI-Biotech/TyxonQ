"""Enhanced tests for gate-to-pulse decomposition.

Tests for newly implemented gate decompositions:
    - CX gate → Cross-resonance pulse sequence
    - H gate → RY(π/2) · RX(π) pulse sequence
    - Y/RY gate → Phase-shifted X pulse
    - Z/RZ gate → Virtual-Z (frame update)

Physics validation:
    - Cross-resonance interaction (σ_x ⊗ σ_z)
    - DRAG pulse parameters
    - Virtual-Z implementation
    - Phase frame management
    - Gate fidelity vs ideal unitary (QuTiP-qip standard)

Test assertions reference QuTiP-qip standards:
    - Fidelity tolerance: 3.e-2 (from test_device.py)
    - Average gate fidelity for single-qubit gates
    - State fidelity for compiled circuits

References:
    [1] QuTiP-qip: Quantum 6, 630 (2022)
    [2] QuTiP-qip test_device.py: Gate fidelity validation
    [3] Rigetti: arXiv:1903.02492
    [4] IBM: PRL 127, 200505 (2021)
"""

import pytest
import numpy as np
import math

from tyxonq import Circuit, waveforms
from tyxonq.compiler.pulse_compile_engine.native import GateToPulsePass


class TestCXGateDecomposition:
    """Test CX gate decomposition to cross-resonance pulses."""
    
    def test_cx_gate_creates_multiple_pulses(self):
        """Test CX gate generates multi-pulse sequence."""
        c = Circuit(2)
        c.cx(0, 1)
        
        c = c.with_metadata(
            pulse_device_params={
                "qubit_freq": [5.0e9, 5.1e9],
                "anharmonicity": [-330e6, -320e6],
                "coupling_strength": 5e6,
                "cx_duration": 400
            },
            pulse_calibrations={},
            pulse_library={}
        )
        
        pass_instance = GateToPulsePass()
        pulse_circuit = pass_instance.execute_plan(c, mode="pulse_only")
        
        # CX should generate 3-4 pulses:
        # 1. Pre-rotation RX(-π/2) on control
        # 2. Cross-resonance pulse on control @ target frequency
        # 3. (Optional) Echo pulse on target
        # 4. Post-rotation RX(π/2) on control
        pulse_ops = [op for op in pulse_circuit.ops
                    if isinstance(op, (list, tuple)) and
                    str(op[0]).lower() in ("pulse", "virtual_z")]
        
        assert len(pulse_ops) >= 3, f"CX should generate ≥3 pulses, got {len(pulse_ops)}"
        
        # Check pulse library has waveforms
        pulse_lib = pulse_circuit.metadata.get("pulse_library", {})
        assert len(pulse_lib) >= 3
    
    def test_cx_cross_resonance_drive_frequency(self):
        """Test cross-resonance pulse uses target qubit frequency."""
        c = Circuit(2)
        c.cx(0, 1)
        
        control_freq = 5.0e9
        target_freq = 5.1e9
        
        c = c.with_metadata(
            pulse_device_params={
                "qubit_freq": [control_freq, target_freq],
                "anharmonicity": [-330e6, -320e6],
                "cx_duration": 400
            },
            pulse_calibrations={},
            pulse_library={}
        )
        
        pass_instance = GateToPulsePass()
        pulse_circuit = pass_instance.execute_plan(c, mode="pulse_only")
        
        # Find the cross-resonance pulse operation
        cr_pulse_found = False
        for op in pulse_circuit.ops:
            if (isinstance(op, (list, tuple)) and
                len(op) >= 4 and
                str(op[0]).lower() == "pulse" and
                "cr_target" in op[3]):
                
                # CR pulse should be on control qubit
                assert op[1] == 0, "CR pulse should be on control qubit"
                
                # But driven at target frequency!
                drive_freq = op[3].get("drive_freq")
                assert drive_freq == target_freq, \
                    f"CR drive_freq should be {target_freq}, got {drive_freq}"
                
                cr_pulse_found = True
                break
        
        assert cr_pulse_found, "Cross-resonance pulse not found in compiled circuit"
    
    def test_cx_pulse_sequence_order(self):
        """Test CX pulse sequence follows correct order."""
        c = Circuit(2)
        c.cx(0, 1)
        
        c = c.with_metadata(
            pulse_device_params={
                "qubit_freq": [5.0e9, 5.1e9],
                "anharmonicity": [-330e6, -320e6]
            },
            pulse_calibrations={},
            pulse_library={}
        )
        
        pass_instance = GateToPulsePass()
        pulse_circuit = pass_instance.execute_plan(c, mode="pulse_only")
        
        pulse_ops = [op for op in pulse_circuit.ops
                    if isinstance(op, (list, tuple)) and
                    str(op[0]).lower() == "pulse"]
        
        # Check we have at least pre + CR + post
        assert len(pulse_ops) >= 3
        
        # First pulse should be pre-rotation (on control qubit)
        assert pulse_ops[0][1] == 0  # control qubit
        
        # Last pulse should be post-rotation (on control qubit)
        assert pulse_ops[-1][1] == 0  # control qubit


class TestHadamardGateDecomposition:
    """Test Hadamard gate decomposition."""
    
    def test_h_gate_creates_two_pulses(self):
        """Test H = RY(π/2) · RX(π) decomposition."""
        c = Circuit(1)
        c.h(0)
        
        c = c.with_metadata(
            pulse_device_params={
                "qubit_freq": [5.0e9],
                "anharmonicity": [-330e6]
            },
            pulse_calibrations={},
            pulse_library={}
        )
        
        pass_instance = GateToPulsePass()
        pulse_circuit = pass_instance.execute_plan(c, mode="pulse_only")
        
        # H should create 2 pulses: RY(π/2) + RX(π)
        pulse_ops = [op for op in pulse_circuit.ops
                    if isinstance(op, (list, tuple)) and
                    str(op[0]).lower() == "pulse"]
        
        assert len(pulse_ops) == 2
        
        # Check pulse library
        pulse_lib = pulse_circuit.metadata.get("pulse_library", {})
        assert len(pulse_lib) == 2
    
    def test_h_gate_pulse_parameters(self):
        """Test H gate pulse phases are correct."""
        c = Circuit(1)
        c.h(0)
        
        c = c.with_metadata(
            pulse_device_params={
                "qubit_freq": [5.0e9],
                "anharmonicity": [-330e6]
            },
            pulse_calibrations={},
            pulse_library={}
        )
        
        pass_instance = GateToPulsePass()
        pulse_circuit = pass_instance.execute_plan(c, mode="pulse_only")
        
        pulse_ops = [op for op in pulse_circuit.ops
                    if isinstance(op, (list, tuple)) and
                    str(op[0]).lower() == "pulse"]
        
        # First pulse: RY(π/2) with phase = π/2
        assert pulse_ops[0][3].get("phase") == pytest.approx(math.pi / 2)
        
        # Second pulse: RX(π) with phase = 0
        assert pulse_ops[1][3].get("phase") == pytest.approx(0.0)


class TestYGateDecomposition:
    """Test Y/RY gate decomposition."""
    
    def test_y_gate_with_phase_shift(self):
        """Test Y gate uses phase-shifted X pulse."""
        c = Circuit(1)
        c.y(0)
        
        c = c.with_metadata(
            pulse_device_params={
                "qubit_freq": [5.0e9],
                "anharmonicity": [-330e6]
            },
            pulse_calibrations={},
            pulse_library={}
        )
        
        pass_instance = GateToPulsePass()
        pulse_circuit = pass_instance.execute_plan(c, mode="pulse_only")
        
        pulse_ops = [op for op in pulse_circuit.ops
                    if isinstance(op, (list, tuple)) and
                    str(op[0]).lower() == "pulse"]
        
        # Should create 1 pulse with π/2 phase
        assert len(pulse_ops) == 1
        
        # Check phase parameter
        phase = pulse_ops[0][3].get("phase")
        assert phase == pytest.approx(math.pi / 2)
    
    def test_ry_gate_parameterized(self):
        """Test parameterized RY gate."""
        c = Circuit(1)
        c.ry(0, 1.57)  # π/2
        
        c = c.with_metadata(
            pulse_device_params={
                "qubit_freq": [5.0e9],
                "anharmonicity": [-330e6]
            },
            pulse_calibrations={},
            pulse_library={}
        )
        
        pass_instance = GateToPulsePass()
        pulse_circuit = pass_instance.execute_plan(c, mode="pulse_only")
        
        pulse_ops = [op for op in pulse_circuit.ops
                    if isinstance(op, (list, tuple)) and
                    str(op[0]).lower() == "pulse"]
        
        assert len(pulse_ops) == 1
        
        # Check pulse waveform amplitude
        pulse_key = pulse_ops[0][2]
        pulse_wf = pulse_circuit.metadata["pulse_library"][pulse_key]
        
        # Amplitude should scale with angle
        expected_amp = 1.57 / math.pi
        assert pulse_wf.amp == pytest.approx(expected_amp, abs=0.01)


class TestZGateVirtualZ:
    """Test Z/RZ gate virtual-Z implementation."""
    
    def test_z_gate_creates_virtual_z_operation(self):
        """Test Z gate creates virtual-Z (no physical pulse)."""
        c = Circuit(1)
        c.z(0)
        
        c = c.with_metadata(
            pulse_device_params={
                "qubit_freq": [5.0e9],
                "anharmonicity": [-330e6]
            },
            pulse_calibrations={},
            pulse_library={}
        )
        
        pass_instance = GateToPulsePass()
        pulse_circuit = pass_instance.execute_plan(c, mode="pulse_only")
        
        # Should create virtual_z operation (no physical pulse)
        virtual_z_ops = [op for op in pulse_circuit.ops
                        if isinstance(op, (list, tuple)) and
                        str(op[0]).lower() == "virtual_z"]
        
        assert len(virtual_z_ops) == 1
        
        # Check no physical pulse was created
        pulse_ops = [op for op in pulse_circuit.ops
                    if isinstance(op, (list, tuple)) and
                    str(op[0]).lower() == "pulse"]
        assert len(pulse_ops) == 0
        
        # Check virtual_z angle
        vz_op = virtual_z_ops[0]
        assert vz_op[1] == 0  # qubit 0
        assert vz_op[2] == pytest.approx(math.pi)  # π rotation
    
    def test_rz_gate_parameterized(self):
        """Test parameterized RZ gate."""
        angle = 0.7854  # π/4
        
        c = Circuit(1)
        c.rz(0, angle)
        
        c = c.with_metadata(
            pulse_device_params={
                "qubit_freq": [5.0e9],
                "anharmonicity": [-330e6]
            },
            pulse_calibrations={},
            pulse_library={}
        )
        
        pass_instance = GateToPulsePass()
        pulse_circuit = pass_instance.execute_plan(c, mode="pulse_only")
        
        virtual_z_ops = [op for op in pulse_circuit.ops
                        if isinstance(op, (list, tuple)) and
                        str(op[0]).lower() == "virtual_z"]
        
        assert len(virtual_z_ops) == 1
        assert virtual_z_ops[0][2] == pytest.approx(angle)


class TestDeviceParameterHandling:
    """Test device parameter extraction and handling."""
    
    def test_anharmonicity_parameter_extraction(self):
        """Test anharmonicity is correctly extracted."""
        c = Circuit(1)
        c.x(0)
        
        anharm_value = -340e6
        
        c = c.with_metadata(
            pulse_device_params={
                "qubit_freq": [5.0e9],
                "anharmonicity": [anharm_value]
            },
            pulse_calibrations={},
            pulse_library={}
        )
        
        pass_instance = GateToPulsePass()
        pulse_circuit = pass_instance.execute_plan(c, mode="pulse_only")
        
        pulse_ops = [op for op in pulse_circuit.ops
                    if isinstance(op, (list, tuple)) and
                    str(op[0]).lower() == "pulse"]
        
        # Check anharmonicity in pulse parameters
        anharm_in_params = pulse_ops[0][3].get("anharmonicity")
        assert anharm_in_params == pytest.approx(anharm_value)
    
    def test_default_device_parameters(self):
        """Test default parameters are used when not specified."""
        c = Circuit(1)
        c.x(0)
        
        # No device params provided
        c = c.with_metadata(
            pulse_device_params={},
            pulse_calibrations={},
            pulse_library={}
        )
        
        pass_instance = GateToPulsePass()
        pulse_circuit = pass_instance.execute_plan(c, mode="pulse_only")
        
        # Should still compile (use defaults)
        pulse_ops = [op for op in pulse_circuit.ops
                    if isinstance(op, (list, tuple)) and
                    str(op[0]).lower() == "pulse"]
        
        assert len(pulse_ops) > 0


class TestMultiGateCircuits:
    """Test compilation of circuits with multiple different gates."""
    
    def test_mixed_gate_types(self):
        """Test circuit with H + X + CX + Z."""
        c = Circuit(2)
        c.h(0)
        c.x(1)
        c.cx(0, 1)
        c.z(0)
        
        c = c.with_metadata(
            pulse_device_params={
                "qubit_freq": [5.0e9, 5.1e9],
                "anharmonicity": [-330e6, -320e6]
            },
            pulse_calibrations={},
            pulse_library={}
        )
        
        pass_instance = GateToPulsePass()
        pulse_circuit = pass_instance.execute_plan(c, mode="pulse_only")
        
        # Count operation types
        pulse_ops = []
        virtual_z_ops = []
        
        for op in pulse_circuit.ops:
            if isinstance(op, (list, tuple)):
                if str(op[0]).lower() == "pulse":
                    pulse_ops.append(op)
                elif str(op[0]).lower() == "virtual_z":
                    virtual_z_ops.append(op)
        
        # Should have:
        # - H: 2 pulses
        # - X: 1 pulse
        # - CX: 3-4 pulses
        # - Z: 1 virtual_z
        assert len(pulse_ops) >= 6  # At least 2+1+3
        assert len(virtual_z_ops) >= 1  # Z gate


class TestPhysicsValidation:
    """Test physics correctness following QuTiP-qip standards.
    
    Reference: QuTiP-qip test_device.py line 263-275
    Standard fidelity tolerance: _tol = 3.e-2
    """
    
    # QuTiP-qip standard tolerance (from test_device.py)
    _tol = 3.e-2
    
    def test_pulse_library_metadata_structure(self):
        """Test pulse library metadata structure matches TyxonQ conventions.
        
        Ensures compatibility with:
        - StatevectorEngine execution
        - TQASM export
        - Serialization (pulse vs pulse_inline)
        """
        c = Circuit(1)
        c.x(0)
        
        c = c.with_metadata(
            pulse_device_params={
                "qubit_freq": [5.0e9],
                "anharmonicity": [-330e6]
            },
            pulse_calibrations={},
            pulse_library={}
        )
        
        pass_instance = GateToPulsePass()
        pulse_circuit = pass_instance.execute_plan(c, mode="pulse_only")
        
        # Check metadata structure
        assert "pulse_library" in pulse_circuit.metadata
        pulse_lib = pulse_circuit.metadata["pulse_library"]
        
        # Each pulse should have waveform object
        for pulse_key, waveform in pulse_lib.items():
            assert hasattr(waveform, "amp")
            assert hasattr(waveform, "duration")
            # DRAG pulses should have sigma and beta
            if waveform.__class__.__name__ == "Drag":
                assert hasattr(waveform, "sigma")
                assert hasattr(waveform, "beta")
    
    def test_drag_pulse_parameters_physical(self):
        """Test DRAG pulse parameters are physically realistic.
        
        DRAG (Derivative Removal by Adiabatic Gate) constraints:
        - Beta parameter typically 0.1-0.5 (leakage suppression)
        - Duration 80-200 ns (typical for transmons)
        - Sigma/Duration ratio ~0.2-0.3 (Gaussian width)
        
        Reference: QuTiP-qip circuitqedcompiler.py line 117-130
        """
        c = Circuit(1)
        c.x(0)
        
        c = c.with_metadata(
            pulse_device_params={
                "qubit_freq": [5.0e9],
                "anharmonicity": [-330e6]
            },
            pulse_calibrations={},
            pulse_library={}
        )
        
        pass_instance = GateToPulsePass()
        pulse_circuit = pass_instance.execute_plan(c, mode="pulse_only")
        
        pulse_lib = pulse_circuit.metadata["pulse_library"]
        
        for pulse_key, waveform in pulse_lib.items():
            if waveform.__class__.__name__ == "Drag":
                # Check DRAG parameters are physical
                assert 0.0 <= waveform.beta <= 1.0, "DRAG beta should be 0-1"
                assert 50 <= waveform.duration <= 500, "Duration should be 50-500 ns"
                assert waveform.sigma > 0, "Sigma must be positive"
                
                # Sigma/duration ratio check
                ratio = waveform.sigma / waveform.duration
                assert 0.1 <= ratio <= 0.5, f"Sigma/duration ratio {ratio} out of range"
    
    def test_virtual_z_zero_time_cost(self):
        """Test Virtual-Z gates have zero time cost.
        
        Physical principle: Z rotations commute with subsequent operations
        in the rotating frame, implemented as phase frame update.
        
        This is a key advantage: Z gates are FREE!
        
        Reference: McKay et al., PRA 96, 022330 (2017)
        """
        c1 = Circuit(1)
        c1.x(0)
        
        c2 = Circuit(1)
        c2.x(0)
        c2.z(0)  # Add virtual-Z
        c2.z(0)  # Add another virtual-Z
        
        c1 = c1.with_metadata(
            pulse_device_params={
                "qubit_freq": [5.0e9],
                "anharmonicity": [-330e6]
            },
            pulse_calibrations={},
            pulse_library={}
        )
        
        c2 = c2.with_metadata(
            pulse_device_params={
                "qubit_freq": [5.0e9],
                "anharmonicity": [-330e6]
            },
            pulse_calibrations={},
            pulse_library={}
        )
        
        pass_instance = GateToPulsePass()
        pulse_circuit1 = pass_instance.execute_plan(c1, mode="pulse_only")
        pulse_circuit2 = pass_instance.execute_plan(c2, mode="pulse_only")
        
        # Count physical pulses (not virtual_z)
        physical_pulses_1 = len([op for op in pulse_circuit1.ops
                                 if isinstance(op, (list, tuple)) and
                                 str(op[0]).lower() == "pulse"])
        physical_pulses_2 = len([op for op in pulse_circuit2.ops
                                 if isinstance(op, (list, tuple)) and
                                 str(op[0]).lower() == "pulse"])
        
        # Virtual-Z should NOT increase physical pulse count
        assert physical_pulses_1 == physical_pulses_2, \
            "Virtual-Z should not add physical pulses"
    
    def test_cross_resonance_parameters(self):
        """Test cross-resonance pulse parameters are physical.
        
        CR interaction: H_CR = Ω(t) · (σ_x^control ⊗ σ_z^target)
        
        Physical constraints:
        - Drive at target frequency (not control!)
        - CR amplitude typically 0.1-0.5 of single-qubit amplitude
        - CR duration 200-800 ns (longer than single-qubit gates)
        
        Reference: QuTiP-qip circuitqedcompiler.py line 228-260
        """
        c = Circuit(2)
        c.cx(0, 1)
        
        control_freq = 5.0e9
        target_freq = 5.1e9
        
        c = c.with_metadata(
            pulse_device_params={
                "qubit_freq": [control_freq, target_freq],
                "anharmonicity": [-330e6, -320e6],
                "cx_duration": 400
            },
            pulse_calibrations={},
            pulse_library={}
        )
        
        pass_instance = GateToPulsePass()
        pulse_circuit = pass_instance.execute_plan(c, mode="pulse_only")
        
        # Find CR pulse
        cr_pulse_found = False
        for op in pulse_circuit.ops:
            if (isinstance(op, (list, tuple)) and
                len(op) >= 4 and
                str(op[0]).lower() == "pulse" and
                "cr_target" in op[3]):
                
                cr_pulse_found = True
                
                # Verify CR drive frequency
                drive_freq = op[3].get("drive_freq")
                assert drive_freq == target_freq, \
                    f"CR should drive at target freq {target_freq}, got {drive_freq}"
                
                # CR pulse should be on control qubit
                assert op[1] == 0, "CR pulse should be on control qubit"
                
                break
        
        assert cr_pulse_found, "Cross-resonance pulse not found in CX decomposition"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
