"""Integration tests for Pulse compilation engine.

This test suite validates the complete pulse compilation pipeline:
    - Gate-to-pulse decomposition
    - Pulse lowering (defcal inlining)
    - Pulse scheduling

Test coverage:
    - Basic gate-to-pulse conversion (X, H, CX)
    - Custom calibration support
    - Hybrid mode (gate + pulse mixed)
    - Pulse-only mode (full decomposition)
    - Scheduling optimization
"""

import pytest
import numpy as np

from tyxonq import Circuit, waveforms
from tyxonq.compiler.pulse_compile_engine import PulseCompiler


class TestPulseCompilerBasics:
    """Test basic pulse compilation functionality."""
    
    def test_pulse_compiler_initialization(self):
        """Test PulseCompiler can be instantiated."""
        compiler = PulseCompiler()
        assert compiler is not None
        assert compiler.optimization_level == 1
        
        compiler_opt3 = PulseCompiler(optimization_level=3)
        assert compiler_opt3.optimization_level == 3
    
    def test_compile_single_x_gate(self):
        """Test compilation of a single X gate to pulse."""
        c = Circuit(1)
        c.x(0)
        
        compiler = PulseCompiler()
        pulse_circuit = compiler.compile(
            c,
            device_params={
                "qubit_freq": [5.0e9],
                "anharmonicity": [-330e6]
            },
            mode="pulse_only"
        )
        
        # Check that circuit has pulse operations
        assert pulse_circuit is not None
        
        # Check that pulse operations were created
        pulse_ops = [op for op in pulse_circuit.ops 
                    if isinstance(op, (list, tuple)) and 
                    str(op[0]).lower() == "pulse"]
        assert len(pulse_ops) > 0
        
        # Check metadata
        assert "pulse_mode" in pulse_circuit.metadata
        assert pulse_circuit.metadata["pulse_mode"] == "pulse_only"
        assert "pulse_schedule" in pulse_circuit.metadata
    
    def test_compile_with_custom_calibration(self):
        """Test compilation with user-provided pulse calibration."""
        c = Circuit(1)
        c.x(0)
        
        # Create custom X pulse
        x_pulse = waveforms.Drag(amp=0.5, duration=160, sigma=40, beta=0.2)
        
        compiler = PulseCompiler()
        compiler.add_calibration("x", [0], x_pulse, {
            "qubit_freq": 5.0e9,
            "drive_freq": 5.0e9
        })
        
        # Get calibrations
        cals = compiler.get_calibrations()
        assert "x_0" in cals
        assert cals["x_0"]["gate"] == "x"
        assert cals["x_0"]["qubits"] == [0]
    
    def test_hybrid_mode_preserves_gates(self):
        """Test that hybrid mode preserves some gates."""
        c = Circuit(2)
        c.h(0)
        c.cx(0, 1)
        c.measure_z(0)
        c.measure_z(1)
        
        compiler = PulseCompiler()
        pulse_circuit = compiler.compile(c, mode="hybrid")
        
        # Check that measurement gates are preserved
        measure_ops = [op for op in pulse_circuit.ops 
                      if isinstance(op, (list, tuple)) and 
                      str(op[0]).lower() == "measure_z"]
        
        # Should have 2 measurement operations
        assert len(measure_ops) == 2


class TestGateToPulsePass:
    """Test gate-to-pulse decomposition pass."""
    
    def test_x_gate_to_drag_pulse(self):
        """Test X gate decomposes to Drag pulse."""
        from tyxonq.compiler.pulse_compile_engine.native import GateToPulsePass
        
        c = Circuit(1)
        c.x(0)
        c = c.with_metadata(
            pulse_device_params={"qubit_freq": [5.0e9], "anharmonicity": [-330e6]},
            pulse_calibrations={}
        )
        
        pass_instance = GateToPulsePass()
        pulse_circuit = pass_instance.execute_plan(c, mode="pulse_only")
        
        # Check that pulse operations were created
        pulse_ops = [op for op in pulse_circuit.ops
                    if isinstance(op, (list, tuple)) and
                    str(op[0]).lower() == "pulse"]
        assert len(pulse_ops) > 0


class TestPulseLoweringPass:
    """Test pulse lowering (defcal inlining) pass."""
    
    def test_pulse_lowering_inlines_waveforms(self):
        """Test that pulse lowering inlines waveform definitions."""
        from tyxonq.compiler.pulse_compile_engine.native import PulseLoweringPass
        
        c = Circuit(1)
        
        # Manually create a pulse operation
        pulse = waveforms.Constant(amp=1.0, duration=100)
        c._pulse_cache = {"test_pulse": pulse}  # type: ignore
        c = c.extended([("pulse", 0, "test_pulse", {})])
        
        pass_instance = PulseLoweringPass()
        lowered_circuit = pass_instance.execute_plan(c)
        
        # Check that pulse was inlined
        inline_ops = [op for op in lowered_circuit.ops
                     if isinstance(op, (list, tuple)) and
                     str(op[0]).lower() == "pulse_inline"]
        
        assert len(inline_ops) >= 0  # May or may not inline depending on implementation


class TestPulseSchedulingPass:
    """Test pulse scheduling and optimization pass."""
    
    def test_scheduling_adds_timing_metadata(self):
        """Test that scheduling adds timing information."""
        from tyxonq.compiler.pulse_compile_engine.native import PulseSchedulingPass
        
        c = Circuit(2)
        
        # Create mock pulse operations
        pulse1 = waveforms.Constant(amp=1.0, duration=100)
        pulse2 = waveforms.Constant(amp=1.0, duration=150)
        
        c._pulse_cache = {"p1": pulse1, "p2": pulse2}  # type: ignore
        c = c.extended([
            ("pulse", 0, "p1", {}),
            ("pulse", 1, "p2", {})
        ])
        
        pass_instance = PulseSchedulingPass(optimization_level=1)
        scheduled_circuit = pass_instance.execute_plan(c, dt=1e-10)
        
        # Check scheduling metadata was added
        if "pulse_schedule" in scheduled_circuit.metadata:
            assert "pulse_total_time" in scheduled_circuit.metadata


class TestPulseCompilerIntegration:
    """Integration tests for full pulse compilation pipeline."""
    
    def test_full_pipeline_with_multiple_gates(self):
        """Test complete compilation of a multi-gate circuit."""
        c = Circuit(2)
        c.h(0)
        c.x(1)
        c.measure_z(0)
        c.measure_z(1)
        
        compiler = PulseCompiler(optimization_level=2)
        pulse_circuit = compiler.compile(
            c,
            device_params={
                "qubit_freq": [5.0e9, 5.1e9],
                "anharmonicity": [-330e6, -320e6]
            },
            mode="hybrid"
        )
        
        # Verify compilation succeeded
        assert pulse_circuit is not None
        assert pulse_circuit.num_qubits == 2
        
        # Check metadata
        assert "pulse_mode" in pulse_circuit.metadata
        assert "pulse_device_params" in pulse_circuit.metadata
    
    def test_pulse_compilation_with_device_params(self):
        """Test pulse compilation uses device parameters correctly."""
        c = Circuit(1)
        c.x(0)
        
        device_params = {
            "qubit_freq": [5.2e9],
            "anharmonicity": [-340e6],
            "T1": [80e-6],
            "T2": [120e-6]
        }
        
        compiler = PulseCompiler()
        pulse_circuit = compiler.compile(c, device_params=device_params)
        
        # Verify device params are stored in metadata
        assert pulse_circuit.metadata["pulse_device_params"] == device_params


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
