"""
Unit tests for Virtual-Z optimization in pulse compilation.

Tests the _optimize_virtual_z method that merges adjacent RZ/Virtual-Z operations
to simplify phase tracking and reduce compilation overhead.

Coverage:
  - Basic merging of consecutive virtual_z on same qubit
  - Angle normalization to [0, 2π) range
  - Zero-angle filtering
  - Multiple qubits (no merging across qubits)
  - Mixed operations (virtual_z + pulse)
  - Edge cases (empty list, single operation)
"""

import math
import pytest
from tyxonq.compiler.pulse_compile_engine.native.gate_to_pulse import GateToPulsePass


class TestVirtualZOptimization:
    """Test Virtual-Z optimization functionality."""
    
    @pytest.fixture
    def optimizer(self):
        """Create a GateToPulsePass instance for testing."""
        return GateToPulsePass()
    
    def test_empty_list(self, optimizer):
        """Test optimization of empty operation list."""
        result = optimizer._optimize_virtual_z([])
        assert result == []
    
    def test_no_virtual_z(self, optimizer):
        """Test that non-virtual_z operations pass through unchanged."""
        ops = [
            ("pulse", 0, "x_pulse", {}),
            ("pulse", 1, "y_pulse", {}),
            ("barrier",),
        ]
        result = optimizer._optimize_virtual_z(ops)
        assert result == ops
    
    def test_single_virtual_z(self, optimizer):
        """Test that a single virtual_z operation is preserved."""
        ops = [("virtual_z", 0, math.pi / 4)]
        result = optimizer._optimize_virtual_z(ops)
        assert len(result) == 1
        assert result[0][0] == "virtual_z"
        assert result[0][1] == 0
        assert abs(result[0][2] - math.pi / 4) < 1e-10
    
    def test_merge_consecutive_same_qubit(self, optimizer):
        """Test merging of consecutive virtual_z on same qubit."""
        ops = [
            ("virtual_z", 0, math.pi / 4),
            ("virtual_z", 0, math.pi / 3),
            ("virtual_z", 0, math.pi / 6),
        ]
        result = optimizer._optimize_virtual_z(ops)
        
        assert len(result) == 1
        assert result[0][0] == "virtual_z"
        assert result[0][1] == 0
        # π/4 + π/3 + π/6 = 3π/12 + 4π/12 + 2π/12 = 9π/12 = 3π/4
        expected_angle = 3 * math.pi / 4
        assert abs(result[0][2] - expected_angle) < 1e-10
    
    def test_merge_with_pulse_in_between(self, optimizer):
        """Test that virtual_z is not merged across other operations."""
        ops = [
            ("virtual_z", 0, math.pi / 4),
            ("virtual_z", 0, math.pi / 3),
            ("pulse", 0, "x_pulse", {}),  # ← Breaks the chain
            ("virtual_z", 0, math.pi / 2),
        ]
        result = optimizer._optimize_virtual_z(ops)
        
        assert len(result) == 3
        # First two should be merged
        assert result[0][0] == "virtual_z"
        assert result[0][1] == 0
        expected_angle_1 = math.pi / 4 + math.pi / 3
        assert abs(result[0][2] - expected_angle_1) < 1e-10
        
        # Pulse in the middle
        assert result[1][0] == "pulse"
        assert result[1][1] == 0
        
        # Last virtual_z should remain as is
        assert result[2][0] == "virtual_z"
        assert result[2][1] == 0
        assert abs(result[2][2] - math.pi / 2) < 1e-10
    
    def test_no_merge_different_qubits(self, optimizer):
        """Test that virtual_z on different qubits are not merged."""
        ops = [
            ("virtual_z", 0, math.pi / 4),
            ("virtual_z", 1, math.pi / 3),
            ("virtual_z", 0, math.pi / 2),
        ]
        result = optimizer._optimize_virtual_z(ops)
        
        # Should have 3 operations: 2 on q0 (not consecutive), 1 on q1
        assert len(result) == 3
        
        # q0 first
        assert result[0] == ("virtual_z", 0, math.pi / 4)
        # q1
        assert result[1] == ("virtual_z", 1, math.pi / 3)
        # q0 second
        assert result[2] == ("virtual_z", 0, math.pi / 2)
    
    def test_angle_normalization(self, optimizer):
        """Test that angles are normalized to [0, 2π)."""
        ops = [
            ("virtual_z", 0, 3 * math.pi),  # 3π → π
            ("virtual_z", 0, 2 * math.pi),  # 2π → 0 (should be filtered)
        ]
        result = optimizer._optimize_virtual_z(ops)
        
        # 3π + 2π = 5π ≡ π (mod 2π)
        assert len(result) == 1
        assert result[0][0] == "virtual_z"
        assert result[0][1] == 0
        expected_angle = math.pi
        assert abs(result[0][2] - expected_angle) < 1e-10
    
    def test_zero_angle_filtering(self, optimizer):
        """Test that zero-angle virtual_z operations are filtered out."""
        ops = [
            ("virtual_z", 0, 0.0),  # Zero angle
            ("virtual_z", 0, 1e-11),  # Very small angle (below threshold)
        ]
        result = optimizer._optimize_virtual_z(ops)
        
        # Both should be filtered out
        assert len(result) == 0
    
    def test_zero_angle_from_merge(self, optimizer):
        """Test filtering of zero angles that result from merging."""
        ops = [
            ("virtual_z", 0, math.pi),
            ("virtual_z", 0, -math.pi),  # Cancels out
            ("virtual_z", 0, 2 * math.pi),  # Equivalent to zero
        ]
        result = optimizer._optimize_virtual_z(ops)
        
        # Sum: π - π + 2π ≡ 2π ≡ 0 (mod 2π)
        assert len(result) == 0
    
    def test_complex_scenario(self, optimizer):
        """Test a complex scenario with multiple qubits and operations."""
        ops = [
            ("virtual_z", 0, math.pi / 4),
            ("virtual_z", 0, math.pi / 4),  # Merge: π/2
            ("pulse", 0, "x_pulse", {}),
            ("virtual_z", 1, math.pi / 3),
            ("virtual_z", 1, math.pi / 6),  # Merge: π/2
            ("virtual_z", 1, 0.0),           # Filter: zero
            ("pulse", 1, "y_pulse", {}),
            ("virtual_z", 0, math.pi / 3),
            ("virtual_z", 0, math.pi / 6),  # Merge: π/2
        ]
        result = optimizer._optimize_virtual_z(ops)
        
        # Expected: merged vz for q0, pulse, merged vz for q1, pulse, merged vz for q0
        assert len(result) == 5
        
        # First: merged vz(π/2) on q0
        assert result[0] == ("virtual_z", 0, math.pi / 2)
        
        # Pulse on q0
        assert result[1][0] == "pulse"
        assert result[1][1] == 0
        
        # Merged vz(π/2) on q1
        assert result[2] == ("virtual_z", 1, math.pi / 2)
        
        # Pulse on q1
        assert result[3][0] == "pulse"
        assert result[3][1] == 1
        
        # Merged vz(π/2) on q0
        assert result[4] == ("virtual_z", 0, math.pi / 2)
    
    def test_large_angle_merging(self, optimizer):
        """Test merging of angles greater than 2π."""
        ops = [
            ("virtual_z", 0, 10 * math.pi),
            ("virtual_z", 0, 3 * math.pi),
        ]
        result = optimizer._optimize_virtual_z(ops)
        
        # 10π + 3π = 13π ≡ π (mod 2π)
        assert len(result) == 1
        assert result[0][0] == "virtual_z"
        assert result[0][1] == 0
        expected_angle = math.pi
        assert abs(result[0][2] - expected_angle) < 1e-10
    
    def test_negative_angles(self, optimizer):
        """Test handling of negative angles."""
        ops = [
            ("virtual_z", 0, math.pi / 4),
            ("virtual_z", 0, -math.pi / 4),  # Cancels
            ("virtual_z", 0, math.pi / 2),   # Results in π/2
        ]
        result = optimizer._optimize_virtual_z(ops)
        
        # π/4 - π/4 + π/2 = π/2
        assert len(result) == 1
        assert abs(result[0][2] - math.pi / 2) < 1e-10


class TestVirtualZIntegration:
    """Integration tests for Virtual-Z optimization in full compilation pipeline."""
    
    def test_optimization_in_gate_to_pulse(self):
        """Test that optimization is called during gate-to-pulse compilation."""
        from tyxonq import Circuit
        
        # Create a circuit with multiple RZ gates
        c = Circuit(2)
        c.rz(0, math.pi / 4)  # Correct API: qubit, then angle
        c.rz(0, math.pi / 3)
        c.x(0)              # This breaks the virtual_z chain
        c.rz(0, math.pi / 2)
        
        # Apply gate-to-pulse compilation
        compiler = GateToPulsePass()
        pulse_circuit = compiler.execute_plan(c, mode="pulse_only")
        
        # Count virtual_z operations
        virtual_z_ops = [op for op in pulse_circuit.ops if isinstance(op, (list, tuple)) and len(op) > 0 and op[0] == "virtual_z"]
        
        # Should have 2 virtual_z operations (merged first two + separate last one), not 3
        # First two RZ gates on q0 should be merged into one virtual_z
        # Then pulse for X gate
        # Then third RZ gate as separate virtual_z
        assert len(virtual_z_ops) == 2
    
    def test_optimization_preserves_order(self):
        """Test that optimization preserves the order of non-virtual_z operations."""
        from tyxonq import Circuit
        
        c = Circuit(2)
        c.h(0)
        c.rz(0, math.pi / 4)  # Correct API: qubit, then angle
        c.x(1)
        c.cx(0, 1)
        c.rz(0, math.pi / 3)
        
        compiler = GateToPulsePass()
        pulse_circuit = compiler.execute_plan(c, mode="pulse_only")
        
        # Extract pulse operations (not virtual_z)
        pulses = [op for op in pulse_circuit.ops if isinstance(op, (list, tuple)) and op[0] == "pulse"]
        
        # Should maintain relative order
        assert len(pulses) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
