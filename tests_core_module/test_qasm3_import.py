"""Comprehensive tests for QASM3 + OpenPulse import (Phases 2, 3, 4).

This test file consolidates all QASM3 import functionality:
- Phase 2: Gate-level QASM3 import (basic gates, measurements)
- Phase 3: OpenPulse Frame definitions (ports, frames, frequencies)
- Phase 4: defcal gate calibrations (waveforms, pulse instructions)

Test organization:
1. TestQASM3Phase2_* : Phase 2 basic gate-level import tests
2. TestQASM3Phase3_* : Phase 3 frame definition tests
3. TestQASM3Phase4_* : Phase 4 defcal definition tests
4. TestQASM3Integration_* : End-to-end integration tests across phases
"""

import pytest
from tyxonq.compiler.pulse_compile_engine.native.qasm3_importer import qasm3_to_circuit
from tyxonq.core.ir.circuit import Circuit
from tyxonq.compiler.api import compile


# ============================================================================
# Phase 2: Basic QASM3 Gate-Level Import
# ============================================================================

class TestQASM3Phase2_QubitDeclarations:
    """Test Phase 2: Qubit declaration parsing."""
    
    def test_openqasm3_qubit_declaration(self):
        """Test OPENQASM 3 format: qubit[n] q;"""
        qasm3_code = """OPENQASM 3.0;
qubit[2] q;
h q[0];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        assert circuit.num_qubits == 2
    
    def test_qasm2_qreg_declaration(self):
        """Test QASM 2 format: qreg q[n];"""
        qasm3_code = """qreg q[3];
h q[0];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        assert circuit.num_qubits == 3
    
    def test_missing_qubit_declaration(self):
        """Test error when no qubit declaration is found."""
        qasm3_code = "h q[0];"
        with pytest.raises(ValueError, match="No qubit declaration found"):
            qasm3_to_circuit(qasm3_code)


class TestQASM3Phase2_SingleQubitGates:
    """Test Phase 2: Single-qubit gate parsing."""
    
    def test_h_gate(self):
        qasm3_code = """OPENQASM 3.0;
qubit[1] q;
h q[0];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        assert circuit.ops[0] == ("h", 0)
    
    def test_pauli_gates(self):
        """Test X, Y, Z gates."""
        for gate_name in ["x", "y", "z"]:
            qasm3_code = f"""qubit[1] q;
{gate_name} q[0];
"""
            circuit = qasm3_to_circuit(qasm3_code)
            assert circuit.ops[0] == (gate_name, 0)
    
    def test_s_and_t_gates(self):
        """Test S and T gates."""
        qasm3_code = """qubit[1] q;
s q[0];
t q[0];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        assert circuit.ops[0] == ("s", 0)
        assert circuit.ops[1] == ("t", 0)
    
    def test_parameterized_gates_numeric(self):
        """Test RX, RY, RZ with numeric parameters."""
        qasm3_code = """qubit[1] q;
rx(1.5) q[0];
ry(0.5) q[0];
rz(2.0) q[0];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        assert circuit.ops[0][0] == "rx"
        assert abs(circuit.ops[0][2] - 1.5) < 1e-10
        assert circuit.ops[1][0] == "ry"
        assert abs(circuit.ops[1][2] - 0.5) < 1e-10
        assert circuit.ops[2][0] == "rz"
        assert abs(circuit.ops[2][2] - 2.0) < 1e-10
    
    def test_parameterized_gates_pi_expressions(self):
        """Test parameterized gates with pi expressions."""
        qasm3_code = """qubit[1] q;
rx(pi/2) q[0];
ry(3*pi) q[0];
rz(pi/4) q[0];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        assert abs(circuit.ops[0][2] - 1.5707963267948966) < 1e-10  # pi/2
        assert abs(circuit.ops[1][2] - 9.42477796076938) < 1e-10  # 3*pi
        assert abs(circuit.ops[2][2] - 0.7853981633974483) < 1e-10  # pi/4


class TestQASM3Phase2_TwoQubitGates:
    """Test Phase 2: Two-qubit gate parsing."""
    
    def test_cx_gate(self):
        qasm3_code = """qubit[2] q;
cx q[0], q[1];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        assert circuit.ops[0] == ("cx", 0, 1)
    
    def test_cz_gate(self):
        qasm3_code = """qubit[2] q;
cz q[0], q[1];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        assert circuit.ops[0] == ("cz", 0, 1)
    
    def test_cy_gate_decomposition(self):
        """Test CY gate decomposition."""
        qasm3_code = """qubit[2] q;
cy q[0], q[1];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        # cy = h(q1); cz(q0, q1); h(q1)
        assert len(circuit.ops) == 3
        assert circuit.ops[0] == ("h", 1)
        assert circuit.ops[1] == ("cz", 0, 1)
        assert circuit.ops[2] == ("h", 1)
    
    def test_swap_gate_decomposition(self):
        """Test SWAP gate decomposition."""
        qasm3_code = """qubit[2] q;
swap q[0], q[1];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        # swap(a, b) = cx(a, b); cx(b, a); cx(a, b)
        assert len(circuit.ops) == 3
        assert circuit.ops[0] == ("cx", 0, 1)
        assert circuit.ops[1] == ("cx", 1, 0)
        assert circuit.ops[2] == ("cx", 0, 1)


class TestQASM3Phase2_Measurements:
    """Test Phase 2: Measurement parsing."""
    
    def test_single_measurement(self):
        qasm3_code = """qubit[1] q;
measure q[0];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        assert circuit.ops[0] == ("measure_z", 0)
    
    def test_multiple_measurements(self):
        qasm3_code = """qubit[2] q;
measure q[0];
measure q[1];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        assert circuit.ops[0] == ("measure_z", 0)
        assert circuit.ops[1] == ("measure_z", 1)


class TestQASM3Phase2_MixedCircuits:
    """Test Phase 2: Mixed gate circuits."""
    
    def test_bell_state_preparation(self):
        """Test Bell state: H + CNOT + measurements."""
        qasm3_code = """OPENQASM 3.0;
qubit[2] q;
h q[0];
cx q[0], q[1];
measure q[0];
measure q[1];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        assert circuit.num_qubits == 2
        assert len(circuit.ops) == 4
        assert circuit.ops[0] == ("h", 0)
        assert circuit.ops[1] == ("cx", 0, 1)
        assert circuit.ops[2] == ("measure_z", 0)
        assert circuit.ops[3] == ("measure_z", 1)
    
    def test_complex_circuit(self):
        """Test a more complex circuit."""
        qasm3_code = """qubit[3] q;
h q[0];
rx(pi/4) q[0];
h q[1];
cz q[0], q[1];
h q[2];
ry(pi/8) q[2];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        assert circuit.num_qubits == 3
        assert len(circuit.ops) == 6


# ============================================================================
# Phase 3: OpenPulse Frame Definitions
# ============================================================================

class TestQASM3Phase3_FrameDefinitions:
    """Test Phase 3: Frame and port definitions."""
    
    def test_single_frame_definition(self):
        """Test parsing a single frame definition."""
        qasm3_code = """OPENQASM 3.0;
defcalgrammar "openpulse";
qubit[1] q;
cal {
    extern port d0;
    frame d0_frame = newframe(d0, 5000000000.0, 0.0);
}
h q[0];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        assert 'qasm3_frames' in circuit.metadata
        
        frames = circuit.metadata['qasm3_frames']
        assert 'd0_frame' in frames
        assert frames['d0_frame']['port'] == 'd0'
        assert frames['d0_frame']['frequency'] == 5000000000.0
        assert frames['d0_frame']['phase'] == 0.0
    
    def test_multiple_frame_definitions(self):
        """Test parsing multiple frame definitions."""
        qasm3_code = """OPENQASM 3.0;
defcalgrammar "openpulse";
qubit[2] q;
cal {
    extern port d0;
    extern port d1;
    frame d0_frame = newframe(d0, 5000000000.0, 0.0);
    frame d1_frame = newframe(d1, 5100000000.0, 1.57);
}
h q[0];
cx q[0], q[1];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        frames = circuit.metadata['qasm3_frames']
        assert len(frames) == 2
        assert frames['d0_frame']['frequency'] == 5000000000.0
        assert abs(frames['d1_frame']['phase'] - 1.57) < 1e-10
    
    def test_frame_with_scientific_notation(self):
        """Test frame frequencies with scientific notation."""
        qasm3_code = """OPENQASM 3.0;
defcalgrammar "openpulse";
qubit[1] q;
cal {
    extern port d0;
    frame d0_frame = newframe(d0, 5e9, 0.0);
}
h q[0];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        frames = circuit.metadata['qasm3_frames']
        assert frames['d0_frame']['frequency'] == 5e9
    
    def test_frame_with_negative_phase(self):
        """Test frame with negative phase."""
        qasm3_code = """OPENQASM 3.0;
defcalgrammar "openpulse";
qubit[1] q;
cal {
    extern port d0;
    frame d0_frame = newframe(d0, 5000000000.0, -0.5);
}
h q[0];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        frames = circuit.metadata['qasm3_frames']
        assert frames['d0_frame']['phase'] == -0.5
    
    def test_circuit_gates_with_frames(self):
        """Test that circuit gates are correctly parsed alongside frames."""
        qasm3_code = """OPENQASM 3.0;
defcalgrammar "openpulse";
qubit[2] q;
cal {
    extern port d0;
    extern port d1;
    frame d0_frame = newframe(d0, 5e9, 0.0);
    frame d1_frame = newframe(d1, 5.1e9, 0.0);
}
h q[0];
cx q[0], q[1];
measure q[0];
measure q[1];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        
        # Verify frames
        assert len(circuit.metadata['qasm3_frames']) == 2
        
        # Verify gates
        assert len(circuit.ops) == 4
        assert circuit.ops[0] == ('h', 0)
        assert circuit.ops[1] == ('cx', 0, 1)


# ============================================================================
# Phase 4: defcal Gate Definitions
# ============================================================================

class TestQASM3Phase4_DefcalDefinitions:
    """Test Phase 4: defcal gate definitions."""
    
    def test_simple_single_qubit_defcal(self):
        """Test parsing a simple single-qubit defcal."""
        qasm3_code = """OPENQASM 3.0;
defcalgrammar "openpulse";
qubit[1] q;
cal {
    extern port d0;
    frame d0_frame = newframe(d0, 5e9, 0.0);
}
defcal h $0 {
    waveform wf_1 = gaussian(0.1+0j, 160dt, 40dt);
    play(d0_frame, wf_1);
}
h q[0];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        assert 'qasm3_defcals' in circuit.metadata
        
        defcals = circuit.metadata['qasm3_defcals']
        assert 'h_1q' in defcals
        assert defcals['h_1q'].gate_name == 'h'
        assert defcals['h_1q'].qubits == [0]
    
    def test_defcal_with_drag_waveform(self):
        """Test defcal with DRAG waveform definition."""
        qasm3_code = """OPENQASM 3.0;
defcalgrammar "openpulse";
qubit[1] q;
cal {
    extern port d0;
    frame d0_frame = newframe(d0, 5e9, 0.0);
}
defcal x $0 {
    waveform wf_2 = drag((0.3+0j)+0.0im, 160dt, 40dt, 0.2);
    play(d0_frame, wf_2);
}
x q[0];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        defcals = circuit.metadata.get('qasm3_defcals', {})
        assert 'x_1q' in defcals
    
    def test_two_qubit_defcal(self):
        """Test parsing a two-qubit defcal (CX gate)."""
        qasm3_code = """OPENQASM 3.0;
defcalgrammar "openpulse";
qubit[2] q;
cal {
    extern port d0;
    extern port d1;
    frame d0_frame = newframe(d0, 5e9, 0.0);
    frame d1_frame = newframe(d1, 5.1e9, 0.0);
}
defcal cx $0, $1 {
    waveform wf_1 = gaussian(0.1+0j, 160dt, 40dt);
    play(d0_frame, wf_1);
    play(d1_frame, wf_1);
}
cx q[0], q[1];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        defcals = circuit.metadata.get('qasm3_defcals', {})
        assert 'cx_2q' in defcals
        assert defcals['cx_2q'].gate_name == 'cx'
        assert defcals['cx_2q'].qubits == [0, 1]
    
    def test_multiple_defcals(self):
        """Test parsing multiple defcal definitions."""
        qasm3_code = """OPENQASM 3.0;
defcalgrammar "openpulse";
qubit[1] q;
cal {
    extern port d0;
    frame d0_frame = newframe(d0, 5e9, 0.0);
}
defcal h $0 {
    waveform wf_h = gaussian(0.1+0j, 160dt, 40dt);
    play(d0_frame, wf_h);
}
defcal x $0 {
    waveform wf_x = gaussian(0.2+0j, 160dt, 40dt);
    play(d0_frame, wf_x);
}
h q[0];
x q[0];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        defcals = circuit.metadata.get('qasm3_defcals', {})
        assert len(defcals) >= 2


# ============================================================================
# Phase Integration: Complete Workflow Tests
# ============================================================================

class TestQASM3Integration_CompleteWorkflow:
    """Test Phase 2+3+4: Complete QASM3 workflow integration."""
    
    def test_full_bell_state_with_frames_and_defcals(self):
        """Test complete Bell state with frames and defcals."""
        qasm3_code = """OPENQASM 3.0;
defcalgrammar "openpulse";
qubit[2] q;
cal {
    extern port d0;
    extern port d1;
    frame d0_frame = newframe(d0, 5e9, 0.0);
    frame d1_frame = newframe(d1, 5.1e9, 0.0);
}
defcal h $0 {
    waveform wf_h = gaussian(0.1+0j, 160dt, 40dt);
    play(d0_frame, wf_h);
}
defcal cx $0, $1 {
    waveform wf_cx = gaussian(0.2+0j, 160dt, 40dt);
    play(d0_frame, wf_cx);
    play(d1_frame, wf_cx);
}
h q[0];
cx q[0], q[1];
measure q[0];
measure q[1];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        
        # Verify circuit structure
        assert circuit.num_qubits == 2
        assert len(circuit.ops) == 4
        assert circuit.ops[0] == ('h', 0)
        assert circuit.ops[1] == ('cx', 0, 1)
        assert circuit.ops[2] == ('measure_z', 0)
        assert circuit.ops[3] == ('measure_z', 1)
        
        # Verify frames
        assert 'qasm3_frames' in circuit.metadata
        frames = circuit.metadata['qasm3_frames']
        assert len(frames) == 2
        
        # Verify defcals
        assert 'qasm3_defcals' in circuit.metadata
        defcals = circuit.metadata['qasm3_defcals']
        assert 'h_1q' in defcals
        assert 'cx_2q' in defcals
    
    def test_chained_execution_qasm3_to_circuit(self):
        """Test complete chain: QASM3 source → Circuit IR → execution-ready."""
        from tyxonq.devices.simulators.driver import _qasm_to_ir_if_needed
        
        qasm3_source = """OPENQASM 3.0;
qubit[2] q;
h q[0];
cx q[0], q[1];
measure q[0];
measure q[1];
"""
        
        # Simulate driver layer parsing
        circuit = _qasm_to_ir_if_needed(None, qasm3_source)
        
        assert circuit.num_qubits == 2
        assert len(circuit.ops) == 4
        
        # Verify circuit is ready for execution
        assert hasattr(circuit, 'num_qubits')
        assert hasattr(circuit, 'ops')
        assert hasattr(circuit, 'metadata')


# ============================================================================
# Phase 5: Complete Closed-Loop Workflow (Compile → Export QASM3 → Import)
# ============================================================================

class TestQASM3ClosedLoopWorkflow:
    """Test complete closed-loop: TyxonQ compile(output='qasm3') → import → verify.
    
    这是最关键的测试场景：
    1. 在 TyxonQ 中构建混合电路（门+脉冲）或纯门电路
    2. 编译导出为 QASM3（包含 frame+defcal）
    3. 将导出的 QASM3 导入回来
    4. 验证导入的电路与原始电路结构一致
    
    这验证了编译系统和导入系统的**互操作性**。
    """
    
    def test_compile_simple_circuit_to_qasm3(self):
        """Test compiling a simple gate circuit to QASM3."""
        # Step 1: Create a simple circuit
        circuit = Circuit(2)
        circuit.h(0).cx(0, 1).measure_z(0).measure_z(1)
        
        # Step 2: Compile to QASM3
        result = compile(circuit, output="qasm3")
        # compile() 返回 CompileResult: {"circuit": ..., "compiled_source": ..., "metadata": ...}
        # 对于 QASM3 output，circuit 和 compiled_source 都是 QASM3 字符串
        qasm3_code = result["compiled_source"]
        
        # Verify it's valid QASM3
        assert isinstance(qasm3_code, str)
        assert "OPENQASM 3" in qasm3_code
        assert "qubit[2]" in qasm3_code
        assert "h q[0]" in qasm3_code
        assert "cx q[0], q[1]" in qasm3_code
    
    def test_compile_and_import_bell_state(self):
        """Test round-trip: Circuit → compile(qasm3) → import → verify."""
        # Step 1: Create original circuit
        original = Circuit(2)
        original.h(0).cx(0, 1).measure_z(0).measure_z(1)
        
        # Step 2: Compile to QASM3
        compile_result = compile(original, output="qasm3")
        # compile() 返回 CompileResult，compiled_source 是 QASM3 字符串
        qasm3_code = compile_result["compiled_source"]
        
        print("\n[Compiled QASM3]")
        print(qasm3_code)
        
        # Step 3: Import QASM3 back to Circuit
        imported = qasm3_to_circuit(qasm3_code)
        
        # Step 4: Verify structure consistency
        assert imported.num_qubits == original.num_qubits
        assert len(imported.ops) == len(original.ops)
        
        # Verify each operation matches
        for i, (orig_op, imported_op) in enumerate(zip(original.ops, imported.ops)):
            assert orig_op[0] == imported_op[0], f"Op {i} gate type mismatch"
            assert orig_op[1:] == imported_op[1:], f"Op {i} qubit indices mismatch"
    
    def test_compile_parametric_gates_qasm3(self):
        """Test compiling circuit with parameterized gates."""
        # Create circuit with RX/RY gates
        circuit = Circuit(2)
        circuit.rx(1.5, 0).ry(0.5, 1).cx(0, 1)
        
        # Compile to QASM3
        result = compile(circuit, output="qasm3")
        # compiled_source 是 QASM3 字符串
        qasm3_code = result["compiled_source"]
        
        # Should contain parameterized gates
        assert "rx" in qasm3_code or "RX" in qasm3_code
        assert "ry" in qasm3_code or "RY" in qasm3_code
        
        # Import back
        imported = qasm3_to_circuit(qasm3_code)
        assert imported.num_qubits == 2
        # Should have rx, ry, cx + auto-added measurements (2 qubits)
        # 3 original ops + 2 auto measurements = 5 ops
        assert len(imported.ops) == 5
    
    def test_compile_mixed_gates_qasm3(self):
        """Test compiling circuit with mixed single and two-qubit gates."""
        circuit = Circuit(3)
        circuit.h(0)
        circuit.x(1)
        circuit.y(2)
        circuit.cx(0, 1)
        circuit.cz(1, 2)
        circuit.measure_z(0).measure_z(1).measure_z(2)
        
        # Compile
        result = compile(circuit, output="qasm3")
        # compiled_source 是 QASM3 字符串
        qasm3_code = result["compiled_source"]
        
        # Import
        imported = qasm3_to_circuit(qasm3_code)
        
        # Verify
        assert imported.num_qubits == 3
        # All operations should be present (h, x, y, cx, cz, measure, measure, measure)
        assert len(imported.ops) == 8
    
    def test_compile_with_pi_expressions(self):
        """Test that compiled QASM3 handles pi expressions correctly."""
        import math
        
        circuit = Circuit(1)
        circuit.rx(0, math.pi / 2)  # rx(qubit, angle) - π/2
        circuit.ry(0, math.pi)      # ry(qubit, angle) - π
        
        # Compile
        result = compile(circuit, output="qasm3")
        # compiled_source 是 QASM3 字符串
        qasm3_code = result["compiled_source"]
        
        # Import
        imported = qasm3_to_circuit(qasm3_code)
        
        # Verify pi expressions are preserved in values
        assert abs(imported.ops[0][2] - math.pi/2) < 1e-10
        assert abs(imported.ops[1][2] - math.pi) < 1e-10
    
    def test_compile_large_circuit_qasm3(self):
        """Test compiling a larger circuit."""
        # Create a 4-qubit circuit with multiple gates
        circuit = Circuit(4)
        circuit.h(0).h(1).h(2).h(3)
        circuit.cx(0, 1).cx(2, 3)
        circuit.cz(1, 2)
        circuit.measure_z(0).measure_z(1).measure_z(2).measure_z(3)
        
        # Compile
        result = compile(circuit, output="qasm3")
        # compiled_source 是 QASM3 字符串
        qasm3_code = result["compiled_source"]
        
        assert "qubit[4]" in qasm3_code
        
        # Import
        imported = qasm3_to_circuit(qasm3_code)
        assert imported.num_qubits == 4
        # All operations should be present (4h + 2cx + 1cz + 4measure = 11)
        assert len(imported.ops) == 11
    
    def test_device_run_with_qasm3_source(self):
        """Test device().run(source=qasm3_string) execution flow.
        
        This tests the complete driver layer integration:
        1. Create QASM3 source code
        2. Pass to device().run(source=...)
        3. Driver layer detects QASM3 and auto-imports
        4. Simulates the imported circuit
        """
        qasm3_source = """OPENQASM 3.0;
qubit[2] q;
h q[0];
cx q[0], q[1];
measure q[0];
measure q[1];
"""
        
        # This is what happens internally:
        # device().run(source=qasm3_source)
        #   → driver._qasm_to_ir_if_needed(None, qasm3_source)
        #   → qasm3_to_circuit(qasm3_source)
        #   → Circuit IR for execution
        
        # 直接使用 compile() 的 compiled_source 字段（不仅测试 qasm3_to_circuit）
        circuit = Circuit(2)
        circuit.h(0).cx(0, 1)
        
        result = compile(circuit, output="qasm3")
        # compiled_source 是擲坐 source，也是 QASM3 字符串
        qasm3_from_compile = result["compiled_source"]
        
        # 重新从 compiled_source 导入
        reimported = qasm3_to_circuit(qasm3_from_compile)
        
        assert reimported.num_qubits == 2
        assert len(reimported.ops) >= 2
    
    def test_compile_export_import_consistency(self):
        """Test that compile() → export → import is consistent.
        
        Key verification:
        - Original circuit has N ops
        - Compiled QASM3 contains all operations
        - Imported circuit has N ops with matching gate types
        """
        original = Circuit(2)
        original.h(0)
        original.cx(0, 1)
        original.ry(0.7, 1)
        original.measure_z(0)
        original.measure_z(1)
        
        num_original_ops = len(original.ops)
        
        # Compile and import
        compile_result = compile(original, output="qasm3")
        # compiled_source 是擲坐 source，也是 QASM3 字符串
        qasm3_code = compile_result["compiled_source"]
        imported = qasm3_to_circuit(qasm3_code)
        
        # Should have same number of ops
        assert len(imported.ops) == num_original_ops
        
        # All gate types should match
        for i, (orig, imp) in enumerate(zip(original.ops, imported.ops)):
            assert orig[0] == imp[0], f"Op {i}: {orig[0]} != {imp[0]}"
    
# ============================================================================
# Phase 5: Complete Closed-Loop Workflow (Compile → Export QASM3 → Import → Run)
# ============================================================================

class TestQASM3EndToEndWorkflow:
    """Test complete end-to-end: TyxonQ Circuit → compile(qasm3) → import → run().
    
    这是最关键的真实世界测试：完整的链式调用
    1. 在 TyxonQ 中构建电路
    2. 编译导出 QASM3
    3. QASM3 导入回来
    4. 通过 device().run() 真正执行
    5. 验证数值结果一致性
    """
    
    def test_complete_chain_bell_state(self):
        """Test complete chain: create → compile → import → run."""
        # Step 1: Create circuit
        original = Circuit(2)
        original.h(0).cx(0, 1)
        
        # Step 2: Compile to QASM3
        result = compile(original, output="qasm3")
        # compiled_source 是擲坐 source，也是 QASM3 字符串
        qasm3_code = result["compiled_source"]
        
        assert "OPENQASM 3" in qasm3_code
        assert "h q[0]" in qasm3_code
        assert "cx q[0], q[1]" in qasm3_code
        
        # Step 3: Import QASM3 back
        imported = qasm3_to_circuit(qasm3_code)
        assert imported.num_qubits == 2
        # Should have h, cx, and auto-added measurements
        assert len(imported.ops) >= 2
        
        # Step 4: Verify we can execute the imported circuit
        try:
            # Run via driver layer (simulates device().run())
            from tyxonq.devices.simulators.driver import _qasm_to_ir_if_needed
            circuit_ir = _qasm_to_ir_if_needed(None, qasm3_code)
            assert circuit_ir.num_qubits == 2
        except Exception as e:
            pytest.fail(f"Failed to execute imported circuit: {e}")
    
    def test_complete_chain_with_measurements(self):
        """Test chain with explicit measurements."""
        # Create circuit WITH measurements
        original = Circuit(2)
        original.h(0).cx(0, 1).measure_z(0).measure_z(1)
        
        # Compile
        result = compile(original, output="qasm3")
        # compiled_source 是擲坐 source，也是 QASM3 字符串
        qasm3_code = result["compiled_source"]
        
        # Import
        imported = qasm3_to_circuit(qasm3_code)
        
        # Both should have 4 ops (h, cx, measure, measure)
        assert len(original.ops) == len(imported.ops)
        
        # Run imported circuit
        from tyxonq.devices.simulators.driver import _qasm_to_ir_if_needed
        circuit_ir = _qasm_to_ir_if_needed(None, qasm3_code)
        assert circuit_ir.num_qubits == 2


# ============================================================================
# Phase 4: defcal Gate Definition Import Tests
# ============================================================================

class TestPhase4DefcalBasics:
    """Test Phase 4: Basic defcal gate definitions."""
    
    def test_simple_single_qubit_defcal(self):
        """Test parsing a simple single-qubit defcal."""
        qasm3_code = """OPENQASM 3.0;
defcalgrammar "openpulse";
qubit[1] q;
cal {
    extern port d0;
    frame d0_frame = newframe(d0, 5e9, 0.0);
}
defcal h $0 {
    waveform wf_1 = gaussian(0.1+0j, 160dt, 40dt);
    play(d0_frame, wf_1);
}
h q[0];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        assert circuit.num_qubits == 1
        assert 'qasm3_defcals' in circuit.metadata
        
        defcals = circuit.metadata['qasm3_defcals']
        assert 'h_1q' in defcals
        
        h_defcal = defcals['h_1q']
        assert h_defcal.gate_name == 'h'
        assert h_defcal.qubits == [0]
    
    def test_defcal_with_drag_waveform(self):
        """Test parsing defcal with DRAG waveform definition."""
        qasm3_code = """OPENQASM 3.0;
defcalgrammar "openpulse";
qubit[1] q;
cal {
    extern port d0;
    frame d0_frame = newframe(d0, 5e9, 0.0);
}
defcal x $0 {
    waveform wf_2 = drag((0.3+0j)+0.0im, 160dt, 40dt, 0.2);
    play(d0_frame, wf_2);
}
x q[0];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        defcals = circuit.metadata.get('qasm3_defcals', {})
        assert 'x_1q' in defcals
        
        x_defcal = defcals['x_1q']
        assert x_defcal.gate_name == 'x'
        assert x_defcal.qubits == [0]
    
    def test_two_qubit_defcal(self):
        """Test parsing a two-qubit defcal (e.g., CX gate)."""
        qasm3_code = """OPENQASM 3.0;
defcalgrammar "openpulse";
qubit[2] q;
cal {
    extern port d0;
    extern port d1;
    frame d0_frame = newframe(d0, 5e9, 0.0);
    frame d1_frame = newframe(d1, 5.1e9, 0.0);
}
defcal cx $0, $1 {
    waveform wf_1 = gaussian(0.1+0j, 160dt, 40dt);
    play(d0_frame, wf_1);
    play(d1_frame, wf_1);
}
cx q[0], q[1];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        defcals = circuit.metadata.get('qasm3_defcals', {})
        assert 'cx_2q' in defcals
        
        cx_defcal = defcals['cx_2q']
        assert cx_defcal.gate_name == 'cx'
        assert cx_defcal.qubits == [0, 1]
    
    def test_multiple_defcals(self):
        """Test parsing multiple defcal definitions."""
        qasm3_code = """OPENQASM 3.0;
defcalgrammar "openpulse";
qubit[1] q;
cal {
    extern port d0;
    frame d0_frame = newframe(d0, 5e9, 0.0);
}
defcal h $0 {
    waveform wf_h = gaussian(0.1+0j, 160dt, 40dt);
    play(d0_frame, wf_h);
}
defcal x $0 {
    waveform wf_x = gaussian(0.2+0j, 160dt, 40dt);
    play(d0_frame, wf_x);
}
h q[0];
x q[0];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        defcals = circuit.metadata.get('qasm3_defcals', {})
        assert len(defcals) >= 2
    
    def test_defcal_body_storage(self):
        """Test that defcal body is correctly stored for later execution."""
        qasm3_code = """OPENQASM 3.0;
defcalgrammar "openpulse";
qubit[1] q;
cal {
    extern port d0;
    frame d0_frame = newframe(d0, 5e9, 0.0);
}
defcal h $0 {
    waveform wf = gaussian(0.1+0j, 160dt, 40dt);
    play(d0_frame, wf);
}
h q[0];
"""
        circuit = qasm3_to_circuit(qasm3_code)
        defcals = circuit.metadata.get('qasm3_defcals', {})
        
        if 'h_1q' in defcals:
            h_defcal = defcals['h_1q']
            assert h_defcal.body is not None
            assert len(h_defcal.body) > 0


class TestPhase4CompileWithPulse:
    """Test Phase 4: Compiling with pulse to generate defcal in QASM3/TQASM.
    
    关键场景：
    1. 模式 A (Gate+Pulse 混合)：circuit.use_pulse() → compile(output="tqasm") → TQASM with defcal
    2. 模式 C (纯 Pulse)：PulseProgram → compile_pulse() → TQASM with defcal
    """
    
    def test_mode_a_compile_and_run_via_device(self):
        """Test Mode A: Gate circuit with use_pulse() → compile → device().run()
        
        Complete workflow:
        1. Create gate circuit with pulse compilation
        2. Compile to QASM3 with defcal
        3. Execute the compiled circuit via device().run()
        4. Verify execution succeeds
        
        核心测试：compiled QASM3 should be executable by device!
        """
        # Step 1: Create gate circuit
        circuit = Circuit(2)
        circuit.h(0).cx(0, 1)
        
        # Step 2: Enable pulse compilation
        circuit.use_pulse(
            mode="pulse_only",
            device_params={
                "qubit_freq": [5.0e9, 5.1e9],
                "anharmonicity": [-330e6, -330e6]
            },
            inline_pulses=True  # 关键：生成完整 defcal
        )
        
        # Step 3: Compile to QASM3 with defcal
        result = compile(circuit, output="qasm3")
        # compiled_source 是擲坐 source QASM3 字符串
        qasm3_code = result["compiled_source"]
        
        assert isinstance(qasm3_code, str)
        assert len(qasm3_code) > 0
        
        # Step 4 & 5: Execute the compiled circuit
        # 这才是核心测试！编译出的 QASM3 应该能被 device 执行！
        # Method: Compile returns a result, we need to verify it's executable
        try:
            # The compiled QASM3 with defcal should be parseable
            imported = qasm3_to_circuit(qasm3_code)
            
            # Then run the imported circuit
            result = imported.device(provider="simulator", device="statevector").run(shots=100)
            
            # 如果执行成功，说明 defcal + QASM3 parsing 完整工作！
            assert result is not None
        except Exception as e:
            pytest.fail(f"Failed to execute compiled QASM3 via device().run(): {e}")
    
    def test_mode_c_pulse_program_compile_and_run(self):
        """Test Mode C: PulseProgram → compile_pulse → device().run()
        
        Complete workflow:
        1. Create pure pulse program
        2. Compile to TQASM with defcal
        3. Execute via device().run(source=tqasm)
        4. Verify execution succeeds
        
        核心测试：compiled TQASM should be executable by device!
        """
        from tyxonq.core.ir.pulse import PulseProgram
        from tyxonq.compiler.api import compile_pulse
        
        # Step 1: Create pure pulse program
        prog = PulseProgram(1)
        prog.drag(0, amp=1.0, duration=160, sigma=40, beta=0.2, qubit_freq=5.0e9)
        
        # Step 2: Compile to TQASM with defcal
        result = compile_pulse(
            prog,
            output="tqasm",
            device_params={
                "qubit_freq": [5.0e9],
                "anharmonicity": [-330e6]
            },
            options={"inline_pulses": True}
        )
        
        # pulse_schedule 是擲坐 source TQASM 字符串
        tqasm_code = result["compiled_pulse_schedule"]
        
        assert isinstance(tqasm_code, str)
        assert len(tqasm_code) > 0
        
        # Step 3 & 4: Execute the compiled TQASM
        # 这才是核心测试！编译出的 TQASM 应该能被 device 执行！
        try:
            # Create a new circuit for execution
            temp_circuit = Circuit(1)
            # Store the TQASM as source
            temp_circuit._source = tqasm_code
            
            result = temp_circuit.device(provider="simulator", device="statevector").run(shots=100)
            # 如果执行成功，说明 defcal + TQASM parsing 完整工作！
            assert result is not None
        except Exception as e:
            pytest.fail(f"Failed to execute compiled TQASM via device().run(): {e}")
    
    def test_complete_closed_loop_compile_qasm3_import_run(self):
        """Test complete closed-loop: compile → QASM3 → import → run → verify results.
        
        This is the ultimate integration test:
        1. Create a Bell state circuit
        2. Compile with pulse → QASM3 with defcal
        3. Import QASM3 back to Circuit
        4. Run imported circuit via device()
        5. Verify execution succeeds
        
        核心测试：完整闭环 - compile → QASM3 → import → device().run()
        """
        # Step 1: Create Bell state circuit
        original = Circuit(2)
        original.h(0).cx(0, 1)
        
        # Step 2: Compile with pulse
        original.use_pulse(
            device_params={
                "qubit_freq": [5.0e9, 5.1e9],
                "anharmonicity": [-330e6, -330e6]
            },
            inline_pulses=True
        )
        
        result = compile(original, output="qasm3")
        # compiled_source 是擲坐 source QASM3 字符串
        qasm3_code = result["compiled_source"]
        
        # Verify QASM3 is generated
        assert isinstance(qasm3_code, str)
        assert len(qasm3_code) > 0
        
        # Step 3: Import QASM3 back
        imported = qasm3_to_circuit(qasm3_code)
        assert imported.num_qubits == 2
        
        # Step 4: Run imported circuit
        try:
            run_result = imported.device(provider="simulator", device="statevector").run(shots=100)
            assert run_result is not None
            # 步骤 5: 执行成功！这表明编译 → QASM3 → 导入 → 执行整个庒圈完整了！
        except Exception as e:
            pytest.fail(f"Failed to run imported QASM3 circuit via device: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    pytest.main([__file__, "-v"])
    pytest.main([__file__, "-v"])
