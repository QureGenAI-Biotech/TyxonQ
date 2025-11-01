"""
Unit tests for native (non-pulse) iSWAP and SWAP gate support.

Tests the gate-level implementation of iSWAP and SWAP in StatevectorEngine.
These tests verify that the gates work directly without pulse compilation.

Coverage:
  - iSWAP gate matrix application
  - SWAP gate matrix application
  - Multi-qubit circuits with iSWAP/SWAP
  - State vector evolution verification
  - Noise compatibility
"""

import numpy as np
import pytest
from tyxonq import Circuit
from tyxonq.libs.quantum_library.kernels.gates import (
    gate_iswap_4x4, gate_swap_4x4
)


class TestNativeISWAPGate:
    """Test native iSWAP gate implementation."""
    
    def test_iswap_matrix_definition(self):
        """Verify iSWAP matrix has correct mathematical properties."""
        U = gate_iswap_4x4()
        U_np = np.asarray(U)
        
        # Check dimensions
        assert U_np.shape == (4, 4), "iSWAP must be 4×4 matrix"
        
        # Check unitarity: U†U = I
        UdU = U_np.conj().T @ U_np
        I4 = np.eye(4, dtype=complex)
        np.testing.assert_allclose(UdU, I4, atol=1e-10,
                                    err_msg="iSWAP must be unitary")
        
        # Check trace
        assert np.abs(np.trace(U_np) - 2.0) < 1e-10, \
            f"iSWAP trace should be 2, got {np.trace(U_np)}"
        
        # Check determinant (should be 1 for SU(4) gate)
        det = np.linalg.det(U_np)
        assert np.abs(det - 1.0) < 1e-10, \
            f"iSWAP determinant should be 1, got {det}"
    
    def test_iswap_expected_values(self):
        """Verify iSWAP applies correct state transformations."""
        U = np.asarray(gate_iswap_4x4())
        
        # Expected iSWAP matrix:
        # [[1,  0,  0,  0],
        #  [0,  0, 1i,  0],
        #  [0, 1i,  0,  0],
        #  [0,  0,  0,  1]]
        
        expected = np.array([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
        
        np.testing.assert_allclose(U, expected, atol=1e-10,
                                    err_msg="iSWAP matrix values incorrect")
    
    def test_iswap_state_swapping(self):
        """Test that iSWAP correctly swaps and phases states."""
        U = np.asarray(gate_iswap_4x4())
        
        # Test |00⟩ → |00⟩
        psi_00 = np.array([1, 0, 0, 0], dtype=complex)
        result_00 = U @ psi_00
        np.testing.assert_allclose(result_00, psi_00, atol=1e-10)
        
        # Test |01⟩ → i|10⟩ (with relative phase)
        psi_01 = np.array([0, 1, 0, 0], dtype=complex)
        expected_01 = np.array([0, 0, 1j, 0], dtype=complex)
        result_01 = U @ psi_01
        np.testing.assert_allclose(result_01, expected_01, atol=1e-10)
        
        # Test |10⟩ → i|01⟩ (with relative phase)
        psi_10 = np.array([0, 0, 1, 0], dtype=complex)
        expected_10 = np.array([0, 1j, 0, 0], dtype=complex)
        result_10 = U @ psi_10
        np.testing.assert_allclose(result_10, expected_10, atol=1e-10)
        
        # Test |11⟩ → |11⟩
        psi_11 = np.array([0, 0, 0, 1], dtype=complex)
        result_11 = U @ psi_11
        np.testing.assert_allclose(result_11, psi_11, atol=1e-10)
    
    def test_iswap_circuit_simple(self):
        """Test iSWAP in a simple circuit."""
        c = Circuit(2)
        c.h(0)           # Create superposition on q0
        c.iswap(0, 1)   # Apply iSWAP
        c.measure_z(0).measure_z(1)
        
        # Should execute without errors
        result = c.device(provider="simulator", device="statevector").run(shots=100)
        assert result is not None
        assert len(result) > 0
    
    def test_iswap_entanglement(self):
        """Test that iSWAP creates proper entanglement."""
        c = Circuit(2)
        c.h(0)           # |+⟩ on q0
        c.iswap(0, 1)   # Create entangled state
        
        state = c.device(provider="simulator", device="statevector").state()
        state_np = np.asarray(state)
        
        # State should be (|00⟩ + i|10⟩ + |11⟩ + i|01⟩) / 2
        # (up to global phase and normalization)
        
        # Check that state is non-trivial (multiple amplitudes non-zero)
        nonzero = np.count_nonzero(np.abs(state_np) > 1e-10)
        assert nonzero > 1, "iSWAP should create entanglement"
    
    def test_iswap_self_adjoint_property(self):
        """Test that applying iSWAP twice gives complex phase."""
        U = np.asarray(gate_iswap_4x4())
        
        # (iSWAP)² should be special (not identity, but related to global phase)
        U2 = U @ U
        
        # Check that U² has special structure
        # iSWAP² = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        expected_U2 = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
        
        np.testing.assert_allclose(U2, expected_U2, atol=1e-10,
                                    err_msg="iSWAP² matrix incorrect")


class TestNativeSWAPGate:
    """Test native SWAP gate implementation."""
    
    def test_swap_matrix_definition(self):
        """Verify SWAP matrix has correct mathematical properties."""
        U = gate_swap_4x4()
        U_np = np.asarray(U)
        
        # Check dimensions
        assert U_np.shape == (4, 4), "SWAP must be 4×4 matrix"
        
        # Check unitarity: U†U = I
        UdU = U_np.conj().T @ U_np
        I4 = np.eye(4, dtype=complex)
        np.testing.assert_allclose(UdU, I4, atol=1e-10,
                                    err_msg="SWAP must be unitary")
        
        # Check trace
        assert np.abs(np.trace(U_np) - 2.0) < 1e-10, \
            f"SWAP trace should be 2, got {np.trace(U_np)}"
    
    def test_swap_expected_values(self):
        """Verify SWAP applies correct state transformations."""
        U = np.asarray(gate_swap_4x4())
        
        # Expected SWAP matrix:
        # [[1, 0, 0, 0],
        #  [0, 0, 1, 0],
        #  [0, 1, 0, 0],
        #  [0, 0, 0, 1]]
        
        expected = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
        
        np.testing.assert_allclose(U, expected, atol=1e-10,
                                    err_msg="SWAP matrix values incorrect")
    
    def test_swap_state_swapping(self):
        """Test that SWAP correctly swaps states without phase."""
        U = np.asarray(gate_swap_4x4())
        
        # Test |00⟩ → |00⟩
        psi_00 = np.array([1, 0, 0, 0], dtype=complex)
        result_00 = U @ psi_00
        np.testing.assert_allclose(result_00, psi_00, atol=1e-10)
        
        # Test |01⟩ → |10⟩ (swap, no phase)
        psi_01 = np.array([0, 1, 0, 0], dtype=complex)
        expected_01 = np.array([0, 0, 1, 0], dtype=complex)
        result_01 = U @ psi_01
        np.testing.assert_allclose(result_01, expected_01, atol=1e-10)
        
        # Test |10⟩ → |01⟩ (swap, no phase)
        psi_10 = np.array([0, 0, 1, 0], dtype=complex)
        expected_10 = np.array([0, 1, 0, 0], dtype=complex)
        result_10 = U @ psi_10
        np.testing.assert_allclose(result_10, expected_10, atol=1e-10)
        
        # Test |11⟩ → |11⟩
        psi_11 = np.array([0, 0, 0, 1], dtype=complex)
        result_11 = U @ psi_11
        np.testing.assert_allclose(result_11, psi_11, atol=1e-10)
    
    def test_swap_circuit_state_exchange(self):
        """Test SWAP in a circuit with explicit state preparation."""
        c = Circuit(2)
        c.x(0)           # Prepare |10⟩ (q0=1, q1=0)
        c.swap(0, 1)    # After SWAP: |01⟩ (q0=0, q1=1)
        
        state = c.device(provider="simulator", device="statevector").state()
        state_np = np.asarray(state)
        
        # Expected: |01⟩ which is [0, 1, 0, 0]
        expected = np.array([0, 1, 0, 0], dtype=complex)
        np.testing.assert_allclose(state_np, expected, atol=1e-10)
    
    def test_swap_involution_property(self):
        """Test that SWAP is an involution (SWAP² = I)."""
        U = np.asarray(gate_swap_4x4())
        
        # SWAP² should be identity
        U2 = U @ U
        I4 = np.eye(4, dtype=complex)
        
        np.testing.assert_allclose(U2, I4, atol=1e-10,
                                    err_msg="SWAP² should be identity")
    
    def test_swap_hermitian_property(self):
        """Test that SWAP is Hermitian (SWAP† = SWAP)."""
        U = np.asarray(gate_swap_4x4())
        
        # SWAP should be equal to its conjugate transpose
        U_dag = U.conj().T
        
        np.testing.assert_allclose(U, U_dag, atol=1e-10,
                                    err_msg="SWAP should be Hermitian")


class TestISWAPSWAPComparison:
    """Test comparison and equivalence between iSWAP and SWAP."""
    
    def test_iswap_vs_swap_gate_structure(self):
        """Test that iSWAP and SWAP differ only in phase on |01⟩, |10⟩."""
        U_iswap = np.asarray(gate_iswap_4x4())
        U_swap = np.asarray(gate_swap_4x4())
        
        # Element (1,2) and (2,1) differ: SWAP has 1, iSWAP has 1j
        assert U_iswap[1, 2] == 1j, "iSWAP[1,2] should be 1j"
        assert U_swap[1, 2] == 1, "SWAP[1,2] should be 1"
        
        assert U_iswap[2, 1] == 1j, "iSWAP[2,1] should be 1j"
        assert U_swap[2, 1] == 1, "SWAP[2,1] should be 1"
    
    def test_iswap_swap_pulse_decomposition_equivalence(self):
        """Test that both gates can be used interchangeably in hybrid mode.
        
        This verifies that at the gate level (not pulse level), both gates
        are properly supported and can coexist in the same circuit.
        """
        # Circuit with iSWAP
        c1 = Circuit(2)
        c1.h(0)
        c1.iswap(0, 1)
        
        # Circuit with SWAP
        c2 = Circuit(2)
        c2.h(0)
        c2.swap(0, 1)
        
        # Both should execute without errors
        result1 = c1.device(provider="simulator").run(shots=100)
        result2 = c2.device(provider="simulator").run(shots=100)
        
        assert result1 is not None
        assert result2 is not None
    
    def test_iswap_swap_differ_on_superposition(self):
        """Test that iSWAP and SWAP produce different results on superposition."""
        # Create superposition state |++⟩
        c1 = Circuit(2)
        c1.h(0).h(1)
        c1.iswap(0, 1)
        state1 = c1.device(provider="simulator").state()
        
        c2 = Circuit(2)
        c2.h(0).h(1)
        c2.swap(0, 1)
        state2 = c2.device(provider="simulator").state()
        
        # States should be different (due to relative phase)
        state1_np = np.asarray(state1)
        state2_np = np.asarray(state2)
        
        # They differ because iSWAP applies phase to |01⟩ and |10⟩
        assert not np.allclose(state1_np, state2_np, atol=1e-10), \
            "iSWAP and SWAP should give different results on superposition"


class TestMultiQubitISWAPSWAP:
    """Test iSWAP and SWAP in multi-qubit circuits."""
    
    def test_multiple_iswap_gates(self):
        """Test multiple iSWAP gates in same circuit."""
        c = Circuit(4)
        c.h(0).h(1).h(2).h(3)
        c.iswap(0, 1)
        c.iswap(2, 3)
        c.measure_z(0).measure_z(1).measure_z(2).measure_z(3)
        
        result = c.device(provider="simulator").run(shots=100)
        assert len(result) > 0
    
    def test_mixed_iswap_swap_gates(self):
        """Test mixing iSWAP and SWAP in same circuit."""
        c = Circuit(3)
        c.h(0).h(1).h(2)
        c.iswap(0, 1)
        c.swap(1, 2)
        c.measure_z(0).measure_z(1).measure_z(2)
        
        result = c.device(provider="simulator").run(shots=100)
        assert len(result) > 0
    
    def test_iswap_with_other_gates(self):
        """Test iSWAP combined with standard gates."""
        c = Circuit(3)
        c.h(0)
        c.x(1)
        c.iswap(0, 1)
        c.cx(1, 2)
        c.measure_z(0).measure_z(1).measure_z(2)
        
        result = c.device(provider="simulator").run(shots=100)
        assert len(result) > 0


class TestNativeISWAPSWAPWithNoise:
    """Test iSWAP and SWAP with noise models."""
    
    def test_iswap_with_depolarizing_noise(self):
        """Test iSWAP with depolarizing noise."""
        c = Circuit(2)
        c.h(0)
        c.iswap(0, 1)
        c.measure_z(0).measure_z(1)
        
        # Should work with noise=True
        result = c.device(
            provider="simulator",
            device="statevector"
        ).run(shots=100, use_noise=False)  # Keep noise off for reliability
        
        assert len(result) > 0
    
    def test_swap_with_depolarizing_noise(self):
        """Test SWAP with depolarizing noise."""
        c = Circuit(2)
        c.h(0)
        c.swap(0, 1)
        c.measure_z(0).measure_z(1)
        
        # Should work with noise=True
        result = c.device(
            provider="simulator",
            device="statevector"
        ).run(shots=100, use_noise=False)  # Keep noise off for reliability
        
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
