"""Test Circuit.unitary() method implementation.

This test suite validates the Circuit.unitary() method across all layers:
1. Circuit class API (user interface)
2. quantum_library kernels (core numerical implementation)
3. StatevectorEngine executor (execution engine)
4. Chem module integration (backward compatibility)
"""

import numpy as np
import pytest
import tyxonq as tq
from tyxonq.libs.quantum_library.kernels.gates import gate_ry


def test_single_qubit_unitary():
    """Test single-qubit unitary application (√X gate)."""
    c = tq.Circuit(1)
    sqrt_x = np.array([[0.5+0.5j, 0.5-0.5j],
                       [0.5-0.5j, 0.5+0.5j]], dtype=np.complex128)
    c.unitary(0, matrix=sqrt_x)
    state = c.state()
    
    # √X|0⟩ should give (1+i)|0⟩/2 + (1-i)|1⟩/2
    expected = np.array([0.5+0.5j, 0.5-0.5j], dtype=np.complex128)
    assert np.allclose(state, expected), f"Expected {expected}, got {state}"


def test_two_qubit_unitary():
    """Test two-qubit unitary application (iSWAP gate)."""
    c = tq.Circuit(2)
    c.x(0)  # Prepare |10⟩ state
    
    # iSWAP gate
    iswap = np.array([[1, 0, 0, 0],
                      [0, 0, 1j, 0],
                      [0, 1j, 0, 0],
                      [0, 0, 0, 1]], dtype=np.complex128)
    c.unitary(0, 1, matrix=iswap)
    state = c.state()
    
    # iSWAP|10⟩ = i|01⟩
    expected_state = np.array([0, 1j, 0, 0], dtype=np.complex128)
    assert np.allclose(state, expected_state), f"Expected {expected_state}, got {state}"


def test_parameterized_unitary():
    """Test unitary with parameterized gates from kernels library."""
    c = tq.Circuit(1)
    theta = np.pi / 4
    c.unitary(0, matrix=gate_ry(theta))
    state = c.state()
    
    # RY(π/4)|0⟩ = cos(π/8)|0⟩ + sin(π/8)|1⟩
    expected = np.array([np.cos(theta/2), np.sin(theta/2)], dtype=np.complex128)
    assert np.allclose(state, expected), f"Expected {expected}, got {state}"
    
    # Check probabilities
    probs = np.abs(state)**2
    assert np.isclose(np.sum(probs), 1.0), "Probabilities should sum to 1"


def test_unitary_integration_with_standard_gates():
    """Test unitary integration in circuit with standard gates."""
    c = tq.Circuit(2)
    c.h(0)
    c.cx(0, 1)  # Create Bell state
    
    # Apply custom rotation on qubit 1
    custom_rot = np.array([[np.cos(0.1), -1j*np.sin(0.1)],
                           [-1j*np.sin(0.1), np.cos(0.1)]], dtype=np.complex128)
    c.unitary(1, matrix=custom_rot)
    state = c.state()
    
    # Check state is normalized
    assert np.isclose(np.linalg.norm(state), 1.0), "State should be normalized"
    assert state.shape == (4,), f"Expected shape (4,), got {state.shape}"


def test_chaining_multiple_unitaries():
    """Test chaining multiple unitary operations."""
    c = tq.Circuit(2)
    
    # Hadamard gates on both qubits
    h_gate = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    c.unitary(0, matrix=h_gate)
    c.unitary(1, matrix=h_gate)
    
    # Apply iSWAP
    iswap = np.array([[1, 0, 0, 0],
                      [0, 0, 1j, 0],
                      [0, 1j, 0, 0],
                      [0, 0, 0, 1]], dtype=np.complex128)
    c.unitary(0, 1, matrix=iswap)
    
    state = c.state()
    
    # H⊗H|00⟩ = (|00⟩+|01⟩+|10⟩+|11⟩)/2
    # iSWAP(H⊗H|00⟩) = (|00⟩+i|10⟩+i|01⟩+|11⟩)/2
    expected = np.array([0.5, 0.5j, 0.5j, 0.5], dtype=np.complex128)
    assert np.allclose(state, expected), f"Expected {expected}, got {state}"


def test_unitary_matrix_validation():
    """Test unitary method validates input parameters."""
    c = tq.Circuit(2)
    
    # Test invalid matrix shape for 1-qubit
    with pytest.raises(ValueError, match="Matrix shape.*incompatible"):
        wrong_matrix = np.eye(4)  # 4x4 for 1 qubit
        c.unitary(0, matrix=wrong_matrix)
    
    # Test invalid matrix shape for 2-qubit
    with pytest.raises(ValueError, match="Matrix shape.*incompatible"):
        wrong_matrix = np.eye(2)  # 2x2 for 2 qubits
        c.unitary(0, 1, matrix=wrong_matrix)
    
    # Test invalid qubit index
    with pytest.raises(ValueError, match="Invalid qubit index"):
        c.unitary(5, matrix=np.eye(2))  # Qubit 5 doesn't exist


def test_unitary_preserves_normalization():
    """Test that unitary operations preserve state normalization."""
    c = tq.Circuit(3)
    
    # Apply series of random unitaries
    np.random.seed(42)
    for _ in range(5):
        # Random single-qubit unitary
        q = np.random.randint(0, 3)
        random_u = np.linalg.qr(np.random.randn(2, 2) + 1j*np.random.randn(2, 2))[0]
        c.unitary(q, matrix=random_u)
    
    state = c.state()
    norm = np.linalg.norm(state)
    assert np.isclose(norm, 1.0), f"State norm should be 1.0, got {norm}"


def test_unitary_chem_module_integration():
    """Test that chem module's _apply_kqubit_unitary uses unified implementation."""
    from tyxonq.applications.chem.chem_libs.quantum_chem_library.statevector_ops import _apply_kqubit_unitary
    
    # Test that chem module function works correctly
    n_qubits = 3
    state = np.zeros(2**n_qubits, dtype=np.complex128)
    state[0] = 1.0  # |000⟩
    
    # Apply Hadamard on qubit 0
    h_matrix = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    state_out = _apply_kqubit_unitary(state, h_matrix, [0], n_qubits)
    
    # In LSB-first ordering (axis 0 = qubit 0):
    # H on qubit 0: |000⟩ → (|000⟩ + |100⟩)/√2
    # Binary: |000⟩ = index 0, |100⟩ = index 4 (100 in binary = 4)
    expected = np.zeros(8, dtype=np.complex128)
    expected[0] = 1/np.sqrt(2)  # |000⟩
    expected[4] = 1/np.sqrt(2)  # |100⟩ (bit 2 set, LSB-first means index 4)
    
    assert np.allclose(state_out, expected), f"Expected {expected}, got {state_out}"


if __name__ == "__main__":
    # Allow running as script for quick testing
    pytest.main([__file__, "-v"])
