"""Test Circuit.expectation() method for Pauli observables."""

import pytest
import numpy as np
import tyxonq as tq
from tyxonq.libs.quantum_library.kernels.gates import gate_x, gate_y, gate_z


def test_expectation_single_qubit_x():
    """Test expectation of X on |+⟩ state."""
    tq.set_backend("numpy")
    c = tq.Circuit(1)
    c.h(0)  # |+⟩ state
    
    exp_x = c.expectation((gate_x(), [0]))
    assert np.isclose(exp_x, 1.0, atol=1e-10), f"Expected 1.0, got {exp_x}"


def test_expectation_single_qubit_z():
    """Test expectation of Z on |0⟩ state."""
    tq.set_backend("numpy")
    c = tq.Circuit(1)  # |0⟩ state
    
    exp_z = c.expectation((gate_z(), [0]))
    assert np.isclose(exp_z, 1.0, atol=1e-10), f"Expected 1.0, got {exp_z}"


def test_expectation_two_qubit_zz_bell():
    """Test expectation of Z⊗Z on Bell state."""
    tq.set_backend("numpy")
    c = tq.Circuit(2)
    c.h(0).cx(0, 1)  # |Φ+⟩ = (|00⟩ + |11⟩)/√2
    
    exp_zz = c.expectation((gate_z(), [0]), (gate_z(), [1]))
    # For |Φ+⟩: ⟨Z_0 Z_1⟩ = 1 (both |00⟩ and |11⟩ have same parity)
    assert np.isclose(exp_zz, 1.0, atol=1e-10), f"Expected 1.0, got {exp_zz}"


def test_expectation_two_qubit_xx_bell():
    """Test expectation of X⊗X on Bell state."""
    tq.set_backend("numpy")
    c = tq.Circuit(2)
    c.h(0).cx(0, 1)  # |Φ+⟩
    
    exp_xx = c.expectation((gate_x(), [0]), (gate_x(), [1]))
    # For |Φ+⟩: ⟨X_0 X_1⟩ = 1
    assert np.isclose(exp_xx, 1.0, atol=1e-10), f"Expected 1.0, got {exp_xx}"


def test_expectation_tfi_energy():
    """Test TFI Hamiltonian energy calculation."""
    tq.set_backend("numpy")
    n = 3
    c = tq.Circuit(n)
    
    # Simple product state |+++⟩
    for i in range(n):
        c.h(i)
    
    # H = -Σ X_i + Σ Z_i Z_{i+1}
    # For |+++⟩: ⟨X_i⟩ = 1, ⟨Z_i Z_{i+1}⟩ = 0
    energy = 0.0
    for i in range(n):
        energy -= c.expectation((gate_x(), [i]))
    for i in range(n - 1):
        energy += c.expectation((gate_z(), [i]), (gate_z(), [i + 1]))
    
    expected = -3.0  # -1 * 3 qubits in X basis
    assert np.isclose(energy, expected, atol=1e-10), f"Expected {expected}, got {energy}"


def test_expectation_with_mps_simulator():
    """Test expectation works with MPS simulator backend."""
    tq.set_backend("numpy")
    n = 4
    
    # Configure MPS simulator via device()
    c = tq.Circuit(n)
    c.device(provider="simulator", device="matrix_product_state", max_bond=16)
    
    # Create simple state
    for i in range(n):
        c.h(i)
    
    # Test single-qubit expectation
    exp_x = c.expectation((gate_x(), [0]))
    assert np.isclose(exp_x, 1.0, atol=1e-10), f"Expected 1.0, got {exp_x}"
    
    # Test two-qubit expectation
    exp_zz = c.expectation((gate_z(), [0]), (gate_z(), [1]))
    assert np.isclose(exp_zz, 0.0, atol=1e-10), f"Expected 0.0, got {exp_zz}"  # Independent qubits in |+⟩


def test_state_auto_selects_mps_engine():
    """Test that state() automatically uses MPS engine when device is configured."""
    tq.set_backend("numpy")
    n = 3
    
    c = tq.Circuit(n)
    c.device(provider="simulator", device="matrix_product_state", max_bond=8)
    c.h(0).cx(0, 1).cx(1, 2)
    
    # state() should automatically use MPS engine
    psi = c.state()
    assert psi.shape == (2**n,)
    
    # Compare with explicit statevector
    c2 = tq.Circuit(n)
    c2.h(0).cx(0, 1).cx(1, 2)
    psi2 = c2.state(engine="statevector")
    
    np.testing.assert_allclose(psi, psi2, atol=1e-10)


def test_state_explicit_engine_override():
    """Test that explicit engine parameter overrides device config."""
    tq.set_backend("numpy")
    n = 2
    
    c = tq.Circuit(n)
    c.device(provider="simulator", device="matrix_product_state")  # MPS config
    c.h(0).cx(0, 1)
    
    # Explicitly request statevector
    psi_sv = c.state(engine="statevector")
    assert psi_sv.shape == (4,)
    
    # Explicitly request MPS
    psi_mps = c.state(engine="mps")
    assert psi_mps.shape == (4,)
    
    # Both should give same result
    np.testing.assert_allclose(psi_sv, psi_mps, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
