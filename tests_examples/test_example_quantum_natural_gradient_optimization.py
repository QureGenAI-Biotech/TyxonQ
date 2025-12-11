"""
Tests for examples/quantum_natural_gradient_optimization.py

This test suite validates the quantum natural gradient optimization example.
"""

import pytest
import importlib.util
from pathlib import Path
import numpy as np


def load_example_module(name):
    """Dynamically load example module"""
    example_path = Path(__file__).parent.parent / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, example_path)
    if spec is None or spec.loader is None:
        pytest.skip(f"Example file {example_path} not found")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hamiltonian_construction():
    """Test TFIM Hamiltonian construction"""
    try:
        module = load_example_module("quantum_natural_gradient_optimization")
        
        n = 4
        H = module.build_tfim_hamiltonian(n, J=1.0, h=-1.0)
        
        # Check Hamiltonian is Hermitian
        assert H.shape == (2**n, 2**n)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-10)
        
        # Check Hamiltonian is real (TFIM should be real)
        assert np.allclose(np.imag(H), 0, atol=1e-10)
        
    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}")


def test_ansatz_construction():
    """Test hardware-efficient ansatz circuit construction"""
    try:
        module = load_example_module("quantum_natural_gradient_optimization")
        import tyxonq as tq
        
        tq.set_backend("numpy")
        K = tq.get_backend()
        
        n = 4
        nlayers = 2
        params = K.ones([nlayers, n, 2])
        
        c = module.hardware_efficient_ansatz(n, nlayers, params)
        
        # Verify circuit has gates
        assert len(c.ops) > 0
        
        # Should have Hadamards, CNOTs, and rotations
        gate_types = [op[0] for op in c.ops]
        assert 'h' in gate_types
        assert 'cnot' in gate_types or 'cx' in gate_types
        assert 'ry' in gate_types
        assert 'rz' in gate_types
        
    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}")


def test_state_function():
    """Test state function returns valid state vector"""
    try:
        module = load_example_module("quantum_natural_gradient_optimization")
        import tyxonq as tq
        
        tq.set_backend("numpy")
        K = tq.get_backend()
        
        # Override module constants for test
        old_n, old_l = module.N_QUBITS, module.N_LAYERS
        module.N_QUBITS, module.N_LAYERS = 3, 1
        
        params = K.ones([1, 3, 2])
        state = module.state_function(params)
        
        # Restore
        module.N_QUBITS, module.N_LAYERS = old_n, old_l
        
        # Check state is normalized
        assert state.shape == (2**3,)
        norm = np.linalg.norm(state)
        np.testing.assert_allclose(norm, 1.0, atol=1e-6)
        
    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}")


def test_energy_computation():
    """Test energy computation"""
    try:
        module = load_example_module("quantum_natural_gradient_optimization")
        import tyxonq as tq
        
        tq.set_backend("numpy")
        K = tq.get_backend()
        
        n = 3
        params = K.zeros([1, n, 2])
        H = module.build_tfim_hamiltonian(n)
        
        # Override for test
        old_n, old_l = module.N_QUBITS, module.N_LAYERS
        module.N_QUBITS, module.N_LAYERS = n, 1
        
        energy = module.compute_energy(params, H)
        
        # Restore
        module.N_QUBITS, module.N_LAYERS = old_n, old_l
        
        # Energy should be real
        assert np.isreal(energy) or np.abs(np.imag(energy)) < 1e-10
        
    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}")


@pytest.mark.slow
def test_standard_gradient_descent():
    """Test standard gradient descent runs"""
    try:
        module = load_example_module("quantum_natural_gradient_optimization")
        import tyxonq as tq
        
        tq.set_backend("pytorch")
        
        # Small test case
        n, nlayers = 3, 1
        params_init = np.random.randn(nlayers, n, 2) * 0.1
        H = module.build_tfim_hamiltonian(n)
        
        # Run short optimization
        e_final, _, energies, _ = module.standard_gradient_descent(
            params_init, H, lr=0.05, n_steps=5
        )
        
        # Check energy decreased
        assert energies[0] > energies[-1] or abs(energies[0] - energies[-1]) < 0.1
        
        # Check energy is real
        assert np.isreal(e_final)
        
    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}")


@pytest.mark.slow  
def test_quantum_natural_gradient():
    """Test QNG optimization runs"""
    try:
        module = load_example_module("quantum_natural_gradient_optimization")
        import tyxonq as tq
        
        tq.set_backend("numpy")
        
        # Small test case
        n, nlayers = 3, 1
        params_init = np.random.randn(nlayers, n, 2) * 0.1
        H = module.build_tfim_hamiltonian(n)
        
        # Override module constants
        old_n, old_l = module.N_QUBITS, module.N_LAYERS
        module.N_QUBITS, module.N_LAYERS = n, nlayers
        
        # Run short QNG
        e_final, _, energies, _ = module.quantum_natural_gradient(
            params_init, H, lr=0.05, n_steps=5, reg=1e-3
        )
        
        # Restore
        module.N_QUBITS, module.N_LAYERS = old_n, old_l
        
        # Check energy is real
        assert np.isreal(e_final)
        
        # Check optimization ran
        assert len(energies) == 5
        
    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}")


@pytest.mark.slow
def test_compare_optimizers():
    """Integration test: compare optimizers"""
    try:
        module = load_example_module("quantum_natural_gradient_optimization")
        
        # Override for faster test
        old_n, old_l, old_epochs = module.N_QUBITS, module.N_LAYERS, module.N_EPOCHS
        module.N_QUBITS, module.N_LAYERS, module.N_EPOCHS = 3, 1, 5
        
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            results = module.compare_optimizers()
        
        # Restore
        module.N_QUBITS, module.N_LAYERS, module.N_EPOCHS = old_n, old_l, old_epochs
        
        # Check results structure
        assert 'energies_sgd' in results
        assert 'energies_qng' in results
        assert 'exact' in results
        
        # Check energies are reasonable
        assert len(results['energies_sgd']) == 5
        assert len(results['energies_qng']) == 5
        
    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}")
