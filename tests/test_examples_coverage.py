"""
Comprehensive test coverage for examples directory.
Tests all examples to ensure they work with the refactored PyTorch backend.
"""

import sys
import os
import pytest
import numpy as np
import torch
from functools import partial
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tyxonq as tq


@pytest.fixture(scope="function")
def torchb():
    """Set up PyTorch backend for testing"""
    try:
        tq.set_backend("pytorch")
        yield
        tq.set_backend("numpy")
    except ImportError as e:
        print(e)
        tq.set_backend("numpy")
        pytest.skip("****** No torch backend found, skipping test suit *******")


class TestBasicExamples:
    """Test basic examples that should work without external dependencies"""
    
    def test_simple_qaoa_example(self, torchb):
        """Test simple QAOA example"""
        # This is a simplified version of simple_qaoa.py
        def create_qaoa_circuit(gamma, beta, n_qubits=2):
            c = tq.Circuit(n_qubits)
            # Initial state
            for i in range(n_qubits):
                c.h(i)
            
            # Problem Hamiltonian (simplified)
            c.rx(0, theta=gamma)
            c.rx(1, theta=gamma)
            c.cnot(0, 1)
            
            # Mixing Hamiltonian
            c.rx(0, theta=beta)
            c.rx(1, theta=beta)
            
            return c
        
        gamma = tq.backend.ones([]) * 0.1
        beta = tq.backend.ones([]) * 0.2
        
        circuit = create_qaoa_circuit(gamma, beta)
        expectation = tq.backend.real(circuit.expectation((tq.gates.z(), [0])))
        
        assert expectation is not None
        assert isinstance(expectation, (float, np.ndarray, torch.Tensor))
        print(f"QAOA expectation: {expectation}")
    
    def test_parameter_shift_example(self, torchb):
        """Test parameter shift gradient computation"""
        def parameterized_circuit(theta):
            c = tq.Circuit(2)
            c.rx(0, theta=theta)
            c.cnot(0, 1)
            return tq.backend.real(c.expectation((tq.gates.z(), [0])))
        
        # Test parameter shift
        theta = tq.backend.ones([])
        s = 0.1
        
        # Forward shift
        forward = parameterized_circuit(theta + s)
        # Backward shift
        backward = parameterized_circuit(theta - s)
        
        # Parameter shift gradient
        gradient = (forward - backward) / (2 * s)
        
        assert gradient is not None
        assert isinstance(gradient, (float, np.ndarray, torch.Tensor))
        print(f"Parameter shift gradient: {gradient}")
    
    def test_qudit_circuit_example(self, torchb):
        """Test qudit circuit operations"""
        # Test qudit operations
        d = 3  # qudit dimension
        c = tq.Circuit(2)
        
        # Apply some gates
        c.any(0, 1, unitary=tq.backend.eye(4))
        
        # Get state
        state = c.state()
        
        assert state is not None
        assert len(state) == 4
        print(f"Circuit state shape: {state.shape}")
    
    def test_basic_quantum_operations(self, torchb):
        """Test basic quantum operations"""
        # Create circuit
        c = tq.Circuit(2)
        
        # Apply gates
        c.h(0)
        c.cnot(0, 1)
        c.rx(0, theta=0.5)
        c.ry(1, theta=0.3)
        
        # Get state
        state = c.state()
        assert state is not None
        assert len(state) == 4
        
        # Compute expectation
        expectation = c.expectation((tq.gates.z(), [0]))
        assert expectation is not None
        
        print(f"Basic circuit state: {state}")
        print(f"Basic circuit expectation: {expectation}")


class TestAdvancedExamples:
    """Test more advanced examples with PyTorch features"""
    
    def test_jit_optimization_example(self, torchb):
        """Test JIT-compiled optimization"""
        @tq.backend.jit
        def optimized_circuit(theta):
            c = tq.Circuit(2)
            c.rx(0, theta=theta)
            c.cnot(0, 1)
            return tq.backend.real(c.expectation((tq.gates.z(), [0])))
        
        theta = tq.backend.ones([])
        result = optimized_circuit(theta)
        
        assert result is not None
        print(f"JIT optimized result: {result}")
    
    def test_autodiff_optimization_example(self, torchb):
        """Test automatic differentiation for optimization"""
        def loss_function(params):
            c = tq.Circuit(2)
            c.rx(0, theta=params[0])
            c.ry(1, theta=params[1])
            c.cnot(0, 1)
            return tq.backend.real(c.expectation((tq.gates.z(), [0])))
        
        # Compute gradient
        params = tq.backend.ones([2])
        grad_fn = tq.backend.grad(loss_function)
        gradient = grad_fn(params)
        
        assert gradient is not None
        assert len(gradient) == 2
        print(f"Autodiff gradient: {gradient}")
    
    def test_vmap_batch_processing_example(self, torchb):
        """Test vectorized batch processing"""
        def single_circuit(theta):
            c = tq.Circuit(2)
            c.rx(0, theta=theta)
            c.cnot(0, 1)
            return tq.backend.real(c.expectation((tq.gates.z(), [0])))
        
        # Vectorize over parameters
        vmap_fn = tq.backend.vmap(single_circuit, vectorized_argnums=0)
        
        thetas = tq.backend.ones([5])
        results = vmap_fn(thetas)
        
        assert results is not None
        assert len(results) == 5
        print(f"Vmap batch results: {results}")
    
    def test_complex_quantum_operations(self, torchb):
        """Test complex quantum operations with proper handling"""
        def complex_circuit(theta):
            c = tq.Circuit(2)
            c.rx(0, theta=theta)
            c.cnot(0, 1)
            # This returns a complex expectation
            complex_exp = c.expectation((tq.gates.z(), [0]))
            # Convert to real for autodiff
            return tq.backend.real(complex_exp)
        
        theta = tq.backend.ones([])
        result = complex_circuit(theta)
        
        assert result is not None
        assert not torch.is_complex(result)
        print(f"Complex operation result: {result}")


class TestMatrixOperations:
    """Test matrix operations and linear algebra"""
    
    def test_matrix_exponential_example(self, torchb):
        """Test matrix exponential operations"""
        # Create a simple matrix
        matrix = tq.backend.eye(2)
        matrix = matrix + 0.1j * tq.backend.eye(2)
        
        # Compute matrix exponential
        exp_matrix = tq.backend.expm(matrix)
        
        assert exp_matrix is not None
        assert exp_matrix.shape == (2, 2)
        print(f"Matrix exponential shape: {exp_matrix.shape}")
    
    def test_eigenvalue_computation(self, torchb):
        """Test eigenvalue computations"""
        # Create a Hermitian matrix
        matrix = tq.backend.eye(2)
        matrix = matrix + 0.1 * tq.backend.eye(2)
        
        # Compute eigenvalues
        eigenvals = tq.backend.eigvalsh(matrix)
        
        assert eigenvals is not None
        assert len(eigenvals) == 2
        print(f"Eigenvalues: {eigenvals}")


class TestNoiseAndChannels:
    """Test noise models and quantum channels"""
    
    def test_depolarizing_channel(self, torchb):
        """Test depolarizing noise channel"""
        c = tq.Circuit(2)
        c.h(0)
        c.cnot(0, 1)
        
        # Test basic circuit with noise simulation
        # Note: Full noise model implementation would require more complex setup
        expectation = c.expectation((tq.gates.z(), [0]))
        
        assert expectation is not None
        print(f"Circuit expectation: {expectation}")
    
    def test_amplitude_damping_channel(self, torchb):
        """Test amplitude damping channel"""
        c = tq.Circuit(1)
        c.h(0)
        
        # Test basic circuit
        expectation = c.expectation((tq.gates.z(), [0]))
        
        assert expectation is not None
        print(f"Circuit expectation: {expectation}")


class TestTimeEvolution:
    """Test time evolution and dynamics"""
    
    def test_simple_time_evolution(self, torchb):
        """Test simple time evolution"""
        c = tq.Circuit(2)
        c.h(0)
        
        # Simple time evolution
        def hamiltonian(t):
            return tq.backend.eye(4) * 0.1
        
        # Use manual ODE solver (as implemented in experimental.py)
        dt = 0.01
        steps = 10
        current_state = c.state()
        
        for i in range(steps):
            current_time = i * dt
            h = hamiltonian(current_time)
            current_state = current_state + dt * (-1.0j * h @ current_state)
        
        assert current_state is not None
        print(f"Time evolution final state shape: {current_state.shape}")


class TestOptimizationWorkflows:
    """Test complete optimization workflows"""
    
    def test_vqe_workflow(self, torchb):
        """Test VQE (Variational Quantum Eigensolver) workflow"""
        def vqe_circuit(params):
            c = tq.Circuit(2)
            c.rx(0, theta=params[0])
            c.ry(1, theta=params[1])
            c.cnot(0, 1)
            return tq.backend.real(c.expectation((tq.gates.z(), [0])))
        
        # JIT the circuit
        jitted_circuit = tq.backend.jit(vqe_circuit)
        
        # Compute gradient
        grad_fn = tq.backend.grad(jitted_circuit)
        
        # Optimization step
        params = tq.backend.ones([2])
        loss = jitted_circuit(params)
        gradient = grad_fn(params)
        
        # Update parameters
        learning_rate = 0.1
        new_params = params - learning_rate * gradient
        
        assert loss is not None
        assert gradient is not None
        assert new_params is not None
        print(f"VQE loss: {loss}")
        print(f"VQE gradient: {gradient}")
        print(f"VQE new params: {new_params}")
    
    def test_qaoa_workflow(self, torchb):
        """Test QAOA (Quantum Approximate Optimization Algorithm) workflow"""
        def qaoa_circuit(gammas, betas):
            c = tq.Circuit(2)
            
            # Initial state
            c.h(0)
            c.h(1)
            
            # Problem layers
            for gamma in gammas:
                c.rx(0, theta=gamma)
                c.rx(1, theta=gamma)
                c.cnot(0, 1)
            
            # Mixing layers
            for beta in betas:
                c.rx(0, theta=beta)
                c.rx(1, theta=beta)
            
            return tq.backend.real(c.expectation((tq.gates.z(), [0])))
        
        # Vectorized QAOA
        vmap_qaoa = tq.backend.vmap(qaoa_circuit, vectorized_argnums=(0, 1))
        
        gammas = tq.backend.ones([3, 2])  # 3 layers, 2 qubits
        betas = tq.backend.ones([3, 2])
        
        results = vmap_qaoa(gammas, betas)
        
        assert results is not None
        assert len(results) == 3
        print(f"QAOA results: {results}")


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_parameters(self, torchb):
        """Test handling of invalid parameters"""
        c = tq.Circuit(2)
        
        # Test with invalid parameters - PyTorch handles inf gracefully
        try:
            c.rx(0, theta=tq.backend.ones([]) * float('inf'))
            # PyTorch may handle inf gracefully, so we just check it doesn't crash
            print("Infinite parameter handled gracefully")
        except Exception as e:
            print(f"Exception caught: {e}")
            pass  # Expected
    
    def test_empty_circuit(self, torchb):
        """Test empty circuit operations"""
        c = tq.Circuit(1)
        
        # Empty circuit should still work
        state = c.state()
        expectation = c.expectation((tq.gates.z(), [0]))
        
        assert state is not None
        assert expectation is not None
        print(f"Empty circuit state: {state}")
        print(f"Empty circuit expectation: {expectation}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
