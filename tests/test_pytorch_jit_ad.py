"""
Test PyTorch JIT and automatic differentiation capabilities for TyxonQ
"""

import sys
import os
import pytest
import numpy as np
from functools import partial

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tyxonq as tq


@pytest.fixture(scope="function")
def torchb():
    try:
        tq.set_backend("pytorch")
        yield
        tq.set_backend("numpy")
    except ImportError as e:
        print(e)
        tq.set_backend("numpy")
        pytest.skip("****** No torch backend found, skipping test suit *******")


def test_pytorch_jit_basic(torchb):
    """Test basic PyTorch JIT functionality"""
    
    @tq.backend.jit
    def simple_circuit(theta):
        c = tq.Circuit(2)
        c.rx(0, theta=theta)
        c.cnot(0, 1)
        return c.expectation((tq.gates.z(), [0]))
    
    theta = tq.backend.ones([])
    result = simple_circuit(theta)
    assert result is not None
    print(f"JIT result: {result}")


def test_pytorch_jit_with_static_args(torchb):
    """Test PyTorch JIT with static arguments"""
    
    @partial(tq.backend.jit, static_argnums=(1, 2))
    def parameterized_circuit(theta, n_qubits, n_layers):
        c = tq.Circuit(n_qubits)
        for i in range(n_qubits):
            c.h(i)
        for layer in range(n_layers):
            for i in range(n_qubits - 1):
                c.cnot(i, i + 1)
            for i in range(n_qubits):
                c.rx(i, theta=theta[layer, i])
        return c.expectation((tq.gates.z(), [0]))
    
    theta = tq.backend.ones([3, 4])  # 3 layers, 4 qubits
    result = parameterized_circuit(theta, 4, 3)
    assert result is not None
    print(f"JIT with static args result: {result}")


def test_pytorch_jit_compile(torchb):
    """Test PyTorch torch.compile functionality"""
    
    @partial(tq.backend.jit, jit_compile=True)
    def compiled_circuit(theta):
        c = tq.Circuit(2)
        c.rx(0, theta=theta)
        c.ry(1, theta=theta)
        c.cnot(0, 1)
        return c.expectation((tq.gates.z(), [0]))
    
    theta = tq.backend.ones([])
    result = compiled_circuit(theta)
    assert result is not None
    print(f"torch.compile result: {result}")


def test_pytorch_automatic_differentiation(torchb):
    """Test PyTorch automatic differentiation"""
    
    def circuit_function(theta):
        c = tq.Circuit(2)
        c.rx(0, theta=theta)
        c.cnot(0, 1)
        return c.expectation((tq.gates.z(), [0]))
    
    # Test grad
    theta = tq.backend.ones([])
    grad_fn = tq.backend.grad(circuit_function)
    gradient = grad_fn(theta)
    assert gradient is not None
    print(f"Gradient: {gradient}")
    
    # Test value_and_grad
    value_grad_fn = tq.backend.value_and_grad(circuit_function)
    value, grad = value_grad_fn(theta)
    assert value is not None
    assert grad is not None
    print(f"Value: {value}, Gradient: {grad}")


def test_pytorch_jit_with_autodiff(torchb):
    """Test PyTorch JIT combined with automatic differentiation"""
    
    @tq.backend.jit
    def jitted_circuit(theta):
        c = tq.Circuit(2)
        c.rx(0, theta=theta)
        c.ry(1, theta=theta)
        c.cnot(0, 1)
        return c.expectation((tq.gates.z(), [0]))
    
    # JIT the gradient function
    grad_fn = tq.backend.grad(jitted_circuit)
    jitted_grad_fn = tq.backend.jit(grad_fn)
    
    theta = tq.backend.ones([])
    gradient = jitted_grad_fn(theta)
    assert gradient is not None
    print(f"JIT + Autodiff gradient: {gradient}")


def test_pytorch_vmap(torchb):
    """Test PyTorch vmap functionality"""
    
    def single_circuit(theta):
        c = tq.Circuit(2)
        c.rx(0, theta=theta)
        c.cnot(0, 1)
        return c.expectation((tq.gates.z(), [0]))
    
    # Vectorize over theta
    vmap_fn = tq.backend.vmap(single_circuit, vectorized_argnums=0)
    
    thetas = tq.backend.ones([5])  # 5 different theta values
    results = vmap_fn(thetas)
    assert results is not None
    assert len(results) == 5
    print(f"Vmap results: {results}")


def test_pytorch_vmap_with_jit(torchb):
    """Test PyTorch vmap combined with JIT"""
    
    @tq.backend.jit
    def jitted_circuit(theta):
        c = tq.Circuit(2)
        c.rx(0, theta=theta)
        c.cnot(0, 1)
        return c.expectation((tq.gates.z(), [0]))
    
    # Vectorize the JIT-compiled function
    vmap_fn = tq.backend.vmap(jitted_circuit, vectorized_argnums=0)
    
    thetas = tq.backend.ones([3])
    results = vmap_fn(thetas)
    assert results is not None
    assert len(results) == 3
    print(f"Vmap + JIT results: {results}")


def test_pytorch_vectorized_value_and_grad(torchb):
    """Test PyTorch vectorized value and gradient computation"""
    
    def circuit_function(theta):
        c = tq.Circuit(2)
        c.rx(0, theta=theta)
        c.cnot(0, 1)
        return c.expectation((tq.gates.z(), [0]))
    
    # Vectorized value and gradient
    vvag_fn = tq.backend.vectorized_value_and_grad(
        circuit_function, 
        argnums=0, 
        vectorized_argnums=0
    )
    
    thetas = tq.backend.ones([4])
    values, gradients = vvag_fn(thetas)
    assert values is not None
    assert gradients is not None
    assert len(values) == 4
    assert len(gradients) == 4
    print(f"Vectorized values: {values}")
    print(f"Vectorized gradients: {gradients}")


def test_pytorch_optimization_workflow(torchb):
    """Test complete optimization workflow with PyTorch"""
    
    def loss_function(params):
        c = tq.Circuit(2)
        c.rx(0, theta=params[0])
        c.ry(1, theta=params[1])
        c.cnot(0, 1)
        expectation = c.expectation((tq.gates.z(), [0]))
        return expectation
    
    # JIT the loss function
    jitted_loss = tq.backend.jit(loss_function)
    
    # JIT the gradient function
    grad_fn = tq.backend.grad(jitted_loss)
    jitted_grad = tq.backend.jit(grad_fn)
    
    # Initial parameters
    params = tq.backend.ones([2])
    
    # Simple gradient descent
    learning_rate = 0.1
    for step in range(10):
        loss = jitted_loss(params)
        grad = jitted_grad(params)
        params = params - learning_rate * grad
        print(f"Step {step}: Loss = {loss}, Params = {params}")


def test_pytorch_jit_fallback(torchb):
    """Test PyTorch JIT fallback when compilation fails"""
    
    # This function might fail JIT compilation due to complex control flow
    def complex_circuit(theta, condition):
        c = tq.Circuit(2)
        if condition > 0.5:
            c.rx(0, theta=theta)
        else:
            c.ry(0, theta=theta)
        c.cnot(0, 1)
        return c.expectation((tq.gates.z(), [0]))
    
    # Should fallback to original function if JIT fails
    jitted_fn = tq.backend.jit(complex_circuit)
    
    theta = tq.backend.ones([])
    condition = tq.backend.ones([])
    
    result = jitted_fn(theta, condition)
    assert result is not None
    print(f"JIT fallback result: {result}")


if __name__ == "__main__":
    pytest.main([__file__])
