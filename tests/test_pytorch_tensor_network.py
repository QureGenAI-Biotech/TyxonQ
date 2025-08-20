#!/usr/bin/env python3
"""
Test script for PyTorch Tensor Network implementation
"""

import sys
import os
sys.path.insert(0, '.')

import torch
import numpy as np
from src.tyxonq.backends.pytorch_tensor_network import (
    TensorNode, TensorEdge, contract_between, contract, copy,
    quantum_expectation, create_quantum_state, create_quantum_gate
)


def test_basic_operations():
    """Test basic tensor network operations"""
    print("Testing basic tensor network operations...")
    
    # Create test tensors
    a = torch.randn(2, 2)
    b = torch.randn(2, 2)
    
    # Create nodes
    node_a = TensorNode(a, name="A")
    node_b = TensorNode(b, name="B")
    
    # Test contraction
    result = contract_between(node_a, node_b)
    expected = torch.matmul(a, b)
    
    assert torch.allclose(result.tensor, expected), "Contraction failed"
    print("✅ Basic contraction test passed")
    
    # Test copy
    node_dict, edge_dict = copy([node_a, node_b])
    assert len(node_dict) == 2, "Copy failed"
    print("✅ Copy test passed")


def test_quantum_operations():
    """Test quantum-specific operations"""
    print("Testing quantum operations...")
    
    # Create quantum state
    state = create_quantum_state((2,), dtype=torch.complex64)
    assert state.tensor[0] == 1.0, "Quantum state creation failed"
    print("✅ Quantum state creation passed")
    
    # Create quantum gate (Hadamard)
    hadamard = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
    gate = create_quantum_gate(hadamard, name="H")
    print("✅ Quantum gate creation passed")
    
    # Test quantum expectation
    # For ground state and identity operator, expectation should be 1
    identity = torch.eye(2, dtype=torch.complex64)
    identity_node = TensorNode(identity, name="I")
    expectation = quantum_expectation(state, identity_node)
    # Convert to scalar for comparison
    expectation_scalar = expectation.item() if expectation.numel() == 1 else expectation.flatten()[0].item()
    assert abs(expectation_scalar - 1.0) < 1e-6, "Quantum expectation failed"
    print("✅ Quantum expectation test passed")


def test_complex_operations():
    """Test complex tensor operations"""
    print("Testing complex operations...")
    
    # Create complex tensors
    a = torch.randn(2, 3, dtype=torch.complex64)
    b = torch.randn(3, 2, dtype=torch.complex64)
    
    node_a = TensorNode(a, name="A")
    node_b = TensorNode(b, name="B")
    
    # Test contraction with complex tensors
    result = contract_between(node_a, node_b)
    expected = torch.matmul(a, b)
    
    assert torch.allclose(result.tensor, expected), "Complex contraction failed"
    print("✅ Complex tensor operations passed")


def test_multiple_contractions():
    """Test contracting multiple nodes"""
    print("Testing multiple contractions...")
    
    # Create multiple nodes
    nodes = []
    for i in range(3):
        tensor = torch.randn(2, 2)
        node = TensorNode(tensor, name=f"Node_{i}")
        nodes.append(node)
    
    # Contract all nodes
    result = contract(nodes)
    
    # Manual contraction for verification
    expected = torch.matmul(torch.matmul(nodes[0].tensor, nodes[1].tensor), nodes[2].tensor)
    
    assert torch.allclose(result.tensor, expected), "Multiple contractions failed"
    print("✅ Multiple contractions test passed")


def main():
    """Run all tests"""
    print("🧪 Testing PyTorch Tensor Network Implementation")
    print("=" * 50)
    
    try:
        test_basic_operations()
        test_quantum_operations()
        test_complex_operations()
        test_multiple_contractions()
        
        print("\n🎉 All tests passed!")
        print("PyTorch Tensor Network implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
