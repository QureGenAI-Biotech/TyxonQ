#!/usr/bin/env python3
"""
Unified test script for tensornetwork replacement
Tests equivalence between original tensornetwork and our PyTorch implementation
for both src directory usage and examples directory usage
"""

import sys
import os
sys.path.insert(0, '.')

import torch
import numpy as np
from typing import List, Tuple, Any


def test_our_implementation():
    """Test our PyTorch implementation"""
    print("Testing our PyTorch implementation...")
    
    try:
        # Import our implementation
        sys.path.insert(0, 'src/tyxonq/backends')
        from pytorch_tensor_network import tn as tn_ours
        
        # Test 1: Node creation
        tensor = torch.randn(2, 2)
        node_ours = tn_ours.Node(tensor, name="test_node")
        assert node_ours.name == "test_node"
        assert torch.allclose(node_ours.tensor, tensor)
        
        # Test 2: Copy functionality
        node_dict, edge_dict = tn_ours.copy([node_ours], conjugate=False)
        assert len(node_dict) == 1
        assert id(node_ours) in node_dict
        
        # Test 3: Contract_between
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)
        node_a = tn_ours.Node(a, name="A")
        node_b = tn_ours.Node(b, name="B")
        result = tn_ours.contract_between(node_a, node_b)
        expected = torch.matmul(a, b)
        assert torch.allclose(result.tensor, expected)
        
        # Test 4: Quantum operations
        hadamard = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
        gate_node = tn_ours.Node(hadamard, name="H")
        
        state = torch.tensor([1.0, 0.0], dtype=torch.complex64)
        state_node = tn_ours.Node(state, name="state")
        
        result = tn_ours.contract_between(state_node, gate_node, allow_outer_product=True)
        expected = torch.matmul(hadamard, state)
        assert torch.allclose(result.tensor, expected)
        
        print("✅ Our PyTorch implementation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Our PyTorch implementation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_equivalence():
    """Test equivalence between original and our implementation"""
    print("Testing equivalence between implementations...")
    
    try:
        import tensornetwork as tn_original
        
        # Import our implementation
        sys.path.insert(0, 'src/tyxonq/backends')
        from pytorch_tensor_network import tn as tn_ours
        
        # Test 1: Basic node creation equivalence
        tensor = torch.randn(2, 2)
        node_orig = tn_original.Node(tensor.numpy(), name="test")
        node_ours = tn_ours.Node(tensor, name="test")
        
        assert node_orig.name == node_ours.name
        assert np.allclose(node_orig.tensor, node_ours.tensor.numpy())
        
        # Test 2: Copy functionality equivalence
        node_dict_orig, edge_dict_orig = tn_original.copy([node_orig], conjugate=False)
        node_dict_ours, edge_dict_ours = tn_ours.copy([node_ours], conjugate=False)
        
        assert len(node_dict_orig) == len(node_dict_ours)
        
        # Test 3: Contract_between equivalence
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)
        
        node_a_orig = tn_original.Node(a.numpy(), name="A")
        node_b_orig = tn_original.Node(b.numpy(), name="B")
        node_a_orig[1] ^ node_b_orig[0]
        result_orig = tn_original.contract_between(node_a_orig, node_b_orig)
        
        node_a_ours = tn_ours.Node(a, name="A")
        node_b_ours = tn_ours.Node(b, name="B")
        result_ours = tn_ours.contract_between(node_a_ours, node_b_ours)
        
        assert np.allclose(result_orig.tensor, result_ours.tensor.numpy())
        
        print("✅ Equivalence test passed")
        return True
        
    except Exception as e:
        print(f"❌ Equivalence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration with actual src code patterns"""
    print("Testing integration with src code patterns...")
    
    try:
        # Import our implementation
        sys.path.insert(0, 'src/tyxonq/backends')
        from pytorch_tensor_network import tn
        
        # Simulate circuit.py pattern: create gate nodes
        gate_tensor = torch.eye(2, dtype=torch.complex64)
        gate_node = tn.Node(gate_tensor, name="gate")
        
        # Simulate circuit.py pattern: create measurement nodes
        measurement_tensor = torch.randn(2, 2)
        measurement_node = tn.Node(measurement_tensor, name="measurement")
        
        # Simulate circuit.py pattern: copy nodes
        node_dict, edge_dict = tn.copy([gate_node, measurement_node], conjugate=False)
        assert len(node_dict) == 2
        
        # Simulate cons.py pattern: contract nodes
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)
        node_a = tn.Node(a, name="A")
        node_b = tn.Node(b, name="B")
        result = tn.contract_between(node_a, node_b)
        expected = torch.matmul(a, b)
        assert torch.allclose(result.tensor, expected)
        
        # Simulate gates.py pattern: quantum gate operations
        hadamard = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
        gate = tn.Node(hadamard, name="H")
        assert gate.tensor.shape == (2, 2)
        
        print("✅ Integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_examples_compatibility():
    """Test compatibility with examples usage patterns"""
    print("Testing examples compatibility...")
    
    try:
        # Import our implementation
        sys.path.insert(0, 'src/tyxonq/backends')
        from pytorch_tensor_network import tn, create_finite_tfi_mpo
        
        # Test MPO creation (used in examples)
        Jx = torch.tensor([1.0, 1.0, 1.0])
        Bz = torch.tensor([-1.0, -1.0, -1.0, -1.0])
        
        # Create MPO cores
        cores = create_finite_tfi_mpo(Jx, Bz, dtype=torch.complex64)
        assert len(cores) == 4  # 4 sites
        
        # Test that we can contract them
        result = tn.contract(cores)
        assert result.tensor.shape == (1, 2, 2, 2, 2, 1)
        
        print("✅ Examples compatibility test passed")
        return True
        
    except Exception as e:
        print(f"❌ Examples compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all unified tests"""
    print("🧪 Unified TensorNetwork Replacement Testing")
    print("=" * 50)
    
    tests = [
        test_our_implementation,
        test_equivalence,
        test_integration,
        test_examples_compatibility,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All unified tests passed!")
        print("✅ TensorNetwork replacement is working correctly!")
        print("✅ Ready to replace tensornetwork in src and examples!")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
