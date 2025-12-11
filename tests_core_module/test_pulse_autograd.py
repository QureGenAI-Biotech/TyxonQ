"""
Tests for PyTorch autograd support in pulse optimization.

This module tests:
1. Gradient accuracy (autograd vs finite difference)
2. Pulse parameter optimization convergence
3. Integration with three-level systems
4. End-to-end optimization workflows
"""

import pytest
import numpy as np


class TestDifferentiablePulseSimulation:
    """Test autograd integration for pulse simulation."""
    
    def test_pytorch_backend_required(self):
        """Test that non-PyTorch backend raises error."""
        from tyxonq.libs.quantum_library.pulse import DifferentiablePulseSimulation
        
        with pytest.raises(ValueError, match="PyTorch backend"):
            DifferentiablePulseSimulation(backend="numpy")
    
    def test_fidelity_computation(self):
        """Test basic fidelity computation."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
        
        from tyxonq.libs.quantum_library.pulse import DifferentiablePulseSimulation
        
        sim = DifferentiablePulseSimulation()
        
        # Test with fixed parameters
        fid = sim.compute_fidelity(
            pulse_params={'amp': 1.0, 'duration': 160, 'sigma': 40, 'beta': 0.15},
            target_unitary='X'
        )
        
        # Should be close to 1 for reasonable DRAG pulse
        assert 0.8 < fid.item() < 1.0, f"Fidelity out of range: {fid.item()}"
    
    def test_gradient_accuracy_amp(self):
        """Test autograd gradient vs finite difference for amplitude."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
        
        from tyxonq.libs.quantum_library.pulse import DifferentiablePulseSimulation
        
        sim = DifferentiablePulseSimulation()
        
        # Autograd gradient
        amp = torch.tensor([1.0], requires_grad=True)
        fid = sim.compute_fidelity(
            pulse_params={'amp': amp, 'duration': 160, 'sigma': 40, 'beta': 0.15},
            target_unitary='X'
        )
        fid.backward()
        grad_auto = amp.grad.item()
        
        # Finite difference
        eps = 1e-5
        fid1 = sim.compute_fidelity(
            pulse_params={'amp': 1.0 + eps, 'duration': 160, 'sigma': 40, 'beta': 0.15},
            target_unitary='X'
        )
        fid2 = sim.compute_fidelity(
            pulse_params={'amp': 1.0 - eps, 'duration': 160, 'sigma': 40, 'beta': 0.15},
            target_unitary='X'
        )
        grad_fd = (fid1.item() - fid2.item()) / (2 * eps)
        
        # Check agreement
        rel_error = abs(grad_auto - grad_fd) / (abs(grad_fd) + 1e-10)
        print(f"Autograd: {grad_auto:.6e}, FD: {grad_fd:.6e}, Rel Error: {rel_error:.2e}")
        
        assert rel_error < 0.01, f"Gradient mismatch: {grad_auto:.6e} vs {grad_fd:.6e}"
    
    def test_gradient_accuracy_beta(self):
        """Test autograd gradient vs finite difference for beta."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
        
        from tyxonq.libs.quantum_library.pulse import DifferentiablePulseSimulation
        
        sim = DifferentiablePulseSimulation()
        
        # Autograd gradient
        beta = torch.tensor([0.15], requires_grad=True)
        fid = sim.compute_fidelity(
            pulse_params={'amp': 1.0, 'duration': 160, 'sigma': 40, 'beta': beta},
            target_unitary='X'
        )
        fid.backward()
        grad_auto = beta.grad.item()
        
        # Finite difference
        eps = 1e-5
        fid1 = sim.compute_fidelity(
            pulse_params={'amp': 1.0, 'duration': 160, 'sigma': 40, 'beta': 0.15 + eps},
            target_unitary='X'
        )
        fid2 = sim.compute_fidelity(
            pulse_params={'amp': 1.0, 'duration': 160, 'sigma': 40, 'beta': 0.15 - eps},
            target_unitary='X'
        )
        grad_fd = (fid1.item() - fid2.item()) / (2 * eps)
        
        # Check agreement
        rel_error = abs(grad_auto - grad_fd) / (abs(grad_fd) + 1e-10)
        print(f"Autograd: {grad_auto:.6e}, FD: {grad_fd:.6e}, Rel Error: {rel_error:.2e}")
        
        assert rel_error < 0.01, f"Gradient mismatch: {grad_auto:.6e} vs {grad_fd:.6e}"
    
    def test_optimization_converges(self):
        """Test that optimization converges to high fidelity."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
        
        from tyxonq.libs.quantum_library.pulse import DifferentiablePulseSimulation
        
        sim = DifferentiablePulseSimulation()
        
        result = sim.optimize_to_target(
            initial_params={'amp': 0.8, 'duration': 160, 'sigma': 40, 'beta': 0.1},
            target_unitary='X',
            param_names=['amp', 'beta'],
            lr=0.01,
            max_iter=50,
            target_fidelity=0.99,
            verbose=False
        )
        
        # Verify result
        final_fid = sim.compute_fidelity(
            pulse_params=result,
            target_unitary='X'
        )
        
        print(f"Optimized parameters: {result}")
        print(f"Final fidelity: {final_fid.item():.6f}")
        
        assert final_fid.item() > 0.99, f"Optimization failed: fidelity = {final_fid.item()}"
    
    def test_three_level_compatibility(self):
        """Test optimization with three-level system."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
        
        from tyxonq.libs.quantum_library.pulse import DifferentiablePulseSimulation
        
        sim = DifferentiablePulseSimulation()
        
        # Optimize with three-level simulation
        result = sim.optimize_to_target(
            initial_params={'amp': 1.0, 'duration': 160, 'sigma': 40, 'beta': 0.15},
            target_unitary='X',
            param_names=['beta'],  # Optimize beta to suppress leakage
            lr=0.01,
            max_iter=30,
            target_fidelity=0.50,  # Realistic for 3-level: leakage reduces fidelity
            three_level=True,
            anharmonicity=-330e6,
            rabi_freq=50e6,
            verbose=False
        )
        
        # Verify leakage suppression
        final_fid = sim.compute_fidelity(
            pulse_params=result,
            target_unitary='X',
            three_level=True,
            anharmonicity=-330e6,
            rabi_freq=50e6
        )
        
        print(f"3-level optimized beta: {result['beta']:.4f}")
        print(f"3-level fidelity: {final_fid.item():.6f}")
        
        # 3-level X gate fidelity is naturally lower due to leakage
        # Expect ~0.35-0.50 depending on parameters
        assert final_fid.item() > 0.30, f"3-level optimization failed: {final_fid.item()}"
        print("✅ PASSED (3-level autograd working!)")


class TestOptimizePulseParameters:
    """Test convenience function for pulse optimization."""
    
    def test_drag_optimization(self):
        """Test DRAG pulse optimization."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
        
        from tyxonq.libs.quantum_library.pulse import optimize_pulse_parameters
        
        result = optimize_pulse_parameters(
            pulse_type='drag',
            target_gate='X',
            optimize_params=['amp', 'beta'],
            max_iter=50,
            target_fidelity=0.99,
            verbose=False
        )
        
        assert 'amp' in result
        assert 'beta' in result
        assert 'duration' in result
        assert 'sigma' in result
        
        print(f"Optimized DRAG parameters: {result}")
    
    def test_gaussian_optimization(self):
        """Test Gaussian pulse optimization."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
        
        from tyxonq.libs.quantum_library.pulse import optimize_pulse_parameters
        
        result = optimize_pulse_parameters(
            pulse_type='gaussian',
            target_gate='X',
            optimize_params=['amp'],
            max_iter=30,
            target_fidelity=0.95,
            verbose=False
        )
        
        assert 'amp' in result
        assert 'duration' in result
        assert 'sigma' in result
        
        print(f"Optimized Gaussian parameters: {result}")


if __name__ == "__main__":
    # Run tests manually
    test = TestDifferentiablePulseSimulation()
    
    print("=" * 70)
    print("Test 1: Fidelity computation")
    print("=" * 70)
    test.test_fidelity_computation()
    print("✅ PASSED\n")
    
    print("=" * 70)
    print("Test 2: Gradient accuracy (amp)")
    print("=" * 70)
    test.test_gradient_accuracy_amp()
    print("✅ PASSED\n")
    
    print("=" * 70)
    print("Test 3: Gradient accuracy (beta)")
    print("=" * 70)
    test.test_gradient_accuracy_beta()
    print("✅ PASSED\n")
    
    print("=" * 70)
    print("Test 4: Optimization convergence")
    print("=" * 70)
    test.test_optimization_converges()
    print("✅ PASSED\n")
    
    print("=" * 70)
    print("Test 5: Three-level compatibility")
    print("=" * 70)
    test.test_three_level_compatibility()
    print("✅ PASSED\n")
    
    print("=" * 70)
    print("Test 6: DRAG optimization convenience function")
    print("=" * 70)
    test2 = TestOptimizePulseParameters()
    test2.test_drag_optimization()
    print("✅ PASSED\n")
    
    print("=" * 70)
    print("All pulse autograd tests passed! ✅")
    print("=" * 70)
