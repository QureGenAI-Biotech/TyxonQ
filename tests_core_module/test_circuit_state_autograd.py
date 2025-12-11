"""Test Circuit.state() PyTorch autograd support.

This test suite verifies that Circuit.state() properly returns backend tensors
that preserve gradients for automatic differentiation, enabling VQE and other
variational quantum algorithms.

Key Design Decisions:
1. **Default behavior**: c.state() returns backend tensor (preserves autograd)
2. **Explicit numpy**: c.state(form="numpy") returns numpy array (breaks autograd)
3. **Legacy compat**: c.state(form="ket") returns backend tensor (same as default)

Architecture:
- Circuit.state() → Engine.state() → kernel functions → backend tensors
- No float() conversions on parameters (preserves gradient chain)
- Bottom-up autograd support throughout the stack
"""

import pytest
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestCircuitStateAutograd:
    """Test autograd support in Circuit.state()."""
    
    def test_default_returns_backend_tensor(self):
        """Test that c.state() returns backend tensor by default."""
        import tyxonq as tq
        
        tq.set_backend("pytorch")
        c = tq.Circuit(2)
        c.h(0).cx(0, 1)
        
        psi = c.state()
        
        assert isinstance(psi, torch.Tensor), "Default should return torch.Tensor"
        assert psi.shape == (4,), f"Wrong shape: {psi.shape}"
        assert psi.dtype == torch.complex128
    
    def test_form_ket_compatibility(self):
        """Test that form='ket' returns backend tensor (legacy compat)."""
        import tyxonq as tq
        
        tq.set_backend("pytorch")
        c = tq.Circuit(2)
        c.h(0).cx(0, 1)
        
        psi_default = c.state()
        psi_ket = c.state(form="ket")
        psi_tensor = c.state(form="tensor")
        
        assert isinstance(psi_ket, torch.Tensor)
        assert isinstance(psi_tensor, torch.Tensor)
        assert torch.allclose(psi_default, psi_ket)
        assert torch.allclose(psi_default, psi_tensor)
    
    def test_form_numpy_conversion(self):
        """Test that form='numpy' returns numpy array."""
        import tyxonq as tq
        
        tq.set_backend("pytorch")
        c = tq.Circuit(2)
        c.h(0).cx(0, 1)
        
        psi_tensor = c.state()
        psi_numpy = c.state(form="numpy")
        
        assert isinstance(psi_numpy, np.ndarray)
        assert psi_numpy.shape == (4,)
        assert np.allclose(psi_numpy, psi_tensor.detach().numpy())
    
    def test_single_parameter_gradient(self):
        """Test gradient computation for single-parameter circuit."""
        import tyxonq as tq
        
        tq.set_backend("pytorch")
        
        theta = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        
        def energy(param):
            c = tq.Circuit(1)
            c.rx(0, param[0])
            psi = c.state()
            
            # <psi|Z|psi>
            Z = torch.tensor([[1+0j, 0+0j], [0+0j, -1+0j]], dtype=torch.complex128)
            psi = psi.to(dtype=torch.complex128)
            return torch.real(torch.conj(psi) @ Z @ psi)
        
        E = energy(theta)
        E.backward()
        
        assert theta.grad is not None, "Gradient should be computed"
        # For |0> -> Rx(0.5)|0>, <Z> = cos(0.5) ≈ 0.877
        expected_energy = np.cos(0.5)
        assert abs(E.item() - expected_energy) < 1e-6
    
    def test_multi_parameter_vqe_optimization(self):
        """Test VQE optimization with multiple parameters."""
        import tyxonq as tq
        
        tq.set_backend("pytorch")
        
        def vqe_energy(params):
            """2-qubit VQE circuit with simplified Hamiltonian."""
            c = tq.Circuit(2)
            c.rx(0, params[0])
            c.ry(1, params[1])
            c.cx(0, 1)
            c.rz(1, params[2])
            
            psi = c.state()
            
            # H = 0.5*Z0 + 0.3*Z1 - 0.2*Z0Z1
            Z = torch.tensor([[1+0j, 0+0j], [0+0j, -1+0j]], dtype=torch.complex128)
            I = torch.eye(2, dtype=torch.complex128)
            
            Z0 = torch.kron(Z, I)
            Z1 = torch.kron(I, Z)
            Z0Z1 = torch.kron(Z, Z)
            
            psi = psi.to(dtype=torch.complex128)
            E0 = torch.real(torch.conj(psi) @ Z0 @ psi)
            E1 = torch.real(torch.conj(psi) @ Z1 @ psi)
            E01 = torch.real(torch.conj(psi) @ Z0Z1 @ psi)
            
            return 0.5 * E0 + 0.3 * E1 - 0.2 * E01
        
        # Initialize and optimize
        params = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([params], lr=0.1)
        
        initial_energy = vqe_energy(params).item()
        
        for _ in range(20):
            opt.zero_grad()
            E = vqe_energy(params)
            E.backward()
            opt.step()
        
        final_energy = vqe_energy(params).item()
        
        # Energy should decrease
        assert final_energy < initial_energy, f"Energy should decrease: {initial_energy:.4f} -> {final_energy:.4f}"
        # Should converge to reasonable value
        assert final_energy < 0.3, f"Energy should converge: {final_energy:.4f}"
    
    def test_batched_parameters(self):
        """Test gradient with batched parameters (batch training)."""
        import tyxonq as tq
        
        tq.set_backend("pytorch")
        
        # Batch of 3 parameter sets
        theta_batch = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64, requires_grad=True)
        
        def batch_energy(params):
            """Compute energy for a batch of parameters."""
            energies = []
            for theta in params:
                c = tq.Circuit(1)
                c.rx(0, theta)
                psi = c.state()
                
                Z = torch.tensor([[1+0j, 0+0j], [0+0j, -1+0j]], dtype=torch.complex128)
                psi = psi.to(dtype=torch.complex128)
                E = torch.real(torch.conj(psi) @ Z @ psi)
                energies.append(E)
            
            return torch.stack(energies)
        
        E_batch = batch_energy(theta_batch)
        loss = E_batch.mean()
        loss.backward()
        
        assert theta_batch.grad is not None
        assert theta_batch.grad.shape == (3,)
        
        # Gradients should be computed (non-zero for non-zero theta)
        # Note: gradient values depend on circuit structure, just verify they exist
        assert theta_batch.grad[0].abs().item() < 1e-6  # θ=0 has zero gradient
        assert theta_batch.grad[1].abs().item() > 1e-6  # θ=0.5 has non-zero gradient
        assert theta_batch.grad[2].abs().item() > 1e-6  # θ=1.0 has non-zero gradient
    
    def test_complex_circuit_autograd(self):
        """Test autograd through complex circuit with multiple gate types."""
        import tyxonq as tq
        
        tq.set_backend("pytorch")
        
        params = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True)
        
        def complex_circuit_energy(p):
            c = tq.Circuit(3)
            c.h(0).h(1).h(2)
            c.rx(0, p[0])
            c.ry(1, p[1])
            c.rz(2, p[2])
            c.cx(0, 1).cx(1, 2)
            
            psi = c.state()
            
            # Simple observable: sum of Z on all qubits
            Z = torch.tensor([[1+0j, 0+0j], [0+0j, -1+0j]], dtype=torch.complex128)
            I = torch.eye(2, dtype=torch.complex128)
            
            # Z0 ⊗ I ⊗ I
            Z0 = torch.kron(torch.kron(Z, I), I)
            # I ⊗ Z1 ⊗ I
            Z1 = torch.kron(torch.kron(I, Z), I)
            # I ⊗ I ⊗ Z2
            Z2 = torch.kron(torch.kron(I, I), Z)
            
            psi = psi.to(dtype=torch.complex128)
            E = torch.real(torch.conj(psi) @ (Z0 + Z1 + Z2) @ psi)
            return E
        
        E = complex_circuit_energy(params)
        E.backward()
        
        assert params.grad is not None
        assert params.grad.shape == (3,)
        # Note: gradients may be zero if circuit structure makes energy insensitive to params
        # The important thing is that autograd works (no errors), not the specific values
    
    def test_numpy_backend_no_autograd(self):
        """Test that numpy backend doesn't have autograd (expected behavior)."""
        import tyxonq as tq
        
        tq.set_backend("numpy")
        
        c = tq.Circuit(2)
        c.h(0).cx(0, 1)
        psi = c.state()
        
        # Should return numpy array with numpy backend
        assert isinstance(psi, np.ndarray)
        assert psi.shape == (4,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
