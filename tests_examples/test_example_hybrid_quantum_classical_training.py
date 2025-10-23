"""
Tests for examples/hybrid_quantum_classical_training.py

This test suite validates the hybrid quantum-classical training example.
"""

import pytest
import importlib.util
from pathlib import Path


def load_example_module(name):
    """Dynamically load example module"""
    example_path = Path(__file__).parent.parent / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, example_path)
    if spec is None or spec.loader is None:
        pytest.skip(f"Example file {example_path} not found")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quantum_circuit_forward():
    """Test quantum circuit forward function"""
    try:
        module = load_example_module("hybrid_quantum_classical_training")
        import torch
        import tyxonq as tq
        
        tq.set_backend("pytorch")
        
        # Create dummy inputs
        x = torch.randn(9, dtype=torch.float32)
        weights = torch.randn(2 * 2, 9, dtype=torch.float32)  # 2 layers, 9 qubits
        
        # Run forward pass
        output = module.quantum_circuit_forward(x, weights)
        
        # Check output shape
        assert output.shape == (9,)
        
        # Check output is real (expectations should be real)
        assert torch.all(torch.abs(torch.imag(output)) < 1e-6)
        
    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}")


def test_model_construction():
    """Test hybrid model can be constructed"""
    try:
        module = load_example_module("hybrid_quantum_classical_training")
        import torch
        
        model = module.HybridQuantumClassicalModel(n_qubits=9, n_layers=2)
        
        # Check model has expected components
        assert hasattr(model, 'quantum_layer')
        assert hasattr(model, 'fc')
        assert hasattr(model, 'sigmoid')
        
    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}")


def test_model_forward_pass():
    """Test model forward pass"""
    try:
        module = load_example_module("hybrid_quantum_classical_training")
        import torch
        
        model = module.HybridQuantumClassicalModel(n_qubits=9, n_layers=2)
        
        # Create batch input
        batch_size = 4
        x = torch.randn(batch_size, 9, dtype=torch.float32)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 1)
        
        # Check output is in [0, 1] (sigmoid output)
        assert torch.all(output >= 0) and torch.all(output <= 1)
        
    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}")


@pytest.mark.slow
def test_load_and_preprocess_mnist():
    """Test MNIST data loading and preprocessing"""
    try:
        module = load_example_module("hybrid_quantum_classical_training")
        
        # This downloads MNIST, so mark as slow test
        x_train, y_train = module.load_and_preprocess_mnist()
        
        # Check data shapes
        assert x_train.shape[1] == 9  # 3x3 = 9 features
        assert len(y_train) == len(x_train)
        
        # Check labels are binary
        import torch
        unique_labels = torch.unique(y_train)
        assert len(unique_labels) <= 2
        assert torch.all((y_train == 0) | (y_train == 1))
        
    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}")


@pytest.mark.slow
@pytest.mark.skipif(not __import__('torch').cuda.is_available(), 
                    reason="GPU not available")
def test_training_on_gpu():
    """Test training loop on GPU (if available)"""
    try:
        module = load_example_module("hybrid_quantum_classical_training")
        import torch
        
        # Load small dataset
        x_train, y_train = module.load_and_preprocess_mnist()
        
        # Run very short training (1 epoch)
        import io
        import contextlib
        
        # Temporarily override N_EPOCHS
        old_epochs = module.N_EPOCHS
        module.N_EPOCHS = 1
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            model, losses, times = module.train_hybrid_model(x_train, y_train)
        
        module.N_EPOCHS = old_epochs
        
        # Check training ran
        assert len(losses) == 1
        assert len(times) == 1
        
    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}")


@pytest.mark.slow
def test_full_example_main():
    """Integration test: run full example with short training"""
    try:
        module = load_example_module("hybrid_quantum_classical_training")
        
        # Temporarily reduce training epochs for test
        old_epochs = module.N_EPOCHS
        old_samples = module.N_TRAIN_SAMPLES
        module.N_EPOCHS = 2
        module.N_TRAIN_SAMPLES = 50
        
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            module.main()
        
        # Restore original values
        module.N_EPOCHS = old_epochs
        module.N_TRAIN_SAMPLES = old_samples
        
        output = f.getvalue()
        assert "Training completed" in output or "completed successfully" in output
        
    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}")
