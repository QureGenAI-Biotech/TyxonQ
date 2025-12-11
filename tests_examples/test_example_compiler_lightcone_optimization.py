"""
Tests for examples/compiler_lightcone_optimization.py

This test suite validates the compiler lightcone optimization example.
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


@pytest.mark.slow
def test_correctness_validation():
    """Test lightcone optimization correctness validation"""
    try:
        module = load_example_module("compiler_lightcone_optimization")
        
        # Run smaller test case
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            module.correctness_validation(n=5, nlayers=2)
        
        output = f.getvalue()
        assert "passed" in output.lower() or "✓" in output
        
    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}")


def test_brickwall_ansatz_construction():
    """Test brickwall ansatz circuit construction"""
    try:
        module = load_example_module("compiler_lightcone_optimization")
        import tyxonq as tq
        
        K = tq.set_backend("numpy")
        n = 4
        nlayers = 2
        
        c = tq.Circuit(n)
        params = K.ones([nlayers, n, 2])
        c = module.brickwall_ansatz(c, params, 'rzz', nlayers)
        
        # Verify circuit has gates
        assert len(c.ops) > 0
        
        # Should have 2-qubit gates and parameterized gates
        gate_types = [op[0] for op in c.ops]
        assert 'rzz' in gate_types
        
    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}")


def test_loss_function_with_lightcone():
    """Test loss function with lightcone enabled"""
    try:
        module = load_example_module("compiler_lightcone_optimization")
        import tyxonq as tq
        
        K = tq.set_backend("numpy")
        n = 4
        nlayers = 2
        params = K.ones([nlayers * n * 2])
        
        # Should run without error
        loss_lc = module.loss_function(params, n, nlayers, enable_lightcone=True)
        loss_no_lc = module.loss_function(params, n, nlayers, enable_lightcone=False)
        
        # Results should be close
        np.testing.assert_allclose(float(loss_lc), float(loss_no_lc), rtol=1e-4)
        
    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}")


@pytest.mark.slow
def test_benchmark_efficiency_runs():
    """Smoke test: benchmark runs without error"""
    try:
        module = load_example_module("compiler_lightcone_optimization")
        
        # This is slow, so we just verify it can start
        # Full benchmark should be run manually
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            # Just run a very small case
            import tyxonq as tq
            K = tq.set_backend("pytorch")
            K.set_dtype("complex64")
            
            vg = K.jit(K.value_and_grad(module.loss_function), static_argnums=(1, 2, 3))
            params = K.ones([2 * 4 * 2])
            _ = vg(params, 4, 2, True)
        
        # Should complete without error
        assert True
        
    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}")
