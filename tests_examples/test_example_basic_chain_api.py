"""
Tests for examples/basic_chain_api.py

This test suite validates the basic chain API demonstration example.
"""

import pytest
import sys
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


def test_basic_chain_demo_runs():
    """Smoke test: basic chain demo runs without error"""
    try:
        module = load_example_module("basic_chain_api")
        result = module.demo_basic_chain()
        assert result is not None
        assert isinstance(result, (list, dict))
    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}")


def test_global_defaults_demo_runs():
    """Test global defaults configuration demo"""
    try:
        module = load_example_module("basic_chain_api")
        result = module.demo_global_defaults()
        assert result is not None
    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}")


def test_auto_completion_demo_runs():
    """Test auto-completion mechanism demo"""
    try:
        module = load_example_module("basic_chain_api")
        result = module.demo_auto_completion()
        assert result is not None
    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}")


def test_multi_backend_demo_runs():
    """Test multi-backend support demo"""
    try:
        module = load_example_module("basic_chain_api")
        result = module.demo_multi_backend()
        assert result is not None
    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}")


def test_example_main_runs():
    """Integration test: run full example"""
    try:
        module = load_example_module("basic_chain_api")
        # This will run all demos
        # Just ensure it doesn't crash
        import io
        import contextlib
        
        # Capture stdout to avoid cluttering test output
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            # Run the demos (excluding cloud which requires credentials)
            module.demo_basic_chain()
            module.demo_global_defaults()
            module.demo_auto_completion()
            module.demo_multi_backend()
        
        output = f.getvalue()
        assert len(output) > 0  # Should have printed something
        
    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}")
