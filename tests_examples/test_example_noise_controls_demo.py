from __future__ import annotations
import importlib.util, pathlib


def test_example_runs_with_noise_controls():
    p = pathlib.Path('examples/noise_controls_demo.py')
    spec = importlib.util.spec_from_file_location(p.stem, str(p))
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)

    # Ensure top-level API exists
    assert hasattr(mod.tq, 'enable_noise')
    assert hasattr(mod.tq, 'is_noise_enabled')
    assert hasattr(mod.tq, 'get_noise_config')

    # Switch to local simulator for offline testing
    mod.tq.enable_noise(True, {"type": "depolarizing", "p": 0.0})
    # Should not raise
    mod.demo_noise_controls()


