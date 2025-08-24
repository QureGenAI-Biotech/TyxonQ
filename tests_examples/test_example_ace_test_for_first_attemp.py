from __future__ import annotations
import importlib.util, pathlib

def test_example_runs_on_simulator():
    p = pathlib.Path('examples-ng/ace_test_for_first_attemp.py')
    spec = importlib.util.spec_from_file_location(p.stem, str(p))
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)

    # Force simulator provider/device for offline test
    setattr(mod, "provider", "simulator")
    setattr(mod, "device", "statevector")

    # Should not raise
    mod.quantum_hello_world()
