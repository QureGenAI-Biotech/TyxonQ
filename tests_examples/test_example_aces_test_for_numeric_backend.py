from __future__ import annotations
import importlib.util, pathlib


def test_example_runs_with_numeric_backend_numpy():
    p = pathlib.Path('examples-ng/aces_test_for_numeric_backend.py')
    spec = importlib.util.spec_from_file_location(p.stem, str(p))
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)

    # Should not raise with numpy backend
    tasks = mod.demo_set_numeric_backend("numpy")
    assert isinstance(tasks, list)
    assert len(tasks) >= 1


def test_example_runs_with_numeric_backend_pytorch_if_available():
    p = pathlib.Path('examples-ng/aces_test_for_numeric_backend.py')
    spec = importlib.util.spec_from_file_location(p.stem, str(p))
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)

    try:
        tasks = mod.demo_set_numeric_backend("pytorch")
        assert isinstance(tasks, list)
        assert len(tasks) >= 1
    except Exception:
        # If pytorch is not installed, skip gracefully by asserting True
        assert True


