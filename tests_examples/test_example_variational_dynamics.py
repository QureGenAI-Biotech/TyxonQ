from __future__ import annotations
import importlib.util, sys, pathlib
def test_example_compiles():
    p = pathlib.Path('examples-ng/variational_dynamics.py')
    spec = importlib.util.spec_from_file_location(p.stem, str(p))
    mod = importlib.util.module_from_spec(spec)
    # Only compile module; do not execute top-level code if any heavy run guarded by __main__
    assert spec is not None and spec.loader is not None
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        # Allow examples to require optional deps; mark as xfail-like by asserting message contains common optional hints
        # We still fail fast so breakages are visible; can refine with pytest marks later.
        raise
