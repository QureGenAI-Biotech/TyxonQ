from __future__ import annotations
import importlib.util, pathlib


def test_example_compiles():
    p = pathlib.Path('examples/circuit_chain_demo.py')
    spec = importlib.util.spec_from_file_location(p.stem, str(p))
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)

