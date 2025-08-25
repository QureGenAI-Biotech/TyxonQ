from __future__ import annotations

from tyxonq.numerics import get_backend, set_backend, use_backend


def test_set_backend_by_name_affects_get_backend_none():
    set_backend("numpy")
    b = get_backend(None)
    assert getattr(b, "name", "") == "numpy"


def test_use_backend_context_manager_restores_previous():
    set_backend("numpy")
    b0 = get_backend(None)
    assert getattr(b0, "name", "") == "numpy"
    with use_backend("numpy"):
        b1 = get_backend(None)
        assert getattr(b1, "name", "") == "numpy"
    b2 = get_backend(None)
    assert getattr(b2, "name", "") == "numpy"


