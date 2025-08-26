from __future__ import annotations

from tyxonq.config import (
    PACKAGE_NAME,
    DEFAULT_COMPLEX_DTYPE_STR,
    DEFAULT_REAL_DTYPE_STR,
    SUPPORTED_BACKENDS,
    default_dtypes,
    normalize_backend_name,
    is_valid_vectorization_policy,
)


def test_defaults_and_supported_backends():
    assert PACKAGE_NAME == "tyxonq"
    assert DEFAULT_COMPLEX_DTYPE_STR in {"complex64", "complex128"}
    assert DEFAULT_REAL_DTYPE_STR in {"float32", "float64"}
    assert "numpy" in SUPPORTED_BACKENDS
    c, r = default_dtypes()
    assert c == DEFAULT_COMPLEX_DTYPE_STR and r == DEFAULT_REAL_DTYPE_STR


def test_normalize_backend_and_vectorization_policy():
    assert normalize_backend_name("cpu") == "numpy"
    assert normalize_backend_name("torch") == "pytorch"
    assert is_valid_vectorization_policy("auto")
    assert not is_valid_vectorization_policy("invalid")


