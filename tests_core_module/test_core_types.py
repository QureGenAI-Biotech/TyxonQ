# 核心 types / config 测试合并：config 默认值与后端名归一化、向量化策略校验，
# core.types.Problem 结构，以及 core.errors 的异常继承层级。
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
from tyxonq.core.types import Problem
from tyxonq.core.errors import TyxonQError, CompilationError, DeviceExecutionError


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


def test_core_types_problem_and_aliases():
    pb = Problem(kind="hamiltonian", payload={"terms": [("Z", 0, 1.0)]})
    assert pb.kind == "hamiltonian"
    assert normalize_backend_name("pytorch") == "pytorch"
    assert normalize_backend_name("cpu") == "numpy"  # alias for numpy
    assert normalize_backend_name("gpu") == "cupynumeric"  # alias for cupynumeric
    assert normalize_backend_name("numpy(cpu)") == "numpy"
    assert normalize_backend_name("cupynumeric(gpu)") == "cupynumeric"
    assert normalize_backend_name("torch") == "pytorch"
    assert is_valid_vectorization_policy("auto") is True


def test_core_errors_hierarchy():
    try:
        raise CompilationError("failed to build comiple_plan")
    except TyxonQError as e:
        assert "failed" in str(e)
    try:
        raise DeviceExecutionError("timeout")
    except TyxonQError as e:
        assert "timeout" in str(e)
