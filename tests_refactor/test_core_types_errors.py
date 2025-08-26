from tyxonq.core.types import (
    Problem
)
from tyxonq.config import normalize_backend_name, is_valid_vectorization_policy
from tyxonq.core.errors import TyxonQError, CompilationError, DeviceExecutionError


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
        raise CompilationError("failed to build pipeline")
    except TyxonQError as e:
        assert "failed" in str(e)
    try:
        raise DeviceExecutionError("timeout")
    except TyxonQError as e:
        assert "timeout" in str(e)


