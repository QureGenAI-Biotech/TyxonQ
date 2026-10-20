"""复制包含自定义门的线路后，验证实际量子态保持一致。"""

import numpy as np
import pytest

from tyxonq.core.ir import Circuit


@pytest.mark.parametrize("operation", ["metadata", "extended"])
def test_copy_preserves_custom_unitary_state(operation):
    circuit = Circuit(2)
    circuit.h(0)
    matrix = np.diag([1.0, 1.0j, -1.0, -1.0j])
    circuit.unitary(0, 1, matrix=matrix)
    copied = (
        circuit.with_metadata(label="copy")
        if operation == "metadata"
        else circuit.extended([])
    )
    np.testing.assert_allclose(copied.state(), circuit.state(), atol=1e-12)
    assert copied._unitary_cache is not circuit._unitary_cache
