from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from tyxonq.applications.chem.algorithms.sqd.gpu.davidson import davidson_lowest

pytestmark = pytest.mark.skipif(not torch.mps.is_available(), reason="PyTorch MPS is unavailable")


def test_mps_davidson_uses_explicit_host_small_eigh():
    diagonal = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float32, device="mps")
    initial = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device="mps")
    result = davidson_lowest(
        lambda vector: vector * diagonal, diagonal, initial_vector=initial,
        tolerance=1e-6, max_cycles=4, max_space=3,
    )
    torch.mps.synchronize()
    assert result.converged
    assert result.energy == pytest.approx(0.0, abs=1e-6)
    assert result.residual_norm < 1e-6
    assert result.coefficients.device.type == "mps"
    np.testing.assert_allclose(result.coefficients.cpu().numpy(), [1.0, 0.0, 0.0], atol=1e-6)
