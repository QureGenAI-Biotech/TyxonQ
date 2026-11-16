import math

import pytest


def test_ucc_device_execution_smoke():
    pytest.importorskip("pyscf")
    from tyxonq.applications.chem.algorithms.vqe.uccsd import UCCSD
    from tyxonq.applications.chem.molecule import h2

    ucc = UCCSD(h2)
    energy = ucc.energy(shots=1024, provider="simulator", device="statevector")
    assert isinstance(energy, float) and math.isfinite(energy)
