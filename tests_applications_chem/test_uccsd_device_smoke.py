import math
import pytest


def _has_pyscf():
    try:
        import pyscf  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_pyscf(), reason="PySCF not installed; skipping chem UCCSD device smoke test")
def test_uccsd_energy_device_h2_smoke():
    import tyxonq as tq
    # Use legacy UCCSD explicitly from static path
    from tyxonq.applications.chem.algorithms.uccsd import UCCSD
    from tyxonq.applications.chem import molecule

    # Build UCCSD with preset H2 molecule (RHF default)
    ucc = UCCSD(molecule.h2)

    # Evaluate energy via device chain API (counts-based) on simulator
    e = ucc.energy(shots=2048, runtime="device", provider="simulator", device="statevector")
    # Gradient (parameter-shift) sanity via energy_and_grad(engine="device")
    e2, g = ucc.energy_and_grad([0.0, -0.07260814651571323], engine="device")
    assert isinstance(e2, float)
    assert g.shape == (ucc.n_params,)

    # Kernel with device engine should run if configured
    ucc.engine = "device"
    e_opt = ucc.kernel()
    assert isinstance(e_opt, float)

    # Basic sanity: finite float and within reasonable chemical range
    assert isinstance(e, float)
    assert math.isfinite(e)
    assert -3.0 < e < 0.0


