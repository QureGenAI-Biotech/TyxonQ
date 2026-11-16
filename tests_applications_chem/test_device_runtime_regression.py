"""VQE 扁平执行入口回归：位序、采样、梯度与 shots 消歧。"""

from __future__ import annotations

import time

import numpy as np
import pytest

_T0 = time.time()


def _step(msg):
    print(f"  [{time.time() - _T0:6.1f}s] {msg}", flush=True)


def _has_pyscf():
    try:
        import pyscf  # noqa: F401
        return True
    except Exception:
        return False


needs_pyscf = pytest.mark.skipif(not _has_pyscf(), reason="PySCF not installed; skipping.")
_H4_CACHE: dict = {}


def _h4_uccsd_ref():
    if not _H4_CACHE:
        from tyxonq.applications.chem.algorithms.vqe.uccsd import UCCSD
        from tyxonq.applications.chem.molecule import h4

        u = UCCSD(h4)
        u.kernel(shots=0)
        p = np.asarray(u.params).copy()
        _H4_CACHE.update(u=u, p=p, e_num=u.energy(p, shots=0, numeric_engine="statevector"))
    return _H4_CACHE


@needs_pyscf
def test_large_system_sampling_bit_order():
    """H4 UCCSD 与水 CAS(4,4) HEA：正 shots 对照各自数值参考。"""
    ref = _h4_uccsd_ref()
    _step("UCCSD H4 采样档 4096 shots")
    e_s = ref["u"].energy(ref["p"], shots=4096, provider="simulator", device="statevector")
    assert abs(e_s - ref["e_num"]) < 2e-2

    from pyscf import gto
    from tyxonq.applications.chem.algorithms.vqe.hea import HEA

    _step("HEA 水 CAS(4,4) 数值优化 + 采样档")
    mol = gto.M(atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
                basis="sto-3g", unit="Angstrom")
    hea = HEA(molecule=mol, layers=1, mapping="parity", active_space=(4, 4))
    hea.kernel(shots=0)
    p = np.asarray(hea.params).copy()
    e_num = hea.energy(p, shots=0)
    e_s = hea.energy(p, shots=4096, provider="simulator", device="statevector")
    assert abs(e_s - e_num) < 2e-2


@needs_pyscf
def test_shots_zero_is_numeric_and_unambiguous(monkeypatch):
    """零 shots 走算法数值层；硬件配置仅警告，且不得进入 Circuit.run。"""
    from tyxonq.applications.chem.algorithms.vqe.execution import measurement_ops
    from tyxonq.applications.chem.algorithms.vqe.hea import HEA
    from tyxonq.applications.chem.algorithms.vqe.uccsd import UCCSD
    from tyxonq.applications.chem.molecule import h2
    from tyxonq.core.ir.circuit import Circuit
    from tyxonq.libs.hamiltonian_encoding.hamiltonian_grouping import (
        group_hamiltonian_pauli_terms,
    )

    hea = HEA(n_qubits=2, layers=1,
              hamiltonian=[(-1.0, [("Z", 0)]), (0.2, [("X", 0)])])
    params = np.linspace(0.1, 0.6, hea.n_params)
    expected = hea.energy(params, shots=0)

    identity, groups = group_hamiltonian_pauli_terms(hea.hamiltonian, hea.n_qubits)
    backend_energy = float(identity)
    for bases, items in groups.items():
        circuit = hea.get_circuit(params).extended(
            list(measurement_ops(hea.n_qubits, bases))
        ).postprocessing(method="expval_pauli_sum", identity_const=0.0, items=items)
        payload = circuit.run(provider="simulator", device="statevector", shots=0)[0]
        backend_energy += payload["postprocessing"]["result"]["energy"]
    assert backend_energy == pytest.approx(expected, abs=1e-10)

    sampled = hea.energy(params, shots=32, provider=None, device="statevector")
    assert np.isfinite(sampled)

    def forbidden_run(*args, **kwargs):
        raise AssertionError("shots=0 must not call Circuit.run")

    monkeypatch.setattr(Circuit, "run", forbidden_run)
    assert hea.energy(params, shots=0) == pytest.approx(expected)
    energy, gradient = hea.energy_and_grad(params, shots=0)
    assert energy == pytest.approx(expected)
    assert gradient.shape == (hea.n_params,)

    ucc = UCCSD(h2)
    ucc_expected = ucc.energy(shots=0, numeric_engine="civector")
    assert np.isfinite(ucc_expected)
    assert ucc.energy_and_grad(shots=0)[1].shape == (ucc.n_params,)

    with pytest.warns(UserWarning, match="algorithm-level numeric_engine"):
        assert hea.energy(params, shots=0, provider="lqcloud") == pytest.approx(expected)
    with pytest.warns(UserWarning, match="not Circuit.run"):
        assert ucc.energy(
            shots=0, provider="lqcloud", numeric_engine="civector"
        ) == pytest.approx(ucc_expected)
    assert ucc.numeric_engine == "civector"

    def converged_gradient(x, **opts):
        assert opts["shots"] == 0
        assert opts["provider"] == "simulator"
        assert opts["device"] == "statevector"
        return expected, np.zeros_like(x)

    monkeypatch.setattr(hea, "energy_and_grad", converged_gradient)
    with pytest.warns(UserWarning, match="algorithm-level numeric_engine") as caught:
        hea.kernel(shots=0, provider="lqcloud")
    assert len(caught) == 1

    with pytest.raises(ValueError, match="Circuit-only"):
        hea.energy(params, shots=0, noise={"type": "readout"})
    with pytest.raises(ValueError, match="numeric_engine"):
        hea.energy(params, shots=16, numeric_engine="statevector")
    with pytest.raises(TypeError, match="shots=0"):
        hea.energy(params, shots=0, runtime="numeric")
    for invalid in (-1, True, 1.5):
        with pytest.raises((TypeError, ValueError)):
            hea.energy(params, shots=invalid)


@needs_pyscf
def test_counts_vs_numeric_consistency():
    """H4 高 shots 计数聚合与数值能量口径一致。"""
    ref = _h4_uccsd_ref()
    _step("H4 32768 shots 计数聚合")
    e_s = ref["u"].energy(ref["p"], shots=32768, provider="simulator", device="statevector")
    assert abs(e_s - ref["e_num"]) < 1e-2


@needs_pyscf
def test_circuit_parameter_shift_gradient():
    """UCC 两点移位与 HEA ±π/2 梯度在正 shots 下对照数值梯度。"""
    from tyxonq.applications.chem.algorithms.vqe.uccsd import UCCSD
    from tyxonq.applications.chem.molecule import h2

    u = UCCSD(h2)
    u.kernel(shots=0)
    p = np.asarray(u.params).copy() + 0.13
    _, g_num = u.energy_and_grad(p, shots=0, numeric_engine="statevector")
    assert np.max(np.abs(g_num)) > 0.1

    for trotter in (False, True):
        u.trotter = trotter
        _step(f"H2 UCC sampling gradient, trotter={trotter}")
        _, g_circuit = u.energy_and_grad(
            p, shots=16384, provider="simulator", device="statevector"
        )
        assert np.max(np.abs(np.asarray(g_circuit) - g_num)) < 0.12
