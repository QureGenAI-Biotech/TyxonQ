"""Device runtime 三个基础回归测试，封堵历史测试盲区。

背景：旋转位序双重镜像（q→n-1-q）与解析聚合缺陷曾长期未被发现，
因为既有测试集恰好落在三个盲区里：

- 盲区①：所有对比测试都是 2 比特 H2——镜像变换在 2 比特下自抵消，
  位序错误只在 ≥3 比特、非回文体系才显形（水 CAS(4,4) +0.25 Ha、H4 +0.116 Ha）；
- 盲区②：``UCC.energy``/``HEA.energy`` 在 ``shots==0`` 时短路到 numeric
  runtime，device runtime 的解析分支从未被执行过（曾把多比特 ``ZZ`` 项
  因子化为单比特 ⟨Z⟩ 连乘，只对乘积态成立）；
- 盲区③：采样测试容差过宽（``2/√shots + 0.02``），小系统偏差被吸收。

判据全部使用目标算法（UCCSD/HEA）；体系取 H4（8 比特）与水 CAS(4,4)（6 比特）。
"""

from __future__ import annotations

import numpy as np
import pytest


def _has_pyscf():
    try:
        import pyscf  # noqa: F401
        return True
    except Exception:
        return False


needs_pyscf = pytest.mark.skipif(not _has_pyscf(), reason="PySCF not installed; skipping.")

# H4 UCCSD 数值参考（模块级共享，避免三个测试重复优化）
_H4_CACHE: dict = {}


def _h4_uccsd_ref():
    if not _H4_CACHE:
        from tyxonq.applications.chem.algorithms.uccsd import UCCSD
        from tyxonq.applications.chem.molecule import h4

        u = UCCSD(h4)
        u.kernel(shots=0, runtime="numeric")
        p = np.asarray(u.params).copy()
        e_num = float(u.energy(p, runtime="numeric", numeric_engine="statevector"))
        _H4_CACHE.update(u=u, p=p, e_num=e_num)
    return _H4_CACHE


def _h4_device_runtime():
    from tyxonq.applications.chem.runtimes.ucc_device_runtime import UCCDeviceRuntime

    ref = _h4_uccsd_ref()
    u = ref["u"]
    return UCCDeviceRuntime(u.n_qubits, u.n_elec_s, u.h_qubit_op, mode=u.mode,
                            ex_ops=u.ex_ops, param_ids=u.param_ids, init_state=None)


@needs_pyscf
def test_large_system_sampling_bit_order():
    """盲区①：≥3 比特体系采样档（counts>0）对照数值参考，UCCSD 与 HEA 各一。

    旋转位序双重镜像在 H4（8 比特）上的偏差约 +0.116 Ha；
    4096 shots 纯统计波动实测可达 ~7e-3，容差给 2e-2 防 flaky，
    仍能抓住 0.1 Ha 量级的位序回归；2 比特 H2 下该镜像自抵消，
    故必须用大体系把关。``hea_device_runtime`` 与 ``ucc_device_runtime``
    有同款缺陷史，故两者都要直连采样档验证。
    """
    ref = _h4_uccsd_ref()
    rt = _h4_device_runtime()
    e_s = rt.energy(ref["p"], shots=4096, provider="simulator", device="statevector")
    assert abs(e_s - ref["e_num"]) < 2e-2

    # HEA：水 CAS(4,4) parity（6 比特），固定收敛参数直连 device 采样档。
    from pyscf import gto
    from tyxonq.applications.chem.algorithms.hea import HEA

    mol = gto.M(atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
                basis="sto-3g", unit="Angstrom")
    hea = HEA(molecule=mol, layers=1, mapping="parity", runtime="device",
              active_space=(4, 4))
    hea.kernel(shots=0, provider="simulator", device="statevector")
    ph = np.asarray(hea.params).copy()
    e_num_h = float(hea.energy(ph, shots=0))
    e_s_h = hea.energy(ph, shots=4096, provider="simulator", device="statevector")
    assert abs(e_s_h - e_num_h) < 2e-2


@needs_pyscf
def test_device_shots0_analytic_path():
    """盲区②：直连 device runtime 的 shots=0 解析分支（绕过 API 层 numeric 短路）。

    曾把多比特 ZZ 乘积因子化为单比特 ⟨Z⟩ 连乘（H4 上偏 +0.116 Ha）。
    UCCSD 用 H4（8 比特），HEA 用水 CAS(4,4)（6 比特）。
    注：UCCSD 侧用 ``trotter=True`` 电路（与数值路径态制备精确等价），
    默认门级分解存在已知的 ~1e-4 ansatz 等价性缺口（见 ucc_device_runtime
    backlog），不在本测试职责内。
    """
    from pyscf import gto
    from tyxonq.applications.chem.runtimes.ucc_device_runtime import UCCDeviceRuntime

    ref = _h4_uccsd_ref()
    u = ref["u"]
    rt = UCCDeviceRuntime(u.n_qubits, u.n_elec_s, u.h_qubit_op, mode=u.mode,
                          ex_ops=u.ex_ops, param_ids=u.param_ids, init_state=None,
                          trotter=True)
    e_a = rt.energy(ref["p"], shots=0, provider="simulator", device="statevector")
    assert e_a == pytest.approx(ref["e_num"], rel=1e-8)

    from tyxonq.applications.chem.algorithms.hea import HEA
    from tyxonq.applications.chem.runtimes.hea_device_runtime import HEADeviceRuntime

    mol = gto.M(atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
                basis="sto-3g", unit="Angstrom")
    hea = HEA(molecule=mol, layers=1, mapping="parity", runtime="device",
              active_space=(4, 4))
    hea.kernel(shots=0, provider="simulator", device="statevector")
    p = np.asarray(hea.params).copy()
    e_num = float(hea.energy(p, shots=0))
    rt_h = HEADeviceRuntime(hea.n_qubits, hea.layers, hea.hamiltonian,
                            n_elec_s=hea.n_elec_s, mapping=hea.mapping,
                            circuit_template=hea.circuit_template,
                            qop=getattr(hea, "_qop_cached", None))
    e_a = rt_h.energy(p, shots=0, provider="simulator", device="statevector")
    assert e_a == pytest.approx(e_num, rel=1e-8)


@needs_pyscf
def test_counts_vs_analytic_consistency():
    """盲区③：同一批电路，高 shot 计数聚合与解析概率聚合必须口径一致。

    两者若存在位序/约定不一致会产生 ~0.1 Ha 量级系统偏差；
    32768 shots 的纯统计波动典型 ~3e-3，容差 1e-2 足以区分。
    """
    ref = _h4_uccsd_ref()
    rt = _h4_device_runtime()
    e_a = rt.energy(ref["p"], shots=0, provider="simulator", device="statevector")
    e_s = rt.energy(ref["p"], shots=32768, provider="simulator", device="statevector")
    assert abs(e_s - e_a) < 1e-2
