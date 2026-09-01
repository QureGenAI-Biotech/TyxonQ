"""HEA 走 qmmm 链路的真机形态测试：采样档（shots>0）穿透。

HEA 存在的目的就是上真机；``test_qc_scanner_hea.py`` 只覆盖了 ``shots=0``
解析回退档，本文件补采样档端到端：``qc_scanner(..., method="hea",
solver_kwargs={shots/provider/device})`` 必须真实走 ``HEADeviceRuntime`` →
``devices.base.run`` 提交链路（simulator 档验证机制，真机实测留待资源到位）。

判据：
- ``shots=0`` 解析档与 ``runtime="numeric"`` 参考一致；
- ``shots=4096`` 采样档全链路可跑通，能量落在统计容差内（证明 device 路径真实执行）；
- ``fcisolver.instance`` 上 ``shots/provider/device`` 属性证明选项穿透；
- 嵌入场景下采样档能量位移存在。

全程 ``method="hea"``（目标量子算法）。
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

B2A = 0.52917721092
# 水几何（Å）与 E6b/E9 同款嵌入体系：QM 水 + tip3p MM 水
H2O_ANG = np.array([(0.0, 0.0, 0.1173), (0.0, 0.7572, -0.4692), (0.0, -0.7572, -0.4692)])
H2O_BOHR = H2O_ANG / B2A
MM_POS_BOHR = (H2O_ANG + np.array([2.9, 0.8, 0.3])) / B2A
MM_CHARGES = np.array([-0.834, 0.417, 0.417])
SPEC = [("O", tuple(H2O_ANG[0])), ("H", tuple(H2O_ANG[1])), ("H", tuple(H2O_ANG[2]))]


@needs_pyscf
def test_hea_scanner_sampling_passthrough():
    """shots=0 与 numeric 一致；shots>0 采样档全链路跑通且落在统计容差内。"""
    from tyxonq.applications.chem.interfaces import qc_scanner

    scan_ref = qc_scanner(SPEC, basis="sto-3g", active_space=(4, 4), unit="Bohr",
                          method="hea", solver_kwargs={"runtime": "numeric"})
    e_ref, _ = scan_ref(H2O_BOHR)

    scan = qc_scanner(SPEC, basis="sto-3g", active_space=(4, 4), unit="Bohr",
                      method="hea",
                      solver_kwargs={"n_layers": 1, "runtime": "device",
                                     "provider": "simulator", "device": "statevector",
                                     "shots": 0})
    e_dev, _ = scan(H2O_BOHR)
    assert float(e_dev) == pytest.approx(float(e_ref), rel=1e-8)

    # 采样档：HEA 的真实工作形态（真机提交同款链路）。
    # 带噪优化下偏差典型 ~1e-3、尾部可达 ~3e-2，容差给 3e-2 防 flaky，
    # 对曾经的 0.25 Ha 量级位序回归仍有 8 倍余量。
    scan_s = qc_scanner(SPEC, basis="sto-3g", active_space=(4, 4), unit="Bohr",
                        method="hea",
                        solver_kwargs={"n_layers": 1, "runtime": "device",
                                       "provider": "simulator", "device": "statevector",
                                       "shots": 4096})
    e_s, _ = scan_s(H2O_BOHR)
    assert abs(float(e_s) - float(e_ref)) < 3e-2

    inst = scan_s.fcisolver.instance
    assert inst is not None
    assert inst.shots == 4096
    assert inst.provider == "simulator"
    assert inst.device == "statevector"


@needs_pyscf
def test_hea_scanner_embedding_sampling():
    """method='hea' + 静电嵌入 + 采样档：能量位移存在，MM 电荷进入哈密顿量。"""
    from tyxonq.applications.chem.interfaces import qc_scanner

    scan = qc_scanner(SPEC, basis="sto-3g", active_space=(4, 4), unit="Bohr",
                      method="hea", mm_charges=(MM_POS_BOHR, MM_CHARGES),
                      solver_kwargs={"n_layers": 1, "runtime": "device",
                                     "provider": "simulator", "device": "statevector",
                                     "shots": 4096})
    e_emb, de = scan(H2O_BOHR)

    scan_bare = qc_scanner(SPEC, basis="sto-3g", active_space=(4, 4), unit="Bohr",
                           method="hea", solver_kwargs={"runtime": "numeric"})
    e_bare, _ = scan_bare(H2O_BOHR)

    assert abs(float(e_emb) - float(e_bare)) > 1e-4, "MM 电荷未进入哈密顿量"
    assert np.asarray(de).shape == (3, 3)
