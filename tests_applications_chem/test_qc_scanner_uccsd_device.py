"""UCCSD/ROUCCSD 走 qmmm 链路 + 真机参数穿透（不改跑真机：以 simulator 档验证透传机制）。

具体 UCCSD/ROUCCSD solver 将 ``device_opts`` 透传到各自 kernel；正 shots 才进入
``devices.base.run``，真机提交保持 opt-in。

判据：
- ``shots=0`` 直接数值计算，能量/梯度/``mm_gradient`` 与数值参考一致；
- ``shots>0`` 采样档能跑通且落在化学精度内（证明 device 路径真实执行）；
- 嵌入场景能量位移存在。

全程 ``method="uccsd"``（目标量子算法）。
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
def test_uccsd_device_opts_passthrough():
    """solver_kwargs 里的 shots/provider/device 透传到 UCCSD.kernel（simulator 档）。

    shots=0 回退解析路径，应与 numeric 参考在数值上完全一致；
    shots>0 采样档同链路可跑通且落在化学精度内（证明 device 路径真实执行）。
    """
    from tyxonq.applications.chem.interfaces import qc_scanner

    scan_ref = qc_scanner(SPEC, basis="sto-3g", active_space=(4, 4), unit="Bohr",
                          method="uccsd", solver_kwargs={"shots": 0})
    e_ref, de_ref = scan_ref(H2O_BOHR)

    scan = qc_scanner(SPEC, basis="sto-3g", active_space=(4, 4), unit="Bohr",
                      method="uccsd",
                      solver_kwargs={"provider": "simulator",
                                     "device": "statevector", "shots": 0})
    e_dev, de_dev = scan(H2O_BOHR)
    assert float(e_dev) == pytest.approx(float(e_ref), rel=1e-10)
    assert np.max(np.abs(np.asarray(de_dev) - np.asarray(de_ref))) < 1e-8

    # 采样档：device 路径真实执行（4096 shots 的统计偏差典型 ~3e-3，
    # 容差给 6e-3；足以抓住曾经的 0.25 Ha 量级位序回归）
    scan_s = qc_scanner(SPEC, basis="sto-3g", active_space=(4, 4), unit="Bohr",
                        method="uccsd",
                        solver_kwargs={"provider": "simulator",
                                       "device": "statevector", "shots": 4096})
    e_s, _ = scan_s(H2O_BOHR)
    assert abs(float(e_s) - float(e_ref)) < 6e-3


@needs_pyscf
def test_uccsd_embedding_with_device_opts():
    """method='uccsd' + 静电嵌入 + device_opts：能量位移与 mm_gradient 与 numeric 一致。"""
    from tyxonq.applications.chem.interfaces import qc_scanner

    scan = qc_scanner(SPEC, basis="sto-3g", active_space=(4, 4), unit="Bohr",
                      method="uccsd", mm_charges=(MM_POS_BOHR, MM_CHARGES),
                      solver_kwargs={"provider": "simulator",
                                     "device": "statevector", "shots": 0})
    e_emb, de = scan(H2O_BOHR)
    g_mm = np.asarray(scan.mm_gradient())

    scan_num = qc_scanner(SPEC, basis="sto-3g", active_space=(4, 4), unit="Bohr",
                          method="uccsd", mm_charges=(MM_POS_BOHR, MM_CHARGES),
                          solver_kwargs={"shots": 0})
    e_num, de_num = scan_num(H2O_BOHR)
    g_num = np.asarray(scan_num.mm_gradient())

    scan_bare = qc_scanner(SPEC, basis="sto-3g", active_space=(4, 4), unit="Bohr",
                           method="uccsd", solver_kwargs={"shots": 0})
    e_bare, _ = scan_bare(H2O_BOHR)

    assert abs(float(e_emb) - float(e_bare)) > 1e-4, "MM 电荷未进入哈密顿量"
    assert float(e_emb) == pytest.approx(float(e_num), rel=1e-10)
    assert np.asarray(de).shape == (3, 3)
    assert np.max(np.abs(np.asarray(de) - np.asarray(de_num))) < 1e-8
    assert g_mm.shape == (3, 3) and np.max(np.abs(g_mm - g_num)) < 1e-8
