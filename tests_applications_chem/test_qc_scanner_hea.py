"""HEA 走 qmmm 链路 + 真机参数穿透（不改跑真机：以 simulator 档验证透传机制）。

背景：``HEA.as_pyscf_solver`` 增加 ``device_opts``（shots/provider/device）后，
``qc_scanner(..., method="hea", solver_kwargs={...})`` 的运行选项应能到达
``HEA.kernel`` → ``devices.base.run``（真机提交入口）；真机实测留待有资源时。

判据：
- 能量与 ``runtime="numeric"`` 参考一致（``shots=0`` 回退解析路径）；
- 嵌入场景（``mm_charges``）能量/``mm_gradient`` 可用；
- ``fcisolver.instance`` 上的 ``shots/provider/device`` 属性证明选项确实穿透。

全程 ``method="hea"``（目标算法之一，与 uccsd 并列）。
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
def test_hea_device_opts_passthrough():
    """solver_kwargs 里的 shots/provider/device 透传到 HEA.kernel（simulator 档）。"""
    from tyxonq.applications.chem.interfaces import qc_scanner

    scan = qc_scanner(SPEC, basis="sto-3g", active_space=(4, 4), unit="Bohr",
                      method="hea",
                      solver_kwargs={"runtime": "device", "provider": "simulator",
                                     "device": "statevector", "shots": 0})
    e_dev, _ = scan(H2O_BOHR)

    # 选项确实到达 HEA 实例（kernel 内持久化为实例属性）
    inst = scan.fcisolver.instance
    assert inst is not None
    assert inst.shots == 0
    assert inst.provider == "simulator"
    assert inst.device == "statevector"

    # shots=0 回退解析路径：与 numeric 参考一致
    scan_ref = qc_scanner(SPEC, basis="sto-3g", active_space=(4, 4), unit="Bohr",
                          method="hea", solver_kwargs={"runtime": "numeric"})
    e_ref, _ = scan_ref(H2O_BOHR)
    assert float(e_dev) == pytest.approx(float(e_ref), rel=1e-8)


@needs_pyscf
def test_hea_embedding_with_device_opts():
    """method='hea' + 静电嵌入 + 真机参数占位：能量位移与 mm_gradient 可用。"""
    from tyxonq.applications.chem.interfaces import qc_scanner

    scan = qc_scanner(SPEC, basis="sto-3g", active_space=(4, 4), unit="Bohr",
                      method="hea",
                      mm_charges=(MM_POS_BOHR, MM_CHARGES),
                      solver_kwargs={"runtime": "device", "provider": "simulator",
                                     "device": "statevector", "shots": 0})
    e_emb, de = scan(H2O_BOHR)
    g_mm = np.asarray(scan.mm_gradient())

    scan_bare = qc_scanner(SPEC, basis="sto-3g", active_space=(4, 4), unit="Bohr",
                           method="hea", solver_kwargs={"runtime": "numeric"})
    e_bare, _ = scan_bare(H2O_BOHR)

    assert abs(float(e_emb) - float(e_bare)) > 1e-4, "MM 电荷未进入哈密顿量"
    assert np.asarray(de).shape == (3, 3)
    assert g_mm.shape == (3, 3) and np.linalg.norm(g_mm) > 1e-4
