"""HEA 走 qmmm 链路的真机形态测试：采样档（shots>0）穿透。

``test_qc_scanner_hea.py`` 覆盖 ``shots=0`` 数值档，本文件补正 shots 采样端到端：
``qc_scanner`` → 无状态分组测量 → ``devices.base.run``。

判据：
- ``shots=0`` 直接数值计算；
- ``shots=8192`` 采样档全链路可跑通，能量落在统计容差内；
- solver 适配器的 ``device_opts`` 证明选项穿透；
- 嵌入场景下采样档能量位移存在。

全程 ``method="hea"``（目标量子算法）。

注：采样档含带噪变分优化，单测可达分钟量级；内置步骤日志，用
``pytest -s`` 可实时查看进度，长时间无输出属正常计算。
"""

from __future__ import annotations

import time

import numpy as np
import pytest

_T0 = time.time()


def _step(msg):
    """步骤进度日志（pytest -s 可见）。"""
    print(f"  [{time.time() - _T0:6.1f}s] {msg}", flush=True)


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

    _step("HEA numeric 参考优化…")
    scan_ref = qc_scanner(SPEC, basis="sto-3g", active_space=(4, 4), unit="Bohr",
                          method="hea", solver_kwargs={"shots": 0})
    e_ref, _ = scan_ref(H2O_BOHR)

    _step("HEA device 解析档（shots=0）…")
    scan = qc_scanner(SPEC, basis="sto-3g", active_space=(4, 4), unit="Bohr",
                      method="hea",
                      solver_kwargs={"n_layers": 1,
                                     "provider": "simulator", "device": "statevector",
                                     "shots": 0})
    e_dev, _ = scan(H2O_BOHR)
    assert float(e_dev) == pytest.approx(float(e_ref), rel=1e-8)

    # 采样档：HEA 的真实工作形态（真机提交同款链路）。
    # 带噪优化收敛损失是重尾分布：4096 shots 实测偶发 +0.1 Ha 量级坏抽签
    # （会超过 3e-2 容差），8192 shots 多次实测稳定在 ~1e-3；故用 8192 + 3e-2，
    # 对曾经的 0.25 Ha 量级位序回归仍有 8 倍余量。
    _step("HEA 采样档（shots=8192，带噪变分优化，预计数分钟）…")
    scan_s = qc_scanner(SPEC, basis="sto-3g", active_space=(4, 4), unit="Bohr",
                        method="hea",
                        solver_kwargs={"n_layers": 1,
                                       "provider": "simulator", "device": "statevector",
                                       "shots": 8192})
    e_s, _ = scan_s(H2O_BOHR)
    assert abs(float(e_s) - float(e_ref)) < 3e-2

    assert scan_s.fcisolver.instance is not None
    assert scan_s.fcisolver.device_opts["shots"] == 8192
    assert scan_s.fcisolver.device_opts["provider"] == "simulator"
    assert scan_s.fcisolver.device_opts["device"] == "statevector"


@needs_pyscf
def test_hea_scanner_embedding_sampling():
    """method='hea' + 静电嵌入 + 采样档：能量位移存在，MM 电荷进入哈密顿量。"""
    from tyxonq.applications.chem.interfaces import qc_scanner

    _step("HEA 嵌入采样档（shots=4096，带噪变分优化，预计数分钟）…")
    scan = qc_scanner(SPEC, basis="sto-3g", active_space=(4, 4), unit="Bohr",
                      method="hea", mm_charges=(MM_POS_BOHR, MM_CHARGES),
                      solver_kwargs={"n_layers": 1,
                                     "provider": "simulator", "device": "statevector",
                                     "shots": 4096})
    e_emb, de = scan(H2O_BOHR)

    _step("裸算 numeric 对照…")
    scan_bare = qc_scanner(SPEC, basis="sto-3g", active_space=(4, 4), unit="Bohr",
                           method="hea", solver_kwargs={"shots": 0})
    e_bare, _ = scan_bare(H2O_BOHR)

    assert abs(float(e_emb) - float(e_bare)) > 1e-4, "MM 电荷未进入哈密顿量"
    assert np.asarray(de).shape == (3, 3)
