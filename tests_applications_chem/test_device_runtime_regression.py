"""Device runtime 三个基础回归测试，封堵历史测试盲区。

背景：旋转位序双重镜像（q→n-1-q）与解析聚合缺陷曾长期未被发现，
因为既有测试集恰好落在三个盲区里：

- 盲区①：所有对比测试都是 2 比特 H2——镜像变换在 2 比特下自抵消，
  位序错误只在 ≥3 比特、非回文体系才显形（水 CAS(4,4) +0.25 Ha、H4 +0.116 Ha）；
- 盲区②：``UCC.energy``/``HEA.energy`` 在 ``shots==0`` 时短路到 numeric
  runtime，device runtime 的解析分支从未被执行过（曾把多比特 ``ZZ`` 项
  因子化为单比特 ⟨Z⟩ 连乘，只对乘积态成立）；
- 盲区③：采样测试容差过宽（``2/√shots + 0.02``），小系统偏差被吸收。
- 盲区④：device PSR 梯度从未对照过数值梯度——旧 ±π/2 移位规则在
  UCC 偶谐波能量面上恒为零，采样档 L-BFGS 一直在拿纯噪声梯度优化
  （靠 MP2 初值已近最优才未暴露）。
- 盲区⑤：StatevectorEngine.state() 的 op 分发循环缺 "cry" 分支且未知
  op 静默跳过，shots=0 解析档丢弃所有 cry 门——门级单激发块
  （cx+parity+cry+parity⁻¹+cx）退化为恒等，曾被误判为“ansatz 分解
  缺陷”（backlog 7）。同款分发循环存在于多处（run()/state()/各引擎），
  引擎层回归见 tests_core_module/test_statevector_engine_probs.py。

判据全部使用目标算法（UCCSD/HEA）；体系取 H4（8 比特）、水 CAS(4,4)（6 比特）
与 H2（4 比特，盲区④）。

注：device 档单测可达分钟量级（变分优化 + 高 shots 采样）；各测试内置
步骤日志，用 ``pytest -s`` 可实时查看进度，长时间无输出属正常计算。
"""

from __future__ import annotations

import time

import numpy as np
import pytest

_T0 = time.time()


def _step(msg):
    """步骤进度日志（pytest -s 可见）：采样档单步可能数十秒，
    没有输出时很容易被误认为挂死。"""
    print(f"  [{time.time() - _T0:6.1f}s] {msg}", flush=True)


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
    _step("盲区① UCCSD H4 采样档 4096 shots…")
    e_s = rt.energy(ref["p"], shots=4096, provider="simulator", device="statevector")
    assert abs(e_s - ref["e_num"]) < 2e-2

    # HEA：水 CAS(4,4) parity（6 比特），固定收敛参数直连 device 采样档。
    from pyscf import gto
    from tyxonq.applications.chem.algorithms.hea import HEA

    _step("盲区① HEA 水 CAS(4,4) numeric 优化 + 采样档…")
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
    注：UCCSD 侧用默认门级分解——state() 补上 cry 分支后（盲区⑤）
    门级能量面与数值路径严格等价（实测 H4 随机参数点 ~1e-14）。
    """
    from pyscf import gto
    from tyxonq.applications.chem.runtimes.ucc_device_runtime import UCCDeviceRuntime

    ref = _h4_uccsd_ref()
    u = ref["u"]
    _step("盲区② UCCSD H4 门级档 shots=0 解析分支…")
    rt = UCCDeviceRuntime(u.n_qubits, u.n_elec_s, u.h_qubit_op, mode=u.mode,
                          ex_ops=u.ex_ops, param_ids=u.param_ids, init_state=None)
    e_a = rt.energy(ref["p"], shots=0, provider="simulator", device="statevector")
    assert e_a == pytest.approx(ref["e_num"], rel=1e-8)

    from tyxonq.applications.chem.algorithms.hea import HEA
    from tyxonq.applications.chem.runtimes.hea_device_runtime import HEADeviceRuntime

    _step("盲区② HEA 水 CAS(4,4) numeric 优化 + shots=0 解析分支…")
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
    _step("盲区③ H4 解析聚合 + 32768 shots 计数聚合（重活，分钟量级）…")
    e_a = rt.energy(ref["p"], shots=0, provider="simulator", device="statevector")
    e_s = rt.energy(ref["p"], shots=32768, provider="simulator", device="statevector")
    assert abs(e_s - e_a) < 1e-2


@needs_pyscf
def test_device_psr_gradient_two_shift_rule():
    """盲区④：device PSR 梯度必须与数值梯度一致。

    历史缺陷（backlog 8，已修复）：UCC 能量面只含偶次谐波 {2,4,...}
    （exp(θA) 且 A²=−I ⇒ 态幅 ~cos θ, sin θ），旧 ±π/2 移位规则
    sin(2k·π/2)=0 → 所有参数梯度恒为 ~0。现用两点移位规则
    g = 2·D(π/8) + (1−√2)·D(π/4)（对谐波 {2,4} 精确）。

    - trotter 档与门级档：能量面均与数值路径严格等价（门级等价依赖
      盲区⑤的 state() cry 修复），两档都直接对照 numeric 梯度；
      H2 单激发含 trotter 化的 {2,4} 谐波，恰好覆盖两点规则的权重；
    - 采样档 smoke：8192 shots 梯度落在统计噪声带内。
    """
    from tyxonq.applications.chem.algorithms.uccsd import UCCSD
    from tyxonq.applications.chem.molecule import h2
    from tyxonq.applications.chem.runtimes.ucc_device_runtime import UCCDeviceRuntime

    u = UCCSD(h2)
    u.kernel(shots=0, runtime="numeric")
    p = np.asarray(u.params).copy() + 0.13
    _step("盲区④ H2 numeric 梯度参考…")
    _, g_num = u.energy_and_grad(p, runtime="numeric", numeric_engine="statevector")
    g_num = np.asarray(g_num)
    # 前提：数值梯度非零（否则本测试失去拦截力）
    assert np.max(np.abs(g_num)) > 0.1

    def _rt(trotter: bool):
        return UCCDeviceRuntime(u.n_qubits, u.n_elec_s, u.h_qubit_op, mode=u.mode,
                                ex_ops=u.ex_ops, param_ids=u.param_ids, init_state=None,
                                trotter=trotter)

    # trotter 档：与数值梯度直接对照（若 PSR 规则回归到 ±π/2，diff 将达 O(0.5)）
    _step("盲区④ trotter 档 PSR vs numeric…")
    _, g_t = _rt(True).energy_and_grad(p, shots=0, provider="simulator", device="statevector")
    assert np.max(np.abs(np.asarray(g_t) - g_num)) < 1e-5

    # 门级档：同样直接对照 numeric（盲区⑤修复前此处只能做 PSR vs FD 自洽，
    # 因 cry 被 state() 丢弃导致门级能量面错误；修复后实测 ~2e-6）
    _step("盲区④ 门级档 PSR vs numeric…")
    _, g_gate = _rt(False).energy_and_grad(p, shots=0, provider="simulator", device="statevector")
    assert np.max(np.abs(np.asarray(g_gate) - g_num)) < 1e-5

    # 采样档 smoke：真实梯度 + 统计噪声
    _step("盲区④ 8192-shot 采样梯度 smoke…")
    _, g_s = _rt(True).energy_and_grad(p, shots=8192, provider="simulator", device="statevector")
    assert np.max(np.abs(np.asarray(g_s) - g_num)) < 4.0 / np.sqrt(8192) + 0.02
