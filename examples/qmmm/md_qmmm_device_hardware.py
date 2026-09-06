"""E10：QM/MM 上真机——UCCSD 与 HEA 经 qc_scanner 直达量子硬件（含模拟器档）。

E1 展示了 ``qc_scanner`` 的本地数值路径（``runtime="numeric"``）。本教程
展示同一条 QM/MM 链路如何**把哈密顿量的测量任务提交到量子设备**：
先用 ``provider="simulator"`` 走与真机完全相同的提交链路验证机制，
再给出切换真机（TyxonQ/QCOS/Quafu）的一行改动模板。

三档运行模式::

    runtime="numeric"                          本地精确求值（确定性，无噪声）
    runtime="device", provider="simulator"     设备提交链路 + 本地模拟器
                                               （与真机同款代码路径，本教程主线）
    runtime="device", provider="tyxonq"        真机（需 ``tq.set_token``，第 4 节）

关键参数经 ``solver_kwargs`` 穿透：``shots``（0 = 解析回退档，
>0 = 采样档）、``provider``、``device``，一路到达 ``devices.base.run``。

两条算法路径的定位：
  - ``method="uccsd"``：化学精度优先；解析档（shots=0）与 numeric 一致，
    采样档用于验证真机链路；
  - ``method="hea"``：**为真机而生**——浅电路、硬件原生门集友好，
    采样档（shots>0）是它的正常工作形态。

所需环境：tyxonq + pyscf>=2.10（不需要 ase / openmm / ipi / mdi）。
预期运行时间：约 10 分钟量级（水 / STO-3G / CAS(4,4)，多个采样档变分
优化；采样档是重活——每步梯度要提交数百个采样电路，脚本带累计耗时
日志逐段报告进度，长时间无输出的段落属正常计算而非挂死）。
依赖缺失时本脚本优雅退出，不报错。
"""

import sys
import time

import numpy as np

# ---- 依赖守卫：缺 pyscf 时优雅退出（CI 友好） ----
try:
    from pyscf import gto
except ImportError:
    print("本教程需要 pyscf>=2.10：pip install 'pyscf>=2.10'")
    sys.exit(0)

from tyxonq.applications.chem.interfaces.scanner import qc_scanner

# ---------------------------------------------------------------------------
# 演示体系：QM 水分子 + 一个 tip3p MM 水（静电嵌入），与测试用例同款。
# 约定：QM 几何与 MM 位置一律用 Bohr 交给 scanner（几何构建可用 Å，见 E1）。
# ---------------------------------------------------------------------------
B2A = 0.52917721092
H2O_ANG = np.array([(0.0, 0.0, 0.1173), (0.0, 0.7572, -0.4692), (0.0, -0.7572, -0.4692)])
H2O_BOHR = H2O_ANG / B2A
MM_POS_BOHR = (H2O_ANG + np.array([2.9, 0.8, 0.3])) / B2A   # MM 水位置（Bohr）
MM_CHARGES = np.array([-0.834, 0.417, 0.417])               # tip3p 电荷
SPEC = [("O", tuple(H2O_ANG[0])), ("H", tuple(H2O_ANG[1])), ("H", tuple(H2O_ANG[2]))]
BASIS, ACTIVE_SPACE = "sto-3g", (4, 4)

_T_START = time.time()


def _log(msg):
    """带累计耗时的进度日志：采样档是重变分优化，单段可能数分钟，
    没有进度输出时很容易被误认为挂死。"""
    print(f"[{time.time() - _T_START:6.1f}s] {msg}", flush=True)


def _device_kwargs(shots):
    """设备档标准三件套：所有真机/模拟器档都走这组选项。"""
    return {"runtime": "device", "provider": "simulator",
            "device": "statevector", "shots": shots}


def _cap_sampling_opt(instance):
    """限制采样档 L-BFGS 评估次数（真机实践要点）。

    采样档成本模型（实测）：单次 energy_and_grad 要提交
    (1+4×n_params)×n_groups 个电路，耗时由**电路数**主导、与 shots 几乎
    无关（水 CAS(4,4)，512 与 2048 shots 同为 ~156s/次）。两个后果：
    ① 真梯度下噪声地板（~1e-3 Ha）使严格 ftol/gtol 永不满足；
    ② L-BFGS 线搜索在带噪目标上会额外吃评估（maxiter 帽不住，
    maxfun 才是硬帽）。故用 maxfun=2 把每个几何点限制在
    “初值点 + 1 次线搜索评估”，验证提交链路同时控制成本；
    完整变分优化请按预算放宽 maxfun（每评估 ~分钟量级）或改用
    SPSA 类梯度免费优化器。经 as_pyscf_solver(config_function=...)
    注入，kernel 前调用。
    """
    instance.scipy_minimize_options = {"ftol": 1e-7, "gtol": 1e-4,
                                       "maxiter": 1, "maxfun": 2}


# ===========================================================================
# 第 1 节：UCCSD 上设备——解析档与采样档
# ===========================================================================
# shots=0 是解析回退档：不采样，从态矢量概率直接聚合期望值，
# 应与 runtime="numeric" 在机器精度内一致（回归测试以 1e-8 把关）。
print("=" * 72)
print("第 1 节：UCCSD 设备链路（method='uccsd'）")
print("=" * 72)

_log("numeric 参考扫描…")
scan_num = qc_scanner(SPEC, basis=BASIS, active_space=ACTIVE_SPACE,
                      method="uccsd", solver_kwargs={"runtime": "numeric"})
e_num, _ = scan_num(H2O_BOHR)

_log("device 解析档（shots=0）…")
scan_dev0 = qc_scanner(SPEC, basis=BASIS, active_space=ACTIVE_SPACE,
                       method="uccsd", solver_kwargs=_device_kwargs(shots=0))
e_dev0, de_dev0 = scan_dev0(H2O_BOHR)
print(f"numeric 参考 ：E = {e_num:.10f} Ha")
print(f"device shots=0（解析档）：E = {e_dev0:.10f} Ha，差 {e_dev0 - e_num:+.2e}")
assert abs(e_dev0 - e_num) < 1e-6, "解析档应回退到与 numeric 一致的精确聚合"

# shots>0 是采样档：对哈密顿量分组做基旋转 + Z 测量，从计数估计期望值。
# 统计偏差典型 ~1e-3（2048 shots）；容差给 3e-2，仍远小于位序类回归的量级。
_log("device 采样档（shots=2048，受限评估次数，预计 ~5 分钟）…")
scan_dev = qc_scanner(SPEC, basis=BASIS, active_space=ACTIVE_SPACE,
                      method="uccsd",
                      solver_kwargs={**_device_kwargs(shots=2048),
                                     "config_function": _cap_sampling_opt})
e_dev, de_dev = scan_dev(H2O_BOHR)
print(f"device shots=2048（采样档）：E = {e_dev:.10f} Ha，差 {e_dev - e_num:+.2e}")
assert abs(e_dev - e_num) < 3e-2, "采样档偏差超出统计容差"
print(f"采样档核梯度范数 |dE/dR| = {np.linalg.norm(de_dev):.6f} Ha/Bohr"
      f"（解析档 {np.linalg.norm(de_dev0):.6f}）")

# 选项确实穿透到了提交层：UCCSD 路径的 device_opts 直接透传给
# instance.kernel（不持久化为实例属性；HEA 路径会持久化，见第 2 节）。


# ===========================================================================
# 第 2 节：HEA 上设备——采样档是它的正常工作形态
# ===========================================================================
# HEA（硬件高效 ansatz）存在的目的就是上真机：浅电路、门集友好。
# 带噪优化下能量偏差典型 ~1e-3；但收敛损失是重尾分布——4096 shots 实测
# 偶发 +0.1 Ha 量级的坏抽签，8192 shots 多次实测稳定在 ~1e-3（与第 4 节
# 嵌入档结论一致）：HEA 采样优化建议 8192 shots 起步。
print()
print("=" * 72)
print("第 2 节：HEA 设备链路（method='hea'，采样档为真实工作形态）")
print("=" * 72)

_log("HEA numeric 参考…")
scan_hea_num = qc_scanner(SPEC, basis=BASIS, active_space=ACTIVE_SPACE,
                          method="hea", solver_kwargs={"runtime": "numeric"})
e_hea_num, _ = scan_hea_num(H2O_BOHR)

_log("HEA 采样档（shots=8192，带变分优化，预计数分钟）…")
scan_hea = qc_scanner(SPEC, basis=BASIS, active_space=ACTIVE_SPACE,
                      method="hea",
                      solver_kwargs={"n_layers": 1, **_device_kwargs(shots=8192)})
e_hea, de_hea = scan_hea(H2O_BOHR)
print(f"HEA numeric 参考 ：E = {e_hea_num:.10f} Ha")
print(f"HEA shots=8192   ：E = {e_hea:.10f} Ha，差 {e_hea - e_hea_num:+.2e}")
assert abs(e_hea - e_hea_num) < 3e-2, "HEA 采样档偏差超出统计容差"
print(f"HEA 核梯度范数 |dE/dR| = {np.linalg.norm(de_hea):.6f} Ha/Bohr")
# 选项穿透自检：HEA 把 shots/provider/device 持久化在实例上，
# 这是真机提交前确认参数到位的直接手段。
inst = scan_hea.fcisolver.instance
print(f"选项穿透自检：shots={inst.shots}, provider={inst.provider}, device={inst.device}")
assert inst.shots == 8192 and inst.provider == "simulator"
print("注：RY-only ansatz 表达能力有限，E_HEA 高于 E_UCCSD 属预期（变分上界）。")


# ===========================================================================
# 第 3 节：静电嵌入 + 设备档——MM 电荷进入量子哈密顿量（UCCSD）
# ===========================================================================
# 嵌入只需在构造时多给一个 mm_charges=(位置, 电荷)；设备选项照旧穿透。
# 判据：嵌入后能量相对裸算有位移（>1e-4 Ha），且采样档与解析档位移一致。
print()
print("=" * 72)
print("第 3 节：静电嵌入 + 设备档（mm_charges）")
print("=" * 72)

_log("嵌入采样档（UCCSD shots=2048 + MM 电荷，受限评估次数，预计 ~5 分钟）…")
scan_emb_dev = qc_scanner(SPEC, basis=BASIS, active_space=ACTIVE_SPACE,
                          method="uccsd", mm_charges=(MM_POS_BOHR, MM_CHARGES),
                          solver_kwargs={**_device_kwargs(shots=2048),
                                         "config_function": _cap_sampling_opt})
e_emb, de_emb = scan_emb_dev(H2O_BOHR)
_log("嵌入档完成，计算 mm_gradient…")
g_mm = np.asarray(scan_emb_dev.mm_gradient())   # MM 反作用力（供 MD 耦合用）

shift = e_emb - e_dev
print(f"嵌入档（shots=2048）：E = {e_emb:.10f} Ha，位移 {shift:+.6f} Ha")
print(f"mm_gradient 范数 = {np.linalg.norm(g_mm):.6f} Ha/Bohr（MM 原子反作用力）")
assert abs(shift) > 1e-4, "MM 电荷未进入哈密顿量"

_log("嵌入 numeric 参考…")
scan_emb_num = qc_scanner(SPEC, basis=BASIS, active_space=ACTIVE_SPACE,
                          method="uccsd", mm_charges=(MM_POS_BOHR, MM_CHARGES),
                          solver_kwargs={"runtime": "numeric"})
e_emb_num, _ = scan_emb_num(H2O_BOHR)
print(f"嵌入档 numeric 参考：E = {e_emb_num:.10f} Ha，差 {e_emb - e_emb_num:+.2e}")
assert abs(e_emb - e_emb_num) < 3e-2, "嵌入采样档偏差超出统计容差"


# ===========================================================================
# 第 4 节：真机实践要点——HEA 嵌入档需要高 shots（带噪优化收敛性）
# ===========================================================================
# 重要结论：采样档的总偏差 = 能量估计的统计噪声 + 带噪优化的收敛损失。
# 后者才是主导：梯度本身带噪，低 shots 下优化器提前停滞。实测本体系：
#   shots=2048 → 嵌入档偏差可达 ~0.1~0.25 Ha（优化停滞）
#   shots=8192 → 偏差回落到 ~2e-3（固定参数对照只有 ~1.6e-3，证明聚合链正确）
# 因此真机上用 HEA 做嵌入计算时，shots 要从高起步（≥8192），
# 或采用梯度累积/自适应 shots 策略；UCCSD 因 ansatz 结构好，2048 已够。
print()
print("=" * 72)
print("第 4 节：HEA 嵌入档的 shots 收敛性（真机实践要点）")
print("=" * 72)

_log("HEA 嵌入采样档（shots=8192，带变分优化，预计数分钟）…")
scan_hea_emb = qc_scanner(SPEC, basis=BASIS, active_space=ACTIVE_SPACE,
                          method="hea", mm_charges=(MM_POS_BOHR, MM_CHARGES),
                          solver_kwargs={"n_layers": 1, **_device_kwargs(shots=8192)})
e_hea_emb, _ = scan_hea_emb(H2O_BOHR)

_log("HEA 嵌入 numeric 参考…")
scan_hea_emb_num = qc_scanner(SPEC, basis=BASIS, active_space=ACTIVE_SPACE,
                              method="hea", mm_charges=(MM_POS_BOHR, MM_CHARGES),
                              solver_kwargs={"runtime": "numeric"})
e_hea_emb_num, _ = scan_hea_emb_num(H2O_BOHR)
print(f"HEA 嵌入档 shots=8192：E = {e_hea_emb:.10f} Ha")
print(f"HEA 嵌入档 numeric  ：E = {e_hea_emb_num:.10f} Ha，差 {e_hea_emb - e_hea_emb_num:+.2e}")
assert abs(e_hea_emb - e_hea_emb_num) < 3e-2, "HEA 高 shots 嵌入档偏差超出统计容差"


# ===========================================================================
# 第 5 节：切换真机——只改 provider/device，提前设置 token
# ===========================================================================
# 真机提交与上面所有模拟器档走**同一条代码路径**（devices.base.run），
# 因此模拟器档验证通过后，切换只是换掉三件套里的两个字符串。
print()
print("=" * 72)
print("第 5 节：真机模板（默认注释；资源到位后放开即用）")
print("=" * 72)
print("""
    import tyxonq as tq
    tq.set_token("YOUR_TOKEN", provider="tyxonq", device="homebrew_s2")

    scan_real = qc_scanner(SPEC, basis=BASIS, active_space=ACTIVE_SPACE,
                           method="hea",          # 真机首选 HEA
                           solver_kwargs={"runtime": "device",
                                          "provider": "tyxonq",   # 或 qcos / quafu
                                          "device": "homebrew_s2",
                                          "shots": 2048})
    e, de = scan_real(H2O_BOHR)
""")
print("提示：")
print("  - provider 取值同 tyxonq.devices.base.run：tyxonq / qcos / quafu / simulator；")
print("  - 真机 shots 建议：UCCSD 从 2048 起；HEA（尤其嵌入/梯度场景）从 8192 起；")
print("  - 带嵌入的 MD（ASE/i-PI/OpenMM/MDI）只需把上面的 scanner 换进对应适配层，")
print("    参见 E3（ASE）、E4（i-PI）、E-EE/E6（OpenMM）、E9（MDI）。")

print("\nE10 完成。")
_log("总耗时见上行时间戳。")
