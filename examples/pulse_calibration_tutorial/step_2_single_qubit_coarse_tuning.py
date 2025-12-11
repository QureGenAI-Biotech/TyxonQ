"""
========================================================================
TyxonQ 脉冲调教实战教程 - 第 2 步：单比特门粗调
========================================================================

本步骤是实际调教工作的开始，目的是：
✅ 从零开始设计单比特门的脉冲
✅ 进行粗调找到可行工作点
✅ 理解脉冲参数对旋转角的影响

学习预期时间：45 分钟
难度：★★☆☆☆

========================================================================
核心概念
========================================================================

"粗调"意味着什么？
──────────────

在精细的参数优化之前，我们首先需要找到"可用的工作点"。
这个工作点应该满足：
  ✅ 脉冲能够制造出接近目标旋转的效果
  ✅ 泄漏率可接受（< 10%）
  ✅ 保真度合理（> 80%）

粗调的步骤：
  1️⃣ 选择脉冲类型（通常 Gaussian 作为起点）
  2️⃣ 估计所需的脉冲振幅和时长
  3️⃣ 扫描脉冲参数，找到最好的粗略工作点
  4️⃣ 粗略优化 DRAG 参数（beta）
  5️⃣ 验证概念可行性

========================================================================
实践：X 门的粗调
========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from tyxonq import Circuit, waveforms
from tyxonq.core.ir.pulse import PulseProgram

print("""
╔════════════════════════════════════════════════════════════════════════╗
║  TyxonQ 脉冲调教教程 - 第 2 步：单比特门粗调                           ║
╚════════════════════════════════════════════════════════════════════════╝
""")

# ========================================================================
# 第 2.1 部分：参数估计（理论计算）
# ========================================================================

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2.1 理论参数估计
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

目的：在盲目扫描之前，用理论值指导我们选择合理的参数范围
""")

print("\n📐 Rabi 旋转理论")
print("─" * 60)

print("""
Rabi 旋转的核心公式：

  旋转角：θ = Ω · t
  
  其中：
  • Ω: Rabi 频率（rad/s）= γ · A
  • γ: 旋转偶合常数（rad/(s·V)）
  • A: 脉冲振幅（V）
  • t: 脉冲时长（s）

对于 π 旋转（完全翻转 |0⟩ → |1⟩）：
  π = γ · A · t
  
  因此：
  A = π / (γ · t)

或者反过来：
  t = π / (γ · A)
""")

# 假设的硬件参数
gamma = 2 * np.pi * 50e6  # 约 50 MHz/V （典型值）
target_angle = np.pi       # π 旋转（X 门）

print(f"\n🔧 假设条件：")
print(f"   • 旋转偶合常数 γ = {gamma / (2*np.pi) / 1e6:.0f} MHz/V")
print(f"   • 目标旋转角 θ = π 弧度")
print(f"   • 脉冲类型：Gaussian（推荐起点）")

# 计算不同时长下所需的振幅
pulse_durations = [80, 120, 160, 200, 240]

print(f"\n📊 不同脉冲时长所需的振幅：")
print("─" * 60)
print(f"{'时长(ns)':<12} {'所需振幅':<15} {'特点':<30}")
print("─" * 60)

for t_ns in pulse_durations:
    t_s = t_ns * 1e-9
    needed_amp = target_angle / (gamma * t_s)
    
    # 评价
    if needed_amp > 1.0:
        comment = "⚠️ 超过振幅限制 (>1)"
    elif needed_amp < 0.3:
        comment = "⚠️ 太弱，噪声敏感"
    else:
        comment = "✅ 合理范围"
    
    print(f"{t_ns:<12} {needed_amp:<15.3f} {comment:<30}")

print("\n💡 建议选择：")
print("   • 首选时长：160 ns（振幅约 0.8 - 在合理范围内）")
print("   • 备选时长：120 ns（振幅约 1.05 - 略高但可用）")

# ========================================================================
# 第 2.2 部分：参数扫描（找到粗调工作点）
# ========================================================================

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2.2 参数扫描：寻找最佳的粗调工作点
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

策略：固定脉冲类型和时长，扫描振幅
      找到能产生最接近 π 旋转的振幅值
""")

# 扫描参数
pulse_duration = 160        # ns，从上面的理论计算选定
sigma = pulse_duration / 4  # 高斯宽度（推荐值）
amplitudes = np.linspace(0.5, 1.0, 11)  # 振幅扫描范围

print(f"\n🔬 扫描条件：")
print(f"   • 脉冲类型：Gaussian")
print(f"   • 脉冲时长：{pulse_duration} ns")
print(f"   • 高斯宽度：{sigma} ns")
print(f"   • 扫描范围：{amplitudes[0]} 到 {amplitudes[-1]} V")
print(f"   • 扫描步数：{len(amplitudes)}")

print(f"\n开始扫描...")
print("─" * 80)

scan_results = []

for amp in amplitudes:
    # 创建脉冲程序
    prog = PulseProgram(num_qubits=1)
    prog.set_device_params(
        qubit_freq=[5.0e9],
        anharmonicity=[-330e6],
        T1=[80e-6],
        T2=[120e-6]
    )
    
    # 创建 Gaussian 脉冲
    pulse = waveforms.Gaussian(
        amp=amp,
        duration=pulse_duration,
        sigma=sigma
    )
    
    # 添加脉冲
    prog.add_pulse(0, pulse, qubit_freq=5.0e9)
    
    # 执行模拟
    state = prog.state(backend="numpy")
    
    # 提取结果
    pop_0 = abs(state[0])**2  # |0⟩ 的概率
    pop_1 = abs(state[1])**2  # |1⟩ 的概率
    leakage = 1 - pop_0 - pop_1  # 泄漏到 |2⟩
    
    # 推断旋转角（从人口比）
    # P_1 = sin²(θ/2)，所以 θ = 2·arcsin(√P_1)
    if pop_1 <= 1.0:
        inferred_angle = 2 * np.arcsin(np.sqrt(pop_1))
    else:
        inferred_angle = np.nan
    
    # 计算与 π 的偏差
    angle_error = abs(inferred_angle - np.pi)
    
    scan_results.append({
        'amp': amp,
        'pop_0': pop_0,
        'pop_1': pop_1,
        'leakage': leakage,
        'angle': inferred_angle,
        'angle_error': angle_error
    })
    
    # 打印进度
    status = "✅" if angle_error < 0.2 and leakage < 0.05 else "⚠️" if leakage < 0.1 else "❌"
    print(f"{status} 振幅 {amp:.2f}V: P₁={pop_1:.4f}  θ_inferred={inferred_angle/np.pi:.3f}π  "
          f"误差={angle_error/np.pi:.3f}π  泄漏={leakage:.4f}")

# 找到最佳工作点
best_idx = min(range(len(scan_results)), 
               key=lambda i: scan_results[i]['angle_error'])
best_result = scan_results[best_idx]

print("\n" + "=" * 80)
print("🎯 粗调最佳工作点：")
print("=" * 80)
print(f"   振幅：{best_result['amp']:.3f} V")
print(f"   推断旋转角：{best_result['angle']/np.pi:.4f}π（目标 1.0000π）")
print(f"   角度误差：{best_result['angle_error']/np.pi:.4f}π（相对误差：{best_result['angle_error']/np.pi*100:.2f}%）")
print(f"   |1⟩ 人口：{best_result['pop_1']:.4f}（理想值 1.0）")
print(f"   泄漏率：{best_result['leakage']:.4f}（可接受 < 5%）")

# ========================================================================
# 第 2.3 部分：初步 DRAG 优化
# ========================================================================

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2.3 初步 DRAG 优化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DRAG 参数 β 的作用：
  • β = 0：纯 Gaussian（无 DRAG 效应）
  • β > 0：加入导数项，抑制泄漏到 |2⟩
  • β 越大：泄漏抑制越强，但需要最优值（过大反而增加错误）

目标：找到能最小化泄漏的 β 值
""")

print(f"\n🔬 固定最佳振幅 {best_result['amp']:.3f}V，扫描 DRAG 参数 β")
print("─" * 80)

# 扫描 DRAG 参数
betas = np.linspace(0.0, 0.4, 9)

print(f"{'β 值':<8} {'|1⟩ 人口':<12} {'泄漏率':<12} {'相对变化':<15} {'评价':<15}")
print("─" * 80)

drag_results = []

for beta in betas:
    # 创建 DRAG 脉冲
    pulse = waveforms.Drag(
        amp=best_result['amp'],
        duration=pulse_duration,
        sigma=sigma,
        beta=beta
    )
    
    # 执行模拟
    prog = PulseProgram(num_qubits=1)
    prog.set_device_params(
        qubit_freq=[5.0e9],
        anharmonicity=[-330e6],
        T1=[80e-6],
        T2=[120e-6]
    )
    prog.add_pulse(0, pulse, qubit_freq=5.0e9)
    state = prog.state(backend="numpy")
    
    pop_0 = abs(state[0])**2
    pop_1 = abs(state[1])**2
    leakage = 1 - pop_0 - pop_1
    
    drag_results.append({
        'beta': beta,
        'pop_1': pop_1,
        'leakage': leakage
    })
    
    # 与 β=0 的对比
    if beta == 0:
        baseline_leakage = leakage
    
    leakage_reduction = (baseline_leakage - leakage) / baseline_leakage * 100 if beta > 0 else 0
    
    # 评价
    if leakage < 0.01:
        rating = "✅ 优秀"
    elif leakage < 0.05:
        rating = "✅ 良好"
    elif leakage < 0.10:
        rating = "⚠️ 可接受"
    else:
        rating = "❌ 需要改进"
    
    print(f"{beta:<8.2f} {pop_1:<12.4f} {leakage:<12.4f} {leakage_reduction:>+14.1f}% {rating:<15}")

# 找到最优 β
best_beta_idx = np.argmin([r['leakage'] for r in drag_results])
best_beta_result = drag_results[best_beta_idx]

print("\n" + "=" * 80)
print("🎯 最优 DRAG 参数：")
print("=" * 80)
print(f"   最优 β：{best_beta_result['beta']:.2f}")
print(f"   泄漏率：{best_beta_result['leakage']:.4f}（相对改善：{(baseline_leakage - best_beta_result['leakage'])/baseline_leakage*100:.1f}%）")
print(f"   |1⟩ 人口：{best_beta_result['pop_1']:.4f}")

# ========================================================================
# 总结
# ========================================================================

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 第 2 步总结与关键结果
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

print(f"""
✅ 获得的 X 门粗调参数：

   脉冲参数：
     • 类型：DRAG
     • 振幅：{best_result['amp']:.3f} V
     • 时长：{pulse_duration} ns
     • 高斯宽度：{sigma} ns
     • DRAG β：{best_beta_result['beta']:.2f}

   性能指标：
     • |1⟩ 人口：{best_beta_result['pop_1']:.4f}（目标 1.0）
     • 泄漏率：{best_beta_result['leakage']:.4f}（目标 < 1%）
     • 保真度（初步）：~{(1 - best_beta_result['leakage'])*best_beta_result['pop_1']*100:.1f}%

⚠️ 当前状态：
   • ✅ 已找到可行的工作点
   • ✅ 基本的泄漏抑制已生效
   • ⚠️ 还需要精细优化以达到 99.9% 保真度
   • ⚠️ 还需要表征 Rabi 频率的精确性

🎯 下一步（第 3 步）：
   • 精细调整振幅，使 |1⟩ 人口更接近 1.0
   • 更精细的 DRAG 参数优化
   • 进行 Rabi 振荡测量，精确校准旋转角
   • 达到 99.9% 保真度的目标

📊 数据保存建议：
   # 保存这个粗调结果
   coarse_tuning_result_x_gate = {{
       'pulse_type': 'DRAG',
       'amplitude': {best_result['amp']:.3f},
       'duration_ns': {pulse_duration},
       'sigma_ns': {sigma},
       'beta': {best_beta_result['beta']:.2f},
       'measured_leakage': {best_beta_result['leakage']:.4f}
   }}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

print("\n✨ 第 2 步完成！现在可以进入更深层的优化...")
