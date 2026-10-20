"""运行自旋链的基态制备、Trotter 演化、AQC 压缩、扰动响应和谱分析。

安装可选依赖：pip install -e '.[spin]'。从仓库根目录运行小算例：
    export PYTHONPATH=src
    python examples/run_spin_dynamics.py --help
    python examples/run_spin_dynamics.py --n 4 --steps 2 --segments 2:1 --hva-layers 1 --ground-maxiter 20 --aqc-maxiter 20 --output /tmp/spin-demo

默认自旋数为 10；--simulator statevector 使用态矢量，quimb_mps 使用
矩阵乘积态，精度由 --max-bond 和 --cutoff 控制。--no-aqc 可跳过线路压缩。
模型采用物理自旋 S=σ/2，时间与耦合常数单位需相容。
优化达到轮数上限不保证保真度达标，应读取实际 fidelity 和优化器状态。
结果写入 --output 指定目录，区分基态误差、Trotter 误差和 AQC 误差。
谱的镜像或截断选项仅用于显示；本示例只进行本地模拟。"""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np

from tyxonq.applications.spin import (AQCOptions, GroundStateOptions, SpinChainModel,
                                     run_spin_dynamics, spectrum_for_plot)


def report_progress(event):
    """输出可读进度，长计算时可以看到当前分支和保真度。"""
    stage = event["stage"]
    if stage == "ground_state_start":
        print("开始 DMRG 参考态与 HVA 制备优化……", flush=True)
    elif stage == "ground_state_done":
        print(f"基态制备完成：E0={event['energy']:.12f}，F={event['fidelity']:.8f}", flush=True)
    elif stage in ("aqc_step", "aqc_iteration"):
        iteration = event.get("iterations", event.get("iteration"))
        print(f"AQC {event['branch']}，步 {event['step']}，迭代 {iteration}，"
              f"F={event['fidelity']:.8f}", flush=True)
    elif stage == "complete":
        print(f"计算完成：验收通过={event['acceptance_passed']}，耗时 {event['elapsed_seconds']:.1f} 秒", flush=True)


def save_result(result, output, *, mirror=False, clip_negative=False):
    """保存原始数据、绘图副本、参数及可重新执行的数值线路。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    arrays = dict(result.arrays)
    keys = ("exact_ground_dsf", "full_trotter_dsf", "dsf")
    for key in keys:
        arrays[key + "_normalized_plot"] = spectrum_for_plot(
            arrays[key], mirror=mirror, clip_negative=clip_negative, normalize=True)
    np.savez_compressed(output / "results.npz", **arrays)

    # 参数单独保存，便于查看每次初始化、延续及分段切换。
    parameters = {"hva": result.ground_state.parameters}
    for branch, aqc in result.aqc_results.items():
        for record in aqc.records:
            step = record["step"]
            parameters[f"{branch}_step_{step}_initial"] = np.array(record["initial_parameters"])
            parameters[f"{branch}_step_{step}_final"] = aqc.parameters[step]
    np.savez_compressed(output / "parameters.npz", **parameters)

    # JSON 线路只含 H/X/S/SDG/CX/RX/RY/RZ 和数值角度，无未绑定参数。
    circuits = {name: {branch: [{"num_qubits": c.num_qubits, "ops": c.ops} for c in cs]
                       for branch, cs in branches.items()}
                for name, branches in (("executed", result.circuits),
                                        ("full_trotter", result.baseline_circuits))}
    (output / "circuits.json").write_text(json.dumps(circuits, indent=2), encoding="utf-8")
    summary = dict(result.summary)
    summary["runtime"] = {name: version(name) for name in ("numpy", "scipy", "torch", "quimb", "cotengra", "cotengrust")}
    summary["runtime"]["python"] = platform.python_version()
    summary["plot_processing"] = {"mirror": mirror, "clip_negative": clip_negative,
                                   "normalized_by": "each spectrum's maximum absolute value"}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    config_keys = ("model", "dt", "steps", "order", "center", "perturbation_axis", "measurement_axis",
                   "simulator", "ground_state_options", "aqc_options")
    (output / "config.json").write_text(json.dumps({k: summary[k] for k in config_keys},
                                                   ensure_ascii=False, indent=2), encoding="utf-8")

    # 绘图使用英文轴标签，避免用户机器缺少中文字体；含义在中文 notebook 中解释。
    titles = ("Exact ground + exact evolution", "HVA + full Trotter",
              "HVA + AQC + tail" if result.aqc_results else "HVA + full Trotter (baseline)")
    t, q, w = arrays["times"], arrays["momenta"], arrays["frequencies"]
    response_keys = ("exact_ground_green", "full_trotter_green", "green")
    response_scale = max(np.max(np.abs(arrays[k])) for k in response_keys) or 1
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    for ax, key, title in zip(axes, response_keys, titles):
        im = ax.pcolormesh(arrays["sites"], t, arrays[key], shading="nearest", cmap="RdBu_r",
                           vmin=-response_scale, vmax=response_scale)
        ax.set(title=title, xlabel="Site j", ylabel="Time t")
    fig.colorbar(im, ax=axes, label="Retarded spin response")
    fig.savefig(output / "response.png", dpi=170)
    plt.close(fig)

    for normalized in (False, True):
        data = [arrays[key + "_normalized_plot"] if normalized else arrays[key] for key in keys]
        scale = max(np.max(np.abs(a)) for a in data) or 1
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
        for ax, a, title in zip(axes, data, titles):
            im = ax.pcolormesh(q / np.pi, w, a, shading="nearest", cmap="RdBu_r", vmin=-scale, vmax=scale)
            ax.set(title=title, xlabel="Momentum q / pi", ylabel="Frequency omega")
        fig.colorbar(im, ax=axes, label="Normalized plot" if normalized else "S(q, omega)")
        fig.savefig(output / ("spectra_normalized.png" if normalized else "spectra_physical.png"), dpi=170)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    for ax, key, title in zip(axes, ("num_2q_gates", "depth_2q", "depth"),
                              ("CNOT count", "Two-qubit depth", "Total base-gate depth")):
        for branch, style in (("plus", "-"), ("minus", "--")):
            records = summary["circuit_stats"][branch]
            ax.plot(t, [r["executed"][key] for r in records], style, label=f"Executed {branch}")
        ax.plot(t, [r["full_trotter"][key] for r in summary["circuit_stats"]["plus"]], ":", label="Full Trotter")
        ax.set(xlabel="Time t", ylabel=title)
        ax.legend(fontsize=8)
    fig.savefig(output / "cost.png", dpi=170)
    plt.close(fig)
    return output


def main():
    """命令行默认启用 AQC；--no-aqc 显式运行未压缩基准。"""
    import torch
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "heisenberg10")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--jxy", type=float, default=1.0)
    parser.add_argument("--jz", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=.6)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--order", type=int, choices=(1, 2), default=2)
    parser.add_argument("--center", type=int)
    parser.add_argument("--perturbation-axis", choices=("X", "Y", "Z"), default="Z")
    parser.add_argument("--measurement-axis", choices=("X", "Y", "Z"), default="Z")
    parser.add_argument("--hva-layers", type=int, default=3)
    parser.add_argument("--initial-state", choices=("singlet", "neel"), default="singlet")
    parser.add_argument("--ground-maxiter", type=int, default=300)
    parser.add_argument("--aqc-maxiter", type=int, default=300)
    parser.add_argument("--fidelity-target", type=float, default=.99)
    parser.add_argument("--segments", default="3:1,2:2", help="各段的时间点数:种子步数，用逗号分隔")
    parser.add_argument("--max-bond", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--simulator", choices=("statevector", "quimb_mps"), default="statevector")
    parser.add_argument("--no-aqc", action="store_true")
    parser.add_argument("--plot-mirror", action="store_true")
    parser.add_argument("--plot-clip-negative", action="store_true")
    args = parser.parse_args()
    ground_options = GroundStateOptions(layers=args.hva_layers, initial_state=args.initial_state,
        maxiter=args.ground_maxiter, max_bond=args.max_bond, cutoff=args.cutoff, seed=args.seed)
    try:
        segments = tuple(tuple(map(int, item.split(":"))) for item in args.segments.split(","))
        aqc_options = None if args.no_aqc else AQCOptions(segments=segments, maxiter=args.aqc_maxiter,
            fidelity_target=args.fidelity_target, max_bond=args.max_bond, cutoff=args.cutoff)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    result = run_spin_dynamics(SpinChainModel(args.n, args.jxy, args.jz), dt=args.dt, steps=args.steps,
        order=args.order, ground_state_options=ground_options, aqc_options=aqc_options,
        simulator=args.simulator, center=args.center, perturbation_axis=args.perturbation_axis,
        measurement_axis=args.measurement_axis, progress=report_progress)
    output = save_result(result, args.output, mirror=args.plot_mirror, clip_negative=args.plot_clip_negative)
    print(f"结果已保存到 {output.resolve()}")
    # 未达标准仍保存真实结果，以退出码提示调用者，绝不替换为未压缩线路。
    return 0 if result.summary["acceptance_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
