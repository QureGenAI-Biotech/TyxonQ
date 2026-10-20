"""自旋链专用 AQC：quimb 收缩 + PyTorch 梯度 + SciPy 参数优化。"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .circuits import (ParameterAngle, blocks_to_circuit,
                       circuit_stats, evolution_blocks, make_spin_ansatz)
from .simulation import mps_state


@dataclass(frozen=True)
class AQCOptions:
    """segments 中每项为（本段压缩时间点数，浅层种子的演化步数）。"""

    segments: tuple = ((3, 1), (2, 2))
    fidelity_target: float = 0.99
    maxiter: int = 300
    max_bond: int = 32
    cutoff: float = 1e-10

    def __post_init__(self):
        if not self.segments:
            raise ValueError("segments must not be empty; use aqc_options=None to disable AQC")
        for segment in self.segments:
            if len(segment) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 1 for x in segment):
                raise ValueError("each segment must contain two positive integers")
        if not 0 < self.fidelity_target <= 1:
            raise ValueError("fidelity_target must be in (0, 1]")
        for value in (self.maxiter, self.max_bond):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError("maxiter and max_bond must be positive integers")
        if not np.isfinite(self.cutoff) or self.cutoff < 0:
            raise ValueError("cutoff must be finite and nonnegative")


@dataclass
class AQCResult:
    """circuits 包含 t=0、压缩时间点和后续混合演化；records 只对应压缩点。"""

    circuits: list
    parameters: dict
    records: list
    compressed_steps: int


class FidelityObjective:
    """直接对原生参数向量求导，避免经过 Qiskit 的参数或 Circuit 适配。

    目标 MPS 显式归一化；ansatz 由幺正门作用于归一化 |0> 得到，范数恒为 1。
    仅目标态生成使用 MPS 截断；梯度路径采用未截断的门张量网络，避免 SVD 简并梯度。
    """

    def __init__(self, ansatz, target_mps):
        self.ansatz = ansatz
        self.target = target_mps.copy()
        self.target.normalize()
        self.path = None

    def value_and_grad(self, parameters):
        """返回 SciPy 所需的（实损失，实梯度），统一使用 CPU 双精度。"""
        import quimb.tensor as qtn
        import torch

        x = torch.tensor(np.asarray(parameters), dtype=torch.float64, requires_grad=True)
        circ = qtn.Circuit(self.ansatz.n)
        for op in self.ansatz.ops:
            name = op[0]
            if name in ("rx", "ry", "rz"):
                value = op[2]
                if isinstance(value, ParameterAngle):
                    angle = value.scale * x[value.index] + value.offset
                    # 直接传 tensor 参数数组；把 tensor 包进 Python 元组会被 quimb 转成 NumPy。
                    circ.apply_gate(name.upper(), params=angle.reshape(1), qubits=(op[1],), parametrize=True)
                else:
                    circ.apply_gate(name.upper(), value, op[1])
            elif name == "cx":
                circ.apply_gate("CNOT", op[1], op[2])
            else:
                circ.apply_gate("SDG" if name == "sdg" else name.upper(), op[1])
        network = self.target.H & circ.psi
        # 常量门、|0> 和目标 MPS 转换到相同后端；已有 tensor 用 to 保留梯度。
        network.apply_to_arrays(lambda array: array.to(dtype=torch.complex128) if torch.is_tensor(array)
                                else torch.tensor(array, dtype=torch.complex128))
        # 结构固定，只计算一次收缩路径；greedy 对首版小体系成本可控且可复现。
        if self.path is None:
            self.path = network.contraction_path(optimize="greedy")
        overlap = network.contract(all, optimize=self.path, backend="torch")
        loss = 1 - torch.abs(overlap) ** 2
        loss.backward()
        return float(loss.detach()), x.grad.detach().numpy().copy()


def compress_dynamics(model, initial_blocks, dt, steps, aqc_options=None, *, order=2, progress=None):
    """逐时刻压缩同一初态（已含扰动）的动力学；不同扰动应分别调用。"""
    opts = aqc_options or AQCOptions()
    total = sum(segment[0] for segment in opts.segments)
    if total > steps or any(segment[1] > steps for segment in opts.segments):
        raise ValueError("compressed steps and seed steps must not exceed steps")
    initial = blocks_to_circuit(model.n, initial_blocks)
    circuits, parameters, records = [initial], {}, []
    k = 0
    for segment_index, (n_steps, seed_steps) in enumerate(opts.segments):
        seed = list(initial_blocks) + evolution_blocks(model, dt, seed_steps, order)
        ansatz = make_spin_ansatz(model.n, seed)
        x0 = ansatz.initial_parameters.copy()
        for _ in range(n_steps):
            k += 1
            target_circuit = blocks_to_circuit(model.n, list(initial_blocks) + evolution_blocks(model, dt, k, order))
            target = mps_state(target_circuit, opts.max_bond, opts.cutoff)
            objective = FidelityObjective(ansatz, target)
            initial_loss, _ = objective.value_and_grad(x0)
            best = {"loss": initial_loss, "x": x0.copy()}
            history = [float(1 - initial_loss)]

            def evaluate(x):
                value, gradient = objective.value_and_grad(x)
                if value < best["loss"]:
                    best.update(loss=value, x=x.copy())
                return value, gradient

            def callback(intermediate_result):
                history.append(float(1 - intermediate_result.fun))
                if progress is not None and len(history) % 50 == 0:
                    progress({"stage": "aqc_iteration", "step": k,
                              "iteration": len(history) - 1, "fidelity": history[-1]})
                if intermediate_result.fun <= 1 - opts.fidelity_target:
                    raise StopIteration

            if initial_loss <= 1 - opts.fidelity_target:
                iterations, success, message = 0, True, "Initial parameters meet fidelity target"
            else:
                result = minimize(evaluate, x0, jac=True, method="L-BFGS-B", callback=callback,
                                  options={"maxiter": opts.maxiter, "ftol": 1e-12, "gtol": 1e-8})
                iterations, success, message = int(result.nit), bool(result.success), str(result.message)
            bound = ansatz.bind(best["x"])
            # 独立执行绑定线路后复核，不只相信优化器内部的损失。
            actual = mps_state(bound, opts.max_bond, opts.cutoff)
            fidelity = float(abs(target.H @ actual) ** 2)
            record = {
                "step": k, "segment": segment_index, "seed_steps": seed_steps,
                "parameter_count": len(x0), "initial_fidelity": float(1 - initial_loss),
                "fidelity": fidelity, "objective_fidelity": float(1 - best["loss"]),
                "target_met": bool(fidelity >= opts.fidelity_target - 1e-10),
                "optimizer_success": success, "optimizer_message": message, "iterations": iterations,
                "initial_parameters": x0.tolist(), "history": history,
                "full_trotter": circuit_stats(target_circuit), "aqc": circuit_stats(bound),
            }
            circuits.append(bound)
            parameters[k] = best["x"].copy()
            records.append(record)
            x0 = best["x"].copy()
            if progress is not None:
                progress({"stage": "aqc_step", "step": k, "fidelity": fidelity,
                          "iterations": iterations, "target_met": record["target_met"]})
    for tail in range(1, steps - total + 1):
        circuits.append(blocks_to_circuit(model.n, circuits[total].ops + evolution_blocks(model, dt, tail, order)))
    return AQCResult(circuits, parameters, records, total)
