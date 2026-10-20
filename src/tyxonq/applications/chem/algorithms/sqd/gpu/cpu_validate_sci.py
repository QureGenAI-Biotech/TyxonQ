#!/usr/bin/env python3
"""在独立无 CUDA Python 进程中复核 selected-CI 算符像并构造 RDM。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from time import perf_counter
from pathlib import Path


def _watch_parent_stdin():
    """父进程独持管道写端；即使被强杀，EOF 也会终止本验证进程。"""
    try:
        while os.read(0, 1):
            pass
    finally:
        os._exit(125)


# 仅显式 CLI 模式启动，普通 import 不创建线程；在加载数值库前覆盖父进程退出。
if __name__ == "__main__" and "--watch-parent-stdin" in sys.argv:
    threading.Thread(target=_watch_parent_stdin, daemon=True).start()

import numpy as np
from pyscf import lib
from pyscf.fci import addons, direct_spin1, selected_ci


def add_zero_coupling_spectator(ci_strings, h1e, eri, norb, nelec):
    """临时添加零耦合旁观轨道和固定电子，返回维度不变的 Hamiltonian 输入。

    增加的轨道积分均为零，且旁观电子只有一个固定组态，因此 CI 维度及
    物理 Hamiltonian 的矩阵完全不变；RDM 和自旋仍在原始空间计算。
    """
    extended_norb = norb + 1
    extended_h1 = np.zeros((extended_norb,) * 2, dtype=np.float64)
    extended_h1[:norb, :norb] = h1e
    extended_eri = np.zeros((extended_norb,) * 4, dtype=np.float64)
    extended_eri[:norb, :norb, :norb, :norb] = eri
    # PySCF C 内核按 uint64 解释标签；第 63 位通过 view 保留完整位模式。
    spectator = np.asarray([1 << norb], dtype=np.uint64).view(np.int64)
    # PySCF 的 ctypes 内核不读取 NumPy stride，必须显式连续化切片标签。
    extended_strings = tuple(spectator if n == 0 else np.ascontiguousarray(strings, dtype=np.int64)
                             for n, strings in zip(nelec, ci_strings))
    extended_nelec = tuple(max(1, n) for n in nelec)
    return extended_strings, extended_h1, extended_eri, extended_norb, extended_nelec


def polarized_hamiltonian_image(coefficients, ci_strings, h1e, eri, norb, nelec):
    """使用零耦合旁观电子绕过 PySCF selected-CI 的空自旋分量除零。"""
    extended_strings, extended_h1, extended_eri, extended_norb, extended_nelec = (
        add_zero_coupling_spectator(ci_strings, h1e, eri, norb, nelec))
    vector = selected_ci._as_SCIvector(np.asarray(coefficients).reshape(
        len(ci_strings[0]), len(ci_strings[1])), extended_strings)
    h2 = direct_spin1.absorb_h1e(extended_h1, extended_eri, extended_norb, extended_nelec, .5)
    return np.asarray(selected_ci.contract_2e(h2, vector, extended_norb, extended_nelec)).reshape(-1)


def polarized_spin_penalty(sci_vector, norb, nelec, spin_sq, spin_shift):
    """在原空间计算与 fix_spin_ 一致的线性或平方 S² 惩罚。"""
    image = selected_ci.contract_ss(sci_vector, norb, nelec)
    # SCIvector 的非原位 NumPy 运算会丢失 _strs；第二次 S² 收缩须保留标签。
    residual = image
    residual -= float(spin_sq) * sci_vector
    minimum_spin = .5 * abs(nelec[0] - nelec[1])
    if float(spin_sq) >= minimum_spin * (minimum_spin + 1.) + .1:
        residual = selected_ci.contract_ss(residual, norb, nelec) - float(spin_sq) * residual
    return spin_shift * np.asarray(residual).reshape(-1)


def main() -> int:
    """读取无 pickle 输入，完成 PySCF 计算并原子写出无 pickle 结果。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--watch-parent-stdin", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    started = perf_counter()
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖 CPU validation 输出：{args.output}")
    lib.num_threads(args.threads)

    with np.load(args.input, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        coefficients = np.asarray(archive["coefficients"], dtype=np.float64)
        ci_strings = (
            np.asarray(archive["ci_strs_a"], dtype=np.int64),
            np.asarray(archive["ci_strs_b"], dtype=np.int64),
        )
        h1e = np.asarray(archive["h1e"], dtype=np.float64)
        eri = np.asarray(archive["eri"], dtype=np.float64)
    norb = int(metadata["norb"])
    nelec = tuple(int(value) for value in metadata["nelec"])
    spin_sq = metadata["spin_sq"]
    spin_shift = float(metadata["spin_shift"])

    sci_vector = selected_ci._as_SCIvector(
        coefficients.reshape(len(ci_strings[0]), len(ci_strings[1])), ci_strings
    )
    operator_started = perf_counter()
    if metadata.get("include_operator", True):
        if min(nelec) == 0:
            bare_image = polarized_hamiltonian_image(coefficients, ci_strings, h1e, eri, norb, nelec)
        else:
            link_index = selected_ci._all_linkstr_index(ci_strings, norb, nelec)
            h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec, 0.5)
            bare_image = np.asarray(
                selected_ci.contract_2e(
                    h2e, sci_vector, norb, nelec, link_index=link_index
                ), dtype=np.float64).reshape(-1)
        if spin_sq is None:
            cpu_image = bare_image
        elif min(nelec) == 0:
            cpu_image = bare_image + polarized_spin_penalty(sci_vector, norb, nelec, spin_sq, spin_shift)
        else:
            spin_solver = addons.fix_spin_(
                selected_ci.SelectedCI(), shift=spin_shift, ss=float(spin_sq)
            )
            cpu_image = np.asarray(
                spin_solver.contract_2e(
                    h2e, sci_vector, norb, nelec, link_index=link_index
                ), dtype=np.float64).reshape(-1)
        bare_rayleigh = float(np.dot(coefficients, bare_image))
    else:
        # gpu 验收配合 CPU RDM 时只构造密度，不暗中重复 CPU Hc。
        cpu_image = np.empty(0, dtype=np.float64)
        bare_rayleigh = float("nan")
    operator_seconds = perf_counter() - operator_started

    rdm_started = perf_counter()
    solver = selected_ci.SelectedCI()
    rdm1s = solver.make_rdm1s(sci_vector, norb, nelec)
    rdm2s = solver.make_rdm2s(sci_vector, norb, nelec)
    # PySCF 的总 RDM 接口会再次计算上面的自旋分量；直接按同一约定求和。
    # 从副本开始累加，避免覆盖后续还要保存和验收的 aa/ab/bb 分量。
    rdm1 = rdm1s[0] + rdm1s[1]
    rdm2 = rdm2s[0].copy()
    rdm2 += rdm2s[2]
    rdm2 += rdm2s[1]
    rdm2 += rdm2s[1].transpose(2, 3, 0, 1)
    physical_energy = np.einsum("pr,pr->", rdm1, h1e) + 0.5 * np.einsum(
        "prqs,prqs->", rdm2, eri
    )
    rdm_seconds = perf_counter() - rdm_started

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            cpu_image=cpu_image,
            bare_cpu_rayleigh=np.asarray(bare_rayleigh),
            operator_seconds=np.asarray(operator_seconds),
            operator_uses_spectator=np.asarray(metadata.get("include_operator", True) and min(nelec) == 0),
            rdm_seconds=np.asarray(rdm_seconds),
            physical_energy=np.asarray(float(np.real_if_close(physical_energy))),
            rdm1=rdm1,
            rdm2=rdm2,
            rdm1_alpha=rdm1s[0],
            rdm1_beta=rdm1s[1],
            rdm2_aa=rdm2s[0],
            rdm2_ab=rdm2s[1],
            rdm2_bb=rdm2s[2],
        )
    temporary.replace(args.output)
    print(json.dumps({"status": "complete", "threads": args.threads,
                      "elapsed_seconds": perf_counter() - started,
                      "rdm_components_reused": True}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
