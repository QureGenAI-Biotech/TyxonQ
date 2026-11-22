"""化学量子动力学包（时间演化 + 谱学 + 实验对照），与 ``algorithms/``（电子结构）平级。

本包收纳（quantum-dynamic 分支扩建，直观算法模块、无运行时抽象）：

- ``evolution``：数值时间演化（``DynamicsNumericRuntime``，VQD/pVQD）；
- ``observables`` / ``exact`` / ``circuits`` / ``spectroscopy``：观测量纯函数、
  数值精确档、线路档（Trotter → Circuit 链式 ``.run()``）、谱学后处理；
- ``pulse_dynamics``：波形档演化（``PulseProgram`` 链式 IR，长时间窗路线）；
- ``models/``：模型哈密顿量（Pyrazine、SBM 既有；spin_lattice 磁材料、
  exciton OLED 为 Tier 1 新增）、``benchmarks/``：实验锚点对照层。

惰性导出：重依赖（renormalizer / scipy）在子模块内延迟导入，导入本包
只拉 ``evolution`` 的轻路径（与既有行为一致，零回归）。
"""

from .evolution import DynamicsNumericRuntime  # noqa: F401


def __getattr__(name: str):  # PEP 562 惰性导出
    _lazy = {
        "build_pauli_matrix": ".observables",
        "pauli_expectation": ".observables",
        "site_populations": ".observables",
        "bitstring_to_state": ".observables",
        "evolve_exact": ".exact",
        "correlation_exact": ".exact",
        "expectation_trajectory": ".exact",
        "trotter_ops": ".circuits",
        "build_trotter_circuit": ".circuits",
        "measure_pauli_expectation": ".circuits",
        "trotter_expectation_trajectory": ".circuits",
        "compute_dsf": ".spectroscopy",
        "correlation_to_spectrum": ".spectroscopy",
        "apply_window": ".spectroscopy",
        "error_metrics": ".spectroscopy",
        "evolve_with_pulses": ".pulse_dynamics",
        "compute_spin_correlations": ".sqw",
        "compute_sqw": ".sqw",
        "compute_sqw_kicked": ".sqw",
        "sqw_from_correlations": ".sqw",
        "population_trajectory": ".exciton_dynamics",
        "mean_squared_displacement": ".exciton_dynamics",
        "diffusion_length": ".exciton_dynamics",
        "rabi_frequency_two_site": ".exciton_dynamics",
    }
    if name in _lazy:
        import importlib
        return getattr(importlib.import_module(_lazy[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DynamicsNumericRuntime",
    "build_pauli_matrix", "pauli_expectation", "site_populations", "bitstring_to_state",
    "evolve_exact", "correlation_exact", "expectation_trajectory",
    "trotter_ops", "build_trotter_circuit", "measure_pauli_expectation",
    "trotter_expectation_trajectory",
    "compute_dsf", "correlation_to_spectrum", "apply_window", "error_metrics",
    "evolve_with_pulses",
    "compute_spin_correlations", "compute_sqw", "compute_sqw_kicked",
    "sqw_from_correlations",
    "population_trajectory", "mean_squared_displacement", "diffusion_length",
    "rabi_frequency_two_site",
]
