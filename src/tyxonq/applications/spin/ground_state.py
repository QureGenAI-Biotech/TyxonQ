"""DMRG 参考基态和 Hamiltonian variational ansatz（HVA）的经典拟合。"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .circuits import blocks_to_circuit, hva_blocks
from .simulation import mps_state


@dataclass(frozen=True)
class GroundStateOptions:
    """小体系默认配置；随机种子同时控制 DMRG 初态和 HVA 初始化。"""

    layers: int = 3
    initial_state: str = "singlet"
    maxiter: int = 300
    max_bond: int = 32
    cutoff: float = 1e-10
    dmrg_tol: float = 1e-10
    dmrg_max_sweeps: int = 20
    seed: int = 12345

    def __post_init__(self):
        for name in ("layers", "maxiter", "max_bond", "dmrg_max_sweeps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.initial_state not in ("singlet", "neel"):
            raise ValueError("initial_state must be 'singlet' or 'neel'")
        if not np.isfinite([self.cutoff, self.dmrg_tol]).all() or self.cutoff < 0 or self.dmrg_tol <= 0:
            raise ValueError("cutoff must be nonnegative and dmrg_tol must be positive")


@dataclass
class GroundStateResult:
    """同时保留参考 MPS、制备电路、参数和质量信息，避免混淆两种基态。"""

    circuit: object
    blocks: list
    parameters: np.ndarray
    reference_mps: object
    reference_energy: float
    prepared_energy: float
    fidelity: float
    dmrg_converged: bool
    optimizer_success: bool
    optimizer_message: str
    optimizer_iterations: int


def dmrg_ground_state(model, options=None):
    """求参考 MPS；简并情况下返回 DMRG 选中的一个基态。"""
    import quimb.tensor as qtn

    opts = options or GroundStateOptions()
    mpo = model.mpo()
    p0 = qtn.MPS_rand_state(model.n, min(4, opts.max_bond), seed=opts.seed, dtype="float64")
    bonds = sorted({min(b, opts.max_bond) for b in (4, 8, 16, opts.max_bond)})
    dmrg = qtn.DMRG2(mpo, bond_dims=bonds, cutoffs=opts.cutoff, p0=p0)
    converged = dmrg.solve(tol=opts.dmrg_tol, max_sweeps=opts.dmrg_max_sweeps, verbosity=0)
    state = dmrg.state.copy()
    state.normalize()
    return state, float(np.real(dmrg.energy)), bool(converged)


def prepare_ground_state(model, options=None) -> GroundStateResult:
    """通过 MPS overlap 拟合 HVA；这是经典保真度优化，不是硬件 VQE。"""
    import quimb.tensor as qtn

    opts = options or GroundStateOptions()
    target, energy, converged = dmrg_ground_state(model, opts)
    rng = np.random.default_rng(opts.seed)
    x0 = np.tile([0.0, np.pi / 2], opts.layers) + rng.normal(0, 0.1, 2 * opts.layers)

    def objective(parameters):
        circuit = blocks_to_circuit(model.n, hva_blocks(model, parameters, opts.initial_state))
        psi = mps_state(circuit, opts.max_bond, opts.cutoff)
        return float(1 - abs(psi.H @ target) ** 2)

    result = minimize(objective, x0, method="COBYQA", options={"maxiter": opts.maxiter})
    blocks = hva_blocks(model, result.x, opts.initial_state)
    circuit = blocks_to_circuit(model.n, blocks)
    psi = mps_state(circuit, opts.max_bond, opts.cutoff)
    # 显式传入收缩策略；部分 quimb 版本的一维便捷接口会强制使用 auto 搜索。
    bra, mpo, ket = qtn.tensor_network_align(psi.H, model.mpo(), psi)
    prepared_energy = float(np.real((bra | mpo | ket).contract(all, optimize="greedy")))
    return GroundStateResult(
        circuit=circuit, blocks=blocks, parameters=result.x.copy(), reference_mps=target,
        reference_energy=energy, prepared_energy=prepared_energy,
        fidelity=float(abs(psi.H @ target) ** 2), dmrg_converged=converged,
        optimizer_success=bool(result.success), optimizer_message=str(result.message),
        optimizer_iterations=int(result.nit),
    )
