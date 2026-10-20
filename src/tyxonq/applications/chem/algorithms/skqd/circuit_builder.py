"""参考态和二阶 Trotter Krylov 线路，不隐式执行设备。"""

from __future__ import annotations

from operator import index

import numpy as np
from scipy.linalg import expm

from tyxonq.core.ir.circuit import Circuit
from ...chem_libs.fermionic_circuits import append_cphase, append_orbital_rotation
from ...chem_libs.fermionic_operators import validate_integrals
from .ensemble import CircuitRecord, SKQDEnsemble


def prepare_slater_reference(norb, nelec, *, occupations=None, orbital_rotation=None):
    """占据列表按 (alpha,beta) 给出；旋转矩阵的列表示变换后的单粒子轨道。"""
    n = index(norb)
    nelec = tuple(index(x) for x in nelec)
    if n < 1 or len(nelec) != 2 or any(x < 0 or x > n for x in nelec):
        raise ValueError("invalid norb or nelec")
    if occupations is None:
        occupations = tuple(tuple(range(k)) for k in nelec)
    if len(occupations) != 2:
        raise ValueError("occupations must contain alpha and beta lists")
    circuit = Circuit(2 * n)
    for spin, occupied in enumerate(occupations):
        occupied = tuple(index(i) for i in occupied)
        if len(set(occupied)) != nelec[spin] or len(occupied) != nelec[spin]:
            raise ValueError("occupation lists must match nelec without duplicates")
        if any(i < 0 or i >= n for i in occupied):
            raise ValueError("occupied orbital is out of range")
        for i in occupied:
            circuit.x(spin * n + i)
    if orbital_rotation is not None:
        rotations = np.asarray(orbital_rotation, complex)
        if rotations.shape == (n, n):
            rotations = np.stack([rotations, rotations])
        if rotations.shape != (2, n, n):
            raise ValueError("orbital_rotation must have shape (n,n) or (2,n,n)")
        for spin, rotation in enumerate(rotations):
            append_orbital_rotation(circuit, rotation, range(spin * n, (spin + 1) * n))
    circuit.metadata["skqd_reference"] = {"norb": n, "nelec": nelec}
    return circuit


def validate_schedule(time_step, krylov_indices):
    """时间由调用者给定；k=0 仅准备参考态。"""
    if not np.isfinite(time_step) or time_step <= 0:
        raise ValueError("time_step must be finite and positive")
    indices = tuple(index(k) for k in krylov_indices)
    if not indices or any(k < 0 for k in indices) or len(set(indices)) != len(indices):
        raise ValueError("krylov_indices must be distinct non-negative integers")
    return float(time_step), indices


def reference_circuit(initial_state, norb, nelec):
    """复制参考态门，防止追加演化时修改用户的输入线路。"""
    initial = prepare_slater_reference(norb, nelec) if initial_state is None else initial_state
    if initial.num_qubits != 2 * norb:
        raise ValueError("reference circuit must have 2*norb qubits")
    if getattr(initial, "_initial_state", None) is not None:
        raise ValueError("reference must use explicit preparation gates, not simulator inputs")
    if initial.instructions:
        raise ValueError("reference must use gate ops; instruction-based preparation is not supported")
    if any(op[0] in {"measure", "measure_z", "reset", "kraus"} for op in initial.ops):
        raise ValueError("reference circuit must contain only unitary preparation operations")
    copied = Circuit(2 * norb, ops=list(initial.ops))
    # Circuit.unitary 的矩阵保存在实例缓存，单独复制 ops 会丢掉参考态旋转。
    copied._unitary_cache = {key: np.array(value, copy=True)
                             for key, value in getattr(initial, "_unitary_cache", {}).items()}
    return copied


def build_trotter_krylov_circuits(
    h1e, eri, norb, nelec, *, time_step, krylov_indices=range(8),
    trotter_substeps=1, initial_state=None, interaction_rotation=None,
    gaussian_fit=None, basis_id="input",
):
    """构造 H2/2-H1-H2/2；H2 必须在给定 interaction_rotation 下为密度相互作用。

    interaction_rotation 的列是当前基底在 onsite 基底中的表示。
    因此每个 H2 半步先施加该旋转，再作 onsite 演化，最后施加逆旋转。
    gaussian_fit 若提供，必须是同一 h1e、delta_t、网络定义下的拟合结果。
    """
    h, g, n, nelec = validate_integrals(h1e, eri, norb, nelec)
    dt, indices = validate_schedule(time_step, krylov_indices)
    steps = index(trotter_substeps)
    if steps < 1:
        raise ValueError("trotter_substeps must be positive")
    small_dt = dt / steps
    transform = np.eye(n) if interaction_rotation is None else np.asarray(interaction_rotation)
    if (transform.shape != (n, n) or np.iscomplexobj(transform) or
        not np.allclose(transform.T @ transform, np.eye(n), atol=1e-10, rtol=0)):
        raise ValueError("interaction_rotation must be a real orthogonal matrix")
    onsite_g = np.einsum("ap,bq,cr,ds,pqrs->abcd", transform, transform,
                         transform, transform, g, optimize=True)
    density_g = np.zeros_like(onsite_g)
    for p in range(n):
        for q in range(n):
            density_g[p, p, q, q] = onsite_g[p, p, q, q]
    if not np.allclose(onsite_g, density_g, atol=1e-10, rtol=0):
        raise ValueError("H2 is not density-density in the supplied interaction basis; use SqDRIFT")
    one_body = expm(-1j * small_dt * h)
    if gaussian_fit is not None and not np.allclose(gaussian_fit.target, one_body, atol=1e-12, rtol=0):
        raise ValueError("gaussian_fit target does not match the requested Trotter substep")

    def half_interaction(circuit):
        for spin in range(2):
            append_orbital_rotation(circuit, transform, range(spin * n, (spin + 1) * n))
        for p in range(2 * n):
            for q in range(p + 1, 2 * n):
                coupling = density_g[p % n, p % n, q % n, q % n]
                if abs(coupling) > 1e-15:
                    append_cphase(circuit, p, q, -0.5 * small_dt * coupling)
        for spin in range(2):
            append_orbital_rotation(circuit, transform.T, range(spin * n, (spin + 1) * n))

    circuits, records = [], []
    identity = tuple(range(2 * n))
    for k in indices:
        circuit = reference_circuit(initial_state, n, nelec)
        for _ in range(k * steps):
            half_interaction(circuit)
            for spin in range(2):
                modes = range(spin * n, (spin + 1) * n)
                if gaussian_fit is None:
                    append_orbital_rotation(circuit, one_body, modes)
                else:
                    gaussian_fit.append_to(circuit, modes)
            half_interaction(circuit)
        record = CircuitRecord(f"trotter-k{k}", k, k * dt, mode_permutation=identity,
                               measurement_map=identity, basis_id=basis_id,
                               diagnostics={"substeps": k * steps,
                                            "gaussian_error": 0.0 if gaussian_fit is None else gaussian_fit.error})
        circuit.metadata["skqd"] = {"circuit_id": record.circuit_id, "k": k, "time": k * dt}
        circuits.append(circuit)
        records.append(record)
    return SKQDEnsemble(circuits, records, n, nelec, "trotter")
