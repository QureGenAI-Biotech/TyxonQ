"""协调分子 RHF/UHF 的分片、建浴、片段求解、能量与密度组装。"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any

import numpy as np
from pyscf import gto, scf

from .bath import build_bath
from .cluster import build_cluster
from .energy import (
    dmet_total_energy,
    fragment_dmet_energy,
    fragment_wf_correlation_energy,
)
from .fragmentation import IAOFragmentation
from .linalg import hstack, metric_overlap, orthonormality_error
from .solver import solve_cluster
from .types import Cluster, EWFResult, FragmentResult, SolverResult

_BATH_DEFAULTS: dict[str, Any] = {
    "bathtype": "mp2",
    "threshold": 1e-8,
    "dmet_threshold": 1e-8,
    "canonicalize": True,
    "project_dmet_order": 2,
    "project_dmet_mode": "squared-entropy",
    "occupation_tolerance": 1e-6,
}
_BATH_TYPES = {"dmet", "mp2", "full"}
_PROJECT_MODES = {
    "full",
    "half",
    "linear",
    "cosine",
    "cosine-half",
    "entropy",
    "sqrt-entropy",
    "squared-entropy",
}
_CCSD_OPTIONS = {
    "max_cycle",
    "conv_tol",
    "conv_tol_normt",
    "diis_space",
    "diis_start_cycle",
    "iterative_damping",
    "level_shift",
    "solve_lambda",
}
_FCI_OPTIONS = {
    "max_cycle",
    "conv_tol",
    "lindep",
    "davidson_only",
    "fix_spin",
    "fix_spin_penalty",
    "threads",
}


def _validated_mean_field(mf):
    """规范化 RHF/UHF；ROHF 按 Vayesta 约定转换成 UHF。"""

    if isinstance(mf, scf.rohf.ROHF):
        if not bool(getattr(mf, "converged", False)):
            raise ValueError("The ROHF calculation must be converged before constructing EWF.")
        warnings.warn(
            "ROHF reference is converted to UHF for open-shell EWF.",
            RuntimeWarning,
            stacklevel=3,
        )
        mf = mf.to_uhf()
    if not isinstance(mf, (scf.hf.RHF, scf.uhf.UHF)):
        raise TypeError("EWF supports molecular RHF, UHF, or ROHF references.")
    if not isinstance(mf.mol, gto.Mole) or hasattr(mf.mol, "dimension"):
        raise TypeError("Periodic cells and k-points are not supported.")
    if not bool(getattr(mf, "converged", False)):
        raise ValueError("The mean-field calculation must be converged before constructing EWF.")
    unrestricted = isinstance(mf, scf.uhf.UHF)
    mo_coeff = np.asarray(mf.mo_coeff)
    mo_occ = np.asarray(mf.mo_occ)
    if np.iscomplexobj(mo_coeff) or np.iscomplexobj(mf.get_ovlp()):
        raise ValueError("Complex molecular orbitals are not supported.")
    if unrestricted:
        if mo_coeff.ndim != 3 or mo_occ.ndim != 2 or mo_coeff.shape[0] != 2:
            raise TypeError("UHF mo_coeff/mo_occ must contain alpha and beta arrays.")
        if not np.all(np.isclose(mo_occ, 0.0) | np.isclose(mo_occ, 1.0)):
            raise ValueError("UHF occupations must be exactly 0 or 1.")
        counts = tuple(int(np.count_nonzero(value > 0)) for value in mo_occ)
        if counts != tuple(int(value) for value in mf.mol.nelec):
            raise ValueError("UHF occupations do not match mol.nelec.")
    else:
        if int(mf.mol.spin) != 0 or int(mf.mol.nelectron) % 2:
            raise ValueError("RHF EWF requires spin=0 and an even electron count.")
        if mo_coeff.ndim != 2 or mo_occ.ndim != 1:
            raise TypeError("RHF mo_coeff/mo_occ shapes are invalid.")
        if not np.all(np.isclose(mo_occ, 0.0) | np.isclose(mo_occ, 2.0)):
            raise ValueError("RHF occupations must be exactly 0 or 2.")
        if int(np.count_nonzero(mo_occ > 0) * 2) != int(mf.mol.nelectron):
            raise ValueError("RHF occupations do not match mol.nelectron.")
    return mf


def _validated_bath_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    values = _BATH_DEFAULTS.copy()
    supplied = dict(options or {})
    unknown = sorted(set(supplied) - set(values))
    if unknown:
        raise TypeError(f"Unsupported bath option(s): {', '.join(unknown)}.")
    values.update(supplied)
    values["bathtype"] = str(values["bathtype"]).lower()
    if values["bathtype"] not in _BATH_TYPES:
        raise ValueError(f"Unsupported bathtype: {values['bathtype']!r}.")
    values["threshold"] = float(values["threshold"])
    values["dmet_threshold"] = float(values["dmet_threshold"])
    values["occupation_tolerance"] = float(values["occupation_tolerance"])
    if values["threshold"] < 0:
        raise ValueError("bath threshold must be non-negative.")
    if not 0 < values["dmet_threshold"] < 0.5:
        raise ValueError("dmet_threshold must be in (0, 0.5).")
    if values["occupation_tolerance"] <= 0:
        raise ValueError("occupation_tolerance must be positive.")
    if not isinstance(values["canonicalize"], (bool, np.bool_)):
        raise TypeError("canonicalize must be a boolean.")
    raw_order = values["project_dmet_order"]
    if isinstance(raw_order, (bool, np.bool_)):
        raise TypeError("project_dmet_order must be an integer 0, 1, or 2.")
    try:
        converted_order = int(raw_order)
    except (TypeError, ValueError) as exc:
        raise TypeError("project_dmet_order must be an integer 0, 1, or 2.") from exc
    if raw_order != converted_order:
        raise TypeError("project_dmet_order must be an integer 0, 1, or 2.")
    values["project_dmet_order"] = converted_order
    if values["project_dmet_order"] not in (0, 1, 2):
        raise ValueError("project_dmet_order must be 0, 1, or 2.")
    values["project_dmet_mode"] = str(values["project_dmet_mode"]).lower()
    if values["project_dmet_mode"] not in _PROJECT_MODES:
        raise ValueError(
            f"Unsupported project_dmet_mode: {values['project_dmet_mode']!r}."
        )
    return values


class EWF:
    """分子 RHF/UHF 的最小 Embedded Wavefunction Theory 驱动。"""

    def __init__(
        self,
        mf,
        solver: str = "CCSD",
        bath_options: Mapping[str, Any] | None = None,
        solver_options: Mapping[str, Any] | None = None,
        energy_functional: str = "wf",
        *,
        cluster_solver: Callable[[Any, Cluster, Any], SolverResult] | None = None,
        density_assembly: str = "rdm_t_lambda",
    ) -> None:
        mf = _validated_mean_field(mf)
        solver = str(solver).upper()
        if solver not in {"CCSD", "FCI"}:
            raise ValueError("solver must be 'CCSD' or 'FCI'.")
        energy_functional = str(energy_functional).lower()
        if energy_functional not in {"wf", "dmet", "density"}:
            raise ValueError("energy_functional must be 'wf', 'dmet', or 'density'.")
        if cluster_solver is not None and not callable(cluster_solver):
            raise TypeError("cluster_solver must be callable.")
        if density_assembly not in {"rdm_t", "rdm_t_lambda"}:
            raise ValueError("density_assembly must be 'rdm_t' or 'rdm_t_lambda'.")
        if energy_functional == "density" and isinstance(mf, scf.uhf.UHF):
            raise NotImplementedError("The density functional currently requires RHF.")
        if energy_functional == "density":
            from .density import _validate_reference

            _validate_reference(mf)

        solver_options = dict(solver_options or {})
        supported = _CCSD_OPTIONS if solver == "CCSD" else _FCI_OPTIONS
        unknown = sorted(set(solver_options) - supported)
        if unknown:
            raise TypeError(f"Unsupported {solver} option(s): {', '.join(unknown)}.")

        self.mf = mf
        self.mol = mf.mol
        self.solver = solver
        self.bath_options = _validated_bath_options(bath_options)
        self.solver_options = solver_options
        self.energy_functional = energy_functional
        self.cluster_solver = cluster_solver
        self.density_assembly = density_assembly
        self.density_result = None
        self.fragments: list[Any] = []
        self.result: EWFResult | None = None
        self.e_mf = float(mf.e_tot)
        self.e_corr: float | None = None
        self.e_tot: float | None = None
        self.converged = False
        self._iao_data = None
        self._has_run = False
        self._incomplete_warning_emitted = False

    def __repr__(self) -> str:
        return f"EWF(mf={self.mf!r}, solver={self.solver!r})"

    def iao_fragmentation(self, *, minao: str = "auto") -> IAOFragmentation:
        """创建 IAO 分片 context manager。"""
        if self._has_run:
            raise RuntimeError("Fragmentation cannot be changed after EWF.kernel().")
        return IAOFragmentation(self, minao=minao)

    fragmentation = iao_fragmentation

    def _fragmentation_diagnostics(self) -> dict[str, Any]:
        if not self.fragments:
            raise RuntimeError("No EWF fragments were constructed.")
        first = self.fragments[0].fragmentation
        overlap = np.asarray(self.mf.get_ovlp(), dtype=float)
        if first.is_unrestricted:
            errors = []
            for spin in range(2):
                owner_coeff = np.asarray(first.owner_coeff[spin], dtype=float)
                active_coeff = hstack(
                    *(fragment.c_frag[spin] for fragment in self.fragments),
                    nrow=owner_coeff.shape[0],
                )
                rotation = metric_overlap(owner_coeff, overlap, active_coeff)
                errors.append(float(np.linalg.norm(np.eye(owner_coeff.shape[1]) - rotation @ rotation.T)))
            projector_error = max(errors)
            n_owner = int(first.owner_coeff[0].shape[1])
        else:
            owner_coeff = np.asarray(first.owner_coeff, dtype=float)
            active_coeff = hstack(*(fragment.c_frag for fragment in self.fragments), nrow=owner_coeff.shape[0])
            rotation = metric_overlap(owner_coeff, overlap, active_coeff)
            defect = np.eye(owner_coeff.shape[1]) - rotation @ rotation.T
            projector_error = float(np.linalg.norm(defect))
            n_owner = int(owner_coeff.shape[1])

        selected: set[int] = set()
        for fragment in self.fragments:
            selected.update(int(x) for x in fragment.active_orbital_indices)
        n_active = len(selected)
        return {
            "fragmentation_complete": projector_error <= 1e-8,
            "owner_projector_error": projector_error,
            "n_owner_orbitals": n_owner,
            "n_active_owner_orbitals": n_active,
            "n_unassigned_owner_orbitals": n_owner - n_active,
        }

    def kernel(self) -> float:
        """构造所有 fragment cluster、运行 solver 并重构分子能量。"""
        started = perf_counter()
        # 失败后不保留上一次的成功标记，避免 scanner 使用陈旧结果。
        self.converged = False
        self.result = None
        self.density_result = None
        if not self.fragments:
            with self.iao_fragmentation() as fragmentation:
                fragmentation.add_all_atomic_fragments()
        self._has_run = True
        for fragment in self.fragments:
            fragment.lock()

        frag_diagnostics = self._fragmentation_diagnostics()
        if (
            not frag_diagnostics["fragmentation_complete"]
            and not self._incomplete_warning_emitted
        ):
            warnings.warn(
                "IAO owner fragmentation is incomplete; EWF energy is computed but "
                f"owner_projector_error={frag_diagnostics['owner_projector_error']:.3e}.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._incomplete_warning_emitted = True

        fragment_results: list[FragmentResult] = []
        for fragment in self.fragments:
            build_started = perf_counter()
            bath = build_bath(fragment, **self.bath_options)
            cluster = build_cluster(
                self.mf,
                bath,
                canonicalize=bool(self.bath_options["canonicalize"]),
            )
            build_seconds = perf_counter() - build_started
            solve_started = perf_counter()
            if self.cluster_solver is None:
                solver_result = solve_cluster(
                    cluster, self.mf, method=self.solver, options=self.solver_options,
                )
            else:
                solver_result = self.cluster_solver(fragment, cluster, self.mf)
                if not isinstance(solver_result, SolverResult):
                    raise TypeError("cluster_solver must return an EWF SolverResult.")
            solve_seconds = perf_counter() - solve_started
            e_corr_wf = None
            if self.energy_functional == "wf":
                e_corr_wf = float(fragment_wf_correlation_energy(
                    self.mf, cluster, solver_result, fragment.c_proj,
                )[2])
            if cluster.is_unrestricted:
                cluster_orthonormality_error = max(
                    orthonormality_error(value, self.mf.get_ovlp())
                    for value in cluster.c_active
                )
            else:
                cluster_orthonormality_error = orthonormality_error(
                    cluster.c_active, self.mf.get_ovlp()
                )
            diagnostics = {
                "cluster_orthonormality_error": cluster_orthonormality_error,
                "full_fragment_orbitals": tuple(fragment.full_orbital_indices),
                "active_fragment_orbitals": tuple(fragment.active_orbital_indices),
                "active_fragment_labels": tuple(fragment.active_labels),
                "build_seconds": build_seconds,
                "solve_seconds": solve_seconds,
                "solver": dict(solver_result.diagnostics),
            }
            result = FragmentResult(
                fragment_id=int(fragment.id),
                name=str(fragment.name),
                atoms=tuple(int(x) for x in fragment.atoms),
                full_orbital_indices=tuple(int(x) for x in fragment.full_orbital_indices),
                active_orbital_indices=tuple(int(x) for x in fragment.active_orbital_indices),
                full_labels=tuple(fragment.full_labels),
                active_labels=tuple(fragment.active_labels),
                bath=bath,
                cluster=cluster,
                solver=solver_result,
                e_corr_wf=e_corr_wf,
                diagnostics=diagnostics,
            )
            fragment.bath = bath
            fragment.cluster = cluster
            fragment.solver_result = solver_result
            fragment.result = result
            fragment_results.append(result)

        if self.energy_functional == "wf":
            e_corr = float(sum(result.e_corr_wf for result in fragment_results))
            e_tot = self.e_mf + e_corr
        elif self.energy_functional == "density":
            from .density import assemble_density

            if not all(result.solver.converged for result in fragment_results):
                raise RuntimeError("Density assembly requires converged fragment solvers.")
            inputs = tuple(
                (result.cluster, result.solver, fragment.c_proj)
                for fragment, result in zip(self.fragments, fragment_results)
            )
            self.density_result = assemble_density(
                self.mf, inputs, assembly=self.density_assembly,
            )
            e_tot = self.density_result.energy
            e_corr = e_tot - self.e_mf
            frag_diagnostics["density"] = dict(self.density_result.diagnostics)
        else:
            inputs = tuple(
                (result.cluster, result.solver, fragment.c_proj)
                for fragment, result in zip(self.fragments, fragment_results)
            )
            per_fragment = tuple(
                fragment_dmet_energy(
                    self.mf,
                    result.cluster,
                    result.solver,
                    fragment.c_proj,
                    part_cumulant=True,
                    approx_cumulant=True,
                )
                for fragment, result in zip(self.fragments, fragment_results)
            )
            e_tot = dmet_total_energy(
                self.mf,
                inputs,
                part_cumulant=True,
                approx_cumulant=True,
            )
            fragment_results = [
                dataclasses.replace(result, e_dmet=float(energy))
                for result, energy in zip(fragment_results, per_fragment)
            ]
            for fragment, result in zip(self.fragments, fragment_results):
                fragment.result = result
            e_corr = float(e_tot - self.e_mf)

        self.e_corr = float(e_corr)
        self.e_tot = float(e_tot)
        self.converged = all(result.solver.converged for result in fragment_results)
        frag_diagnostics["kernel_seconds"] = perf_counter() - started
        self.result = EWFResult(
            e_mf=self.e_mf,
            e_corr=self.e_corr,
            e_tot=self.e_tot,
            energy_functional=self.energy_functional,
            converged=self.converged,
            fragments=tuple(fragment_results),
            fragmentation_complete=bool(frag_diagnostics["fragmentation_complete"]),
            owner_projector_error=float(frag_diagnostics["owner_projector_error"]),
            n_owner_orbitals=int(frag_diagnostics["n_owner_orbitals"]),
            n_active_owner_orbitals=int(frag_diagnostics["n_active_owner_orbitals"]),
            n_unassigned_owner_orbitals=int(frag_diagnostics["n_unassigned_owner_orbitals"]),
            diagnostics=dict(frag_diagnostics),
        )
        return self.e_tot

    def _require_result(self) -> EWFResult:
        if self.result is None:
            raise RuntimeError("Run EWF.kernel() before requesting an energy.")
        return self.result

    def get_wf_energy(self) -> float:
        """返回 first-index wavefunction 投影重构的总能量。"""
        result = self._require_result()
        # 密度求解不需要 C0；只有调用者显式请求 wf 时才尝试转换。
        contributions = [
            x.e_corr_wf if x.e_corr_wf is not None else
            fragment_wf_correlation_energy(self.mf, x.cluster, x.solver, frag.c_proj)[2]
            for frag, x in zip(self.fragments, result.fragments)
        ]
        return float(self.e_mf + sum(contributions))

    def nuclear_gradient(self) -> np.ndarray:
        """返回 frozen-bath 近似核梯度，形状 ``(natm, 3)``、Hartree/Bohr。

        不包含完整浴轨道及片段投影响应，不能视为完整 EWF 能量的严格导数。
        """
        result = self._require_result()
        if self.energy_functional != "density" or self.density_result is None:
            raise ValueError("nuclear_gradient requires energy_functional='density'.")
        if not self.mf.converged or not self.converged:
            raise RuntimeError("nuclear_gradient requires converged SCF and fragment solvers.")
        if not result.fragmentation_complete:
            raise ValueError("nuclear_gradient requires complete fragment coverage.")
        from .gradient import nuclear_gradient

        started = perf_counter()
        gradient = nuclear_gradient(self.mf, self.density_result)
        result.diagnostics["density"] = dict(self.density_result.diagnostics)
        result.diagnostics["gradient_seconds"] = perf_counter() - started
        result.diagnostics["gradient_approximation"] = "frozen_bath"
        return gradient

    def get_dmet_energy(
        self,
        part_cumulant: bool = True,
        approx_cumulant: bool = True,
    ) -> float:
        """返回 democratically partitioned DMET 总能量。"""
        result = self._require_result()
        inputs = tuple(
            (fragment_result.cluster, fragment_result.solver, fragment.c_proj)
            for fragment, fragment_result in zip(self.fragments, result.fragments)
        )
        return float(
            dmet_total_energy(
                self.mf,
                inputs,
                part_cumulant=part_cumulant,
                approx_cumulant=approx_cumulant,
            )
        )

    def get_e_corr(self, functional: str | None = None, **kwargs) -> float:
        """返回指定能量 functional 的分子关联能。"""
        functional = (functional or self.energy_functional).lower()
        if functional == "wf":
            return self.get_wf_energy() - self.e_mf
        if functional == "dmet":
            return self.get_dmet_energy(**kwargs) - self.e_mf
        if functional == "density":
            self._require_result()
            if self.density_result is None:
                raise ValueError("Run kernel with energy_functional='density' first.")
            return float(self.density_result.energy - self.e_mf)
        raise ValueError(f"Unknown energy functional: {functional!r}.")
