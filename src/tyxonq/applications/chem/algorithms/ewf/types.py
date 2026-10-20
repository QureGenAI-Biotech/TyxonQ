"""定义分子嵌入计算的浴、簇、片段和求解结果数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np

IAOLabel = tuple[int, str, str, str]
SpinPair: TypeAlias = tuple[np.ndarray, np.ndarray]
SpinERI: TypeAlias = tuple[np.ndarray, np.ndarray, np.ndarray]
SpinArray: TypeAlias = np.ndarray | SpinPair


def _is_spin_pair(value: object) -> bool:
    """tuple 是 UHF 数据的显式标记；RHF 的 ndarray 语义保持不变。"""

    return isinstance(value, tuple) and len(value) == 2


@dataclass(frozen=True)
class BathData:
    """一个分片的 DMET 与 MP2-BNO bath 数据。"""

    c_dmet: SpinArray
    dmet_occupations: SpinArray
    c_cluster_occ: SpinArray
    c_cluster_vir: SpinArray
    c_env_occ: SpinArray
    c_env_vir: SpinArray
    c_mp2_occ: SpinArray
    c_mp2_vir: SpinArray
    c_frozen_occ: SpinArray
    c_frozen_vir: SpinArray
    mp2_occ_occupations: SpinArray
    mp2_vir_occupations: SpinArray
    mp2_occ_correlation_energy: float | None = None
    mp2_vir_correlation_energy: float | None = None
    bathtype: str = "mp2"
    threshold: float = 1e-8
    dmet_threshold: float = 1e-8
    canonicalize: bool = True
    project_dmet_order: int = 2
    project_dmet_mode: str = "squared-entropy"
    n_mp2_occ_kept: int = 0
    n_mp2_vir_kept: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Cluster:
    """active/frozen occupied/virtual 轨道与嵌入积分。"""

    c_active_occ: SpinArray
    c_active_vir: SpinArray
    c_frozen_occ: SpinArray
    c_frozen_vir: SpinArray
    fock: SpinArray
    h_eff: SpinArray
    eri: np.ndarray | SpinERI
    spin_orbital_overlap: np.ndarray | None = None

    @property
    def is_unrestricted(self) -> bool:
        return _is_spin_pair(self.c_active_occ)

    @property
    def c_active(self) -> SpinArray:
        if self.is_unrestricted:
            return tuple(
                np.hstack((self.c_active_occ[s], self.c_active_vir[s]))
                for s in range(2)
            )
        return np.hstack((self.c_active_occ, self.c_active_vir))

    @property
    def c_total_occ(self) -> SpinArray:
        if self.is_unrestricted:
            return tuple(
                np.hstack((self.c_frozen_occ[s], self.c_active_occ[s]))
                for s in range(2)
            )
        return np.hstack((self.c_frozen_occ, self.c_active_occ))

    @property
    def c_total_vir(self) -> SpinArray:
        if self.is_unrestricted:
            return tuple(
                np.hstack((self.c_active_vir[s], self.c_frozen_vir[s]))
                for s in range(2)
            )
        return np.hstack((self.c_active_vir, self.c_frozen_vir))

    @property
    def c_total(self) -> SpinArray:
        if self.is_unrestricted:
            return tuple(
                np.hstack((self.c_total_occ[s], self.c_total_vir[s]))
                for s in range(2)
            )
        return np.hstack((self.c_total_occ, self.c_total_vir))

    @property
    def nocc(self) -> int | tuple[int, int]:
        if self.is_unrestricted:
            return tuple(int(value.shape[1]) for value in self.c_active_occ)
        return int(self.c_active_occ.shape[1])

    @property
    def physical_nvir(self) -> int | tuple[int, int]:
        if self.is_unrestricted:
            return tuple(int(value.shape[1]) for value in self.c_active_vir)
        return int(self.c_active_vir.shape[1])

    @property
    def nvir(self) -> int | tuple[int, int]:
        if self.is_unrestricted:
            return tuple(self.norb - nocc for nocc in self.nocc)
        return int(self.c_active_vir.shape[1])

    @property
    def physical_norb(self) -> int | tuple[int, int]:
        if self.is_unrestricted:
            return tuple(
                int(self.c_active_occ[s].shape[1] + self.c_active_vir[s].shape[1])
                for s in range(2)
            )
        return int(self.nocc + self.nvir)

    @property
    def norb(self) -> int:
        if self.is_unrestricted:
            return max(self.physical_norb)
        return self.nocc + self.nvir

    @property
    def dummy_orbitals(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """返回各 spin channel 为统一 ``norb`` 添加的末尾 dummy virtual。"""

        if not self.is_unrestricted:
            return ((), ())
        return tuple(
            tuple(range(nphysical, self.norb)) for nphysical in self.physical_norb
        )

    @property
    def nelec(self) -> tuple[int, int]:
        if self.is_unrestricted:
            return self.nocc
        return self.nocc, self.nocc


@dataclass(frozen=True)
class SolverResult:
    """cluster CCSD/FCI/selected-CI 求解结果。"""

    method: str
    converged: bool
    energy: float
    rdm1: SpinArray
    rdm2: np.ndarray | SpinERI
    t1: SpinArray | None = None
    t2: np.ndarray | SpinERI | None = None
    l1: SpinArray | None = None
    l2: np.ndarray | SpinERI | None = None
    ci: np.ndarray | None = None
    reference_amplitude: float | complex | None = None
    cisd_c1: SpinArray | None = None
    cisd_c2: np.ndarray | SpinERI | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FragmentResult:
    """单个分片从 bath 到能量重构的完整结果。"""

    fragment_id: int
    name: str
    atoms: tuple[int, ...]
    full_orbital_indices: tuple[int, ...]
    active_orbital_indices: tuple[int, ...]
    full_labels: tuple[IAOLabel, ...]
    active_labels: tuple[IAOLabel, ...]
    bath: BathData
    cluster: Cluster
    solver: SolverResult
    e_corr_wf: float | None
    e_dmet: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EWFResult:
    """一次 EWF 计算的总能量与完整诊断。"""

    e_mf: float
    e_corr: float
    e_tot: float
    energy_functional: str
    converged: bool
    fragments: tuple[FragmentResult, ...]
    fragmentation_complete: bool
    owner_projector_error: float
    n_owner_orbitals: int
    n_active_owner_orbitals: int
    n_unassigned_owner_orbitals: int
    diagnostics: dict[str, Any] = field(default_factory=dict)
