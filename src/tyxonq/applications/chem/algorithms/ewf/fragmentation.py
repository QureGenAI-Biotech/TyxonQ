"""构造分子 RHF/UHF 的 IAO 原子片段、轨道归属及投影。"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
from pyscf import lo
from pyscf.scf.hf import RHF
from pyscf.scf.rohf import ROHF
from pyscf.scf.uhf import UHF

from .linalg import (
    assert_orthonormal,
    eigh,
    fix_orbital_sign,
    hstack,
    metric_overlap,
    symmetric_orthogonalizer,
)
from .types import IAOLabel


_AUTO_MINAO = {
    "sto3g": "sto-3g",
    "sto6g": "sto-6g",
}


def get_default_minao(basis: Any) -> str:
    """返回与 Vayesta 分子分支一致的默认 IAO 参考基组。"""
    if not isinstance(basis, str):
        return "minao"
    normalized = basis.replace("-", "").lower()
    return _AUTO_MINAO.get(normalized, "minao")


def _as_items(value: Any, *, name: str, allow_scalar: bool) -> list[Any]:
    """把用户输入转换成非空列表，同时拒绝易产生歧义的类型。"""
    if isinstance(value, (str, int, np.integer)) and not isinstance(value, bool):
        if not allow_scalar:
            raise TypeError(f"{name} 必须是整数序列或字符串序列。")
        items = [value]
    elif isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise TypeError(f"{name} 必须是一维序列。")
        items = value.tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = list(value)
    else:
        raise TypeError(f"{name} 必须是整数序列或字符串序列。")

    if not items:
        raise ValueError(f"{name} 不能为空。")
    if any(isinstance(item, bool) for item in items):
        raise TypeError(f"{name} 不能包含布尔值。")

    all_integer = all(isinstance(item, (int, np.integer)) for item in items)
    all_string = all(isinstance(item, str) for item in items)
    if not (all_integer or all_string):
        raise TypeError(f"{name} 不能混合整数和字符串。")
    if all_string and any(not item.strip() for item in items):
        raise ValueError(f"{name} 不能包含空字符串。")
    return items


def _format_label(label: IAOLabel) -> str:
    """把 PySCF AO 标签转换成便于展示的单行字符串。"""
    atom, symbol, shell, component = label
    return f"{atom}{symbol:>3} {shell}{component}".strip()


class Fragment:
    """一个 IAO owner fragment 及其可选 active 子空间。"""

    def __init__(
        self,
        fragmentation: IAOFragmentation,
        fragment_id: int,
        name: str,
        full_orbital_indices: Sequence[int],
        *,
        atoms: Sequence[int] = (),
    ) -> None:
        self.fragmentation = fragmentation
        self.base = fragmentation.embedding
        self.fragment_id = int(fragment_id)
        self.id = self.fragment_id
        self.name = str(name)
        self.atoms = tuple(int(atom) for atom in atoms)
        self.full_orbital_indices = tuple(int(index) for index in full_orbital_indices)
        self.active_orbital_indices = self.full_orbital_indices
        self.full_labels = tuple(fragmentation.labels[index] for index in self.full_orbital_indices)
        self.active_labels = self.full_labels

        # driver 在 EWF.kernel 开始时锁定分片，防止 bath 构造后再换空间。
        self._locked = False
        self._active_selection_applied = False
        self.bath = None
        self.cluster = None
        self.solver_result = None
        self.result = None
        self._refresh_coefficients()

    @property
    def mf(self) -> RHF | UHF:
        return self.fragmentation.mf

    @property
    def mol(self):
        return self.mf.mol

    @property
    def c_full(self):
        """完整 owner IAO 系数，不随 active 选择改变。"""
        if self.fragmentation.is_unrestricted:
            return tuple(
                coeff[:, self.full_orbital_indices].copy()
                for coeff in self.fragmentation.coeff
            )
        return self.fragmentation.coeff[:, self.full_orbital_indices].copy()

    @property
    def full_coeff(self):
        return self.c_full

    @property
    def coeff(self):
        """当前 active fragment 系数。"""
        return self.c_frag

    @property
    def orbital_indices(self) -> tuple[int, ...]:
        return self.active_orbital_indices

    @property
    def labels(self) -> tuple[IAOLabel, ...]:
        return self.active_labels

    @property
    def n_frag(self) -> int | tuple[int, int]:
        if self.fragmentation.is_unrestricted:
            return tuple(int(coeff.shape[1]) for coeff in self.c_frag)
        return int(self.c_frag.shape[1])

    @property
    def n_full_orbitals(self) -> int:
        return len(self.full_orbital_indices)

    @property
    def n_active_orbitals(self) -> int:
        return len(self.active_orbital_indices)

    @property
    def has_active_selection(self) -> bool:
        return self._active_selection_applied

    @property
    def nelectron(self) -> float | tuple[float, float]:
        """当前 active fragment 在 mean-field 密度中的平均电子数。"""
        dm1 = self.mf.make_rdm1()
        if self.fragmentation.is_unrestricted:
            return tuple(
                float(
                    np.einsum(
                        "ai,ab,bi->",
                        self.fragmentation.overlap @ self.c_frag[s],
                        np.asarray(dm1[s], dtype=float),
                        self.fragmentation.overlap @ self.c_frag[s],
                        optimize=True,
                    )
                )
                for s in range(2)
            )
        overlap_coeff = self.fragmentation.overlap @ self.c_frag
        dm1 = np.asarray(dm1, dtype=float)
        return float(np.einsum("ai,ab,bi->", overlap_coeff, dm1, overlap_coeff, optimize=True))

    def _refresh_coefficients(self) -> None:
        """由 active 全局 IAO 索引重建 fragment、projector 和环境。"""
        coeff = self.fragmentation.coeff
        if self.fragmentation.is_unrestricted:
            self.c_frag = tuple(
                value[:, self.active_orbital_indices].copy() for value in coeff
            )
            self.c_proj = tuple(value.copy() for value in self.c_frag)
            environments = []
            for value in coeff:
                environment = np.ones(value.shape[1], dtype=bool)
                environment[list(self.active_orbital_indices)] = False
                environments.append(value[:, environment].copy())
            self.c_env = tuple(environments)
        else:
            self.c_frag = coeff[:, self.active_orbital_indices].copy()
            self.c_proj = self.c_frag
            environment = np.ones(coeff.shape[1], dtype=bool)
            environment[list(self.active_orbital_indices)] = False
            self.c_env = coeff[:, environment].copy()

    def set_active_orbitals(self, orbitals: Sequence[int] | Sequence[str]) -> Fragment:
        """用 fragment-local IAO 位置或 PySCF 标签选取 active fragment。

        整数是当前 fragment 的完整 owner IAO 中的局部位置；字符串使用
        ``Mole.search_ao_label`` 的匹配规则。函数不进行能量排序或隐式冻结。
        """
        if self._locked:
            raise RuntimeError("EWF.kernel 已开始，不能再修改 fragment active orbitals。")

        items = _as_items(orbitals, name="orbitals", allow_scalar=False)
        if all(isinstance(item, (int, np.integer)) for item in items):
            local_indices = [int(item) for item in items]
            if len(set(local_indices)) != len(local_indices):
                raise ValueError("active orbital 局部索引不能重复。")
            invalid = [index for index in local_indices if index < 0 or index >= self.n_full_orbitals]
            if invalid:
                raise ValueError(
                    "active orbital 局部索引越界："
                    f"{invalid}；合法范围为 0..{self.n_full_orbitals - 1}。"
                )
            active_indices = tuple(self.full_orbital_indices[index] for index in local_indices)
        else:
            patterns = [str(item) for item in items]
            active_indices = self.fragmentation.search_labels(
                patterns,
                allowed_indices=self.full_orbital_indices,
                input_name="active orbital label",
            )

        self.active_orbital_indices = active_indices
        self.active_labels = tuple(self.fragmentation.labels[index] for index in active_indices)
        self._active_selection_applied = True
        self.bath = None
        self.cluster = None
        self.solver_result = None
        self.result = None
        self._refresh_coefficients()
        return self

    def lock(self) -> None:
        """锁定分片；由 EWF driver 在开始构造 bath 时调用。"""
        self._locked = True


class IAOFragmentation:
    """Vayesta 风格的分子 RHF/UHF IAO fragmentation context manager。"""

    def __init__(self, embedding: Any, *, minao: str = "auto") -> None:
        self.embedding = embedding
        self.mf = getattr(embedding, "mf", embedding)
        self._validate_reference()

        if not isinstance(minao, str) or not minao.strip():
            raise TypeError("minao 必须是非空字符串。")
        self.minao = get_default_minao(self.mf.mol.basis) if minao.lower() == "auto" else minao
        self.overlap = np.asarray(self.mf.get_ovlp(), dtype=float)
        self.refmol = lo.iao.reference_mol(self.mf.mol, minao=self.minao)

        self.coeff = None
        self.owner_coeff = None
        self.virtual_coeff = None
        self.labels: tuple[IAOLabel, ...] = ()
        self.fragments: list[Fragment] = []
        self._registered = False
        self._in_context = False

    @property
    def mol(self):
        return self.mf.mol

    @property
    def is_unrestricted(self) -> bool:
        return isinstance(self.mf, UHF)

    @property
    def n_iao(self) -> int:
        if self.owner_coeff is None:
            return int(self.refmol.nao)
        coeff = self.owner_coeff[0] if self.is_unrestricted else self.owner_coeff
        return int(coeff.shape[1])

    @property
    def n_owner_orbitals(self) -> int:
        return self.n_iao

    @property
    def active_owner_indices(self) -> tuple[int, ...]:
        return tuple(sorted({index for fragment in self.fragments for index in fragment.active_orbital_indices}))

    @property
    def n_active_owner_orbitals(self) -> int:
        return len(self.active_owner_indices)

    @property
    def n_unassigned_owner_orbitals(self) -> int:
        return self.n_owner_orbitals - self.n_active_owner_orbitals

    @property
    def unassigned_owner_indices(self) -> tuple[int, ...]:
        active = set(self.active_owner_indices)
        return tuple(index for index in range(self.n_owner_orbitals) if index not in active)

    @property
    def owner_projector_error(self) -> float:
        """返回 active fragment 并集相对完整 owner IAO 空间的 projector 误差。"""
        self._ensure_initialized()
        if not self.fragments:
            return float("inf")
        if self.is_unrestricted:
            errors = []
            for spin in range(2):
                active = hstack(
                    *(fragment.c_frag[spin] for fragment in self.fragments),
                    nrow=self.coeff[spin].shape[0],
                )
                rotation = metric_overlap(self.owner_coeff[spin], self.overlap, active)
                errors.append(
                    float(np.linalg.norm(np.eye(self.n_owner_orbitals) - rotation @ rotation.T))
                )
            return max(errors)
        active = hstack(*(fragment.c_frag for fragment in self.fragments), nrow=self.coeff.shape[0])
        rotation = metric_overlap(self.owner_coeff, self.overlap, active)
        projector = rotation @ rotation.T
        return float(np.linalg.norm(np.eye(self.n_owner_orbitals) - projector))

    @property
    def fragmentation_complete(self) -> bool:
        return self.owner_projector_error <= 1e-8

    def diagnostics(self) -> dict[str, Any]:
        """返回 owner-space 完备性和 active 选择明细。"""
        return {
            "fragmentation_complete": self.fragmentation_complete,
            "owner_projector_error": self.owner_projector_error,
            "n_owner_orbitals": self.n_owner_orbitals,
            "n_active_owner_orbitals": self.n_active_owner_orbitals,
            "n_unassigned_owner_orbitals": self.n_unassigned_owner_orbitals,
            "unassigned_owner_indices": self.unassigned_owner_indices,
        }

    def __enter__(self) -> IAOFragmentation:
        if self._in_context:
            raise RuntimeError("IAOFragmentation 不支持嵌套进入同一个 context。")
        self.kernel()
        self._in_context = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self._in_context = False
        if exc_type is None:
            self.register()
        return False

    def register(self) -> None:
        """把 context 中创建的分片一次性注册到 EWF driver。"""
        if self._registered:
            return
        fragments = getattr(self.embedding, "fragments", None)
        if fragments is None:
            self.embedding.fragments = []
            fragments = self.embedding.fragments
        if not hasattr(fragments, "extend"):
            raise TypeError("embedding.fragments 必须是可变列表。")
        fragments.extend(self.fragments)
        self._registered = True

    def kernel(self) -> IAOFragmentation:
        """构造正交 IAO、无标签虚补空间和完整分子轨道基。"""
        if self.coeff is not None:
            return self
        if self.is_unrestricted:
            mo_coeff = tuple(np.asarray(value, dtype=float) for value in self.mf.mo_coeff)
            mo_occ = tuple(np.asarray(value, dtype=float) for value in self.mf.mo_occ)
            built = tuple(
                self._make_spin_iao_basis(mo_coeff[s], mo_occ[s], spin=s)
                for s in range(2)
            )
            c_iao = tuple(value[0] for value in built)
            c_virtual = tuple(value[1] for value in built)
            coeff = tuple(value[2] for value in built)
        else:
            mo_coeff = np.asarray(self.mf.mo_coeff, dtype=float)
            mo_occ = np.asarray(self.mf.mo_occ, dtype=float)
            c_iao, c_virtual, coeff = self._make_spin_iao_basis(
                mo_coeff, mo_occ, spin=None
            )

        labels = self._make_labels()
        owner_for_labels = c_iao[0] if self.is_unrestricted else c_iao
        if len(labels) != owner_for_labels.shape[1]:
            raise RuntimeError(
                f"IAO 标签数与 owner IAO 数不一致：{len(labels)} != {owner_for_labels.shape[1]}。"
            )
        self.owner_coeff = c_iao
        self.virtual_coeff = c_virtual
        self.coeff = coeff
        self.labels = labels
        return self

    def add_atomic_fragment(
        self,
        atoms: int | str | Sequence[int] | Sequence[str],
        *,
        orbital_filter: str | Sequence[str] | None = None,
        name: str | None = None,
    ) -> Fragment:
        """添加由一个或多个原子拥有的 IAO fragment。"""
        self._ensure_initialized()
        atom_indices, atom_symbols = self._resolve_atoms(atoms)
        selected_atoms = set(atom_indices)
        indices = tuple(index for index, label in enumerate(self.labels) if label[0] in selected_atoms)

        if orbital_filter is not None:
            filters = _as_items(orbital_filter, name="orbital_filter", allow_scalar=True)
            if not all(isinstance(item, str) for item in filters):
                raise TypeError("orbital_filter 必须是字符串或字符串序列。")
            escaped = [re.escape(str(item)) for item in filters]
            indices = self.search_labels(
                escaped,
                allowed_indices=indices,
                input_name="orbital_filter",
            )

        fragment_name = name if name is not None else "/".join(atom_symbols)
        return self._create_fragment(indices, fragment_name, atoms=atom_indices)

    def add_orbital_fragment(
        self,
        orbitals: int | str | Sequence[int] | Sequence[str],
        *,
        name: str | None = None,
    ) -> Fragment:
        """按全局 owner IAO 索引或 PySCF 标签添加 fragment。"""
        self._ensure_initialized()
        items = _as_items(orbitals, name="orbitals", allow_scalar=True)
        if all(isinstance(item, (int, np.integer)) for item in items):
            indices = tuple(int(item) for item in items)
            if len(set(indices)) != len(indices):
                raise ValueError("orbital 全局索引不能重复。")
            invalid = [index for index in indices if index < 0 or index >= self.n_owner_orbitals]
            if invalid:
                raise ValueError(
                    f"orbital 全局索引越界：{invalid}；合法范围为 0..{self.n_owner_orbitals - 1}。"
                )
            default_name = "/".join(_format_label(self.labels[index]) for index in indices)
        else:
            patterns = [str(item) for item in items]
            indices = self.search_labels(patterns, input_name="orbital label")
            default_name = "/".join(patterns)

        atoms = tuple(dict.fromkeys(self.labels[index][0] for index in indices))
        return self._create_fragment(indices, name if name is not None else default_name, atoms=atoms)

    def add_all_atomic_fragments(self, **kwargs: Any) -> list[Fragment]:
        """为分子中的每个原子各添加一个 IAO fragment。"""
        return [self.add_atomic_fragment(atom, **kwargs) for atom in range(self.mol.natm)]

    def search_labels(
        self,
        patterns: Sequence[str],
        *,
        allowed_indices: Sequence[int] | None = None,
        input_name: str = "orbital label",
    ) -> tuple[int, ...]:
        """按 PySCF 标签规则搜索，并在给定 owner 子空间内稳定去重。"""
        self._ensure_initialized()
        allowed = tuple(range(self.n_owner_orbitals)) if allowed_indices is None else tuple(allowed_indices)
        allowed_set = set(allowed)
        matched: set[int] = set()
        for pattern in patterns:
            global_matches = {int(index) for index in self.refmol.search_ao_label(pattern)}
            local_matches = global_matches & allowed_set
            if not local_matches:
                raise ValueError(f"{input_name} {pattern!r} 在目标 fragment 中没有匹配轨道。")
            matched.update(local_matches)
        return tuple(index for index in allowed if index in matched)

    def lock(self) -> None:
        """锁定全部分片，防止 driver 开始后修改 active 空间。"""
        for fragment in self.fragments:
            fragment.lock()

    def _ensure_initialized(self) -> None:
        if self.coeff is None:
            self.kernel()

    def _create_fragment(
        self,
        indices: Sequence[int],
        name: str,
        *,
        atoms: Sequence[int],
    ) -> Fragment:
        if self._registered:
            raise RuntimeError("IAO fragmentation 已注册，不能再添加 fragment。")
        indices = tuple(int(index) for index in indices)
        if not indices:
            raise ValueError(f"fragment {name!r} 为空。")
        if len(set(indices)) != len(indices):
            raise ValueError(f"fragment {name!r} 包含重复 owner IAO。")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("fragment name 必须是非空字符串。")
        fragment_id = len(getattr(self.embedding, "fragments", ())) + len(self.fragments)
        fragment = Fragment(self, fragment_id, name, indices, atoms=atoms)
        self.fragments.append(fragment)
        return fragment

    def _resolve_atoms(
        self,
        atoms: int | str | Sequence[int] | Sequence[str],
    ) -> tuple[tuple[int, ...], tuple[str, ...]]:
        items = _as_items(atoms, name="atoms", allow_scalar=True)
        if all(isinstance(item, (int, np.integer)) for item in items):
            atom_indices = tuple(int(item) for item in items)
            if len(set(atom_indices)) != len(atom_indices):
                raise ValueError("atom 索引不能重复。")
            invalid = [atom for atom in atom_indices if atom < 0 or atom >= self.mol.natm]
            if invalid:
                raise ValueError(f"atom 索引越界：{invalid}；分子共有 {self.mol.natm} 个原子。")
            symbols = tuple(self.mol.atom_symbol(atom) for atom in atom_indices)
            return atom_indices, symbols

        requested = tuple(str(item) for item in items)
        if len(set(requested)) != len(requested):
            raise ValueError("atom 元素符号不能重复。")
        all_symbols = tuple(self.mol.atom_symbol(atom) for atom in range(self.mol.natm))
        missing = [symbol for symbol in requested if symbol not in all_symbols]
        if missing:
            raise ValueError(f"分子中找不到元素：{missing}。")
        atom_indices = tuple(index for index, symbol in enumerate(all_symbols) if symbol in requested)
        return atom_indices, requested

    def _validate_reference(self) -> None:
        mf = self.mf
        if not isinstance(mf, (RHF, UHF)) or isinstance(mf, ROHF):
            raise TypeError("IAOFragmentation 仅支持分子 RHF 或 UHF。")
        if not bool(getattr(mf, "converged", False)):
            raise ValueError("mean-field 尚未收敛，不能构造 IAO fragmentation。")

        if np.iscomplexobj(mf.mo_coeff):
            raise TypeError("IAOFragmentation 暂不支持复数轨道。")
        mo_coeff = np.asarray(mf.mo_coeff)
        mo_occ = np.asarray(mf.mo_occ)
        expected_coeff_ndim = 3 if self.is_unrestricted else 2
        expected_occ_ndim = 2 if self.is_unrestricted else 1
        if mo_coeff.ndim != expected_coeff_ndim or mo_occ.ndim != expected_occ_ndim:
            raise ValueError("mean-field mo_coeff/mo_occ 形状不合法。")
        allowed = (0.0, 1.0) if self.is_unrestricted else (0.0, 2.0)
        valid_occupations = np.isclose(mo_occ, allowed[0], atol=1e-10) | np.isclose(
            mo_occ, allowed[1], atol=1e-10
        )
        if not np.all(valid_occupations):
            raise ValueError(f"mean-field mo_occ 只能包含 {allowed}。")
        if self.is_unrestricted:
            counts = tuple(int(np.count_nonzero(value > 0)) for value in mo_occ)
            if counts != tuple(int(value) for value in mf.mol.nelec):
                raise ValueError("UHF occupied orbital 数与分子电子数不一致。")
        elif np.count_nonzero(mo_occ > 0) != mf.mol.nelectron // 2:
            raise ValueError("RHF occupied orbital 数与分子电子数不一致。")
        overlap = np.asarray(mf.get_ovlp(), dtype=float)
        if self.is_unrestricted:
            for spin, value in zip(("alpha", "beta"), mo_coeff):
                assert_orthonormal(value, overlap, name=f"UHF {spin} MOs", tolerance=1e-7)
        else:
            assert_orthonormal(mo_coeff, overlap, name="RHF MOs", tolerance=1e-7)

    def _make_spin_iao_basis(
        self,
        mo_coeff: np.ndarray,
        mo_occ: np.ndarray,
        *,
        spin: int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """为一个 spin channel 构造 owner IAO 和正交虚补空间。"""

        c_occ = mo_coeff[:, mo_occ > 0]
        c_iao_raw = np.asarray(lo.iao.iao(self.mol, c_occ, minao=self.minao), dtype=float)
        transform, minimum = symmetric_orthogonalizer(c_iao_raw, self.overlap)
        if minimum < 1e-10:
            raise RuntimeError(f"IAO Lowdin metric 的最小本征值过小：{minimum:.3e}。")
        c_iao = c_iao_raw @ transform
        assert_orthonormal(c_iao, self.overlap, name="IAOs", tolerance=1e-7)
        self._check_iao_electrons(c_iao, spin=spin)
        c_virtual = self._make_virtual_complement(c_iao, mo_coeff)
        coeff = hstack(c_iao, c_virtual, nrow=mo_coeff.shape[0])
        assert_orthonormal(coeff, self.overlap, name="IAO full basis", tolerance=1e-7)
        if coeff.shape[1] != mo_coeff.shape[1]:
            raise RuntimeError(
                "IAO full basis 的轨道数与 MO 数不一致："
                f"{coeff.shape[1]} != {mo_coeff.shape[1]}。"
            )
        return c_iao, c_virtual, coeff

    def _check_iao_electrons(self, c_iao: np.ndarray, *, spin: int | None) -> None:
        dm1_all = self.mf.make_rdm1()
        dm1 = np.asarray(dm1_all[spin] if spin is not None else dm1_all, dtype=float)
        projected = np.linalg.multi_dot((c_iao.T, self.overlap, dm1, self.overlap, c_iao))
        n_iao = float(np.trace(projected))
        n_total = float(np.einsum("ab,ab->", dm1, self.overlap, optimize=True))
        if abs(n_iao - n_total) > 1e-8:
            raise RuntimeError(
                "IAO 空间未包含该 spin channel 的全部电子："
                f"IAO={n_iao:.10f}, total={n_total:.10f}。"
            )

    def _make_virtual_complement(self, c_iao: np.ndarray, mo_coeff: np.ndarray) -> np.ndarray:
        c_iao_mo = metric_overlap(mo_coeff, self.overlap, c_iao)
        projector_rest = np.eye(mo_coeff.shape[1]) - c_iao_mo @ c_iao_mo.T
        values, vectors = eigh(projector_rest)
        mask_iao = values <= 0.5
        mask_rest = values > 0.5

        if np.any(np.abs(values[mask_iao]) > 1e-3):
            raise RuntimeError("IAO complement projector 的 IAO 本征值没有接近 0。")
        if np.any(np.abs(1.0 - values[mask_rest]) > 1e-3):
            raise RuntimeError("IAO complement projector 的虚空间本征值没有接近 1。")
        if np.count_nonzero(mask_rest) + c_iao.shape[1] != mo_coeff.shape[1]:
            raise RuntimeError("IAO 虚补空间的轨道数不正确。")

        c_rest = mo_coeff @ vectors[:, mask_rest]
        return fix_orbital_sign(c_rest)

    def _make_labels(self) -> tuple[IAOLabel, ...]:
        labels_ref = list(self.refmol.ao_labels(None))
        if self.refmol.natm == self.mol.natm:
            return tuple((int(a), str(s), str(nl), str(m)) for a, s, nl, m in labels_ref)

        # ghost atom被参考分子删除时，把 refmol atom id 映射回原分子 atom id。
        ref_to_mol: list[int] = []
        for ref_atom in range(self.refmol.natm):
            ref_coord = self.refmol.atom_coord(ref_atom)
            for atom in range(self.mol.natm):
                if np.allclose(self.mol.atom_coord(atom), ref_coord):
                    ref_to_mol.append(atom)
                    break
            else:
                raise RuntimeError(f"无法把参考分子的 atom {ref_atom} 映射回原分子。")
        return tuple(
            (ref_to_mol[int(atom)], str(symbol), str(shell), str(component))
            for atom, symbol, shell, component in labels_ref
        )


__all__ = ["Fragment", "IAOFragmentation", "get_default_minao"]
