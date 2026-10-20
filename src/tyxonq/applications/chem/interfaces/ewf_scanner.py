"""闭壳层 EWF 的几何扫描；每个新几何重新建立嵌入和求解片段。"""

from __future__ import annotations

from time import perf_counter
import sys

try:
    import resource
except ImportError:  # Windows 无 resource；不可把缺测内存当作零。
    resource = None

import numpy as np
from pyscf import lib, scf

from ..algorithms.ewf import EWF


class EWFScanner:
    """将 Bohr 坐标映射到能量及 frozen-bath 近似梯度。

    ``mol`` 固定元素、基组和电荷。``fragment_setup(emb)`` 可重建自定义
    分片；省略时使用 EWF 默认原子分片。相同坐标复用同一次求解，避免
    优化器分开请求能量和力时重新采样。配方在构造后不应修改。
    """

    def __init__(self, mol, *, ewf_options=None, scf_options=None, fragment_setup=None):
        if getattr(mol, "spin", 0) != 0 or mol.nelectron % 2 or getattr(mol, "dimension", 0):
            raise ValueError("EWFScanner requires a nonperiodic closed-shell molecule.")
        self.mol = mol.copy()
        self.symbols = tuple(mol.atom_symbol(i) for i in range(mol.natm))
        self.ewf_options = dict(ewf_options or {})
        self.ewf_options.setdefault("energy_functional", "density")
        self.ewf_options.setdefault("density_assembly", "rdm_t_lambda")
        if self.ewf_options["energy_functional"] != "density":
            raise ValueError("EWFScanner gradients require energy_functional='density'.")
        self.scf_options = {"conv_tol": 1e-11, **dict(scf_options or {})}
        self.fragment_setup = fragment_setup
        self.embedding = None
        self.history = []
        self._coords = None
        self._gradient = None

    def _evaluate(self, coords_bohr):
        coords = np.asarray(coords_bohr, dtype=float)
        if coords.shape != (self.mol.natm, 3) or not np.isfinite(coords).all():
            raise ValueError(f"coords_bohr must be finite with shape ({self.mol.natm}, 3).")
        if self._coords is not None and np.array_equal(coords, self._coords):
            return self.embedding
        start = perf_counter()
        mol = self.mol.set_geom_(coords, unit="Bohr", inplace=False)
        mf = scf.RHF(mol)
        for name, value in self.scf_options.items():
            if not hasattr(mf, name):
                raise ValueError(f"Unknown RHF option: {name}")
            setattr(mf, name, value)
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("RHF did not converge; EWF geometry evaluation stopped.")
        scf_seconds = perf_counter() - start
        emb = EWF(mf, **self.ewf_options)
        if self.fragment_setup is not None:
            self.fragment_setup(emb)
        emb.kernel()
        if not emb.result.converged:
            raise RuntimeError("EWF fragment or density solve did not converge.")
        if not np.isfinite(emb.result.e_tot):
            raise RuntimeError("EWF returned a non-finite energy.")
        # 维度变化与相同维度下的组态变化分别记录；缺失签名不推断为未变化。
        spaces = tuple((f.name, f.cluster.nocc, f.cluster.norb,
                        f.solver.diagnostics.get("selected_space_signature"))
                       for f in emb.result.fragments)
        previous = self.history[-1]["spaces"] if self.history else None
        self.history.append({
            "coords_bohr": coords.tolist(), "energy_hartree": float(emb.result.e_tot),
            "scf_seconds": scf_seconds, "energy_seconds": perf_counter() - start,
            "rss_mb": float(lib.current_memory()[0]), "spaces": spaces,
            "process_peak_rss_mb": None if resource is None else
                float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                / (1024**2 if sys.platform == "darwin" else 1024),
            "spaces_changed": previous is not None and spaces != previous,
            "diagnostics": emb.result.diagnostics,
            "fragments": [{"id": f.fragment_id, "name": f.name,
                           "norb": f.cluster.norb, "method": f.solver.method,
                           "converged": f.solver.converged,
                           "diagnostics": f.diagnostics} for f in emb.result.fragments],
        })
        self.embedding = emb
        self._coords = coords.copy()
        self._gradient = None
        return emb

    def energy(self, coords_bohr):
        """只计算总能量（Hartree）；有限差分时不重复求梯度。"""
        return float(self._evaluate(coords_bohr).result.e_tot)

    def __call__(self, coords_bohr):
        """返回 ``(energy_Hartree, gradient_Hartree_per_Bohr)``。"""
        emb = self._evaluate(coords_bohr)
        if self._gradient is None:
            start = perf_counter()
            gradient = np.asarray(emb.nuclear_gradient(), dtype=float)
            if gradient.shape != (self.mol.natm, 3) or not np.isfinite(gradient).all():
                raise RuntimeError("EWF returned an invalid nuclear gradient.")
            self._gradient = gradient.copy()
            self.history[-1]["gradient_seconds"] = perf_counter() - start
            self.history[-1]["gradient_hartree_per_bohr"] = gradient.tolist()
            self.history[-1]["max_gradient_hartree_per_bohr"] = float(np.max(np.linalg.norm(gradient, axis=1)))
            self.history[-1]["rms_gradient_hartree_per_bohr"] = float(np.sqrt(np.mean(gradient**2)))
        return float(emb.result.e_tot), self._gradient.copy()
