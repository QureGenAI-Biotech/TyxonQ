"""非绝热 PES → ASE Calculator 薄壳（Tier2：非绝热引擎接入 MD 生态）。

定位（照抄 ``ase_calculator.py`` 母版，20-80 行级）：

- 把 ``models/pes.TullyModel``（1D 核坐标）的绝热面能量/梯度包装为
  ASE Calculator，让 ASE 的 MD 驱动器（VelocityVerlet/Langevin）驱动核；
- 核坐标映射：``atoms.positions[0, 0]``（单原子 x 方向，Å）→ Tully 的
  核坐标 x（Bohr）；力符号取负（ASE 约定 F = -dE/dR）；
- 单位换算常数只在本文件出现一次（与 ase_calculator.py 同源数值）；
- 真实分子的 QcScannerPES 薄壳在后续阶段扩展（本文件只管模型 PES）。

用法::

    from ase import Atoms
    from ase.md.verlet import VelocityVerlet
    from tyxonq.applications.chem.interfaces.dynamics_md import NonadiabaticASECalculator
    from tyxonq.applications.chem.dynamics.models.pes import tully_model

    atoms = Atoms("H", positions=[[-5.0, 0, 0]])  # x 单位 Å
    atoms.calc = NonadiabaticASECalculator(tully_model("A"), surface=0)
    dyn = VelocityVerlet(atoms, timestep=0.1)  # fs
    dyn.run(100)

``interfaces/__init__.py`` 不导出本类（保持包级惰性），按子模块路径导入。
"""

from __future__ import annotations

import numpy as np

# ---- 单位换算（与 ase_calculator.py 同源数值；单一出处） ----
BOHR_TO_ANGSTROM = 0.52917721092   # 1 Bohr = 0.52917721092 Å
HARTREE_TO_EV = 27.211386245988    # 1 Hartree = 27.211386245988 eV

try:
    from ase.calculators.calculator import Calculator as _AseCalculator, all_changes as _all_changes
except ImportError:
    _AseCalculator = None
    _all_changes = None


class NonadiabaticASECalculator(_AseCalculator if _AseCalculator is not None else object):
    """Tully 模型绝热面 → ASE Calculator（1D 核坐标映射 atoms[0].x）。

    Parameters
    ----------
    pes : ``models/pes.TullyModel``（A/B/C）。
    surface : 当前绝热面索引（0/1）；Ehrenfest 平均场由调用方另行组合。
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, pes, *, surface: int = 0, **kwargs):
        if _AseCalculator is None:
            raise ImportError(
                "ASE 未安装：请 `pip install ase` 后使用 NonadiabaticASECalculator。"
            )
        super().__init__(**kwargs)
        self.pes = pes
        self.surface = int(surface)
        if self.surface not in (0, 1):
            raise ValueError("surface 须为 0 或 1。")

    def calculate(self, atoms=None, properties=("energy", "forces"),
                  system_changes=_all_changes):
        """计算当前绝热面的能量与力（1D 核坐标映射）。"""
        super().calculate(atoms, properties, system_changes)
        # 核坐标：atoms[0].x（Å）→ Bohr
        x_bohr = float(atoms.positions[0, 0]) / BOHR_TO_ANGSTROM
        evals, _ = self.pes.adiabatic(x_bohr)
        grad = self.pes.adiabatic_gradient(x_bohr)
        energy_hartree = float(evals[self.surface])
        force_hartree_per_bohr = -float(grad[self.surface])  # F = -dE/dx
        self.results["energy"] = energy_hartree * HARTREE_TO_EV
        # 力：Ha/Bohr → eV/Å；方向沿 x（其余分量 0）
        fx = force_hartree_per_bohr * HARTREE_TO_EV / BOHR_TO_ANGSTROM
        self.results["forces"] = np.zeros((len(atoms), 3))
        self.results["forces"][0, 0] = fx


__all__ = ["NonadiabaticASECalculator"]
