"""将 EWF 几何计算接入 ASE，并转换能量、核梯度和力的单位。"""

import numpy as np

from .ase_calculator import BOHR_TO_ANGSTROM, HARTREE_TO_EV, _AseCalculator, _all_changes


class EWFCalculator(_AseCalculator if _AseCalculator is not None else object):
    """用闭壳层 EWF 近似梯度提供 ASE 能量和力。"""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, scanner, **kwargs):
        if _AseCalculator is None:
            raise ImportError("EWFCalculator requires ASE: pip install 'tyxonq[geomopt]'")
        super().__init__(**kwargs)
        self.scanner = scanner

    def calculate(self, atoms=None, properties=("energy",), system_changes=None):
        super().calculate(atoms, properties, system_changes if system_changes is not None else _all_changes)
        if np.any(self.atoms.pbc):
            raise ValueError("EWFCalculator does not support periodic systems.")
        if tuple(self.atoms.get_chemical_symbols()) != self.scanner.symbols:
            raise ValueError("ASE atom identities/order differ from the EWFScanner molecule.")
        coords = self.atoms.get_positions() / BOHR_TO_ANGSTROM
        if "forces" in properties:
            energy, gradient = self.scanner(coords)
            self.results["forces"] = -gradient * HARTREE_TO_EV / BOHR_TO_ANGSTROM
        else:
            energy = self.scanner.energy(coords)
        self.results["energy"] = float(energy) * HARTREE_TO_EV
        self.results["free_energy"] = self.results["energy"]
