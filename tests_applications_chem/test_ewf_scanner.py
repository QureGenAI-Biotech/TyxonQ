"""几何重建、同几何缓存及 ASE 的力符号/单位验证。"""

from types import SimpleNamespace

import numpy as np
import pytest
from pyscf import gto

from tyxonq.applications.chem.interfaces import ewf_scanner as module
from tyxonq.applications.chem.interfaces.ewf_scanner import EWFScanner


def test_scanner_cache_rebuild_and_invalid_gradient(monkeypatch):
    mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
    calls = []
    class FakeEmbedding:
        def __init__(self, mf, **kwargs):
            self.mf = mf
            calls.append(self)
        def kernel(self):
            coords = self.mf.mol.atom_coords()
            self.result = SimpleNamespace(converged=True, e_tot=float(np.sum(coords**2)),
                                          fragments=(), diagnostics={})
        def nuclear_gradient(self):
            return 2 * self.mf.mol.atom_coords()
    monkeypatch.setattr(module, "EWF", FakeEmbedding)
    scanner = EWFScanner(mol)
    coords = mol.atom_coords()
    energy = scanner.energy(coords)
    assert scanner(coords)[0] == energy
    assert len(calls) == 1
    gradient = scanner(coords)[1]
    gradient[:] = 0
    assert np.allclose(scanner(coords)[1], 2 * coords)
    changed = coords.copy()
    changed[1, 2] += .01
    assert scanner.energy(changed) != energy
    assert len(calls) == len(scanner.history) == 2
    assert calls[0].mf is not calls[1].mf
    assert np.array_equal(mol.atom_coords(), coords)
    with pytest.raises(ValueError, match="finite"):
        scanner.energy(coords * np.nan)
    calls[-1].nuclear_gradient = lambda: np.zeros((1, 3))
    with pytest.raises(RuntimeError, match="invalid nuclear gradient"):
        scanner(changed)


def test_scanner_stops_when_scf_fails(monkeypatch):
    mol = gto.M(atom="H 0 0 0; H 0 0 .75", basis="sto-3g", verbose=0)
    scanner = EWFScanner(mol, scf_options={"max_cycle": 0})
    with pytest.raises(RuntimeError, match="RHF did not converge"):
        scanner.energy(mol.atom_coords())
    assert not scanner.history


def test_ase_energy_force_units_and_identity_guard():
    ase = pytest.importorskip("ase")
    from tyxonq.applications.chem.interfaces.ase_calculator import BOHR_TO_ANGSTROM, HARTREE_TO_EV
    from tyxonq.applications.chem.interfaces.ewf_calculator import EWFCalculator
    class HarmonicScanner:
        symbols = ("H", "H")
        def energy(self, coords):
            return float(np.sum(coords**2) / 2)
        def __call__(self, coords):
            return self.energy(coords), coords.copy()
    atoms = ase.Atoms("HH", positions=[[0, 0, 0], [0, 0, BOHR_TO_ANGSTROM]])
    atoms.calc = EWFCalculator(HarmonicScanner())
    assert atoms.get_potential_energy() == pytest.approx(HARTREE_TO_EV / 2)
    assert atoms.get_forces()[1, 2] == pytest.approx(-HARTREE_TO_EV / BOHR_TO_ANGSTROM)
    atoms.positions[1, 2] *= 2
    assert atoms.get_potential_energy() == pytest.approx(2 * HARTREE_TO_EV)
    atoms.pbc = True
    with pytest.raises(ValueError, match="periodic"):
        atoms.get_forces()
    atoms.pbc = False
    atoms[0].symbol = "He"
    with pytest.raises(ValueError, match="identities/order"):
        atoms.get_potential_energy()
