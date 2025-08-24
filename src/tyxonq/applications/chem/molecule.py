

from typing import Union, Tuple
from itertools import product, repeat

import numpy as np
from pyscf.gto.mole import Mole
from pyscf import M, ao2mo

from .static.hamiltonian import random_integral


# Duck-typing PySCF Mole object. Not supposed to be an external user-interface
class _Molecule(Mole):
    @classmethod
    def random(cls, nao, n_elec, seed=2077):
        int1e, int2e = random_integral(nao, seed)
        return cls(int1e, int2e, n_elec)

    def __init__(
        self,
        int1e,
        int2e,
        n_elec: Union[int, Tuple[int, int]],
        spin: int = 0,
        e_nuc: float = 0,
        ovlp: np.ndarray = None,
    ):
        super().__init__()

        self.nao = len(int1e)
        self.int1e = int1e
        self.int2e = int2e
        self.int2e_s8 = ao2mo.restore(8, self.int2e, self.nao)
        # in PySCF m.nelec returns a tuple and m.nelectron returns an integer
        # So here use n_elec to indicate the difference
        self.n_elec = self.nelectron = n_elec
        self.spin = spin
        self.e_nuc = e_nuc
        if ovlp is None:
            self.ovlp = np.eye(self.nao)
        else:
            self.ovlp = ovlp
        # self.symmetry = True
        # avoid sanity check
        self.verbose = 0
        self.build()
        self.incore_anyway = True

    def intor(self, intor, comp=None, hermi=0, aosym="s1", out=None, shls_slice=None, grids=None):
        if intor == "int1e_kin":
            return np.zeros_like(self.int1e)
        elif intor == "int1e_nuc":
            return self.int1e
        elif intor == "int1e_ovlp":
            return self.ovlp
        elif intor == "int2e":
            assert aosym in ["s8", "s1"]
            if aosym == "s1":
                return self.int2e
            else:
                return self.int2e_s8
        else:
            raise ValueError(f"Unsupported integral type: {intor}")

    intor_symmetric = intor

    def tot_electrons(self):
        return self.n_elec

    def energy_nuc(self):
        return self.e_nuc


# for testing/benchmarking

_random = _Molecule.random


def h_chain(n_h=4, bond_distance=0.8, charge=0, spin=0):
    return M(atom=[["H", 0, 0, bond_distance * i] for i in range(n_h)], charge=charge, spin=spin, symmetry=True)


def h_ring(n_h=4, radius=0.7):
    atoms = []
    for angle in np.linspace(0, 2 * np.pi, n_h + 1)[:-1]:
        atom = ["H", 0, np.cos(angle) * radius, np.sin(angle) * radius]
        atoms.append(atom)
    return M(atom=atoms, symmetry=True)


def h_square(n_row, n_col, d_row=1, d_col=1):
    atoms = []
    for j in range(n_row):
        for k in range(n_col):
            atom = ["H", 0, j * d_row, k * d_col]
            atoms.append(atom)
    return M(atom=atoms, symmetry=True)


def h_cube(n_h_edge=2, d=1):
    atoms = []
    for l, j, k in product(*repeat(range(n_h_edge), 3)):
        atom = ["H", l * d, j * d, k * d]
        atoms.append(atom)
    return M(atom=atoms, symmetry=True)


H2 = h2 = h_chain(2, 0.741)

H3p = h3p = h_chain(3, charge=1)

H4 = h4 = h_chain()

H5p = h5p = h_chain(5, charge=1)

H6 = h6 = h_chain(6)

H8 = h8 = h_chain(8)


def water(bond_length=0.9584, bond_angle=104.45, basis="sto3g"):
    bond_angle = bond_angle / 180 * np.pi
    phi = bond_angle / 2
    r = bond_length
    O = ["O", 0, 0, 0]
    H1 = ["H", -r * np.sin(phi), r * np.cos(phi), 0]
    H2 = ["H", r * np.sin(phi), r * np.cos(phi), 0]
    return M(atom=[O, H1, H2], basis=basis, symmetry=True)


H2O = h2o = water


HeHp = hehp = lambda d=1: M(atom=[["H", 0, 0, 0], ["He", 0, 0, d]], charge=1, symmetry=True)
LiH = lih = lambda d=1.6: M(atom=[["H", 0, 0, 0], ["Li", 0, 0, d]], symmetry=True)
BeH2 = beh2 = lambda d=1.6: M(atom=[["H", 0, 0, -d], ["Be", 0, 0, 0], ["H", 0, 0, d]], symmetry=True)


def nh3(bond_length=1.017, bond_angle=107.8, basis="sto3g"):
    # the bond angle is the angle for H-N-H
    bond_angle = bond_angle / 180 * np.pi
    # bond N-H bond projected on the x-y plane
    projected_bond_length = 2 / np.sqrt(3) * np.sin(bond_angle / 2)
    # the angle between N-H the z axis
    phi = np.arcsin(projected_bond_length)
    z = -bond_length * np.cos(phi)
    N = ["N", 0, 0, 0]
    H1 = ["H", projected_bond_length, 0, z]
    H2 = ["H", -projected_bond_length / 2, projected_bond_length / 2 * np.sqrt(3), z]
    H3 = ["H", -projected_bond_length / 2, -projected_bond_length / 2 * np.sqrt(3), z]
    return M(atom=[N, H1, H2, H3], basis=basis, symmetry=True)


NH3 = nh3


def bh3(bond_length=1.190, basis="sto3g"):
    atom = [
        ["B", 0, 0, 0],
        ["H", 0, bond_length, 0],
        ["H", bond_length / 2 * np.sqrt(3), -bond_length / 2, 0],
        ["H", -bond_length / 2 * np.sqrt(3), -bond_length / 2, 0],
    ]
    return M(atom=atom, basis=basis, symmetry=True)


BH3 = bh3


N2 = n2 = nitrogen = lambda d=1.09: M(atom=[["N", 0, 0, 0], ["N", 0, 0, d]], symmetry=True)

CO = co = lambda d=1.128: M(atom=[["C", 0, 0, 0], ["O", 0, 0, d]], symmetry=True)


def get_tetrahedron_coord(d, center_atom="C"):
    x = d / np.sqrt(3)
    tetrahedron_coord = [
        [center_atom, 0, 0, 0],
        ["H", -x, x, x],
        ["H", x, -x, x],
        ["H", -x, -x, -x],
        ["H", x, x, -x],
    ]
    return tetrahedron_coord


CH4 = ch4 = methane = lambda x=1.09: M(atom=get_tetrahedron_coord(x), symmetry=True)


NH4 = nh4 = NH4p = nh4p = lambda x=1.02: M(atom=get_tetrahedron_coord(x, center_atom="N"), charge=1, symmetry=True)


def hcn(ch_length=1.0640, cn_length=0.1156, basis="sto3g"):
    coords = [
        ["C", 0, 0, 0],
        ["H", 0, 0, ch_length],
        ["N", 0, 0, -cn_length],
    ]
    return M(atom=coords, basis=basis, symmetry=True)


HCN = hcn


def c2h2(cc_length=1.39, ch_length=1.09, basis="sto3g"):
    coords = [
        ["C", 0, 0, 0],
        ["C", 0, 0, cc_length],
        ["H", 0, 0, cc_length + ch_length],
        ["H", 0, 0, -ch_length],
    ]
    return M(atom=coords, basis=basis, symmetry=True)


hcch = HCCH = C2H2 = ethyne = acetylene = c2h2


def h2co(basis="sto3g"):
    coords = [
        ["O", 0.0000, 0.0000, 1.2050],
        ["C", 0.0000, 0.0000, 0.0000],
        ["H", 0.0000, 0.9429, -0.5876],
        ["H", 0.0000, -0.9429, -0.5876],
    ]
    return M(atom=coords, basis=basis, symmetry=True)


H2CO = formaldehyde = h2co


def c4h4(cc1, cc2, ch=1.079, basis="sto3g", symmetry=True):
    h_offset = ch / np.sqrt(2)
    atom = [
        ["C", cc1 / 2, cc2 / 2, 0],
        ["H", cc1 / 2 + h_offset, cc2 / 2 + h_offset, 0],
        ["C", -cc1 / 2, cc2 / 2, 0],
        ["H", -cc1 / 2 - h_offset, cc2 / 2 + h_offset, 0],
        ["C", cc1 / 2, -cc2 / 2, 0],
        ["H", cc1 / 2 + h_offset, -cc2 / 2 - h_offset, 0],
        ["C", -cc1 / 2, -cc2 / 2, 0],
        ["H", -cc1 / 2 - h_offset, -cc2 / 2 - h_offset, 0],
    ]
    return M(atom=atom, basis=basis, symmetry=symmetry)


def benzene(cc_length=1.39, ch_length=1.09):
    a, b = cc_length, ch_length

    hexagon = np.array(
        [
            [a, 0, 0],
            [1 / 2 * a, np.sqrt(3) / 2 * a, 0],
            [-1 / 2 * a, np.sqrt(3) / 2 * a, 0],
            [-a, 0, 0],
            [-1 / 2 * a, -np.sqrt(3) / 2 * a, 0],
            [1 / 2 * a, -np.sqrt(3) / 2 * a, 0],
        ]
    )

    atom = []
    for coord in hexagon:
        atom.append(["C"] + coord.tolist())
        atom.append(["H"] + ((1 + b / a) * coord).tolist())
    return M(atom=atom, symmetry=True)


c6h6 = benzene


def indene(basis="sto3g"):
    atom = [
        ["C", 0.2327, 0.7102, 0.0001],
        ["C", 1.637, 1.229, 0.0001],
        ["C", 0.2498, -0.6917, 0.0002],
        ["C", 2.4192, -0.0627, -0.0005],
        ["C", 1.6292, -1.1479, 0.0001],
        ["C", -0.9575, 1.4149, 0.0001],
        ["C", -0.9264, -1.4208, 0.0001],
        ["C", -2.1497, 0.6872, -0.0001],
        ["C", -2.1344, -0.7183, -0.0002],
        ["H", 1.8471, 1.8132, -0.9002],
        ["H", 1.8474, 1.8125, 0.9009],
        ["H", 3.4995, -0.1004, -0.0009],
        ["H", 1.9402, -2.1791, 0],
        ["H", -0.9706, 2.4991, 0.0001],
        ["H", -0.9208, -2.5055, 0.0001],
        ["H", -3.1012, 1.2122, -0.0002],
        ["H", -3.0744, -1.2638, -0.000],
    ]
    return M(atom=atom, basis=basis)


if __name__ == "__main__":
    n = 6
    h1 = np.zeros((n, n))
    for i in range(n - 1):
        h1[i, i + 1] = h1[i + 1, i] = -1.0
    h1[n - 1, 0] = h1[0, n - 1] = -1.0
    eri = np.zeros((n, n, n, n))
    for i in range(n):
        eri[i, i, i, i] = 2.0

    m = _Molecule(h1, eri, 6)
    from pyscf.scf import RHF

    rhf = RHF(m)
    # avoid serialization warning
    rhf.chkfile = False
    print(rhf.kernel())
