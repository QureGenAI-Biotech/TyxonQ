from __future__ import annotations

"""
Cloud usage demo for UCCSD and HEA with PySCF molecules.

Run:
  python examples/cloud_uccsd_hea_demo.py
"""

import numpy as np
from pyscf import gto

from tyxonq.applications.chem.algorithms.uccsd import UCCSD
from tyxonq.applications.chem.algorithms.hea import HEA
from tyxonq.applications.chem.molecule import h2,h4


def build_h2(basis: str = "sto-3g"):
    m = gto.Mole()
    m.atom = "H 0 0 0; H 0 0 0.74"
    m.basis = basis
    m.charge = 0
    m.spin = 0
    m.build()
    return m


def main():
    mol = build_h2()
    # u_local = UCCSD(h2)

    # UCCSD - local baseline
    u_local = UCCSD(mol)
    e_local = u_local.kernel(shots=0, runtime="device", provider="simulator", device="statevector")
    print("UCCSD local e:", e_local)

    # UCCSD - cloud HF/integrals, same kernel locally
    u_cloud = UCCSD(mol, classical_provider="tyxonq", classical_device="auto")
    e_cloud = u_cloud.kernel(shots=0, runtime="device", provider="simulator", device="statevector")
    print("UCCSD cloud(HF) e:", e_cloud)

    # HEA - init from molecule directly (new path), local baseline
    hea_local = HEA(molecule=mol, layers=2, mapping="parity", runtime="device")
    e_hea_local = hea_local.kernel(shots=0, provider="simulator", device="statevector")
    print("HEA local e:", e_hea_local)

    # HEA - cloud HF/integrals via molecule pathway
    hea_cloud = HEA(molecule=mol, layers=2, mapping="parity", runtime="device", classical_provider="tyxonq", classical_device="auto")
    e_hea_cloud = hea_cloud.kernel(shots=0, provider="simulator", device="statevector")
    print("HEA cloud(HF) e:", e_hea_cloud)


if __name__ == "__main__":
    main()


