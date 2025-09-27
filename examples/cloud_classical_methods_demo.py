from __future__ import annotations

"""
Demo: Cloud-accelerated classical quantum chemistry methods via TyxonQClassicalClient

Run locally:
  python examples-ng/cloud_classical_methods_demo.py

This script compares local vs cloud (mocked) execution for FCI/CCSD/DFT/MP2/CASSCF,
and shows how to request verbose outputs and retrieve artifacts such as HF chkfile.
"""

import json
from typing import Any

from pyscf import gto

from tyxonq.applications.chem.classical_chem_cloud import cloud_classical_methods


def build_molecule() -> Any:
    m = gto.Mole()
    m.atom = "H 0 0 0; H 0 0 0.74"
    m.basis = "cc-pvdz"
    m.charge = 0
    m.spin = 0
    m.build()
    return m


def main() -> None:
    m = build_molecule()

    # Cloud (mocked) with verbose and explicit device
    cloud = cloud_classical_methods(m, classical_provider="tyxonq", classical_device="auto")
    e_fci_cloud = cloud.fci(verbose=True)
    e_ccsd_cloud = cloud.ccsd(verbose=True)
    e_ccsd_t_cloud = cloud.ccsd_t(verbose=True)
    e_mp2_cloud = cloud.mp2(verbose=True)
    e_dft_cloud = cloud.dft("b3lyp", verbose=True)
    e_casscf_cloud = cloud.casscf(ncas=2, nelecas=2, verbose=True)

    print("\nCloud results (mocked):")
    print(json.dumps({
        "fci": e_fci_cloud,
        "ccsd": e_ccsd_cloud,
        "ccsd(t)": e_ccsd_t_cloud,
        "mp2": e_mp2_cloud,
        "dft(b3lyp)": e_dft_cloud,
        "casscf(ncas=2, nelecas=2)": e_casscf_cloud,
    }, indent=2))

    print("\nNotes:")
    print("- When verbose=True, the client returns extra metadata and may include a base64-encoded HF chkfile artifact.")


if __name__ == "__main__":
    main()


