from __future__ import annotations

import numpy as np
import pytest

from tyxonq.applications.chem.classical_chem_cloud.server import cpu_chem, gpu_chem
from tyxonq.applications.chem.algorithms.uccsd import UCCSD
from tyxonq.applications.chem.algorithms.hea import HEA


def _mol_data(atom: str = "H 0 0 0; H 0 0 0.74", basis: str = "cc-pvdz", charge: int = 0, spin: int = 0) -> dict:
    return {
        "atom": atom,
        "basis": basis,
        "charge": int(charge),
        "spin": int(spin),
        "unit": "Angstrom",
    }


gold_results = {
  'hf':-1.128701033136812,
  "fci": -1.1633925611011462,
  "ccsd": -1.1633925672868486,
  "ccsd(t)": -1.1633925672868486,
  "mp2": -1.1550639182397155,
  "dft(b3lyp)": -1.1732811209085696,
}

def test_cpu_classical_methods_smoke():
    payload_base = {"molecule_data": _mol_data(), "classical_device": "cpu"}

    # batch compute: request multiple methods at once to match new cpu_chem API
    res = cpu_chem.compute({**payload_base, "method": ["fci", "ccsd", "ccsd(t)", "mp2", "dft"], "method_options": {"functional": "b3lyp"}})

    assert isinstance(res["hf"]["energy"], float)
    assert abs(res["hf"]["energy"] - gold_results["hf"]) < 1e-7
    
    assert isinstance(res["fci"]["energy"], float)
    assert abs(res["fci"]["energy"] - gold_results["fci"]) < 1e-7

    assert isinstance(res["ccsd"]["energy"], float)
    assert abs(res["ccsd"]["energy"] - gold_results["ccsd"]) < 1e-7

    assert isinstance(res["ccsd(t)"]["energy"], float)
    assert abs(res["ccsd(t)"]["energy"] - gold_results["ccsd(t)"]) < 1e-6

    assert isinstance(res["mp2"]["energy"], float)
    assert abs(res["mp2"]["energy"] - gold_results["mp2"]) < 1e-6

    assert isinstance(res["dft"]["energy"], float)
    assert abs(res["dft"]["energy"] - gold_results["dft(b3lyp)"]) < 1e-6

def _mol_data_for_uccsd_hea(atom: str = "H 0 0 0; H 0 0 0.74", basis: str = "sto-3g", charge: int = 0, spin: int = 0) -> dict:
    return {
        "atom": atom,
        "basis": basis,
        "charge": int(charge),
        "spin": int(spin),
        "unit": "Angstrom",
    }



def test_uccsd_local_vs_cloud_stub_equal():
    # Use local-stub: cloud request is routed to cpu_chem.compute directly, no HTTP server needed

    mol = _mol_data_for_uccsd_hea("sto-3g")

    u_local = UCCSD(mol, classical_provider="local", runtime="device")
    e_local = u_local.kernel(shots=0, provider="simulator", device="statevector")

    u_cloud = UCCSD(mol, classical_provider="tyxonq", classical_device="auto", runtime="device")
    e_cloud = u_cloud.kernel(shots=0, provider="simulator", device="statevector")

    # Close agreement within tight tolerance
    assert abs(e_local - e_cloud) < 1e-8



# def test_cpu_hf_integrals_to_uccsd_and_hea():
#     payload_hf = {"molecule_data": _mol_data_for_hea(), "classical_device": "cpu", "method": "hf_integrals", "verbose": False}
#     res = cpu_chem.compute(payload_hf)
#     int1e = np.asarray(res["int1e"])  # type: ignore[index]
#     int2e = np.asarray(res["int2e"])  # type: ignore[index]
#     e_core = float(res["e_core"])  # type: ignore[index]
#     nelec = int(res.get("nelectron", 2))

#     # UCCSD from integrals
#     u = UCCSD.from_integral(int1e, int2e, nelec, e_core, runtime="device")
#     e_ucc = u.kernel(runtime="device", provider="simulator", device="statevector", shots=0)
#     assert isinstance(e_ucc, float)

#     # HEA from integrals (mapping parity), shallow layers
#     hea = HEA.from_integral(int1e, int2e, nelec, e_core, n_layers=2, mapping="parity", runtime="device")
#     e_hea = hea.kernel(shots=0, provider="simulator", device="statevector")
#     assert isinstance(e_hea, float)

#     # FCI reference via cpu_chem
#     e_fci = cpu_chem.compute({"molecule_data": _mol_data_for_hea(), "classical_device": "cpu", "method": "fci", "method_options": {}})["fci"]["energy"]

#     # Loose tolerance to accommodate small numeric drift
#     assert np.isfinite(e_fci)
#     assert abs(e_ucc - e_fci) < 1e-4
#     assert abs(e_hea - e_fci) < 1e-3


# @pytest.mark.xfail(condition=not getattr(gpu_chem, "gpu_available", False), reason="gpu4pyscf not available")
# def test_gpu_chem_smoke_delegation():
#     payload = {"molecule_data": _mol_data(), "classical_device": "gpu", "verbose": False}

#     # DFT on GPU (or fallback)
#     dft_res = gpu_chem.compute({**payload, "method": "dft", "method_options": {"functional": "b3lyp"}})
#     assert isinstance(dft_res["energy"], float)
#     assert abs(dft_res["energy"] - (-1.1732811209085696)) < 1e-8

#     # MP2 on GPU (or fallback)
#     mp2_res = gpu_chem.compute({**payload, "method": "mp2", "method_options": {}})
#     assert isinstance(mp2_res["energy"], float)
#     assert abs(mp2_res["energy"] - (-1.155071651191337)) < 1e-8

#     # FCI via delegation to CPU
#     fci_res = gpu_chem.compute({**payload, "method": "fci", "method_options": {}})
#     assert isinstance(fci_res["energy"], float)
#     assert abs(fci_res["energy"] - (-1.1633744903192416)) < 1e-8

#     # CCSD on GPU (or fallback logic inside)
#     ccsd_res = gpu_chem.compute({**payload, "method": "ccsd", "method_options": {}})
#     assert isinstance(ccsd_res["energy"], float)
#     assert abs(ccsd_res["energy"] - (-1.1633744964048178)) < 5e-4
#     # CCSD(T) -> GPU RHF then CPU CCSD(T)
#     ccst_res = gpu_chem.compute({**payload, "method": "ccsd(t)", "method_options": {}})
#     assert isinstance(ccst_res["energy"], float)
#     assert abs(ccst_res["energy"] - (-1.1633744964048178)) < 1e-8
#     # CASSCF -> GPU RHF then CPU CASSCF
#     casscf_res = gpu_chem.compute({**payload, "method": "casscf", "method_options": {"ncas": 2, "nelecas": 2}})
#     assert isinstance(casscf_res["energy"], float)
#     assert abs(casscf_res["energy"] - (-1.1468743339673009)) < 1e-8


# @pytest.mark.xfail(condition=not getattr(gpu_chem, "gpu_available", False), reason="gpu4pyscf not available")
# def test_gpu_hf_integrals_ccsd_casstf_paths():
#     payload_cpu = {"molecule_data": _mol_data_for_hea(), "classical_device": "cpu", "method": "hf_integrals", "verbose": False}

#     res = cpu_chem.compute(payload_cpu)
#     int1e = np.asarray(res["int1e"])  # type: ignore[index]
#     int2e = np.asarray(res["int2e"])  # type: ignore[index]
#     e_core = float(res["e_core"])  # type: ignore[index]
#     nelec = int(res.get("nelectron", 2))

#     payload_gpu = {"molecule_data": _mol_data_for_hea(), "classical_device": "cpu", "method": "hf_integrals", "verbose": False}
#     # hf_integrals path (GPU RHF then to CPU integrals)
#     res_gpu = gpu_chem.compute({**payload_gpu, "method": "hf_integrals", "method_options": {}})
#     int1e_gpu = np.asarray(res_gpu["int1e"])  # type: ignore[index]
#     int2e_gpu = np.asarray(res_gpu["int2e"])  # type: ignore[index]
#     e_core_gpu = float(res_gpu["e_core"])  # type: ignore[index]
#     nelec_gpu = int(res_gpu.get("nelectron", 2))
#     assert abs(e_core_gpu - e_core) < 1e-8
#     assert abs(nelec_gpu - nelec) < 1e-8
#     np.testing.assert_allclose(int1e_gpu, int1e, atol=1e-8)
#     np.testing.assert_allclose(int2e_gpu, int2e, atol=1e-8)


if __name__ == "__main__":
    test_cpu_classical_methods_smoke()
    # test_cpu_hf_integrals_to_uccsd_and_hea()
    # test_gpu_chem_smoke_delegation
    # test_gpu_hf_integrals_ccsd_casstf_paths()