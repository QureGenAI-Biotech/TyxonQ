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
  "fci": -1.1633744903192416,
  "ccsd": -1.1633744964048178,
  "ccsd(t)": -1.1633744964048178,
  "mp2": -1.155071651191337,
  "dft(b3lyp)": -1.1732811209085696,
}

def test_cpu_classical_methods_smoke():
    payload_base = {"molecule_data": _mol_data(), "classical_device": "cpu"}

    # batch compute: request multiple methods at once to match new cpu_chem API
    res = cpu_chem.compute({**payload_base, "method": ["fci", "ccsd", "ccsd(t)", "mp2", "dft"], "method_options": {"functional": "b3lyp"}})

    assert isinstance(res["hf"]["energy"], float)
    assert abs(res["hf"]["energy"] - gold_results["hf"]) < 1e-6
    
    assert isinstance(res["fci"]["energy"], float)
    assert abs(res["fci"]["energy"] - gold_results["fci"]) < 1e-6

    assert isinstance(res["ccsd"]["energy"], float)
    assert abs(res["ccsd"]["energy"] - gold_results["ccsd"]) < 1e-6

    assert isinstance(res["ccsd(t)"]["energy"], float)
    assert abs(res["ccsd(t)"]["energy"] - gold_results["ccsd(t)"]) < 1e-6

    assert isinstance(res["mp2"]["energy"], float)
    assert abs(res["mp2"]["energy"] - gold_results["mp2"]) < 1e-6

    assert isinstance(res["dft"]["energy"], float)
    assert abs(res["dft"]["energy"] - gold_results["dft(b3lyp)"]) < 1e-6


def test_gpu_classical_methods_smoke():
    payload_base = {"molecule_data": _mol_data(), "classical_device": "gpu"}

    # batch compute: request multiple methods at once to match new cpu_chem API
    res = gpu_chem.compute({**payload_base, "method": ["fci", "ccsd", "ccsd(t)", "mp2", "dft"], "method_options": {"functional": "b3lyp"}})

    assert isinstance(res["hf"]["energy"], float)
    assert abs(res["hf"]["energy"] - gold_results["hf"]) < 1e-6
    
    assert isinstance(res["fci"]["energy"], float)
    assert abs(res["fci"]["energy"] - gold_results["fci"]) < 1e-6

    assert isinstance(res["ccsd"]["energy"], float)
    assert abs(res["ccsd"]["energy"] - gold_results["ccsd"]) < 1e-6

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



def test_uccsd_local_vs_cloud_equal():
    # Start FastAPI server in a subprocess on a fixed port, run cloud path, then terminate
    import os, sys, time, subprocess, requests
    server_module = "tyxonq.applications.chem.classical_chem_cloud.server.app:app"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONPATH"] = os.pathsep.join([
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")),
        env.get("PYTHONPATH", "")
    ])
    cmd = [sys.executable, "-m", "uvicorn", server_module, "--host", "127.0.0.1", "--port", "8009"]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        # wait server to be ready
        base = "http://127.0.0.1:8009/classical/compute"
        for _ in range(40):
            time.sleep(0.2)
            try:
                requests.post(base, json={"method": "hf", "molecule_data": {"atom":"H 0 0 0; H 0 0 0.74","basis":"sto-3g","charge":0,"spin":0,"unit":"Angstrom"}})
                break
            except Exception:
                continue

        mol = _mol_data_for_uccsd_hea()
        u_local = UCCSD(**mol, classical_provider="local", runtime="device",run_fci=True)
        e_local = u_local.kernel(shots=0, provider="simulator", device="statevector")

        u_cloud = UCCSD(**mol, classical_provider="tyxonq", classical_device="cpu", runtime="device",run_fci=True)
        e_cloud = u_cloud.kernel(shots=0, provider="simulator", device="statevector")

        u_gpu_cloud = UCCSD(**mol, classical_provider="tyxonq", classical_device="gpu", runtime="device",run_fci=True)
        e_gpu_cloud = u_gpu_cloud.kernel(shots=0, provider="simulator", device="statevector")

        u_hea_cloud = HEA(**mol, classical_provider="tyxonq", classical_device="auto", runtime="device",run_fci=True)
        e_hea_cloud = u_hea_cloud.kernel(shots=0, provider="simulator", device="statevector")

        assert abs(e_local - u_local.e_fci) < 1e-6
        assert abs(e_local - e_cloud) < 1e-6
        assert abs(e_local - e_gpu_cloud) < 1e-6
        assert abs(e_local - e_hea_cloud) < 1e-6
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()




if __name__ == "__main__":
    # test_cpu_classical_methods_smoke()
    # test_uccsd_local_vs_cloud_equal()
    test_gpu_classical_methods_smoke()
    # test_cpu_hf_integrals_to_uccsd_and_hea()
    # test_gpu_chem_smoke_delegation
    # test_gpu_hf_integrals_ccsd_casstf_paths()