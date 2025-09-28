from __future__ import annotations

"""
Minimal mock server for classical cloud API using FastAPI.

Start:
  uvicorn tyxonq.applications.chem.classical_chem_cloud.server.mock_classical_server:app --host 0.0.0.0 --port 8009

Note: This is a local mock that executes with PySCF; in production replace
handlers with real scheduling and isolation.
"""

from typing import Any, Dict, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from pyscf import gto, scf
from pyscf import fci as _fci
from pyscf import cc as _cc
from pyscf import mp as _mp
from pyscf import dft as _dft
from pyscf import mcscf as _mcscf
from tyxonq.applications.chem.chem_libs.hamiltonians_chem_library.hamiltonian_builders import get_integral_from_hf

class MoleculeData(BaseModel):
    atom: str
    basis: str = "sto-3g"
    charge: int = 0
    spin: int = 0


class ClassicalRequest(BaseModel):
    method: str
    molecule_data: MoleculeData
    active_space: Optional[tuple[int, int]] = None
    active_orbital_indices: Optional[list[int]] = None
    method_options: Optional[dict] = None
    classical_device: str = "auto"
    verbose: bool = False


app = FastAPI(title="TyxonQ Classical Mock API (legacy)")


def _build_mol(m: MoleculeData):
    mol = gto.Mole()
    mol.atom = m.atom
    mol.basis = m.basis
    mol.charge = m.charge
    mol.spin = m.spin
    mol.build()
    return mol


@app.post("/classical/compute")
def classical_compute(req: ClassicalRequest):
    meth = req.method.lower()
    mol = _build_mol(req.molecule_data)

    # HF always prepared
    mf = scf.RHF(mol)
    mf.chkfile = None
    mf.verbose = 0
    mf.kernel()

    if meth == "hf_integrals":
        int1e, int2e, e_core = get_integral_from_hf(mf, active_space=req.active_space, active_orbital_indices=req.active_orbital_indices)
        return {
            "int1e": np.asarray(int1e).tolist(),
            "int2e": np.asarray(int2e).tolist(),
            "e_core": float(e_core),
            "e_hf": float(getattr(mf, "e_tot", 0.0)),
            "mo_coeff": np.asarray(getattr(mf, "mo_coeff", None)).tolist() if getattr(mf, "mo_coeff", None) is not None else None,
            "nelectron": int(getattr(mol, "nelectron", 0)),
            "nao": int(getattr(mol, "nao", 0)),
            "spin": int(getattr(mol, "spin", 0)),
            "basis": str(getattr(mol, "basis", "")),
        }

    opts = dict(req.method_options or {})

    if meth == "fci":
        # Total energy (includes nuclear)
        e = float(_fci.FCI(mf).kernel(**opts)[0])
    elif meth == "ccsd":
        # Return total energy (HF + CCSD correlation)
        mycc = _cc.CCSD(mf)
        ret = mycc.kernel(**opts)
        e_corr = float(ret[0]) if isinstance(ret, (tuple, list)) else float(ret)
        e_tot_attr = getattr(mycc, "e_tot", None)
        if e_tot_attr is not None:
            e = float(e_tot_attr)
        else:
            e = float(getattr(mf, "e_tot", 0.0) + e_corr)
    elif meth == "ccsd(t)" or meth == 'ccsd_t':
        # CCSD(T) total energy: CCSD total + triples correction
        mycc = _cc.CCSD(mf)
        mycc.kernel()
        et = mycc.ccsd_t()
        e = float(getattr(mycc, "e_tot", 0.0) + et)
    elif meth == "mp2":
        # Return total energy (HF + MP2 correlation)
        mymp = _mp.MP2(mf)
        ret = mymp.kernel(**opts)
        e_corr = float(ret[0]) if isinstance(ret, (tuple, list)) else float(ret)
        e = float(getattr(mf, "e_tot", 0.0) + e_corr)
    elif meth == "dft":
        xc = str(opts.get("functional", "b3lyp"))
        rks = _dft.RKS(mol)
        rks.xc = xc
        rks.verbose = 0
        e = float(rks.kernel())
    elif meth == "casscf":
        ncas = int(opts.get("ncas"))
        nelecas = opts.get("nelecas")
        if isinstance(nelecas, (list, tuple)):
            nele = (int(nelecas[0]), int(nelecas[1]))
        else:
            nele = int(nelecas)
        mycas = _mcscf.CASSCF(mf, ncas, nele)
        e = float(mycas.kernel(**{k: v for k, v in opts.items() if k not in ("ncas", "nelecas")} )[0])
    else:
        return {"error": f"unsupported method: {req.method}"}

    resp: Dict[str, Any] = {
        "energy": float(e),
        "classical_device": req.classical_device,
    }
    if req.verbose:
        resp["verbose"] = {
            "e_hf": float(getattr(mf, "e_tot", 0.0)),
            "mo_coeff": np.asarray(getattr(mf, "mo_coeff", None)).tolist() if getattr(mf, "mo_coeff", None) is not None else None,
            "mo_energy": np.asarray(getattr(mf, "mo_energy", None)).tolist() if getattr(mf, "mo_energy", None) is not None else None,
        }
    return resp


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)
