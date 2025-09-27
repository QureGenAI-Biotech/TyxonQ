from __future__ import annotations

from typing import Any, Dict, Optional
import os
import math
import numpy as np

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional
    psutil = None  # type: ignore

from pyscf import gto, scf
from pyscf import fci
from pyscf import cc
from pyscf import mp
from pyscf import dft
from pyscf import mcscf
from pyscf import lib

from tyxonq.applications.chem.chem_libs.hamiltonians_chem_library.hamiltonian_builders import get_integral_from_hf


def _sys_limits() -> tuple[int, int]:
    cores = max(1, int((os.cpu_count() or 1) * 0.95))
    mem_mb = 4096
    if psutil is not None:
        try:
            mem_total = psutil.virtual_memory().total
            mem_mb = max(1024, int(mem_total * 0.85 / (1024 * 1024)))
        except Exception:
            pass
    return cores, mem_mb


def setup_cpu_resources(mf: scf.hf.RHF | scf.uhf.UHF | scf.rohf.ROHF) -> None:
    cores, mem_mb = _sys_limits()
    lib.num_threads(cores)
    os.environ["OMP_NUM_THREADS"] = str(cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cores)
    os.environ["MKL_NUM_THREADS"] = str(cores)
    try:
        mf.max_memory = int(mem_mb)
    except Exception:
        pass


def _build_mol(mdat: Dict[str, Any]) -> gto.Mole:
    m = gto.Mole()
    m.atom = mdat.get("atom")
    m.basis = mdat.get("basis", "sto-3g")
    m.charge = int(mdat.get("charge", 0))
    m.spin = int(mdat.get("spin", 0))
    unit = str(mdat.get("unit", "Angstrom"))
    m.unit = unit
    m.build()
    return m


def compute(payload: Dict[str, Any]) -> Dict[str, Any]:
    method = str(payload.get("method", "fci")).lower()
    verbose = bool(payload.get("verbose", False))
    mdat = dict(payload.get("molecule_data", {}))
    m = _build_mol(mdat)

    mf = scf.RHF(m)
    mf.chkfile = None
    mf.verbose = 0
    setup_cpu_resources(mf)
    mf.kernel()

    if method == "hf_integrals":
        int1e, int2e, e_core = get_integral_from_hf(
            mf,
            active_space=payload.get("active_space"),
            aslst=payload.get("active_orbital_indices"),
        )
        return {
            "int1e": np.asarray(int1e).tolist(),
            "int2e": np.asarray(int2e).tolist(),
            "e_core": float(e_core),
            "e_hf": float(getattr(mf, "e_tot", 0.0)),
            "mo_coeff": np.asarray(getattr(mf, "mo_coeff", None)).tolist()
            if getattr(mf, "mo_coeff", None) is not None
            else None,
            "nelectron": int(getattr(m, "nelectron", 0)),
            "nao": int(getattr(m, "nao", 0)),
            "spin": int(getattr(m, "spin", 0)),
            "basis": str(getattr(m, "basis", "")),
        }

    opts = dict(payload.get("method_options", {}))

    if method == "fci":
        e = float(fci.FCI(mf).kernel(**opts)[0])
    elif method == "ccsd":
        mycc = cc.CCSD(mf)
        ret = mycc.kernel(**opts)
        e_corr = float(ret[0]) if isinstance(ret, (tuple, list)) else float(ret)
        e_tot_attr = getattr(mycc, "e_tot", None)
        e = float(e_tot_attr) if e_tot_attr is not None else float(getattr(mf, "e_tot", 0.0) + e_corr)
    elif method in ("ccsd(t)", "ccsd_t"):
        mycc = cc.CCSD(mf)
        mycc.kernel()
        et = mycc.ccsd_t()
        e = float(getattr(mycc, "e_tot", 0.0) + et)
    elif method == "mp2":
        mymp = mp.MP2(mf)
        ret = mymp.kernel(**opts)
        e_corr = float(ret[0]) if isinstance(ret, (tuple, list)) else float(ret)
        e = float(getattr(mf, "e_tot", 0.0) + e_corr)
    elif method == "dft":
        xc = str(opts.get("functional", "b3lyp"))
        rks = dft.RKS(m)
        rks.xc = xc
        rks.verbose = 0
        e = float(rks.kernel())
    elif method == "casscf":
        ncas = int(opts.get("ncas"))
        nelecas = opts.get("nelecas")
        if isinstance(nelecas, (list, tuple)):
            nele = (int(nelecas[0]), int(nelecas[1]))
        else:
            nele = int(nelecas)
        mycas = mcscf.CASSCF(mf, ncas, nele)
        e = float(mycas.kernel(**{k: v for k, v in opts.items() if k not in ("ncas", "nelecas")} )[0])
    else:
        return {"error": f"unsupported method: {method}"}

    resp: Dict[str, Any] = {
        "energy": float(e),
        "classical_device": payload.get("classical_device", "cpu"),
    }
    if verbose:
        resp["verbose"] = {
            "e_hf": float(getattr(mf, "e_tot", 0.0)),
            "mo_coeff": np.asarray(getattr(mf, "mo_coeff", None)).tolist() if getattr(mf, "mo_coeff", None) is not None else None,
            "mo_energy": np.asarray(getattr(mf, "mo_energy", None)).tolist() if getattr(mf, "mo_energy", None) is not None else None,
        }
    return resp


