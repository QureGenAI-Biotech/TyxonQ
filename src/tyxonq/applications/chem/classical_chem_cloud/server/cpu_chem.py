from __future__ import annotations

from typing import Any, Dict, Optional
import os
import base64
import tempfile
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
from pyscf.mcscf import CASCI

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


def setup_cpu_resources() -> None:
    cores, mem_mb = _sys_limits()
    lib.num_threads(cores)
    os.environ["OMP_NUM_THREADS"] = str(cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cores)
    os.environ["MKL_NUM_THREADS"] = str(cores)
    os.environ["PYSCF_MAX_MEMORY"] = str(mem_mb)


def build_mol(mdat: Dict[str, Any]) -> gto.Mole:
    m = gto.Mole()
    m.atom = mdat.get("atom")
    m.basis = mdat.get("basis", "sto-3g")
    m.charge = int(mdat.get("charge", 0))
    m.spin = int(mdat.get("spin", 0))
    unit = str(mdat.get("unit", "Angstrom"))
    m.unit = unit
    m.build()
    return m

setup_cpu_resources()

def compute(payload: Dict[str, Any], pre_build_mol: gto.Mole | None = None, pre_compute_hf=None) -> Dict[str, Any]:
    method_payload = payload.get("method", "hf")

    if isinstance(method_payload, str):
        method_list = [method_payload.lower()]
    else:
        method_list = [x.lower() for x in method_payload]
    
    if 'hf' not in method_list:
        method_list.append("hf")

    mdat = dict(payload.get("molecule_data", {}))
    use_density_fit = bool(payload.get("use_density_fit", False))
    m = pre_build_mol if pre_build_mol is not None else build_mol(mdat)

    # Build or reuse HF; run once and reuse for all requested methods
    if pre_compute_hf is not None:
        mf = pre_compute_hf
    else:
        if m.spin == 0:
            mf = scf.RHF(m)
        else:
            mf = scf.ROHF(m)
        if use_density_fit and hasattr(mf, "density_fit"):
            mf = mf.density_fit()
        # ensure chkfile available for artifact export
        with tempfile.NamedTemporaryFile(prefix="hf_", suffix=".chk", delete=False) as tf:
            mf.chkfile = tf.name
        mf.verbose = 0
        mf = mf.run()

    # Ensure we can export chkfile even when HF is precomputed
    chk_b64: str | None = None
    try:
        if not getattr(mf, "chkfile", None):
            with tempfile.NamedTemporaryFile(prefix="hf_", suffix=".chk") as tf:
                mf.chkfile = tf.name
            # dump current SCF state
            mf.dump_chk()
        with open(mf.chkfile, "rb") as fr:
            chk_b64 = base64.b64encode(fr.read()).decode("ascii")
    except Exception:
        chk_b64 = None

    final_result: Dict[str, Any] = {}
    active_space = payload.get('active_space',None)
    if active_space is None:
        active_space = (m.nelectron, int(m.nao))
    active_orbital_indices = payload.get("active_orbital_indices", None)



    inactive_occ = (m.nelectron - active_space[0]) // 2
    assert (m.nelectron - active_space[0]) % 2 == 0
    inactive_vir = m.nao - active_space[1] - inactive_occ

    if active_orbital_indices is None:
        active_orbital_indices = list(range(inactive_occ, m.nao - inactive_vir))
    frozen_idx = None
    if active_orbital_indices is not None:
        frozen_idx = [i for i in range(m.nao) if i not in active_orbital_indices]
    opts = dict(payload.get("method_options", {}))

    for method in method_list:
        method_result: Dict[str, Any] = {}
        if method == "hf":
            method_result["e_hf"] = mf.e_tot
            method_result["mo_coeff"] =np.asarray(mf.mo_coeff).tolist()
            method_result["mo_energy"] =np.asarray(mf.mo_energy).tolist()
            method_result["nao"] = m.nao
            method_result["spin"] = m.spin
            method_result["basis"] = m.basis
            method_result["_eri"] = np.asarray(m.intor("int2e", aosym="s8")).tolist()
            method_result['energy'] = mf.e_tot
            if chk_b64 is not None:
                method_result["chkfile_b64"] = chk_b64
        elif method == "fci":
            fci = CASCI(mf, active_space[1], active_space[0])
            mo = fci.sort_mo(active_orbital_indices, base=0)
            res = fci.kernel(mo)

            #energey here include the e_core
            method_result["e_fci"] = float(res[0])
            method_result["civector_fci"] = np.asarray(res[2]).ravel().tolist()
            method_result['energy'] = float(res[0])
        elif method == "ccsd":
            method_result = {}
            if frozen_idx:
                mycc = cc.CCSD(mf,frozen=frozen_idx).run()
            else:
                mycc = cc.CCSD(mf).run()
            method_result["e_ccsd"] = float(mycc.e_tot)
            method_result["ccsd_t1"] = np.asarray(mycc.t1).tolist()
            method_result["ccsd_t2"] = np.asarray(mycc.t2).tolist()
            method_result['energy'] = float(mycc.e_tot)
        elif method in ("ccsd(t)", "ccsd_t"):

            if frozen_idx:
                mycc = cc.CCSD(mf,frozen=frozen_idx).run()
            else:
                mycc = cc.CCSD(mf).run()
            
            method_result["e_ccsd"] = float(mycc.e_tot)
            method_result["ccsd_t1"] = np.asarray(mycc.t1).tolist()
            method_result["ccsd_t2"] = np.asarray(mycc.t2).tolist()
            et = mycc.ccsd_t()
            method_result["et_ccsd_t"] = et
            method_result["e_ccsd_t"] = float(mycc.e_tot +et)
            method_result['energy'] = float(mycc.e_tot +et)
        elif method == "mp2":
            if frozen_idx:
                mycc = mp.MP2(mf,frozen=frozen_idx).run()
            else:
                mycc = mp.MP2(mf).run()
            
            method_result["e_mp2"] = float(mycc.e_tot)
            method_result["mp2_t2"] = np.asarray(mycc.t2).tolist()
            method_result['energy'] = float(mycc.e_tot)
        elif method == "dft":
            xc = str(opts.get("functional", "b3lyp"))
            rks = dft.RKS(m)
            rks.xc = xc
            e = float(rks.kernel())
            method_result["energy"] = e
            method_result["e_dft"] = e
            method_result["xc"] = xc
        else:
            method_result = {"error": f"unsupported method: {method}"}
        final_result[method] = method_result

    return final_result


