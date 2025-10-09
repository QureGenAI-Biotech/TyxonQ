from __future__ import annotations

from importlib.machinery import BuiltinImporter
from typing import Any, Dict
import numpy as np
from pyscf import gto  # type: ignore
from tyxonq.applications.chem.classical_chem_cloud.server import cpu_chem


# Optional GPU backends resolved once at module load

# gpu4pyscf fallback: https://github.com/pyscf/gpu4pyscf
from pyscf import cc as cc_cpu
from pyscf import mp as mp_cpu
from pyscf import dft as dft_cpu
from pyscf import gto, scf
from pyscf.mcscf import CASCI
try:
    from gpu4pyscf import cc
    from gpu4pyscf import mp
    from gpu4pyscf import dft
    gpu_available = True
except:
    gpu_available = False

def compute(payload: Dict[str, Any],pre_build_mol= None) -> Dict[str, Any]:

    method_payload = payload.get("method", "hf")

    if isinstance(method_payload, str):
        method_list = [method_payload.lower()]
    else:
        method_list = [x.lower() for x in method_payload]

    mdat = dict(payload.get("molecule_data", {}))
    use_density_fit = bool(payload.get("use_density_fit", True))
    if pre_build_mol:
        m = pre_build_mol
    else:
        m = cpu_chem.build_mol(mdat)


    method_result = {}
    for method in method_list:
        pass


    
    if method == "hf":
        if m.spin==0:
            mf = scf.RHF(m).to_gpu().run()
        else:
            mf = scf.ROHF(m).to_gpu().run()

        # mf.kernel()
        method_result["e_hf"] = mf.e_tot
        method_result["mo_coeff"] = mf.mo_coeff
        method_result["nelectron"] = m.nelectorn
        method_result["nao"] = m.nao
        method_result["spin"] = m.spin
        method_result["basis"] = m.basis
        method_result["_eri"] = m.intor("int2e", aosym="s8")
        method_result['energy'] = mf.e_tot
    opts = dict(payload.get("method_options", {}))

    active_orbital_indices = opts.get('active_orbital_indices',None)
    frozen_idx = [i for i in range(m.nao) if i not in active_orbital_indices]
    if method == "fci":
        method_result = {}
        active_space = opts.get('active_space',(2,2))
        active_orbital_indices = opts.get('active_orbital_indices',None)
        fci = CASCI(mf, active_space[1], active_space[0])
        mo = fci.sort_mo(active_orbital_indices, base=0)
        res = fci.kernel(mo)

        #energey here include the e_core
        method_result["e_fci"] = res[0]
        method_result["civector_fci"] = res[2].ravel()

    if method == "ccsd":

        mf = scf.RHF(m).to_gpu().run()

        if frozen_idx:
            ret = cc.ccsd_incore.CCSD(mf,frozen=frozen_idx).run(**opts)
        else:
            ret = cc.ccsd_incore.CCSD(mf).kernel(**opts)  # type: ignore
        e_corr = ret[0]
        e = float(getattr(mf, "e_tot", 0.0) + e_corr)

        method_result["e_ccsd"] = mf.e_tot + e_corr
        method_result["ccsd_t1"] = mycc.t1
        method_result["ccsd_t2"] = mycc.t2
        method_result['energy'] = mf.e_tot + e_corr

    elif method in ("ccsd(t)", "ccsd_t"):
        method_result = {}
        if frozen_idx:
            mycc = cc_cpu.CCSD(mf,frozen=frozen_idx).run(**opts)
        else:
            mycc = cc_cpu.CCSD(mf).run(**opts)
        
        method_result["e_ccsd"] = mycc.e_tot
        method_result["ccsd_t1"] = mycc.t1
        method_result["ccsd_t2"] = mycc.t2
        et = mycc.ccsd_t()
        method_result["et_ccsd_t"] = et
        method_result["e_ccsd_t"] = mycc.e_tot +et
        method_result['energy'] = mycc.e_tot +et
    elif method == "mp2":
        mf = scf.RHF(m).to_gpu().run()
        method_result = {}
        if frozen_idx:
            mymp = mp.MP2(mf,frozen=frozen_idx)
        else:
            mymp = mp.MP2(mf)  # type: ignore
        ret = mymp.kernel(**opts)
        e_corr = float(ret[0]) if isinstance(ret, (tuple, list)) else float(ret)
        e = float(getattr(mf, "e_tot", 0.0) + e_corr)

        method_result["e_mp2"] = mf.e_tot + e_corr
        method_result["mp2_t2"] = mycc.t2
        method_result['energy'] = mf.e_tot + e_corr
    elif method == "dft":
        method_result = {}
        xc = str(opts.get("functional", "b3lyp"))
        rks = dft.rks.RKS(m)  # type: ignore
        rks.xc = xc
        e = float(rks.kernel())
        method_result['energy'] = e
        method_result["e_dft"] = e
        method_result["xc"] = xc
    else:
        return {"error": f"unsupported method: {method}"}

    resp: Dict[str, Any] = {
        "classical_device": payload.get("classical_device", "cpu"),
    }

    return resp


