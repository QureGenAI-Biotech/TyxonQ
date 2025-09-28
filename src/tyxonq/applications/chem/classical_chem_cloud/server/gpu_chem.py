from __future__ import annotations

from importlib.machinery import BuiltinImporter
from typing import Any, Dict
import numpy as np
from pyscf import gto  # type: ignore
from . import cpu_chem

# Optional GPU backends resolved once at module load

# gpu4pyscf fallback: https://github.com/pyscf/gpu4pyscf
from pyscf import cc
from pyscf import mp
from pyscf import dft
from pyscf import gto, scf
try:
    from gpu4pyscf import cc
    from gpu4pyscf import mp
    from gpu4pyscf import dft
    gpu_available = True
except:
    gpu_available = False
# Placeholder for ByteQC GPU backend integration.
# TODO: Replace with actual ByteQC client calls when repository and API are integrated.





def compute(payload: Dict[str, Any]) -> Dict[str, Any]:
    method = str(payload.get("method", "fci")).lower()
    mdat = dict(payload.get("molecule_data", {}))
    use_density_fit = bool(payload.get("use_density_fit", True))
    m = gto.Mole()
    m.atom = mdat.get("atom")
    m.basis = mdat.get("basis", "sto-3g")
    m.charge = int(mdat.get("charge", 0))
    m.spin = int(mdat.get("spin", 0))
    m.unit = str(mdat.get("unit", "Angstrom"))
    m.build()

    # ByteQC integration removed; always use gpu4pyscf path below  
    if method == "hf_integrals":
        # Try GPU-friendly integral path; fallback to CPU helper if needed
        # Delegate to CPU backend if GPU path cannot provide MO integrals
        cpu_payload = dict(payload)
        cpu_payload["classical_device"] = "cpu"
        mf = scf.RHF(m).to_gpu().run()
        try:
            mf = mf.to_cpu()
        except:
            pass
        return cpu_chem.compute(cpu_payload,pre_build_mol=m,pre_compute_hf=mf)

    if method == "fci":
        # Delegate FCI to CPU backend (no GPU FCI available)
        cpu_payload = dict(payload)
        cpu_payload["classical_device"] = "cpu"
        mf = scf.RHF(m).to_gpu().run()
        try:
           mf = mf.to_cpu()
        except:
            pass
        return cpu_chem.compute(cpu_payload,pre_build_mol=m,pre_compute_hf=mf)
    if method == "casscf":
        # Delegate CASSCF to CPU backend due to limited GPU support
        cpu_payload = dict(payload)
        cpu_payload["classical_device"] = "cpu"
        mf = scf.RHF(m).to_gpu().run()
        try:
            mf = mf.to_cpu()
        except:
            pass
        return cpu_chem.compute(cpu_payload,pre_build_mol=m,pre_compute_hf=mf)
    

    mf = scf.RHF(m).to_gpu()
    mf.chkfile = None
    mf.verbose = 0
    if use_density_fit:
        mf.density_fit().kernel()
    else:
        mf.to_gpu().kernel()

    opts = dict(payload.get("method_options", {}))

    if method == "ccsd":
        mf = scf.RHF(m).to_gpu().run()
        ret = cc.ccsd_incore.CCSD(mf).kernel(**opts)  # type: ignore
        e_corr = ret[0]
        e = float(getattr(mf, "e_tot", 0.0) + e_corr)
    elif method in ("ccsd(t)", "ccsd_t"):
        mf = scf.RHF(m).to_gpu().run()
        try:
            mf = mf.to_cpu()
        except Exception:
            pass
        cpu_payload = dict(payload)
        cpu_payload["classical_device"] = "cpu"
        return cpu_chem.compute(cpu_payload, pre_build_mol=m, pre_compute_hf=mf)
    elif method == "mp2":
        mymp = mp.MP2(mf)  # type: ignore
        ret = mymp.kernel(**opts)
        e_corr = float(ret[0]) if isinstance(ret, (tuple, list)) else float(ret)
        e = float(getattr(mf, "e_tot", 0.0) + e_corr)
    elif method == "dft":
        xc = str(opts.get("functional", "b3lyp"))
        rks = dft.rks.RKS(m)  # type: ignore
        rks.xc = xc
        rks.verbose = 0
        e = float(rks.kernel())
    else:
        # Unknown method on GPU path; delegate to CPU
        cpu_payload = dict(payload)
        cpu_payload["classical_device"] = "cpu"
        return cpu_chem.compute(cpu_payload)

    return {"energy": float(e), "classical_device": payload.get("classical_device", "gpu")}


