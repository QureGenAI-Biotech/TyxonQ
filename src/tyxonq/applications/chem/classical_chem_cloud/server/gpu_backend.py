from __future__ import annotations

from typing import Any, Dict
import numpy as np
from pyscf import gto  # type: ignore
from . import cpu_backend

# Optional GPU backends resolved once at module load

# gpu4pyscf fallback: https://github.com/pyscf/gpu4pyscf
from pyscf import cc
from pyscf import mp
from pyscf import dft
from pyscf import gto, scf
try:
    from gpu4pyscf.cc import ccsd_incore
    from gpu4pyscf.mp import mp2
    from gpu4pyscf.dft import rks
    gpu_avaiable = True
except:
    gpu_avaiable = False
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
    try:

        mf = scf.RHF(m)
        mf.chkfile = None
        mf.verbose = 0
        try:
            if use_density_fit:
                mf.density_fit().to_gpu().kernel()
            else:
                mf.to_gpu().kernel()
        except:
            cpu_backend.setup_cpu_resources(mf)
            if use_density_fit:
                mf.density_fit().kernel()
            else:
                mf.kernel()

        if method == "hf_integrals":
            # Try GPU-friendly integral path; fallback to CPU helper if needed
            #TODO GPU Version
            # Delegate to CPU backend if GPU path cannot provide MO integrals
            cpu_payload = dict(payload)
            cpu_payload["classical_device"] = "cpu"
            return cpu_backend.compute(cpu_payload)

        opts = dict(payload.get("method_options", {}))
        if method == "fci":
            # Delegate FCI to CPU backend (no GPU FCI available)
            cpu_payload = dict(payload)
            cpu_payload["classical_device"] = "cpu"
            return cpu_backend.compute(cpu_payload)
        if method == "ccsd":
            mycc = ccsd_incore.CCSD(mf)  # type: ignore
            et = mycc.kernel()
            e = float(getattr(mycc, "e_tot", 0.0) + et)
        elif method in ("ccsd(t)", "ccsd_t"):
            mycc = ccsd_incore.CCSD(mf)  # type: ignore
            et = mycc.kernel()
            et = mycc.ccsd_t()
            e = float(getattr(mycc, "e_tot", 0.0) + et)
        elif method == "mp2":
            mymp = gMP2(mf)  # type: ignore
            ret = mymp.kernel(**opts)
            e_corr = float(ret[0]) if isinstance(ret, (tuple, list)) else float(ret)
            e = float(getattr(mf, "e_tot", 0.0) + e_corr)
        elif method == "dft":
            xc = str(opts.get("functional", "b3lyp"))
            rks = rks.RKS(m)  # type: ignore
            rks.xc = xc
            rks.verbose = 0
            e = float(rks.kernel())
        elif method == "casscf":
            # Delegate CASSCF to CPU backend due to limited GPU support
            cpu_payload = dict(payload)
            cpu_payload["classical_device"] = "cpu"
            return cpu_backend.compute(cpu_payload)
        else:
            # Unknown method on GPU path; delegate to CPU
            cpu_payload = dict(payload)
            cpu_payload["classical_device"] = "cpu"
            return cpu_backend.compute(cpu_payload)

        return {"energy": float(e), "classical_device": payload.get("classical_device", "gpu")}
    except Exception as _:
        raise


