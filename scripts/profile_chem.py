from __future__ import annotations

import cProfile
import pstats
import io
import os


def profile_callable(tag: str, fn):
    pr = cProfile.Profile()
    pr.enable()
    try:
        fn()
    finally:
        pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
    ps.print_stats(30)
    print(f"===== PROFILE [{tag}] (Top 30 by cumtime) =====")
    print(s.getvalue())


def run_uccsd_device():
    # H2, default basis
    from tyxonq.applications.chem.molecule import h4
    from tyxonq.applications.chem import UCCSD
    ucc = UCCSD(h4, runtime="device")
    # device 路径，provider 默认 simulator，kernel 内 shots=0
    _ = ucc.kernel(runtime="device")


def run_hea_device():
    from tyxonq.applications.chem.molecule import h4
    from tyxonq.applications.chem import HEA
    hea = HEA.from_molecule(h4, active_space=(2, 2), n_layers=1, mapping="parity", runtime="device")
    _ = hea.kernel(runtime="device")


def run_uccsd_device_shots():
    # H2 with shots>0 to profile counts path
    from tyxonq.applications.chem.molecule import h4
    from tyxonq.applications.chem import UCCSD
    import numpy as np
    ucc = UCCSD(h4, runtime="device")
    params = np.zeros(ucc.n_params, dtype=float)
    _ = ucc.energy(params, runtime="device", shots=int(os.environ.get("CHEM_PROFILE_SHOTS", 2048)), provider="simulator", device="statevector")


def run_hea_device_shots():
    # H2 with shots>0 to profile counts path
    from tyxonq.applications.chem.molecule import h4
    from tyxonq.applications.chem import HEA
    hea = HEA.from_molecule(h4, active_space=(2, 2), n_layers=1, mapping="parity", runtime="device")
    _ = hea.energy(hea.init_guess, runtime="device", shots=int(os.environ.get("CHEM_PROFILE_SHOTS", 2048)), provider="simulator", device="statevector")


if __name__ == "__main__":
    # 可选：限制 BLAS 线程，避免干扰
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

    profile_callable("UCCSD.device.H4", run_uccsd_device)
    profile_callable("HEA.device.H4", run_hea_device)
    # shots>0 counts path (H2 to keep fast)
    profile_callable("UCCSD.device.shots.H4", run_uccsd_device_shots)
    profile_callable("HEA.device.shots.H4", run_hea_device_shots)


