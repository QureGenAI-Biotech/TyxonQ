from __future__ import annotations

from typing import Tuple, List, Sequence, Union

import numpy as np
from openfermion import QubitOperator
from openfermion.transforms import jordan_wigner
from pyscf.scf.hf import RHF  # type: ignore
from pyscf import fci  # type: ignore
from scipy.optimize import minimize

from ..runtimes.ucc_device_runtime import UCCDeviceRuntime
from ..runtimes.ucc_numeric_runtime import UCCNumericRuntime
from tyxonq.libs.circuits_library.analysis import get_circuit_summary
from tyxonq.libs.circuits_library.ucc import build_ucc_circuit
from tyxonq.applications.chem.chem_libs.hamiltonians_chem_library.hamiltonian_builders import (
    get_integral_from_hf,
    get_hop_from_integral,
    get_h_fcifunc_from_integral
)
from tyxonq.libs.hamiltonian_encoding.pauli_io import reverse_qop_idx
from tyxonq.applications.chem.chem_libs.quantum_chem_library.ci_state_mapping import get_ci_strings, get_addr as _get_addr_ci


class UCC:
    """Minimal UCC 算法入口（device 默认）。

    - 支持将参数化 ansatz（ex_ops/param_ids/init_state）传递给设备 runtime
    - 对外提供 energy/energy_and_grad（device 路径）
    - 后续将扩展 numeric_statevector/numeric_mps
    """

    def __init__(
        self,
        n_qubits: int,
        n_elec_s: Tuple[int, int],
        h_qubit_op: QubitOperator,
        runtime: str = "device",
        mode: str = "fermion",
        *,
        ex_ops: List[tuple] | None = None,
        param_ids: List[int] | None = None,
        init_state: np.ndarray | None = None,
        decompose_multicontrol: bool = False,
        trotter: bool = False,
    ):
        self.n_qubits = int(n_qubits)
        self.n_elec_s = (int(n_elec_s[0]), int(n_elec_s[1]))
        self.h_qubit_op = h_qubit_op
        self.runtime = runtime
        self.mode = mode
        self.e_core: float = 0.0
        self._params: np.ndarray | None = None

        self.ex_ops = list(ex_ops) if ex_ops is not None else None
        self.param_ids = list(param_ids) if param_ids is not None else None
        self.init_state = init_state
        self.decompose_multicontrol = bool(decompose_multicontrol)
        self.trotter = bool(trotter)
        self.scipy_minimize_options: dict | None = None
        # init guess (zeros by default if ex_ops present)
        if self.ex_ops is not None:
            self.init_guess = np.zeros(self.n_params, dtype=np.float64)
        else:
            self.init_guess = np.zeros(0, dtype=np.float64)

    def _runtime(self) -> UCCDeviceRuntime:
        return UCCDeviceRuntime(
            self.n_qubits,
            self.n_elec_s,
            self.h_qubit_op,
            mode=self.mode,
            ex_ops=self.ex_ops,
            param_ids=self.param_ids,
            init_state=self.init_state,
            decompose_multicontrol=self.decompose_multicontrol,
            trotter=self.trotter,
        )

    def energy(self, params: np.ndarray | None = None, **opts) -> float:
        runtime = str(opts.pop("runtime", self.runtime))
        numeric_engine = opts.pop("numeric_engine", None) or getattr(self, "numeric_engine", None)
        
        if runtime == "device":
            rt = self._runtime()
            e = rt.energy(params, **opts)
            return float(e + self.e_core)
        if runtime == "numeric":
            # Construct optional CI Hamiltonian for CI engines
            ci_hamiltonian = None
            try:
                # Pass (na, nb) explicitly to support open-shell
                ci_hamiltonian = get_h_fcifunc_from_integral(self._int1e, self._int2e, self.n_elec_s)
            except Exception:
                ci_hamiltonian = None
            rt = UCCNumericRuntime(
                self.n_qubits,
                self.n_elec_s,
                self.h_qubit_op,
                ex_ops=self.ex_ops,
                param_ids=self.param_ids,
                init_state=self.init_state,
                mode=self.mode,
                numeric_engine=numeric_engine,
                ci_hamiltonian=ci_hamiltonian,
                
            )
            base = np.asarray(params if params is not None else (self.init_guess if getattr(self, "init_guess", None) is not None else np.zeros(self.n_params)), dtype=np.float64)
            return float(rt.energy(base)+self.e_core)
        raise ValueError(f"unknown runtime: {runtime}")

    def energy_and_grad(self, params: np.ndarray | None = None, **opts):
        runtime = str(opts.pop("runtime", self.runtime))
        numeric_engine = opts.pop("numeric_engine", None) or getattr(self, "numeric_engine", None)
        
        if runtime == "device":
            rt = self._runtime()
            e, g = rt.energy_and_grad(params, **opts)
            return float(e+self.e_core), g
        if runtime == "numeric":
            # Construct optional CI Hamiltonian for CI engines
            ci_hamiltonian = None
            try:
                # Pass (na, nb) explicitly to support open-shell
                ci_hamiltonian = get_h_fcifunc_from_integral(self._int1e, self._int2e, self.n_elec_s)
            except Exception:
                ci_hamiltonian = None
            rt = UCCNumericRuntime(
                self.n_qubits,
                self.n_elec_s,
                self.h_qubit_op,
                ex_ops=self.ex_ops,
                param_ids=self.param_ids,
                init_state=self.init_state,
                mode=self.mode,
                numeric_engine=numeric_engine,
                ci_hamiltonian=ci_hamiltonian
                
            )
            base = np.asarray(params if params is not None else (self.init_guess if getattr(self, "init_guess", None) is not None else np.zeros(self.n_params)), dtype=np.float64)
            e0, g = rt.energy_and_grad(base)
            return float(e0+self.e_core), g
        raise ValueError(f"unknown runtime: {runtime}")

    @property
    def e_ucc(self) -> float:
        """Convenience: return current UCC energy (with core) using stored params.

        Tests expect `e_ucc` to be available for both closed- and open-shell.
        """
        return float(self.energy(getattr(self, "_params", None)))

    def kernel(self, **opts) -> float:
        """Optimize parameters via L-BFGS-B.

        Any options in **opts will be forwarded to energy_and_grad (e.g.,
        shots/provider/device for device runtime, numeric_engine for numeric runtime).
        """
        if self.n_params == 0:
            return float(self.e_core)
        x0 = np.asarray(self.init_guess if getattr(self, "init_guess", None) is not None else np.zeros(self.n_params), dtype=np.float64)

        # runtime options (shots/provider/device/numeric_engine/etc.) from caller
        # 默认 shots 统一为 1024（避免 0 导致无法投递到真机）；调用方可覆盖
        runtime_opts = dict(opts)
        if "shots" not in runtime_opts and str(runtime_opts.get("runtime", self.runtime)) == "device":
            runtime_opts["shots"] = 2048

        def _obj(x: np.ndarray):
            e, g = self.energy_and_grad(x, **runtime_opts)
            return e, np.asarray(g, dtype=np.float64)

        minimize_options = self.scipy_minimize_options or {"ftol": 1e-9, "gtol": 1e-6}
        res = minimize(lambda v: _obj(v), x0=x0, jac=True, method="L-BFGS-B", options=minimize_options)
        self._params = np.asarray(getattr(res, "x", x0), dtype=np.float64)
        return float(getattr(res, "fun", self.energy(self._params)))

    # ---- Convenience builders from integrals / molecule (fermion/qubit modes) ----
    @classmethod
    def from_integral(
        cls,
        int1e: np.ndarray,
        int2e: np.ndarray,
        n_elec: Union[int, Tuple[int, int]],
        *,
        mode: str = "fermion",
        runtime: str = "device",
        ex_ops: List[tuple] | None = None,
        param_ids: List[int] | None = None,
        init_state: np.ndarray | None = None,
        decompose_multicontrol: bool = False,
        trotter: bool = False,
    ) -> "UCC":
        if isinstance(n_elec, int):
            assert n_elec % 2 == 0
            n_elec_s = (n_elec // 2, n_elec // 2)
        else:
            n_elec_s = (int(n_elec[0]), int(n_elec[1]))
        if mode in ["fermion", "qubit"]:
            n_qubits = 2 * len(int1e)
            fop = get_hop_from_integral(int1e, int2e)
            hq = reverse_qop_idx(jordan_wigner(fop), n_qubits)
        else:
            n_qubits = len(int1e)
            # 简化：HCB 模式暂不在 algorithms 中构造，后续补充
            raise NotImplementedError("hcb mode not yet supported in algorithms.ucc.from_integral")
        inst = cls(
            n_qubits=n_qubits,
            n_elec_s=n_elec_s,
            h_qubit_op=hq,
            runtime=runtime,
            mode=mode,
            ex_ops=ex_ops,
            param_ids=param_ids,
            init_state=init_state,
            decompose_multicontrol=decompose_multicontrol,
            trotter=trotter,
        )
        return inst

    @classmethod
    def from_molecule(
        cls,
        m,
        *,
        active_space=None,
        aslst=None,
        mode: str = "fermion",
        runtime: str = "device",
        ex_ops: List[tuple] | None = None,
        param_ids: List[int] | None = None,
        init_state: np.ndarray | None = None,
        decompose_multicontrol: bool = False,
        trotter: bool = False,
    ) -> "UCC":
        hf = RHF(m)
        hf.chkfile = None
        hf.verbose = 0
        hf.kernel()
        int1e, int2e, e_core = get_integral_from_hf(hf, active_space=active_space, aslst=aslst)
        n_elec = active_space[0] if active_space is not None else int(m.nelectron)
        inst = cls.from_integral(
            int1e,
            int2e,
            n_elec,
            mode=mode,
            runtime=runtime,
            ex_ops=ex_ops,
            param_ids=param_ids,
            init_state=init_state,
            decompose_multicontrol=decompose_multicontrol,
            trotter=trotter,
        )
        inst.e_core = float(e_core)
        return inst

    # ---- Helpers mirrored from static for compatibility ----
    def _check_params_argument(self, params: Sequence[float] | None, *, strict: bool = False) -> np.ndarray:
        if params is None:
            if hasattr(self, "params") and self.params is not None:
                params = self.params
            else:
                if strict:
                    raise ValueError("Run the `.kernel` method to determine the parameters first")
                if self.n_params == 0:
                    return np.zeros(0, dtype=np.float64)
                params = np.zeros(self.n_params, dtype=np.float64)
        if len(params) != self.n_params:
            raise ValueError(f"Incompatible parameter shape. {self.n_params} is desired. Got {len(params)}")
        return np.asarray(params, dtype=np.float64)

    @property
    def n_params(self) -> int:
        if self.param_ids is None:
            return len(self.ex_ops) if self.ex_ops is not None else 0
        return (max(self.param_ids) + 1) if len(self.param_ids) > 0 else 0

    def get_circuit(self, params: Sequence[float] | None = None, *, decompose_multicontrol: bool | None = None, trotter: bool | None = None):
        p = self._check_params_argument(params, strict=False)
        return build_ucc_circuit(
            p,
            self.n_qubits,
            self.n_elec_s,
            tuple(self.ex_ops) if self.ex_ops is not None else tuple(),
            tuple(self.param_ids) if self.param_ids is not None else None,
            mode=self.mode,
            init_state=self.init_state,
            decompose_multicontrol=self.decompose_multicontrol if decompose_multicontrol is None else bool(decompose_multicontrol),
            trotter=self.trotter if trotter is None else bool(trotter),
        )

    def energy_device(
        self,
        params: Sequence[float] | None = None,
        *,
        shots: int = 8192,
        provider: str = "simulator",
        device: str = "statevector",
        postprocessing: dict | None = None,
    ) -> float:
        if self.ex_ops is None:
            # HF only
            return float(self.e_core)
        p = self._check_params_argument(params, strict=False)
        rt = self._runtime()
        e = rt.energy(p, shots=shots, provider=provider, device=device, postprocessing=postprocessing)
        return float(e + self.e_core)

    # ---- RDM in MO basis (spin-traced) ----
    def make_rdm1(self, params: Sequence[float] | None = None, *, basis: str = "MO") -> np.ndarray:
        if basis != "MO":
            raise NotImplementedError("algorithms.UCC.make_rdm1 currently supports basis='MO' only")
        p = self._check_params_argument(params, strict=False)
        rt = UCCNumericRuntime(
            self.n_qubits,
            self.n_elec_s,
            self.h_qubit_op,
            ex_ops=self.ex_ops,
            param_ids=self.param_ids,
            init_state=self.init_state,
            mode=self.mode,
        )
        psi = rt._state(p)
        cis = get_ci_strings(self.n_qubits, self.n_elec_s, self.mode)
        civ = psi[cis]
        rdm1_cas = fci.direct_spin1.make_rdm1(civ.astype(np.float64), self.n_qubits // 2, self.n_elec_s)
        return np.asarray(rdm1_cas, dtype=np.float64)

    def make_rdm2(self, params: Sequence[float] | None = None, *, basis: str = "MO") -> np.ndarray:
        if basis != "MO":
            raise NotImplementedError("algorithms.UCC.make_rdm2 currently supports basis='MO' only")
        p = self._check_params_argument(params, strict=False)
        rt = UCCNumericRuntime(
            self.n_qubits,
            self.n_elec_s,
            self.h_qubit_op,
            ex_ops=self.ex_ops,
            param_ids=self.param_ids,
            init_state=self.init_state,
            mode=self.mode,
        )
        psi = rt._state(p)
        cis = get_ci_strings(self.n_qubits, self.n_elec_s, self.mode)
        civ = psi[cis]
        rdm2_cas = fci.direct_spin1.make_rdm12(civ.astype(np.float64), self.n_qubits // 2, self.n_elec_s)[1]
        return np.asarray(rdm2_cas, dtype=np.float64)

    # ---- CI helpers ----
    def civector(self, params: Sequence[float] | None = None, *, numeric_engine: str | None = None) -> np.ndarray:
        p = self._check_params_argument(params, strict=False)
        rt = UCCNumericRuntime(
            self.n_qubits,
            self.n_elec_s,
            self.h_qubit_op,
            ex_ops=self.ex_ops,
            param_ids=self.param_ids,
            init_state=self.init_state,
            mode=self.mode,
            numeric_engine=numeric_engine,
        )
        psi = rt._state(p)
        ci_strings = get_ci_strings(self.n_qubits, self.n_elec_s, self.mode)
        return psi[ci_strings]

    def statevector(self, params: Sequence[float] | None = None) -> np.ndarray:
        p = self._check_params_argument(params, strict=False)
        rt = UCCNumericRuntime(
            self.n_qubits,
            self.n_elec_s,
            self.h_qubit_op,
            ex_ops=self.ex_ops,
            param_ids=self.param_ids,
            init_state=self.init_state,
            mode=self.mode,
        )
        return rt._state(p)

    def get_ci_strings(self, strs2addr: bool = False):
        return get_ci_strings(self.n_qubits, self.n_elec_s, self.mode, strs2addr=strs2addr)

    def get_addr(self, bitstring: str) -> int:
        ci_strings, strs2addr = get_ci_strings(self.n_qubits, self.n_elec_s, self.mode, strs2addr=True)
        return int(_get_addr_ci(int(bitstring, 2), self.n_qubits, self.n_elec_s, strs2addr, self.mode))

    # ---- Printing helpers ----
    def print_ansatz(self):
        info = {
            "#qubits": self.n_qubits,
            "#params": self.n_params,
            "#excitations": 0 if self.ex_ops is None else len(self.ex_ops),
            "initial condition": "custom" if self.init_state is not None else "RHF",
        }
        try:
            import pandas as _pd  # type: ignore
            print(_pd.DataFrame([info]).to_string(index=False))
        except Exception:
            print(info)

    def print_circuit(self):
        c = self.get_circuit()
        summary = get_circuit_summary(c)
        try:
            print(summary.to_string(index=False))  # type: ignore[attr-defined]
        except Exception:
            print(summary)

    def print_summary(self, include_circuit: bool = False):
        print("################################ Ansatz ###############################")
        self.print_ansatz()
        if include_circuit:
            print("############################### Circuit ###############################")
            self.print_circuit()

    # ---- Params mapping ----
    @property
    def param_to_ex_ops(self):
        mapping: dict[int, list] = {}
        if self.param_ids is None or self.ex_ops is None:
            return mapping
        for ex, pid in zip(self.ex_ops, self.param_ids):
            mapping.setdefault(int(pid), []).append(ex)
        return mapping

    # ---- Params property ----
    @property
    def params(self) -> np.ndarray | None:
        return self._params

    @params.setter
    def params(self, v: Sequence[float] | None) -> None:
        self._params = None if v is None else np.asarray(v, dtype=np.float64)


