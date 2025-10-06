from __future__ import annotations

from typing import Tuple, List, Sequence, Union

import numpy as np
from openfermion import FermionOperator, QubitOperator
from openfermion.transforms import jordan_wigner
from pyscf.scf import RHF
from pyscf.scf import ROHF  # type: ignore

from pyscf.scf.hf import RHF as RHF_TYPE
from pyscf.scf.rohf import ROHF as ROHF_TYPE
from pyscf.cc.addons import spatial2spin
from pyscf import fci  # type: ignore
from scipy.optimize import minimize
import logging

from ..runtimes.ucc_device_runtime import UCCDeviceRuntime
from ..runtimes.ucc_numeric_runtime import UCCNumericRuntime, apply_excitation as _apply_excitation_numeric
from tyxonq.libs.circuits_library.analysis import get_circuit_summary
from tyxonq.libs.circuits_library.ucc import build_ucc_circuit
from tyxonq.applications.chem.chem_libs.hamiltonians_chem_library.hamiltonian_builders import (
    get_integral_from_hf,
    get_hop_from_integral,
    get_h_from_integral,
    get_hop_hcb_from_integral
)
from tyxonq.libs.hamiltonian_encoding.pauli_io import reverse_qop_idx,canonical_mo_coeff
from tyxonq.applications.chem.chem_libs.quantum_chem_library.ci_state_mapping import get_ci_strings, get_addr as _get_addr_ci
from tyxonq.applications.chem.classical_chem_cloud.config import create_classical_client, CloudClassicalConfig

from tyxonq.applications.chem.chem_libs.quantum_chem_library.ci_state_mapping import statevector_to_civector,civector_to_statevector

from pyscf import gto,scf
from pyscf.mp import MP2 as _mp2
from pyscf.cc import ccsd as _ccsd
from pyscf.mcscf import CASCI
from tyxonq.applications.chem.molecule import _Molecule
from tyxonq.libs.hamiltonian_encoding.pauli_io import get_fermion_phase,rdm_mo2ao
from itertools import product
from tyxonq.numerics import NumericBackend as nb

logger = logging.getLogger(__name__)






class UCC:
    """Minimal UCC 算法入口（device 默认）。

    - 支持将参数化 ansatz（ex_ops/param_ids/init_state）传递给设备 runtime
    - 对外提供 energy/energy_and_grad（device 路径）
    - 后续将扩展 numeric_statevector/numeric_mps
    """

    def __init__(
        self,
        mol,
        *,
        init_method="mp2",
        active_space=None,
        active_orbital_indices=None,
        mo_coeff=None,
        mode="fermion",
        runtime='device',
        numeric_engine=None,
        run_fci=False,
        atom = None,
        basis = "sto-3g",
        unit = "Angstrom",
        charge = 0,
        spin = 0,
        decompose_multicontrol: bool = False,
        trotter: bool = False,
        classical_provider = 'local',
        classical_device = 'auto',
        ex_ops: List[tuple] | None = None,
        param_ids: List[int] | None = None,
        init_state: np.ndarray | None = None,
        **kwargs
    ):
        if atom is not None:
            mm = gto.Mole()
            mm.atom = atom  # accepts string or list spec per PySCF
            mm.unit = str(unit)
            mm.basis = str(basis)
            mm.charge = int(charge)
            mm.spin = int(spin)
            mm.build()
            mol = mm
        

        self.hf = None
        # process mol
        if isinstance(mol, _Molecule):
            self.mol = mol
            self.mol.verbose = 0
            # self.hf =  RHF(mol)
        if isinstance(mol, gto.Mole):
            # Create a local HF container regardless, to keep compatibility with tests accessing self.hf
            self.mol = mol
            self.mol.verbose = 0
            # self.hf = RHF(mol)
        elif isinstance(mol, RHF_TYPE):
            self.hf = mol.copy()
            self.mol = self.hf.mol
        else:
            raise TypeError(
                f"Unknown input type {type(mol)}. If you're performing open shell calculations, "
                "please use ROHF instead."
            )

        if runtime is None:
            runtime='device'

        if active_space is None:
            active_space = (self.mol.nelectron, int(self.mol.nao))

        self.mode = mode
        self.spin = self.mol.spin
        if mode == "hcb":
            assert self.spin == 0
        self.n_qubits = 2 * active_space[1]
        if mode == "hcb":
            self.n_qubits //= 2

        if not self.hf:
            if self.spin == 0:
                self.hf = RHF(self.mol)
            else:
                self.hf = ROHF(self.mol)




        # process activate space
        self.active_space = active_space
        self.n_elec = active_space[0]
        self.active = active_space[1]
        self.inactive_occ = (self.mol.nelectron - active_space[0]) // 2
        assert (self.mol.nelectron - active_space[0]) % 2 == 0
        self.inactive_vir = self.mol.nao - active_space[1] - self.inactive_occ
        if active_orbital_indices is None:
            active_orbital_indices = list(range(self.inactive_occ, self.mol.nao - self.inactive_vir))
        if len(active_orbital_indices) != active_space[1]:
            raise ValueError("sort_mo should have the same length as the number of active orbitals.")
        frozen_idx = [i for i in range(self.mol.nao) if i not in active_orbital_indices]
        self.active_orbital_indices = active_orbital_indices



        # process backend
        self._check_numeric_engine(numeric_engine)

        if numeric_engine is None:
            # no need to be too precise
            if self.n_qubits <= 16:
                numeric_engine = "civector"
            else:
                numeric_engine = "civector-large"
        self.numeric_engine = numeric_engine


        # classical quantum chemistry
        # hf
        #hf 的积分 和kernel没有计算
        if not self.hf.e_tot:
            if str(classical_provider) != "local":
                # Offload HF convergence and MO-basis integrals to cloud
                client = create_classical_client(str(classical_provider), str(classical_device), CloudClassicalConfig())
                mdat = {
                    "atom": getattr(self.mol, "atom", None),
                    "basis": getattr(self.mol, "basis", "sto-3g"),
                    "charge": int(getattr(self.mol, "charge", 0)),
                    "spin": int(getattr(self.mol, "spin", 0)),
                }
                task = {
                    "method": "hf_integrals",
                    "molecule_data": mdat,
                    "active_space": active_space,
                    "active_orbital_indices": active_orbital_indices,
                }
                _res = client.submit_classical_calculation(task)
                # Load results
                int1e = np.asarray(_res["int1e"])  # type: ignore[index]
                int2e = np.asarray(_res["int2e"])  # type: ignore[index]
                e_core = float(_res["e_core"])  # type: ignore[index]
                if _res.get("mo_coeff") is not None:
                    try:
                        self.hf.mo_coeff = canonical_mo_coeff(np.asarray(_res["mo_coeff"]))  # type: ignore[index]
                    except Exception:
                        pass

                self.hf.e_tot = float(_res.get("e_hf", 0.0))  # type: ignore[attr-defined]
                # In cloud mode, skip local HF run
                self.e_hf = float(_res.get("e_hf", 0.0))
            else:
                # Local HF
                # avoid pyscf 1.7+ return fload and assert error of nuc.real
                self.hf.kernel(dump_chk=False)
                self.e_hf = self.hf.e_tot
                self.hf.mo_coeff = canonical_mo_coeff(self.hf.mo_coeff)
        else:
            #外部传入计算好的hf
            self.e_hf = self.hf.e_tot
            self.hf._eri = self.mol.intor("int2e", aosym="s8")
            self.hf.mo_coeff = canonical_mo_coeff(self.hf.mo_coeff)

        
        # mp2
        if init_method == 'mp2' and not isinstance(self.hf, ROHF_TYPE):
            mp2 = _mp2(self.hf)
            if frozen_idx:
                mp2.frozen = frozen_idx
            e_corr_mp2, mp2_t2 = mp2.kernel()
            self.e_mp2 = self.e_hf + e_corr_mp2
        else:
            self.e_mp2 = None
            mp2_t2 = None
        # ccsd
        if init_method == 'ccsd' and not isinstance(self.hf, ROHF_TYPE):
            ccsd = _ccsd.CCSD(self.hf)
            if frozen_idx:
                ccsd.frozen = frozen_idx
            e_corr_ccsd, ccsd_t1, ccsd_t2 = ccsd.kernel()
            self.e_ccsd = self.e_hf + e_corr_ccsd
        else:
            self.e_ccsd = None
            ccsd_t1 = ccsd_t2 = None
        

        # MP2 and CCSD rely on canonical HF orbitals but FCI doesn't
        # so set custom mo_coeff after MP2 and CCSD and before FCI
        if mo_coeff is not None:
            # use user defined coefficient
            self.hf.mo_coeff = canonical_mo_coeff(mo_coeff)


        # Hamiltonian related（先构建以获得 self.e_core 再计算 e_fci）
        self.hamiltonian_lib = {}
        self.int1e = self.int2e = None
        # e_core includes nuclear repulsion energy
        self.hamiltonian, self.e_core, _ = self._get_hamiltonian_and_core(self.numeric_engine)

        # fci（口径：电子能 + e_core，与 kernel/energy 返回一致）
        if run_fci:
            fci = CASCI(self.hf, self.active_space[1], self.active_space[0])
            mo = fci.sort_mo(active_orbital_indices, base=0)
            res = fci.kernel(mo)

            #energey here include the e_core
            self.e_fci = res[0]
            self.civector_fci = res[2].ravel()
        else:
            self.e_fci = None
            self.civector_fci = None

        self.e_nuc = float(self.mol.energy_nuc())



        # initial guess
        self.t1 = np.zeros([self.no, self.nv])
        self.t2 = np.zeros([self.no, self.no, self.nv, self.nv])
        self.init_method = init_method
        if init_method is None or init_method in ["zeros", "zero"]:
            pass
        elif init_method.lower() == "ccsd":
            self.t1, self.t2 = ccsd_t1, ccsd_t2
        elif init_method.lower() == "fe":
            self.t2 = compute_fe_t2(self.no, self.nv, self.int1e, self.int2e)
        elif init_method.lower() == "mp2":
            self.t2 = mp2_t2
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")


        # circuit related
        self.init_state = None
        self.ex_ops = list(ex_ops) if ex_ops is not None else None
        self.param_ids = list(param_ids) if param_ids is not None else None
        self.init_state = init_state



        # optimization related
        self.scipy_minimize_options = None
        self.grad: str = "param-shift"
        # optimization result
        self.opt_res = None
        # for manually set
        self._params = None
        if runtime is None:
            self.runtime = 'device'
        else:
            self.runtime = runtime

        

        self.decompose_multicontrol = bool(decompose_multicontrol)
        self.trotter = bool(trotter)

        #set cache to speedup the shots=0
        hq = self.h_qubit_op


        # init guess (zeros by default if ex_ops present)
        # if self.ex_ops is not None:
        #     self.init_guess = np.zeros(self.n_params, dtype=np.float64)
        # else:
        #     self.init_guess = np.zeros(0, dtype=np.float64)



    def _get_hamiltonian_and_core(self, numeric_engine):
        self._check_numeric_engine(numeric_engine)
        if numeric_engine is None:
            numeric_engine = self.numeric_engine
            hamiltonian = self.hamiltonian
            e_core = self.e_core
        else:
            if numeric_engine.startswith("civector") or numeric_engine == "pyscf":
                htype = "fcifunc"
            else:
                assert numeric_engine in ["statevector"]
                htype = "sparse"
            hamiltonian = self.hamiltonian_lib.get(htype)
            if hamiltonian is None:
                if self.int1e is None:
                    self.int1e, self.int2e, e_core = get_integral_from_hf(self.hf, self.active_space, self.active_orbital_indices)
                else:
                    e_core = self.e_core
                hamiltonian = get_h_from_integral(self.int1e, self.int2e, self.n_elec_s, self.mode, htype)
                self.hamiltonian_lib[htype] = hamiltonian
            else:
                e_core = self.e_core
        return hamiltonian, e_core, numeric_engine

    def _check_numeric_engine(self, numeric_engine):
        supported_engine = [None, "tensornetwork", "statevector", "civector", "civector-large", "pyscf"]
        if not numeric_engine in supported_engine:
            raise ValueError(f"Numeric engine '{numeric_engine}' not supported")


    # ---------------- TCC-aligned helpers ----------------
    def _sanity_check(self):
        if self.ex_ops is None:
            return
        if self.param_ids is None:
            return
        if len(self.param_ids) != len(self.ex_ops):
            raise ValueError(
                f"Excitation operator size {len(self.ex_ops)} and parameter size {len(self.param_ids)} do not match"
            )

    def apply_excitation(self, state: np.ndarray, ex_op: Tuple, *, numeric_engine: str | None = None) -> np.ndarray:
        self._sanity_check()
        eng = numeric_engine or getattr(self, "numeric_engine", None) or "statevector"
        return _apply_excitation_numeric(state, self.n_qubits, self.n_elec_s, ex_op, self.mode, numeric_engine=eng)

    def energy(self, params: np.ndarray | None = None, **opts) -> float:
        runtime = str(opts.pop("runtime", self.runtime))
        numeric_engine = opts.pop("numeric_engine", None) or getattr(self, "numeric_engine", None)
        
        self._sanity_check()
        params = self._check_params_argument(params)

        if params is self.params and self.opt_res is not None:
            return self.opt_res.e
        # Always fetch the correct Hamiltonian for the requested numeric_engine
        self.hamiltonian, self.e_core, self.numeric_engine = self._get_hamiltonian_and_core(numeric_engine)

        if runtime == "numeric" or int(opts.get('shots',2048)) == 0 :
            logger.info('go to analytic numeric engine path when shots == 0')
            #shots = 0 go to analytic numeric path 
            # Ensure engine-specific Hamiltonian type is used (sparse for statevector, callable for CI)
            rt = UCCNumericRuntime(
                self.n_qubits,
                self.n_elec_s,
                self.h_qubit_op,
                ex_ops=self.ex_ops,
                param_ids=self.param_ids,
                init_state=self.init_state,
                mode=self.mode,
                numeric_engine=numeric_engine,
                hamiltonian=self.hamiltonian
                
            )
            base = np.asarray(params if params is not None else (self.init_guess if getattr(self, "init_guess", None) is not None else np.zeros(self.n_params)), dtype=np.float64)
            return float(rt.energy(base)+self.e_core)

        elif runtime == "device":
            rt = UCCDeviceRuntime(
            self.n_qubits,
            self.n_elec_s,
            self.h_qubit_op,
            mode=self.mode,
            ex_ops=self.ex_ops,
            param_ids=self.param_ids,
            init_state=self.init_state,
            decompose_multicontrol=self.decompose_multicontrol,
            trotter=self.trotter
        )
            e = rt.energy(params, **opts)
            return float(e)
        raise ValueError(f"unknown runtime: {runtime}")

    def energy_and_grad(self, params: np.ndarray | None = None, **opts):
        runtime = opts.get('runtime',self.runtime)
        numeric_engine = opts.pop("numeric_engine", None) or getattr(self, "numeric_engine", None)
        
        self._sanity_check()
        params = self._check_params_argument(params)

        if params is self.params and self.opt_res is not None:
            return self.opt_res.e
        # Always fetch the correct Hamiltonian for the requested numeric_engine
        self.hamiltonian, self.e_core, self.numeric_engine = self._get_hamiltonian_and_core(numeric_engine)


        if runtime == "numeric" or int(opts.get('shots',2048)) == 0 :
            logger.info('go to analytic numeric engine path when shots == 0')
            # Always fetch the correct Hamiltonian for the requested numeric_engine
            #shots = 0 go to analytic numeric path 
            rt = UCCNumericRuntime(
                self.n_qubits,
                self.n_elec_s,
                self.h_qubit_op,
                ex_ops=self.ex_ops,
                param_ids=self.param_ids,
                init_state=self.init_state,
                mode=self.mode,
                numeric_engine=numeric_engine,
                hamiltonian=self.hamiltonian
                
            )
            e0, g = rt.energy_and_grad(params)
            return float(e0+self.e_core), g
        elif runtime == "device":
            rt = UCCDeviceRuntime(
            self.n_qubits,
            self.n_elec_s,
            self.h_qubit_op,
            mode=self.mode,
            ex_ops=self.ex_ops,
            param_ids=self.param_ids,
            init_state=self.init_state,
            decompose_multicontrol=self.decompose_multicontrol,
            trotter=self.trotter
        )
            e, g = rt.energy_and_grad(params, **opts)
            # return float(e+self.e_core), g
            return float(e), g
        
        raise ValueError(f"unknown runtime: {runtime}")

    @property
    def e_ucc(self) -> float:
        """Convenience: return current UCC energy (with core) using stored params.

        Tests expect `e_ucc` to be available for both closed- and open-shell.
        """
        # Use analytic simulator path by default for reproducibility
        return float(self.energy(getattr(self, "_params", None), shots=0, provider="simulator", device="statevector"))

    def kernel(self, **opts) -> float:
        """Optimize parameters via L-BFGS-B.

        Any options in **opts will be forwarded to energy_and_grad (e.g.,
        shots/provider/device for device runtime, numeric_engine for numeric runtime).
        """
        # if self.n_params == 0:
        #     return float(self.e_core)

        if self.init_guess is None:
            self.init_guess = np.zeros(self.n_params)

        # runtime options (shots/provider/device/numeric_engine/etc.) from caller
        # 默认 shots 统一为 1024（避免 0 导致无法投递到真机）；调用方可覆盖
        runtime_opts = dict(opts)
        if "shots" not in runtime_opts:
            if str(runtime_opts.get("runtime", self.runtime)) == "device":
                if  str(runtime_opts.get("provider", 'simulator')) in ["simulator",'local']:
                    # 默认使用解析路径，避免采样噪声影响优化与 RDM
                    shots = 0
                else:
                    # 真机默认使用2048shots
                    shots = 2048
            else:
                shots = 0
            runtime_opts["shots"] = shots
        else:
            shots = runtime_opts['shots']
        self._opt_runtime_opts = dict(runtime_opts)

        func = self.get_opt_function()

        # def _obj(x: np.ndarray):
        #     e, g = self.energy_and_grad(x, **runtime_opts)
        #     return e, np.asarray(g, dtype=np.float64)

        # Merge caller options with a sensible default and allow tests to pass n_tries (ignored here but tolerated)
        # Increase maxiter for analytic/numeric paths to hit tighter tolerances
        if self.scipy_minimize_options is None:
            default_maxiter = 200 if (shots == 0 or str(opts.get("runtime", self.runtime)) == "numeric") else 100
            default_opts = {"ftol": 1e-9, "gtol": 1e-6, "maxiter": default_maxiter}
            minimize_options = dict(default_opts)
        else:
            minimize_options = self.scipy_minimize_options
            
        # res = minimize(lambda v: _obj(v), x0=x0, jac=True, method="L-BFGS-B", options=minimize_options)
        if self.grad == "free":
            res = minimize(lambda x: func(x), x0=self.init_guess, jac=False, method="COBYLA", options=minimize_options)
        else:
            res = minimize(lambda x: func(x)[0], x0=self.init_guess, jac=lambda x: func(x)[1], method="L-BFGS-B", options=minimize_options)
        
        if not res.success:
            logger.warning("Optimization failed. See `.opt_res` for details.")
        # Store optimizer result for downstream algorithms (KUPCCGSD expects .opt_res)
        res['init_guess'] = self.init_guess
        res["e"] = float(res.fun)
        

        self.opt_res = res
        self._params = np.asarray(getattr(res, "x", self.init_guess), dtype=np.float64)
        return res.e

    # ---------- Optimization helper (parity with HEA) ----------
    def get_opt_function(self, *, with_time: bool = False):
        import time as _time

        runtime_opts = getattr(self, "_opt_runtime_opts", {})

        def f_only(x: np.ndarray) -> float:
            return float(self.energy(x, **runtime_opts))

        def f_with_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            e, g = self.energy_and_grad(x, **runtime_opts)
            return float(e), np.asarray(g, dtype=np.float64)

        t1 = _time.time()
        func = f_only if self.grad == "free" else f_with_grad
        # 轻量“预热”，便于可能的 lazy 初始化
        # _ = func(self.init_guess.copy())
        t2 = _time.time()
        if with_time:
            return func, (t2 - t1)
        return func

    # ---- Convenience builders from integrals / molecule (fermion/qubit modes) ----
    @classmethod
    def from_integral(
        cls,
        int1e: np.ndarray,
        int2e: np.ndarray,
        n_elec: Union[int, Tuple[int, int]],
        e_core: float = 0,
        ovlp: np.ndarray = None,
        classical_provider: str = 'local',
        classical_device: str = 'auto',
        **kwargs
    ) -> "UCC":
        if isinstance(n_elec, tuple):
            spin = abs(n_elec[0] - n_elec[1])
            n_elec = n_elec[0] + n_elec[1]
        else:
            assert n_elec % 2 == 0
            spin = 0


        m = _Molecule(int1e, int2e, n_elec, spin, e_core, ovlp)

        return cls(m, classical_provider=classical_provider, classical_device=classical_device, **kwargs)

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

    def get_ex_ops(self, t1: np.ndarray = None, t2: np.ndarray = None):
        """Virtual method to be implemented"""
        raise NotImplementedError

    # ---- Excitation generators (TCC-style, default UCCSD semantics) ----
    def get_ex1_ops(self, t1: np.ndarray | None = None) -> Tuple[List[Tuple], List[int], List[float]]:
        """
        Get one-body excitation operators.

        Parameters
        ----------
        t1: np.ndarray, optional
            Initial one-body amplitudes based on e.g. CCSD

        Returns
        -------
        ex_op: List[Tuple]
            The excitation operators. Each operator is represented by a tuple of ints.
        param_ids: List[int]
            The mapping from excitations to parameters.
        init_guess: List[float]
            The initial guess for the parameters.

        See Also
        --------
        get_ex2_ops: Get two-body excitation operators.
        get_ex_ops: Get one-body and two-body excitation operators for UCCSD ansatz.
        """
        # single excitations
        no, nv = self.no, self.nv
        if t1 is None:
            t1 = self.t1

        if t1.shape == (self.no, self.nv):
            t1 = spatial2spin(t1)
        else:
            assert t1.shape == (2 * self.no, 2 * self.nv)

        ex1_ops = []
        # unique parameters. -1 is a place holder
        ex1_param_ids = [-1]
        ex1_init_guess = []
        for i in range(no):
            for a in range(nv):
                # alpha to alpha
                ex_op_a = (2 * no + nv + a, no + nv + i)
                # beta to beta
                ex_op_b = (no + a, i)
                ex1_ops.extend([ex_op_a, ex_op_b])
                ex1_param_ids.extend([ex1_param_ids[-1] + 1] * 2)
                ex1_init_guess.append(t1[i, a])
        ex1_param_ids = ex1_param_ids[1:]

        # deal with qubit symmetry
        if self.mode == "qubit":
            ex1_ops, ex1_param_ids, ex1_init_guess = self._qubit_phase(ex1_ops, ex1_param_ids, ex1_init_guess)

        return ex1_ops, ex1_param_ids, ex1_init_guess

    def get_ex2_ops(self, t2: np.ndarray | None = None) -> Tuple[List[Tuple], List[int], List[float]]:
        """
        Get two-body excitation operators.

        Parameters
        ----------
        t2: np.ndarray, optional
            Initial two-body amplitudes based on e.g. MP2

        Returns
        -------
        ex_op: List[Tuple]
            The excitation operators. Each operator is represented by a tuple of ints.
        param_ids: List[int]
            The mapping from excitations to parameters.
        init_guess: List[float]
            The initial guess for the parameters.

        See Also
        --------
        get_ex1_ops: Get one-body excitation operators.
        get_ex_ops: Get one-body and two-body excitation operators for UCCSD ansatz.
        """

        # t2 in oovv 1212 format
        no, nv = self.no, self.nv
        if t2 is None:
            t2 = self.t2

        if t2.shape == (self.no, self.no, self.nv, self.nv):
            t2 = spatial2spin(t2)
        else:
            assert t2.shape == (2 * self.no, 2 * self.no, 2 * self.nv, 2 * self.nv)

        def alpha_o(_i):
            return no + nv + _i

        def alpha_v(_i):
            return 2 * no + nv + _i

        def beta_o(_i):
            return _i

        def beta_v(_i):
            return no + _i

        # double excitations
        ex_ops = []
        ex2_param_ids = [-1]
        ex2_init_guess = []
        # 2 alphas or 2 betas
        for i in range(no):
            for j in range(i):
                for a in range(nv):
                    for b in range(a):
                        # i correspond to a and j correspond to b, as in PySCF convention
                        # otherwise the t2 amplitude has incorrect phase
                        # 2 alphas
                        ex_op_aa = (alpha_v(b), alpha_v(a), alpha_o(i), alpha_o(j))
                        # 2 betas
                        ex_op_bb = (beta_v(b), beta_v(a), beta_o(i), beta_o(j))
                        ex_ops.extend([ex_op_aa, ex_op_bb])
                        ex2_param_ids.extend([ex2_param_ids[-1] + 1] * 2)
                        ex2_init_guess.append(t2[2 * i, 2 * j, 2 * a, 2 * b])
        assert len(ex_ops) == 2 * (no * (no - 1) / 2) * (nv * (nv - 1) / 2)
        # 1 alpha + 1 beta
        for i in range(no):
            for j in range(i + 1):
                for a in range(nv):
                    for b in range(a + 1):
                        # i correspond to a and j correspond to b, as in PySCF convention
                        # otherwise the t2 amplitude has incorrect phase
                        if i == j and a == b:
                            # paired
                            ex_op_ab = (beta_v(a), alpha_v(a), alpha_o(i), beta_o(i))
                            ex_ops.append(ex_op_ab)
                            ex2_param_ids.append(ex2_param_ids[-1] + 1)
                            ex2_init_guess.append(t2[2 * i, 2 * i + 1, 2 * a, 2 * a + 1])
                            continue
                        # simple reflection
                        ex_op_ab1 = (beta_v(b), alpha_v(a), alpha_o(i), beta_o(j))
                        ex_op_ab2 = (alpha_v(b), beta_v(a), beta_o(i), alpha_o(j))
                        ex_ops.extend([ex_op_ab1, ex_op_ab2])
                        ex2_param_ids.extend([ex2_param_ids[-1] + 1] * 2)
                        ex2_init_guess.append(t2[2 * i, 2 * j + 1, 2 * a, 2 * b + 1])
                        if (i != j) and (a != b):
                            # exchange alpha and beta
                            ex_op_ab3 = (beta_v(a), alpha_v(b), alpha_o(i), beta_o(j))
                            ex_op_ab4 = (alpha_v(a), beta_v(b), beta_o(i), alpha_o(j))
                            ex_ops.extend([ex_op_ab3, ex_op_ab4])
                            ex2_param_ids.extend([ex2_param_ids[-1] + 1] * 2)
                            ex2_init_guess.append(t2[2 * i, 2 * j + 1, 2 * b, 2 * a + 1])
        ex2_param_ids = ex2_param_ids[1:]

        # deal with qubit symmetry
        if self.mode == "qubit":
            ex_ops, ex2_param_ids, ex2_init_guess = self._qubit_phase(ex_ops, ex2_param_ids, ex2_init_guess)

        return ex_ops, ex2_param_ids, ex2_init_guess

    def _qubit_phase(self, ex_ops, ex_param_ids, ex_init_guess):

        hf_str = np.array(self.get_ci_strings()[:1], dtype=np.uint64)
        iterated_ids = set()
        for i, ex_op in enumerate(ex_ops):
            if ex_param_ids[i] in iterated_ids:
                continue
            phase = get_fermion_phase(ex_op, self.n_qubits, hf_str)[0]
            ex_init_guess[ex_param_ids[i]] *= phase
            iterated_ids.add(ex_param_ids[i])
        return ex_ops, ex_param_ids, ex_init_guess


    # ---- RDM in MO basis (spin-traced) ----
    def make_rdm1(self, params: Sequence[float] | None = None,  statevector= None, basis: str = "AO") -> np.ndarray:
    
        assert self.mode in ["fermion", "qubit"]
        civector = self._statevector_to_civector(statevector).astype(np.float64)

        rdm1_cas = fci.direct_spin1.make_rdm1(civector, self.n_qubits // 2, self.n_elec_s)

        rdm1 = self.embed_rdm_cas(rdm1_cas)

        if basis == "MO":
            return rdm1
        else:
            return rdm_mo2ao(rdm1, self.hf.mo_coeff)

    def make_rdm2(self, params: Sequence[float] | None = None, statevector=None, basis: str = "AO") -> np.ndarray:
    
        assert self.mode in ["fermion", "qubit"]
        civector = self._statevector_to_civector(statevector).astype(np.float64)

        rdm2_cas = fci.direct_spin1.make_rdm12(civector.astype(np.float64), self.n_qubits // 2, self.n_elec_s)[1]

        rdm2 = self.embed_rdm_cas(rdm2_cas)

        if basis == "MO":
            return rdm2
        else:
            return rdm_mo2ao(rdm2, self.hf.mo_coeff)
    
    def embed_rdm_cas(self, rdm_cas):
        """
        Embed CAS RDM into RDM of the whole system
        """
        if self.inactive_occ == 0 and self.inactive_vir == 0:
            # active space approximation not employed
            return rdm_cas
        # slice of indices in rdm corresponding to cas
        slice_cas = slice(self.inactive_occ, self.inactive_occ + len(rdm_cas))
        if rdm_cas.ndim == 2:
            rdm1_cas = rdm_cas
            rdm1 = np.zeros((self.mol.nao, self.mol.nao))
            for i in range(self.inactive_occ):
                rdm1[i, i] = 2
            rdm1[slice_cas, slice_cas] = rdm1_cas
            return rdm1
        else:
            rdm2_cas = rdm_cas
            # active space approximation employed
            rdm1 = self.make_rdm1(basis="MO")
            rdm1_cas = rdm1[slice_cas, slice_cas]
            rdm2 = np.zeros((self.mol.nao, self.mol.nao, self.mol.nao, self.mol.nao))
            rdm2[slice_cas, slice_cas, slice_cas, slice_cas] = rdm2_cas
            for i in range(self.inactive_occ):
                for j in range(self.inactive_occ):
                    rdm2[i, i, j, j] += 4
                    rdm2[i, j, j, i] -= 2
                rdm2[i, i, slice_cas, slice_cas] = rdm2[slice_cas, slice_cas, i, i] = 2 * rdm1_cas
                rdm2[i, slice_cas, slice_cas, i] = rdm2[slice_cas, i, i, slice_cas] = -rdm1_cas
            return rdm2
    

    def _statevector_to_civector(self, statevector=None):
        if statevector is None:
            civector = self.civector()
        else:
            if len(statevector) == self.statevector_size:
                ci_strings = self.get_ci_strings()
                civector = statevector[ci_strings]
            else:
                if len(statevector) == self.civector_size:
                    civector = statevector
                else:
                    raise ValueError(f"Incompatible statevector size: {len(statevector)}")
        return nb.array(civector)

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
        return rt._civector(p)

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

    # ---- PySCF solver adapter (minimal, aligned with HEA style) ----
    @classmethod
    def as_pyscf_solver(cls, config_function = None,runtime: str = "numeric", **kwargs):  # pragma: no cover
        class _FCISolver:
            def __init__(self):
                self.instance: UCC | None = None  # type: ignore[name-defined]
                self.config_function = config_function
                self.instance_kwargs = kwargs
                for arg in ["run_ccsd", "run_fci"]:
                    # keep MP2 for initial guess
                    self.instance_kwargs[arg] = False

            def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
                self.instance = cls.from_integral(h1, h2, nelec, **self.instance_kwargs)
                if self.config_function is not None:
                    self.config_function(self.instance)
                e = self.instance.kernel(shots=kwargs.get("shots",0),runtime=kwargs.get('runtime','numeric'))
                return e + ecore, self.instance.params

            def make_rdm1(self,params, norb, nelec):
                civector = self.instance.civector(params)
                return self.instance.make_rdm1(civector)

            def make_rdm12(self,params,norb, nelec):
                civector = self.instance.civector(params)
                rdm1 = self.instance.make_rdm1(civector)
                rdm2 = self.instance.make_rdm2(civector)
                return rdm1, rdm2

            def spin_square(self, ci, norb, nelec):
                return 0.0, 1.0

        return _FCISolver()

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

    # ---- Dimensions and sizes (parity with TCC) ----
    @property
    def n_elec_s(self):
        """The number of electrons for alpha and beta spin"""
        return (self.n_elec + self.spin) // 2, (self.n_elec - self.spin) // 2
    @property
    def no(self) -> int:
        """The number of occupied orbitals."""
        return self.n_elec // 2

    @property
    def nv(self) -> int:
        """The number of virtual (unoccupied orbitals)."""
        return self.active - self.no

    @property
    def h_fermion_op(self) -> FermionOperator:
        """
        Hamiltonian as openfermion.FermionOperator
        """
        if self.mode == "hcb":
            raise ValueError("No FermionOperator available for hard-core boson Hamiltonian")
        return get_hop_from_integral(self.int1e, self.int2e) + self.e_core

    @property
    def h_qubit_op(self) -> QubitOperator:
        """
        Hamiltonian as openfermion.QubitOperator, mapped by
        Jordan-Wigner transformation.
        """
        if self.mode in ["fermion", "qubit"]:
            return reverse_qop_idx(jordan_wigner(self.h_fermion_op), self.n_qubits)
        else:
            assert self.mode == "hcb"
            return get_hop_hcb_from_integral(self.int1e, self.int2e) + self.e_core

    @property
    def statevector_size(self) -> int:
        return 1 << int(self.n_qubits)

    @property
    def civector_size(self) -> int:
        from math import comb
        na, nb = self.n_elec_s
        n_orb = int(self.n_qubits // 2)
        try:
            return int(comb(n_orb, int(na)) * comb(n_orb, int(nb)))
        except Exception:
            return int(self.statevector_size)

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
        if getattr(self, "_params", None) is not None:
            return self._params
        if getattr(self, "opt_res", None) is not None:
            return getattr(self.opt_res, "x", None)
        return None

    @params.setter
    def params(self, v: Sequence[float] | None) -> None:
        self._params = None if v is None else np.asarray(v, dtype=np.float64)



def compute_fe_t2(no, nv, int1e, int2e):
    n_orb = no + nv

    def translate_o(n):
        if n % 2 == 0:
            return n // 2 + n_orb
        else:
            return n // 2

    def translate_v(n):
        if n % 2 == 0:
            return n // 2 + no + n_orb
        else:
            return n // 2 + no
    t2 = np.zeros((2 * no, 2 * no, 2 * nv, 2 * nv))
    for i, j, k, l in product(range(2 * no), range(2 * no), range(2 * nv), range(2 * nv)):
        # spin not conserved
        if i % 2 != k % 2 or j % 2 != l % 2:
            continue
        a = translate_o(i)
        b = translate_o(j)
        s = translate_v(l)
        r = translate_v(k)
        if len(set([a, b, s, r])) != 4:
            continue
        # r^ s^ b a
        rr, ss, bb, aa = [i % n_orb for i in [r, s, b, a]]
        if (r < n_orb and s < n_orb) or (r >= n_orb and s >= n_orb):
            e_inter = int2e[aa, rr, bb, ss] - int2e[aa, ss, bb, rr]
        else:
            e_inter = int2e[aa, rr, bb, ss]
        if np.allclose(e_inter, 0):
            continue
        e_diff = _compute_e_diff(r, s, b, a, int1e, int2e, n_orb, no)
        if np.allclose(e_diff, 0):
            raise RuntimeError("RHF degenerate ground state")
        theta = np.arctan(-2 * e_inter / e_diff) / 2
        t2[i, j, k, l] = theta
    return t2

def _compute_e_diff(r, s, b, a, int1e, int2e, n_orb, no):
    inert_a = list(range(no))
    inert_b = list(range(no))
    old_a = []
    old_b = []
    for i in [b, a]:
        if i < n_orb:
            inert_b.remove(i)
            old_b.append(i)
        else:
            inert_a.remove(i % n_orb)
            old_a.append(i % n_orb)

    new_a = []
    new_b = []
    for i in [r, s]:
        if i < n_orb:
            new_b.append(i)
        else:
            new_a.append(i % n_orb)

    diag1e = np.diag(int1e)
    diagj = np.einsum("iijj->ij", int2e)
    diagk = np.einsum("ijji->ij", int2e)

    e_diff_1e = diag1e[new_a].sum() + diag1e[new_b].sum() - diag1e[old_a].sum() - diag1e[old_b].sum()
    # fmt: off
    e_diff_j = _compute_j_outer(diagj, inert_a, inert_b, new_a, new_b) \
               - _compute_j_outer(diagj, inert_a, inert_b, old_a, old_b)
    e_diff_k = _compute_k_outer(diagk, inert_a, inert_b, new_a, new_b) \
               - _compute_k_outer(diagk, inert_a, inert_b, old_a, old_b)
    # fmt: on
    return e_diff_1e + 1 / 2 * (e_diff_j - e_diff_k)


def _compute_j_outer(diagj, inert_a, inert_b, outer_a, outer_b):
    # fmt: off
    v = diagj[inert_a][:, outer_a].sum() + diagj[outer_a][:, inert_a].sum() + diagj[outer_a][:, outer_a].sum() \
      + diagj[inert_a][:, outer_b].sum() + diagj[outer_a][:, inert_b].sum() + diagj[outer_a][:, outer_b].sum() \
      + diagj[inert_b][:, outer_a].sum() + diagj[outer_b][:, inert_a].sum() + diagj[outer_b][:, outer_a].sum() \
      + diagj[inert_b][:, outer_b].sum() + diagj[outer_b][:, inert_b].sum() + diagj[outer_b][:, outer_b].sum()
    # fmt: on
    return v


def _compute_k_outer(diagk, inert_a, inert_b, outer_a, outer_b):
    # fmt: off
    v = diagk[inert_a][:, outer_a].sum() + diagk[outer_a][:, inert_a].sum() + diagk[outer_a][:, outer_a].sum() \
      + diagk[inert_b][:, outer_b].sum() + diagk[outer_b][:, inert_b].sum() + diagk[outer_b][:, outer_b].sum()
    # fmt: on
    return v