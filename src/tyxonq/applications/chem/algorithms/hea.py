from __future__ import annotations

from typing import List, Tuple, Sequence, Callable, Union

import numpy as np
from scipy.optimize import minimize

from ..runtimes.hea_device_runtime import HEADeviceRuntime
from ..runtimes.hea_numeric_runtime import HEANumericRuntime
from tyxonq.libs.circuits_library.blocks import build_hwe_ry_ops
from tyxonq.applications.chem.chem_libs.hamiltonians_chem_library.hamiltonian_builders import (
    get_hop_from_integral,
    get_integral_from_hf,
)
from tyxonq.libs.hamiltonian_encoding.fermion_to_qubit import fop_to_qop, parity, binary
from tyxonq.libs.hamiltonian_encoding.pauli_io import reverse_qop_idx
from openfermion import QubitOperator, FermionOperator, hermitian_conjugated
from tyxonq.libs.circuits_library.qiskit_real_amplitudes import (
    real_amplitudes_circuit_template_converter,
)

from tyxonq.applications.chem.classical_chem_cloud.config import create_classical_client, CloudClassicalConfig
from .ucc import UCC
from pyscf import gto,scf
Hamiltonian = List[Tuple[float, List[Tuple[str, int]]]]


class HEA:
    """Hardware-Efficient Ansatz (HEA) implementation for variational quantum algorithms.

    HEA provides a parameterized quantum circuit designed for near-term quantum devices
    with limited connectivity and coherence times. The implementation uses a RY-only
    structure with alternating layers of single-qubit rotations and CNOT entangling chains.

    Circuit Structure:
        - Layer 0: Initial RY rotations RY(θ₀,ᵢ) for each qubit i
        - For each layer l=1...L:
            * CNOT chain: 0→1→2→...→(n-1)
            * RY rotations: RY(θₗ,ᵢ) for each qubit i
        - Barriers inserted between layers for visualization and compilation control

    Key Features:
        - **Parameter count**: (layers + 1) × n_qubits
        - **Hardware-efficient**: Optimized for NISQ device constraints
        - **Gradient support**: Parameter-shift rule for exact gradients
        - **Chemistry integration**: Direct construction from molecular integrals
        - **Runtime flexibility**: Supports both numeric and device execution
        - **RDM computation**: Reduced density matrix calculation for analysis

    Args:
        molecule (object, optional): PySCF Mole object or RHF instance for chemistry.
        n_qubits (int, optional): Number of qubits (required if molecule not provided).
        layers (int, optional): Number of entangling layers (required if molecule not provided).
        hamiltonian (List, optional): Hamiltonian as list of (coeff, pauli_terms) tuples.
        runtime (str, optional): Execution runtime ("device" or "numeric"). Default "device".
        numeric_engine (str, optional): Numeric backend ("statevector", "pytorch", etc.).
        active_space (Tuple[int, int], optional): Active space (n_electrons, n_orbitals).
        mapping (str, optional): Fermion-to-qubit mapping ("parity", "jordan-wigner", "bravyi-kitaev").
        classical_provider (str, optional): Provider for classical calculations. Default "local".
        classical_device (str, optional): Device for classical calculations. Default "auto".
        atom (object, optional): Direct molecular specification for PySCF.
        basis (str, optional): Basis set for quantum chemistry. Default "sto-3g".
        unit (str, optional): Unit for molecular coordinates. Default "Angstrom".
        charge (int, optional): Molecular charge. Default 0.
        spin (int, optional): Molecular spin. Default 0.

    Attributes:
        n_qubits (int): Number of qubits in the circuit.
        layers (int): Number of entangling layers.
        n_params (int): Total number of parameters.
        hamiltonian (List): Hamiltonian terms.
        init_guess (ndarray): Initial parameter guess.
        params (ndarray): Optimized parameters (after kernel()).
        opt_res (dict): Optimization result details.
        runtime (str): Current execution runtime.
        mapping (str): Fermion-to-qubit mapping method.
        grad (str): Gradient computation method ("param-shift" or "free").

    Examples:
        >>> # Basic HEA construction
        >>> hea = HEA(n_qubits=4, layers=2, hamiltonian=[
        ...     (0.5, [('Z', 0)]),
        ...     (0.3, [('X', 0), ('X', 1)])
        ... ])
        >>> energy = hea.energy()  # Evaluate at initial parameters
        >>> optimal_energy = hea.kernel()  # Run optimization
        
        >>> # Chemistry application with H2 molecule
        >>> from pyscf import gto
        >>> mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
        >>> hea = HEA(molecule=mol, layers=1, mapping="parity")
        >>> ground_state_energy = hea.kernel()
        
        >>> # Direct molecular specification
        >>> hea = HEA(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", layers=2)
        >>> hea.kernel(shots=1024)  # Optimize with shot noise
        
        >>> # Custom circuit analysis
        >>> circuit = hea.get_circuit([0.1, 0.2, 0.3, 0.4])  # Specify parameters
        >>> hea.print_circuit()  # Display circuit structure
        >>> hea.print_summary()  # Show optimization results
        
        >>> # Gradient computation
        >>> params = hea.init_guess
        >>> energy, grad = hea.energy_and_grad(params)
        >>> print(f"Energy: {energy}, Gradient norm: {np.linalg.norm(grad)}")
        
        >>> # Reduced density matrices (after optimization)
        >>> rdm1 = hea.make_rdm1()  # 1-RDM for molecular analysis
        >>> rdm2 = hea.make_rdm2()  # 2-RDM for correlation analysis

    Notes:
        - Parameter-shift gradients provide exact derivatives for optimization
        - The parity mapping reduces qubit count by 2 for molecular systems
        - RDM computation requires chemistry metadata from molecular construction
        - Device runtime supports both simulators and quantum hardware
        - Numeric runtime provides exact statevector simulation
        
    See Also:
        UCCSD: Unitary Coupled Cluster Singles and Doubles algorithm.
        UCC: Base class for unitary coupled cluster methods.
        tyxonq.libs.circuits_library: Circuit building utilities.
        tyxonq.applications.chem.runtimes: Execution runtime implementations.
    """
    def __init__(self, 
    molecule: object | None = None, 
    #from integral to build hea
    n_qubits: int | None = None, 
    layers: int | None = None, 
    hamiltonian: Hamiltonian | None = None, 
    runtime: str = "device", 
    numeric_engine: str | None = None, 
    active_space=None, 
    mapping: str = "parity", 
    classical_provider: str = "local", 
    classical_device: str = "auto",
    #build from mole
    atom: object | None = None, 
    basis: str = "sto-3g", 
    unit: str = "Angstrom", 
    charge: int = 0, 
    spin: int = 0,
    **kwargs):
        # Runtime selection
        self.runtime = runtime
        # 可选：外部参数化电路模板（例如来自 Qiskit RealAmplitudes 的转换）
        # 形如 [("ry", q, ("p", idx)), ("cx", c, t), ...]
        self.circuit_template = None
        # Chemistry metadata placeholders
        self.mapping: str | None = None
        self.int1e: np.ndarray | None = None
        self.int2e: np.ndarray | None = None
        self.n_elec: int | None = None
        self.spin: int | None = None
        self.e_core: float | None = None
        # Optimization artifacts
        self.grad: str = "param-shift"
        self.scipy_minimize_options: dict | None = None
        self._params: np.ndarray | None = None
        self.opt_res: dict | None = None
        self.numeric_engine = kwargs.get('numeric_engine','statevector')
        

        # If atom is provided, construct PySCF Mole directly (PySCF style)

        # Initialize from molecule if available (cloud/local HF handled inside)
        if molecule is not None or atom is not None:
            inst = HEA.from_molecule(
                mol = molecule,
                active_space=active_space,
                n_layers=(int(layers) if layers is not None else 3),
                mapping=mapping,
                runtime=runtime,
                classical_provider=classical_provider,
                classical_device=classical_device,
                atom=atom,
                basis=basis,
                unit=unit,
                charge=charge,
                spin=spin,
                **kwargs
            )
            self.n_qubits = int(inst.n_qubits)
            self.layers = int(inst.layers)
            self.hamiltonian = list(inst.hamiltonian)
            self.mapping = inst.mapping
            self.int1e = inst.int1e
            self.int2e = inst.int2e
            self.n_elec = inst.n_elec
            self.spin = inst.spin
            self.e_core = inst.e_core
        else:
            # Otherwise require explicit (n, layers, hamiltonian)
            if n_qubits is None or layers is None or hamiltonian is None:
                raise TypeError("HEA requires either 'molecule'/'atom' or explicit 'n_qubits', 'layers', and 'hamiltonian'")
            self.n_qubits = n_qubits
            self.layers = int(layers)
            self.hamiltonian = list(hamiltonian)

        # RY ansatz: (layers + 1) * n parameters
        self.n_params = (self.layers + 1) * self.n_qubits
        # Use a deterministic non-trivial initial guess to avoid zero-gradient plateaus
        rng = np.random.default_rng(7)
        self.init_guess = rng.random(self.n_params, dtype=np.float64)

    def get_circuit(self, params: Sequence[float] | None = None):
        """构建 HEA 的门级电路（IR Circuit）。

        参数
        ----
        params: 序列
            长度为 (layers + 1) * n。若为空则使用 init_guess。

        返回
        ----
        Circuit
            包含初始 RY 层、每层的 CNOT 链与 RY，以及层间 barrier 指令。
        """
        if params is None:
            params = self.init_guess
        # 优先：若存在外部模板（如 RealAmplitudes 转换得到），实例化为我们的 IR Circuit
        if self.circuit_template is not None:
            from tyxonq.libs.circuits_library.qiskit_real_amplitudes import build_circuit_from_template
            return build_circuit_from_template(self.circuit_template, np.asarray(params, dtype=np.float64), n_qubits=self.n_qubits)
        # 默认：RY-only ansatz
        return build_hwe_ry_ops(self.n_qubits, self.layers, params)

    def energy(self, params: Sequence[float] | None = None, **device_opts) -> float:
        """基于计数的能量评估（设备路径）。

        内部对哈密顿量进行按基分组的测量流程：对每个基组应用相应的基变换（X→H，Y→S^†H），
        然后做 Z 基测量并从计数中估计 <H>。
        """
        if self.runtime == "numeric" or device_opts.get('shots',2048) == 0:
            rt = HEANumericRuntime(self.n_qubits, self.layers, self.hamiltonian, numeric_engine=(self.numeric_engine or "statevector"))
            p = self.init_guess if params is None else params
            return rt.energy(p, self.get_circuit)
        if self.runtime == "device":
            rt = HEADeviceRuntime(self.n_qubits, self.layers, self.hamiltonian, n_elec_s=self.n_elec_s, mapping=self.mapping, circuit_template=self.circuit_template)
            p = self.init_guess if params is None else params
            qop = getattr(self, "_qop_cached", None)
            return HEADeviceRuntime(self.n_qubits, self.layers, self.hamiltonian, n_elec_s=self.n_elec_s, mapping=self.mapping, circuit_template=self.circuit_template, qop=qop).energy(p, **device_opts)

        raise NotImplementedError(f"unsupported runtime: {self.runtime}")

    def energy_and_grad(self, params: Sequence[float] | None = None, **device_opts):
        """参数移位法梯度（设备路径）。

        对每个可移位参数 θ_k 使用标准移位 s=π/2 计算：
            g_k = 0.5 * (E(θ_k+s) - E(θ_k-s))
        与 energy 一样采用计数估计。
        """
        if self.runtime == "numeric" or device_opts.get('shots',2048) == 0:
            rt = HEANumericRuntime(self.n_qubits, self.layers, self.hamiltonian, numeric_engine=(self.numeric_engine or "statevector"))
            p = self.init_guess if params is None else params
            return rt.energy_and_grad(p, self.get_circuit)
        if self.runtime == "device":
            rt = HEADeviceRuntime(self.n_qubits, self.layers, self.hamiltonian, circuit_template=self.circuit_template)
            p = self.init_guess if params is None else params
            qop = getattr(self, "_qop_cached", None)
            return HEADeviceRuntime(self.n_qubits, self.layers, self.hamiltonian, circuit_template=self.circuit_template, qop=qop).energy_and_grad(p, **device_opts)

        raise NotImplementedError(f"unsupported runtime: {self.runtime}")

    # ---------- Optimization (SciPy) ----------
    def get_opt_function(self, *, with_time: bool = False) -> Union[Callable, Tuple[Callable, float]]:
        """返回用于 SciPy 的目标函数封装。

        当 self.grad == "free" 时，仅返回能量函数；否则返回 (能量, 梯度)。
        """
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
        _ = func(self.init_guess.copy())
        t2 = _time.time()
        if with_time:
            return func, (t2 - t1)
        return func

    def kernel(self, **opts) -> float:
        """运行变分优化，返回最优能量并保存 `opt_res` 与 `params`。"""
        # 持久化运行选项（shots/provider/device 等）到优化闭包中
        runtime_opts = dict(opts)
        self.runtime = runtime_opts.get("runtime", self.runtime)
        self.provider = runtime_opts.get("provider",'local')
        self.device = runtime_opts.get("device", 'statevector')
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
            shots = runtime_opts["shots"]
        self.shots = shots
        self._opt_runtime_opts = dict(runtime_opts)
        func = self.get_opt_function()

        default_maxiter = 200 if (shots == 0 or str(opts.get("runtime", self.runtime)) == "numeric") else 100
        default_opts = {"maxiter": default_maxiter}
        if isinstance(self.scipy_minimize_options, dict):
            run_opts = {**default_opts, **self.scipy_minimize_options}
        else:
            run_opts = default_opts
        if self.grad == "free":
            res = minimize(lambda x: func(x), x0=self.init_guess, jac=False, method="COBYLA", options=run_opts)
        else:
            res = minimize(lambda x: func(x)[0], x0=self.init_guess, jac=lambda x: func(x)[1], method="L-BFGS-B", options=run_opts)
        self.opt_res = {
            "success": bool(getattr(res, "success", True)),
            "x": np.asarray(getattr(res, "x", self.init_guess), dtype=np.float64),
            "fun": float(getattr(res, "fun", self.energy(self.init_guess))),
            "message": str(getattr(res, "message", "")),
            "nit": int(getattr(res, "nit", 0)),
            'init_guess': np.asarray(getattr(res, "x", self.init_guess), dtype=np.float64),
        }
        self.params = np.asarray(self.opt_res["x"], dtype=np.float64).copy()
        return float(self.opt_res["fun"])  # type: ignore[index]

    # ---------- Builders from chemistry inputs ----------
    @staticmethod
    def _qop_to_term_list(qop: QubitOperator, n_qubits: int) -> Hamiltonian:
        terms: Hamiltonian = []
        # identity term
        const = qop.terms.get((), 0.0)
        try:
            cval = float(np.real(const)) if hasattr(const, "real") else float(const)
        except Exception:
            cval = float(np.real_if_close(const))
        if cval != 0.0:
            terms.append((cval, []))
        for term, coeff in qop.terms.items():
            if term == ():
                continue
            ops: List[Tuple[str, int]] = []
            for (idx, sym) in term:
                ops.append((sym.upper(), int(idx)))
            try:
                cval = float(np.real(coeff)) if hasattr(coeff, "real") else float(coeff)
            except Exception:
                cval = float(np.real_if_close(coeff))
            terms.append((cval, ops))
        return terms

    # --- Consistent expectation via HEA.energy (TCC-style parity) ---
    def _energy_with_qop(self, qop: QubitOperator, params: Sequence[float]) -> float:
        terms = self._qop_to_term_list(qop, n_qubits=self.n_qubits)
        inst = HEA(
            n_qubits=self.n_qubits,
            layers=self.layers,
            hamiltonian=terms,
            runtime=self.runtime,
            numeric_engine=self.numeric_engine,
        )
        # Reuse the same parameter vector; avoid optimizer artifacts
        p = self._resolve_params(params)
        # For exact expectation on simulator, prefer shots=0 fast path (routes to numeric inside HEA.energy)
        return float(inst.energy(p, shots=0, provider="simulator", device="statevector"))

    @classmethod
    def from_integral(
        cls,
        int1e: np.ndarray,
        int2e: np.ndarray,
        n_elec: Union[int, Tuple[int, int]],
        e_core: float,
        n_layers: int = 1,
        mapping: str = "parity",
        runtime: str = "device",
        **kwargs
    ) -> "HEA":
        """从分子积分构建 HEA。

        流程：
        1) 由 (int1e, int2e) 构造费米子算符 H_f；
        2) 按映射（parity/JW/BK）将 H_f → QubitOperator；
        3) 转为轻量哈密顿量列表 [(coeff, [(P, q), ...]) ...]，加上 e_core；
        4) 以 n_qubits = n_sorb or n_sorb-2（parity 两比特节省）实例化 HEA。
        """
        n_sorb = 2 * len(int1e)
        if isinstance(n_elec, tuple) or isinstance(n_elec, list):
            n_elec_s = n_elec
            n_elec = sum(n_elec)
            spin = abs(n_elec_s[0] - n_elec_s[1])
        else:
            if n_elec % 2 != 0:
                raise ValueError("Odd total electrons: pass (na, nb) tuple instead")
            n_elec_s = (n_elec // 2, n_elec // 2)
            spin = 0
        # TCC: hop 已经包含 e_core 常数项（见 tencirchem/static/hea.py::from_integral）
        fop = get_hop_from_integral(int1e, int2e) + float(e_core)
        if mapping == "jordan-wigner":
            from openfermion.transforms import jordan_wigner as _jw
            qop = reverse_qop_idx(_jw(fop), n_sorb)
        elif mapping == "bravyi-kitaev":
            from openfermion.transforms import bravyi_kitaev as _bk
            qop = reverse_qop_idx(_bk(fop), n_sorb)
        else:
            qop = fop_to_qop(fop, mapping, n_sorb, n_elec_s)
        terms = cls._qop_to_term_list(qop, n_qubits=(n_sorb - 2 if mapping == "parity" else n_sorb))
        n_qubits = (n_sorb - 2) if mapping == "parity" else n_sorb
        # 预映射缓存：记录 QubitOperator，供 runtime shots==0 快径直接消费
        inst = cls(n_qubits=n_qubits, layers=int(n_layers), hamiltonian=terms, runtime=runtime)
        inst._qop_cached = qop
        # record chemistry metadata for downstream features (RDM等)
        inst.mapping = str(mapping)
        inst.int1e = np.array(int1e)
        inst.int2e = np.array(int2e)
        inst.n_elec = n_elec
        inst.spin = spin
        inst.e_core = float(e_core)
        inst.h_qubit_op = qop
        inst.runtime = runtime
        return inst

    @classmethod
    def from_molecule(
        cls,
        mol,
        active_space=None,
        n_layers: int = 3,
        mapping: str = "parity",
        runtime: str = "device",
        classical_provider: str = "local",
        classical_device: str = "auto",
        atom = None,
        basis = "sto-3g",
        unit = "Angstrom",
        charge = 0,
        spin = 0,
        **kwargs
    ) -> "HEA":
        """从 PySCF 分子对象构建 HEA。

        - 自动运行 RHF 得到积分 (int1e, int2e) 与 e_core；
        - 根据分子总电子数与自旋计算 (n_alpha, n_beta)；
        - 复用 from_integral 流程完成映射与实例化。
        """
        # if atom is not None:
        #     mm = gto.Mole()
        #     mm.atom = atom  # accepts string or list spec per PySCF
        #     mm.unit = str(unit)
        #     mm.basis = str(basis)
        #     mm.charge = int(charge)
        #     mm.spin = int(spin)
        #     mm.build()
        #     mol = mm
        
        if isinstance(mol, gto.Mole):
            if mol.spin == 0:
                init_method = "mp2"
            else:
                init_method = "zeros"
        else:
            init_method = "zeros"

        # Expect a PySCF Mole. If you want to pass XYZ coordinates, use HEA(..., atom=..., basis=..., unit=..., charge=..., spin=...)
        # will inherit the local and cloud engine
        ucc_object = UCC(mol=mol, 
        init_method=init_method, 
        active_space=active_space, 
        runtime=runtime, 
        classical_provider=classical_provider, 
        classical_device=classical_device,
        atom=atom,
        basis=basis,
        unit=unit,
        charge=charge,
        spin=spin,
        **kwargs)

        return cls.from_integral(ucc_object.int1e, ucc_object.int2e, ucc_object.n_elec_s, ucc_object.e_core, n_layers=n_layers, mapping=mapping,runtime=runtime, **kwargs)
        # else:
        #     inst = cls.from_integral(ucc_object.int1e, ucc_object.int2e, ucc_object.n_elec_s, ucc_object.e_core, n_layers=n_layers, mapping=mapping, runtime=runtime)
        #     return inst

    @classmethod
    def ry(
        cls,
        int1e: np.ndarray,
        int2e: np.ndarray,
        n_elec: Union[int, Tuple[int, int]],
        e_core: float,
        n_layers: int,
        mapping: str = "parity",
        runtime: str = "device",
        **kwargs
    ) -> "HEA":
        """兼容旧接口的 RY 构造器（等价于 from_integral(..., n_layers, mapping, engine)）。"""
        return cls.from_integral(int1e, int2e, n_elec, e_core, n_layers=n_layers, mapping=mapping, runtime=runtime)

    @classmethod
    def from_qiskit_circuit(
        cls,
        h_qubit_op: QubitOperator,
        circuit: object,
        init_guess: Sequence[float],
        runtime: str = "device",
        **kwargs
    ) -> "HEA":
        """从 QubitOperator 与外部参数化电路构建 HEA（一次性薄封装）。

        约定：针对 Qiskit RealAmplitudes（或兼容的 QuantumCircuit）进行转换，
        将其一次性变为本框架可消费的 `circuit_template`，后续完全走本地逻辑。
        """
        # 1) 归一化 QubitOperator（取实部系数）
        qop_real = QubitOperator()
        for k, v in h_qubit_op.terms.items():
            vv = v.real if hasattr(v, "real") else float(v)
            qop_real.terms[k] = vv

        # 2) 推断比特数
        n_qubits = None
        if hasattr(circuit, "num_qubits"):
            try:
                n_qubits = int(getattr(circuit, "num_qubits"))
            except Exception:
                n_qubits = None
        if n_qubits is None:
            raise TypeError("circuit must be a parameterized QuantumCircuit with 'num_qubits'")

        # 3) 生成哈密顿量项列表并实例化 HEA
        # 若电路与我们的 RY ansatz 等价（如 RealAmplitudes），优先直接映射 reps→layers，避免额外模板路径
        layers = int(getattr(circuit, "reps", 1)) if hasattr(circuit, "reps") else 0
        terms = cls._qop_to_term_list(qop_real, n_qubits=n_qubits)
        inst = cls(n_qubits=n_qubits, layers=layers, hamiltonian=terms, runtime=runtime)
        inst.init_guess = np.asarray(init_guess, dtype=np.float64)
        # 如果无法直接映射（layers==0 或者后续需要更复杂门集），可退回模板方案
        if layers == 0:
            template = real_amplitudes_circuit_template_converter(circuit)
            inst.circuit_template = template
        # 参数个数由模板/外部电路决定，不强制覆盖 n_params；init_guess 已按外部长度设置
        return inst

    # ---------- PySCF solver 适配（可选） ----------
    @classmethod
    def as_pyscf_solver(cls, *, n_layers: int = 1, mapping: str = "parity", runtime: str = "device", config_function: Callable | None = None):
        """返回一个最小 PySCF FCI 求解器兼容对象，内部以 HEA 优化。

        仅实现 kernel/make_rdm1/make_rdm2 所需的最小接口，便于与 CASSCF 对接。
        """

        class _FCISolver:
            def __init__(self):
                self.instance: HEA | None = None

            def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
                self.instance = cls.from_integral(h1, h2, nelec, ecore, n_layers=n_layers, mapping=mapping, runtime=runtime)
                if config_function is not None:
                    config_function(self.instance)
                e = self.instance.kernel()
                return float(e), self.instance.params

            def make_rdm1(self, params, norb, nelec):
                assert self.instance is not None
                return self.instance.make_rdm1(params)

            def make_rdm12(self, params, norb, nelec):
                assert self.instance is not None
                rdm1 = self.instance.make_rdm1(params)
                rdm2 = self.instance.make_rdm2(params)
                return rdm1, rdm2

            def spin_square(self, params, norb, nelec):
                return 0.0, 1.0

        return _FCISolver()

    # ---------- RDM（基于 counts 的期望估计） ----------
    @property
    def n_elec_s(self) -> Tuple[int, int] | None:
        if self.n_elec is None or self.spin is None:
            return None
        na = (self.n_elec + self.spin) // 2
        nb = (self.n_elec - self.spin) // 2
        return int(na), int(nb)

    def _expect_qubit_operator(self, qop: QubitOperator, params: Sequence[float]) -> float:
        terms = self._qop_to_term_list(qop, n_qubits=self.n_qubits)
        rt = HEADeviceRuntime(self.n_qubits, self.layers, terms)
        # Use analytic expectation on simulator to avoid sampling noise in RDM
        return float(rt.energy(params, shots=0, provider="simulator", device="statevector"))

    def make_rdm1(self, params: Sequence[float] | None = None) -> np.ndarray:
        """计算自旋约化的一体 RDM（spin-traced 1RDM）。需要在 from_integral/from_molecule 构建后使用。"""
        params = self._resolve_params(params)
        if self.mapping is None or self.n_elec_s is None:
            raise ValueError("RDM 需要在 from_integral/from_molecule 构建并携带 mapping 与电子数信息")
        mapping = str(self.mapping)
        if mapping == "parity":
            n_sorb = self.n_qubits + 2
        else:
            n_sorb = self.n_qubits
        n_orb = n_sorb // 2
        rdm1 = np.zeros((n_orb, n_orb), dtype=np.float64)
        for i in range(n_orb):
            for j in range(i + 1):
                if int(self.spin or 0) == 0:
                    fop = 2 * FermionOperator(f"{i}^ {j}")
                else:
                    fop = FermionOperator(f"{i}^ {j}") + FermionOperator(f"{i+n_orb}^ {j+n_orb}")
                fop = fop + hermitian_conjugated(fop)
                qop = fop_to_qop(fop, mapping, n_sorb, self.n_elec_s)
                # val = 0.5 * self._energy_with_qop(qop, params)
                terms = self._qop_to_term_list(qop, n_qubits=self.n_qubits)
                inst = HEA(
                    n_qubits=self.n_qubits,
                    layers=self.layers,
                    hamiltonian=terms,
                    runtime=self.runtime
                )
                # Reuse the same parameter vector; avoid optimizer artifacts
                p = self._resolve_params(params)
                # For exact expectation on simulator, prefer shots=0 fast path (routes to numeric inside HEA.energy)
                val = 0.5 *float(inst.energy(p, shots=self.shots, provider=self.provider, device=self.device))
                rdm1[i, j] = rdm1[j, i] = float(val)
        return rdm1

    def make_rdm2(self, params: Sequence[float] | None = None) -> np.ndarray:
        """计算自旋约化的二体 RDM（spin-traced 2RDM）。需要在 from_integral/from_molecule 构建后使用。"""
        params = self._resolve_params(params)
        if self.mapping is None or self.n_elec_s is None:
            raise ValueError("RDM 需要在 from_integral/from_molecule 构建并携带 mapping 与电子数信息")
        mapping = str(self.mapping)
        if mapping == "parity":
            n_sorb = self.n_qubits + 2
        else:
            n_sorb = self.n_qubits
        n_orb = n_sorb // 2
        rdm2 = np.zeros((n_orb, n_orb, n_orb, n_orb), dtype=np.float64)
        calculated: set[Tuple[int, int, int, int]] = set()
        for p in range(n_orb):
            for q in range(n_orb):
                for r in range(n_orb):
                    for s in range(n_orb):
                        if (p, q, r, s) in calculated:
                            continue
                        fop_aaaa = FermionOperator(f"{p}^ {q}^ {r} {s}")
                        fop_abba = FermionOperator(f"{p}^ {q+n_orb}^ {r+n_orb} {s}")
                        if int(self.spin or 0) == 0:
                            fop = 2 * (fop_aaaa + fop_abba)
                        else:
                            fop_bbbb = FermionOperator(f"{p+n_orb}^ {q+n_orb}^ {r+n_orb} {s+n_orb}")
                            fop_baab = FermionOperator(f"{p+n_orb}^ {q}^ {r} {s+n_orb}")
                            fop = fop_aaaa + fop_abba + fop_bbbb + fop_baab
                        fop = fop + hermitian_conjugated(fop)
                        qop = fop_to_qop(fop, mapping, n_sorb, self.n_elec_s)
                        # val = 0.5 * self._energy_with_qop(qop, params)
                        terms = self._qop_to_term_list(qop, n_qubits=self.n_qubits)
                        inst = HEA(
                            n_qubits=self.n_qubits,
                            layers=self.layers,
                            hamiltonian=terms,
                            runtime=self.runtime
                        )
                        # Reuse the same parameter vector; avoid optimizer artifacts
                        theta = self._resolve_params(params)
                        # For exact expectation on simulator, prefer shots=0 fast path (routes to numeric inside HEA.energy)
                        val = 0.5 * float(inst.energy(theta, shots=self.shots, provider=self.provider, device=self.device))
                        idxs = [(p, q, r, s), (s, r, q, p), (q, p, s, r), (r, s, p, q)]
                        for idx in idxs:
                            rdm2[idx] = float(val)
                            calculated.add(idx)
        # 转置到 PySCF 约定：rdm2[p,q,r,s] = <p^+ r^+ s q>
        rdm2 = rdm2.transpose(0, 3, 1, 2)
        return rdm2

    # ---------- 打印辅助 ----------
    def print_circuit(self):
        from tyxonq.libs.circuits_library.analysis import get_circuit_summary
        c = self.get_circuit(self.init_guess)
        summary = get_circuit_summary(c)
        try:
            # 若是 DataFrame-like
            print(summary.to_string(index=False))  # type: ignore[attr-defined]
        except Exception:
            print(summary)

    def print_summary(self):
        print("############################### Circuit ###############################")
        self.print_circuit()
        print("######################### Optimization Result #########################")
        if self.opt_res is None:
            print("Optimization not run yet")
        else:
            print(self.opt_res)

    # ---------- properties ----------
    # @property
    # def grad(self) -> str:
    #     return self._grad

    # @grad.setter
    # def grad(self, v: str) -> None:
    #     if v not in ("param-shift", "free"):
    #         raise ValueError(f"Invalid gradient method {v}")
    #     self._grad = v

    @property
    def params(self) -> np.ndarray | None:
        return self._params

    @params.setter
    def params(self, p: Sequence[float]) -> None:
        self._params = np.asarray(p, dtype=np.float64)

    # small helper to choose params for evaluation (optimized if available)
    def _resolve_params(self, params: Sequence[float] | None) -> np.ndarray:
        if params is not None:
            return np.asarray(params, dtype=np.float64)
        if self.params is not None:
            return np.asarray(self.params, dtype=np.float64)
        return np.asarray(self.init_guess, dtype=np.float64)


# Re-exports for legacy imports convenience
__all__ = [
    "HEA",
    "parity",
    "binary",
]

