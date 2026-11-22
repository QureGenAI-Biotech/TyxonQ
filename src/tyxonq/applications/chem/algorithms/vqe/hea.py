from __future__ import annotations

from typing import List, Tuple, Sequence, Callable, Union
from math import pi

import numpy as np

from .execution import (
    evaluate_grouped_circuits,
    optimize_vqe as _optimize_vqe,
    parameter_shift_gradient,
    prepare_call,
    reject_runtime_option as _reject_runtime_option,
    shifted_parameter_vectors,
)
from tyxonq.libs.circuits_library.blocks import build_hwe_ry_ops
from tyxonq.applications.chem.hamiltonian_builders import (
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
from .ucc_base import UCC
from pyscf import gto,scf
Hamiltonian = List[Tuple[float, List[Tuple[str, int]]]]


def _hea_dense_hamiltonian(model):
    from tyxonq.libs.quantum_library.kernels.pauli import pauli_string_sum_dense

    pauli_ids = {"I": 0, "X": 1, "Y": 2, "Z": 3}
    ls = []
    weights = []
    for coeff, items in model.hamiltonian:
        ls.append([pauli_ids["I"]] * int(model.n_qubits))
        for symbol, qubit in items:
            ls[-1][int(qubit)] = pauli_ids[str(symbol).upper()]
        weights.append(coeff)
    return pauli_string_sum_dense(ls, weights)


def _hea_numeric_value(model, params):
    from tyxonq.libs.quantum_library.dynamics import expectation
    from tyxonq.numerics import NumericBackend as nb

    psi = model.get_circuit(params).state()
    return expectation(psi, _hea_dense_hamiltonian(model), backend=nb)


def _hea_numeric_energy(model, params):
    return float(_hea_numeric_value(model, params))


def _hea_numeric_energy_and_grad(model, params):
    from tyxonq.numerics import NumericBackend as nb

    backend_name = getattr(nb, "backend", "numpy")
    if backend_name not in {"pytorch", "torch"}:
        params = np.asarray(params, dtype=np.float64)
    value_and_grad = nb.value_and_grad(lambda p: _hea_numeric_value(model, p), argnums=0)
    energy, gradient = value_and_grad(params)
    return float(energy), np.asarray(gradient, dtype=np.float64)


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
        - **Execution control**: shots=0 numeric; positive shots circuit measurement
        - **RDM computation**: Reduced density matrix calculation for analysis

    Args:
        molecule (object, optional): PySCF Mole object or RHF instance for chemistry.
        n_qubits (int, optional): Number of qubits (required if molecule not provided).
        layers (int, optional): Number of entangling layers (required if molecule not provided).
        hamiltonian (List, optional): Hamiltonian as list of (coeff, pauli_terms) tuples.
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
        - shots=0 selects direct numeric evaluation; positive shots selects circuit measurement
        - The same parameters work with simulator and explicitly selected hardware providers

    See Also:
        UCCSD: Unitary Coupled Cluster Singles and Doubles algorithm.
        UCC: Base class for unitary coupled cluster methods.
        tyxonq.libs.circuits_library: Circuit building utilities.
    """
    def __init__(self, 
    molecule: object | None = None, 
    #from integral to build hea
    n_qubits: int | None = None, 
    layers: int | None = None, 
    hamiltonian: Hamiltonian | None = None, 
    numeric_engine = 'statevector', 
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
        _reject_runtime_option(kwargs)
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
        # Store UCC object for HOMO-LUMO gap calculation
        self._ucc_object: UCC | None = None
        # Optimization artifacts
        self.grad: str = "param-shift"
        self.scipy_minimize_options: dict | None = None
        self._params: np.ndarray | None = None
        self.opt_res: dict | None = None
        self.numeric_engine = numeric_engine
        

        # If atom is provided, construct PySCF Mole directly (PySCF style)

        # Initialize from molecule if available (cloud/local HF handled inside)
        if molecule is not None or atom is not None:
            inst = HEA.from_molecule(
                mol = molecule,
                active_space=active_space,
                n_layers=(int(layers) if layers is not None else 3),
                mapping=mapping,
                numeric_engine=numeric_engine,
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
            self._ucc_object = inst._ucc_object
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

    def energy(self, params: Sequence[float] | None = None, *, shots=0,
               provider="simulator", device="statevector", postprocessing=None,
               noise=None, numeric_engine=None, **device_opts) -> float:
        """shots=0 走数值解析；shots>0 走分组线路测量。"""
        from tyxonq.libs.hamiltonian_encoding.hamiltonian_grouping import (
            group_hamiltonian_pauli_terms,
        )

        shots, provider, device, pp, noise, _, run_opts = prepare_call(
            shots, provider, device, postprocessing, noise, numeric_engine, device_opts
        )
        params = self._resolve_params(params)
        if shots == 0:
            return _hea_numeric_energy(self, params)
        identity, groups = group_hamiltonian_pauli_terms(self.hamiltonian, self.n_qubits)
        return evaluate_grouped_circuits(
            [lambda: self.get_circuit(params)], groups, identity_const=identity,
            n_qubits=self.n_qubits, shots=shots, provider=provider, device=device,
            postprocessing=pp, noise=noise, device_options=run_opts,
        )[0]

    def energy_and_grad(self, params: Sequence[float] | None = None, *, shots=0,
                        provider="simulator", device="statevector", postprocessing=None,
                        noise=None, numeric_engine=None, **device_opts):
        """数值档用后端梯度；线路档保持 ±π/2 参数移位。"""
        from tyxonq.libs.hamiltonian_encoding.hamiltonian_grouping import (
            group_hamiltonian_pauli_terms,
        )

        shots, provider, device, pp, noise, _, run_opts = prepare_call(
            shots, provider, device, postprocessing, noise, numeric_engine, device_opts
        )
        params = self._resolve_params(params)
        if shots == 0:
            return _hea_numeric_energy_and_grad(self, params)
        base = np.asarray(params, dtype=np.float64)
        vectors = shifted_parameter_vectors(base, [pi / 2])
        builders = [(lambda p=p: self.get_circuit(p)) for p in vectors]
        identity, groups = group_hamiltonian_pauli_terms(self.hamiltonian, self.n_qubits)
        energies = evaluate_grouped_circuits(
            builders, groups, identity_const=identity, n_qubits=self.n_qubits,
            shots=shots, provider=provider, device=device, postprocessing=pp,
            noise=noise, device_options=run_opts,
        )
        return parameter_shift_gradient(
            energies, len(base), shifts=[pi / 2], weights=[0.5]
        )

    def kernel(self, **opts) -> float:
        """优化参数；simulator 缺省 shots=0，显式硬件缺省 shots=2048。"""
        return _optimize_vqe(self, opts, ucc=False)

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
        inst = cls(n_qubits=n_qubits, layers=int(n_layers), hamiltonian=terms, **kwargs)
        inst._qop_cached = qop
        # record chemistry metadata for downstream features (RDM等)
        inst.mapping = str(mapping)
        inst.int1e = np.array(int1e)
        inst.int2e = np.array(int2e)
        inst.n_elec = n_elec
        inst.spin = spin
        inst.e_core = float(e_core)
        inst.h_qubit_op = qop
        return inst

    @classmethod
    def from_molecule(
        cls,
        mol,
        active_space=None,
        n_layers: int = 3,
        mapping: str = "parity",
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
        ucc_object = UCC(
            mol=mol, init_method=init_method, active_space=active_space,
            classical_provider=classical_provider, classical_device=classical_device,
            atom=atom, basis=basis, unit=unit, charge=charge, spin=spin, **kwargs
        )
        inst = cls.from_integral(
            ucc_object.int1e, ucc_object.int2e, ucc_object.n_elec_s, ucc_object.e_core,
            n_layers=n_layers, mapping=mapping, **kwargs
        )
        inst._ucc_object = ucc_object
        return inst

    @classmethod
    def ry(
        cls,
        int1e: np.ndarray,
        int2e: np.ndarray,
        n_elec: Union[int, Tuple[int, int]],
        e_core: float,
        n_layers: int,
        mapping: str = "parity",
        **kwargs
    ) -> "HEA":
        """兼容旧接口的 RY 构造器（等价于 from_integral(..., n_layers, mapping, engine)）。"""
        return cls.from_integral(
            int1e, int2e, n_elec, e_core, n_layers=n_layers, mapping=mapping, **kwargs
        )

    @classmethod
    def from_qiskit_circuit(
        cls,
        h_qubit_op: QubitOperator,
        circuit: object,
        init_guess: Sequence[float],
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
        inst = cls(n_qubits=n_qubits, layers=layers, hamiltonian=terms, **kwargs)
        inst.init_guess = np.asarray(init_guess, dtype=np.float64)
        # 如果无法直接映射（layers==0 或者后续需要更复杂门集），可退回模板方案
        if layers == 0:
            template = real_amplitudes_circuit_template_converter(circuit)
            inst.circuit_template = template
        # 参数个数由模板/外部电路决定，不强制覆盖 n_params；init_guess 已按外部长度设置
        return inst

    # ---------- PySCF solver 适配（可选） ----------
    @classmethod
    def as_pyscf_solver(cls, *, n_layers: int = 1, mapping: str = "parity",
                        config_function: Callable | None = None, device_opts: dict | None = None):
        """返回 PySCF 兼容求解器；执行由 device_opts 中 shots/provider/device 控制。"""
        opts = dict(device_opts or {})
        _reject_runtime_option(opts)

        class _FCISolver:
            def __init__(self):
                self.instance: HEA | None = None
                self.device_opts = dict(opts)

            def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
                self.instance = cls.from_integral(h1, h2, nelec, ecore, n_layers=n_layers, mapping=mapping)
                if config_function is not None:
                    config_function(self.instance)
                run_opts = dict(opts)
                if "shots" in kwargs:
                    run_opts["shots"] = kwargs["shots"]
                e = self.instance.kernel(**run_opts)
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
                # 闭壳单重态的 <S^2> 恒为 0；开壳层情形未实现，避免静默给错值。
                spin = getattr(self.instance, "spin", 0) or 0
                if int(spin) != 0:
                    raise NotImplementedError(
                        "HEA.as_pyscf_solver only supports spin_square for closed-shell "
                        "singlets; use the SQD path (sqd.as_pyscf_solver) for open shells."
                    )
                return 0.0, 1.0

        return _FCISolver()

    # ---------- RDM（数值解析期望） ----------
    @property
    def n_elec_s(self) -> Tuple[int, int] | None:
        if self.n_elec is None or self.spin is None:
            return None
        na = (self.n_elec + self.spin) // 2
        nb = (self.n_elec - self.spin) // 2
        return int(na), int(nb)

    def _expect_qubit_operator(self, qop: QubitOperator, params: Sequence[float]) -> float:
        terms = self._qop_to_term_list(qop, n_qubits=self.n_qubits)
        inst = HEA(
            n_qubits=self.n_qubits, layers=self.layers, hamiltonian=terms,
            numeric_engine=self.numeric_engine,
        )
        inst.circuit_template = self.circuit_template
        return float(inst.energy(params, shots=0))

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
                val = 0.5 * self._expect_qubit_operator(qop, params)
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
                        val = 0.5 * self._expect_qubit_operator(qop, params)
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

    # ---- HOMO-LUMO gap calculation (delegated to UCC) ----
    def get_homo_lumo_gap(self, homo_idx: int | None = None, lumo_idx: int | None = None, 
                          include_ev: bool = False) -> dict:
        """Calculate HOMO-LUMO gap and corresponding orbital energies.

        This method delegates to the internal UCC object for chemistry-related calculations.
        The UCC object is created during from_molecule() construction and contains the
        Hartree-Fock calculation results needed for HOMO-LUMO analysis.

        Parameters
        ----------
        homo_idx : int, optional
            Manual specification of HOMO orbital index (0-based).
            If None, automatically determined from electron count.
        lumo_idx : int, optional
            Manual specification of LUMO orbital index (0-based).
            If None, automatically determined from electron count.
        include_ev : bool, optional
            Whether to include eV conversion in output. Default False.

        Returns
        -------
        dict
            Dictionary containing:
            - 'homo_energy': Energy of HOMO orbital (Hartree)
            - 'lumo_energy': Energy of LUMO orbital (Hartree)
            - 'gap': HOMO-LUMO energy gap (Hartree)
            - 'gap_ev': HOMO-LUMO energy gap (eV) [only if include_ev=True]
            - 'homo_idx': Index of HOMO orbital
            - 'lumo_idx': Index of LUMO orbital
            - 'system_type': 'closed-shell' or 'open-shell'

        Examples
        --------
        >>> from tyxonq.chem import HEA
        >>> from tyxonq.chem.molecule import h2
        >>> hea = HEA(molecule=h2, layers=1)
        >>> gap_info = hea.get_homo_lumo_gap()
        >>> print(f"HOMO-LUMO gap: {gap_info['gap']:.6f} Hartree")
        
        >>> # Include eV conversion
        >>> gap_info = hea.get_homo_lumo_gap(include_ev=True)
        >>> print(f"HOMO-LUMO gap: {gap_info['gap_ev']:.6f} eV")

        Raises
        ------
        RuntimeError
            If HEA was not constructed from molecule (no UCC object available).

        Notes
        -----
        - Only works for HEA constructed via from_molecule() or direct molecule input
        - For HEA built from integrals directly, no HOMO-LUMO gap calculation is possible
        - Uses the same logic as UCC.get_homo_lumo_gap()
        """
        if self._ucc_object is None:
            raise RuntimeError(
                "HOMO-LUMO gap calculation requires HEA to be constructed from molecule. "
                "Use HEA(molecule=mol, ...) or HEA.from_molecule(...) instead of from_integral()."
            )
        return self._ucc_object.get_homo_lumo_gap(homo_idx=homo_idx, lumo_idx=lumo_idx, include_ev=include_ev)

    @property
    def homo_lumo_gap(self) -> float:
        """HOMO-LUMO energy gap in Hartree (property).

        Automatically determines HOMO and LUMO based on molecular system.
        For detailed information including orbital indices and energies,
        use ``get_homo_lumo_gap()`` method.

        Returns
        -------
        float
            HOMO-LUMO gap in Hartree

        Examples
        --------
        >>> from tyxonq.chem import HEA
        >>> from tyxonq.chem.molecule import h2
        >>> hea = HEA(molecule=h2, layers=1)
        >>> gap = hea.homo_lumo_gap
        >>> print(f"Gap: {gap:.6f} Hartree ({gap*27.2114:.4f} eV)")

        See Also
        --------
        get_homo_lumo_gap : Detailed gap calculation with orbital information
        """
        return self.get_homo_lumo_gap()['gap']

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

