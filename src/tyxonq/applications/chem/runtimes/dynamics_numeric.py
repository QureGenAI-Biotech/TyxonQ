from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tyxonq as tq

from tyxonq.libs.hamiltonian_encoding.operator_encoding import (
    qubit_encode_op as _encode_op,
    qubit_encode_basis as _encode_basis,
    get_init_circuit,
    get_init_statevector,
)
from tyxonq.libs.circuits_library.variational import (
    VariationalRuntime,
)


@dataclass
class IvpConfig:
    method: str = "RK45"
    rtol: float = 1e-3
    atol: float = 1e-6


class DynamicsNumericRuntime:
    def __init__(
        self,
        ham_terms,
        basis,
        *,
        boson_encoding: str = "gray",
        init_condition: Optional[Dict] = None,
        n_layers: int = 3,
        eps: float = 1e-5,
        include_phase: bool = False,
        ivp_config: Optional[IvpConfig] = None,
    ):
        if init_condition is None:
            init_condition = {}

        # encode to spin basis and build Hamiltonian matrix
        ham_terms_spin, constant = _encode_op(ham_terms, basis, boson_encoding)
        basis_spin = _encode_basis(basis, boson_encoding)

        # construct dense Hamiltonian with tq Mpo if available, else numeric fallback
        # Here we use tq (same as legacy path) to keep equivalence for tests
        from renormalizer import Model, Mpo

        # reference model in original basis is required by get_init_circuit
        model_ref = Model(basis, ham_terms)

        self.model = Model(basis_spin, ham_terms_spin)
        self.h_mpo = Mpo(self.model)
        _h_dense = self.h_mpo.todense()
        self.h = _h_dense + constant * np.eye(_h_dense.shape[0])

        # initial circuit/state from init_condition mapped from original basis labels
        # prefer statevector path to avoid Circuit(inputs=...) dependency
        try:
            init_state_vec = get_init_statevector(model_ref, self.model, boson_encoding, init_condition)
            self.init_circuit = tq.Circuit(len(self.model.basis))
        except Exception:
            # fallback: get circuit and later fetch state via engine
            self.init_circuit = get_init_circuit(model_ref, self.model, boson_encoding, init_condition)
            from tyxonq.devices.simulators.statevector.engine import StatevectorEngine
            self._eng0 = StatevectorEngine()
            init_state_vec = np.asarray(self._eng0.state(self.init_circuit)).astype(np.complex128)

        # variational ansatz state and jacobian
        # group each ham term as a single evolution op per layer (simple PVQD-style ansatz)
        # ansatz_op_list_grouped will be constructed from self.h_mpo terms if needed; here use MPO dense as source of size
        n_params_per_layer = len(self.model.ham_terms)
        self.n_layers = int(n_layers)
        self.n_params = self.n_layers * n_params_per_layer
        self.eps = float(eps)
        self.include_phase = bool(include_phase)
        self.ivp_config = ivp_config or IvpConfig()

        # simple grouped ops placeholder: evolve each spin-term with parameter
        # We reuse the same order as ham_terms_spin; ansatz_state_fn 返回 ndarray state
        def _ansatz_state(theta: np.ndarray):
            # produce statevector by sequentially applying dense exponentials
            from tyxonq.devices.simulators.statevector.engine import StatevectorEngine
            import scipy.linalg as _la
            eng = StatevectorEngine()
            state = eng.state(self.init_circuit)
            # ensure numpy array for linear algebra
            state = np.asarray(state).astype(np.complex128)
            params = theta.reshape(self.n_layers, n_params_per_layer)
            for i in range(self.n_layers):
                for j, term in enumerate(self.model.ham_terms):
                    mat = Mpo(self.model, term).todense()
                    u = _la.expm(-1j * params[i, j] * mat)
                    state = u @ state
            return state

        # 数值变分运行时（通用）：负责 θ̇、step_vqd/step_pvqd 等
        if 'init_state_vec' not in locals():
            from tyxonq.devices.simulators.statevector.engine import StatevectorEngine
            self._eng0 = StatevectorEngine()
            init_state_vec = np.asarray(self._eng0.state(self.init_circuit)).astype(np.complex128)
        self.var_rt = VariationalRuntime(
            ansatz_state_fn=_ansatz_state,
            hamiltonian=self.h,
            n_params=self.n_params,
            eps=self.eps,
            include_phase=self.include_phase,
            initial_state=init_state_vec,
        )

        self.params_list: List[np.ndarray] = [np.zeros(self.n_params, dtype=np.float64)]
        self.t_list: List[float] = [0.0]
        self.state_list: List[np.ndarray] = [init_state_vec]

        # properties support (optional)
        self.property_mat_dict: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    @property
    def state(self) -> np.ndarray:
        return self.state_list[-1]

    @property
    def params(self) -> np.ndarray:
        return self.params_list[-1]

    @property
    def t(self) -> float:
        return self.t_list[-1]

    def add_property_op(self, key: str, op):
        from renormalizer import Mpo

        mat = Mpo(self.model, op).todense()
        self.property_mat_dict[key] = (mat,)
        # 同步到通用变分运行时
        try:
            self.var_rt.add_property_mat(key, mat)
        except Exception:
            pass

    def properties(self, state: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        if state is None:
            state = self.state_list[-1]
        # ensure numpy array
        state = np.asarray(state).astype(np.complex128)
        res = {}
        for k, (mat,) in self.property_mat_dict.items():
            res[k] = state.conj().T @ (mat @ state)
        return res

    def theta_dot(self, params: Optional[np.ndarray] = None) -> np.ndarray:
        return self.var_rt.theta_dot(self.params if params is None else params)

    def step_vqd(self, delta_t: float) -> np.ndarray:
        new_params = self.var_rt.step_vqd(delta_t)
        self.params_list.append(new_params)
        self.state_list.append(self.var_rt.state_list[-1])
        self.t_list.append(self.var_rt.t)
        return new_params

    def step_pvqd(self, delta_t: float) -> np.ndarray:
        new_params = self.var_rt.step_pvqd(delta_t)
        self.params_list.append(new_params)
        self.state_list.append(self.var_rt.state_list[-1])
        self.t_list.append(self.var_rt.t)
        return new_params


