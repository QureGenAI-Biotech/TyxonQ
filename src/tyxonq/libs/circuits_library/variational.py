from __future__ import annotations

from typing import List, Tuple, Sequence, Callable, Optional

import numpy as np
import tyxonq as tq
from tyxonq.numerics import NumericBackend as nb

from .utils import evolve_pauli_ops


AnsatzOp = Tuple[np.ndarray, complex, str, List[int]]


def _jit_if_available(fn):
    return nb.jit(fn)


def build_layered_pauli_circuit(
    ansatz_op_list_grouped: List[List[AnsatzOp]],
    n_layers: int,
    init_state: Optional[tq.Circuit | np.ndarray],
    params: Sequence[float],
    param_ids: Optional[Sequence[int]] = None,
    *,
    compile_evolution: bool = False,
):
    if param_ids is None:
        param_ids = list(range(len(ansatz_op_list_grouped)))
    params = nb.reshape(params, [n_layers, max(param_ids) + 1])

    if isinstance(init_state, tq.Circuit):
        c = tq.Circuit.from_qir(init_state.to_qir(), circuit_params=init_state.circuit_param)
        n_qubits = c.circuit_param["nqubits"]
    else:
        # infer number of qubits from ansatz ops
        n_qubits = 0
        for group in ansatz_op_list_grouped:
            for _, _, _, qidx in group:
                n_qubits = max(n_qubits, max(qidx) + 1 if qidx else 0)
        c = tq.Circuit(n_qubits, inputs=init_state)

    for i in range(n_layers):
        for j, ansatz_op_list in enumerate(ansatz_op_list_grouped):
            pid = param_ids[j]
            theta = params[i, pid]
            for ansatz_op, coeff, name, qubit_idx_list in ansatz_op_list:
                ceff = coeff.imag if np.isrealobj(coeff) is False and coeff.real == 0 else coeff
                if not compile_evolution:
                    np.testing.assert_allclose(ansatz_op.conj().T @ ansatz_op, np.eye(len(ansatz_op)))
                    opname = f"exp(-iθ{name})"
                    c.exp1(*qubit_idx_list, unitary=ansatz_op, theta=ceff * theta, name=opname)
                else:
                    pauli_string = tuple(zip(qubit_idx_list, name))
                    ops = evolve_pauli_ops(pauli_string, 2 * ceff * theta)
                    c.ops.extend(ops)
    return c


def build_ansatz_state_fn(
    ansatz_op_list_grouped: List[List[AnsatzOp]],
    n_layers: int,
    init_state: Optional[tq.Circuit | np.ndarray],
    param_ids: Optional[Sequence[int]] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    def _fn(theta: np.ndarray):
        c = build_layered_pauli_circuit(ansatz_op_list_grouped, n_layers, init_state, theta, param_ids)
        # simple numeric state for now via statevector engine
        from tyxonq.devices.simulators.statevector.engine import StatevectorEngine
        eng = StatevectorEngine()
        return eng.state(c)

    return _jit_if_available(_fn)


def get_jacobian_func(ansatz_state_fn: Callable[[np.ndarray], np.ndarray]):
    # Finite-difference full Jacobian (vector-output wrt params)
    def _finite_diff(x: np.ndarray, eps: float = 1e-6):
        x = np.asarray(x, dtype=np.float64)
        y0 = ansatz_state_fn(x)
        jac = np.zeros((y0.size, x.size), dtype=np.complex128)
        for i in range(x.size):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            yp = ansatz_state_fn(xp)
            ym = ansatz_state_fn(xm)
            jac[:, i] = (yp - ym) / (2 * eps)
        return jac
    return _finite_diff


def regularized_inversion(m: np.ndarray, eps: float) -> np.ndarray:
    evals, evecs = np.linalg.eigh(m)
    evals = evals + eps * np.exp(-evals / eps)
    new_evals = 1.0 / evals
    return evecs @ np.diag(new_evals) @ evecs.T


def parameter_time_derivative(
    ansatz_state_fn: Callable[[np.ndarray], np.ndarray],
    jacobian_fn: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    hamiltonian: np.ndarray,
    *,
    eps: float = 1e-5,
    include_phase: bool = False,
) -> np.ndarray:
    params = np.asarray(params, dtype=np.float64)
    jacobian = np.asarray(jacobian_fn(params)).astype(np.complex128)
    lhs = jacobian.conj().T @ jacobian
    psi = np.asarray(ansatz_state_fn(params)).astype(np.complex128)
    hpsi = hamiltonian @ psi
    rhs = jacobian.conj().T @ hpsi
    lhs = lhs.real
    rhs = rhs.imag
    if include_phase:
        ovlp = jacobian.conj().T @ psi
        lhs += (ovlp.reshape(-1, 1) * ovlp.reshape(1, -1)).real
        e = psi.conj() @ hpsi
        rhs -= (e * ovlp).imag
    lhs_inv = regularized_inversion(lhs, eps)
    theta_dot = lhs_inv @ rhs
    np.testing.assert_allclose(lhs @ lhs_inv @ rhs, rhs, atol=2e2 * eps)
    np.testing.assert_allclose(theta_dot.imag, 0)
    return theta_dot.real.astype(np.float64)


def pvqd_loss_function(ansatz_state_fn: Callable[[np.ndarray], np.ndarray]):
    def loss(delta_params, params, hamiltonian: np.ndarray, delta_t: float):
        ket = ansatz_state_fn(params + delta_params)
        bra = ansatz_state_fn(params)
        import scipy.linalg as _la
        evolution = _la.expm(1j * delta_t * hamiltonian)
        return 1 - np.linalg.norm(bra.conj() @ (evolution @ ket)) ** 2

    # reuse scipy minimize wrapper via tyxonq if available
    try:
        return tq.interfaces.scipy.scipy_optimize_interface(loss)
    except Exception:
        return loss


__all__ = [
    "build_layered_pauli_circuit",
    "build_ansatz_state_fn",
    "get_jacobian_func",
    "regularized_inversion",
    "parameter_time_derivative",
    "pvqd_loss_function",
]


class VariationalRuntime:
    """Backend-agnostic variational dynamics runtime.

    Parameters
    ----------
    ansatz_state_fn:
        Callable mapping theta -> statevector (ndarray shape (2^n,)).
    hamiltonian:
        Dense Hamiltonian matrix (2^n x 2^n).
    n_params:
        Number of parameters in theta.
    eps:
        Regularization for normal equation inversion.
    include_phase:
        Whether to include global phase term in the equation.
    initial_state:
        Optional initial statevector to seed state_list; if None, uses ansatz_state_fn(zeros).
    """

    def __init__(
        self,
        ansatz_state_fn: Callable[[np.ndarray], np.ndarray],
        hamiltonian: np.ndarray,
        n_params: int,
        *,
        eps: float = 1e-5,
        include_phase: bool = False,
        initial_state: np.ndarray | None = None,
    ) -> None:
        self.ansatz_state_fn = ansatz_state_fn
        self.jacobian_fn = get_jacobian_func(ansatz_state_fn)
        self.h = np.asarray(hamiltonian)
        self.n_params = int(n_params)
        self.eps = float(eps)
        self.include_phase = bool(include_phase)

        self.params_list: list[np.ndarray] = [np.zeros(self.n_params, dtype=np.float64)]
        self.t_list: list[float] = [0.0]
        if initial_state is None:
            s0 = np.asarray(ansatz_state_fn(self.params_list[0]))
        else:
            s0 = np.asarray(initial_state)
        self.state_list: list[np.ndarray] = [s0]

        self.property_mat_dict: dict[str, np.ndarray] = {}

    @property
    def params(self) -> np.ndarray:
        return self.params_list[-1]

    @property
    def t(self) -> float:
        return self.t_list[-1]

    def add_property_mat(self, key: str, mat: np.ndarray) -> None:
        self.property_mat_dict[key] = np.asarray(mat)

    def properties(self, state: np.ndarray | None = None) -> dict[str, np.ndarray]:
        if state is None:
            state = self.state_list[-1]
        res: dict[str, np.ndarray] = {}
        for k, m in self.property_mat_dict.items():
            res[k] = state.conj().T @ (m @ state)
        return res

    def theta_dot(self, params: np.ndarray | None = None) -> np.ndarray:
        if params is None:
            params = self.params
        return parameter_time_derivative(
            self.ansatz_state_fn,
            self.jacobian_fn,
            np.asarray(params, dtype=np.float64),
            self.h,
            eps=self.eps,
            include_phase=self.include_phase,
        )

    def step_vqd(self, delta_t: float) -> np.ndarray:
        dtheta = self.theta_dot(self.params)
        new_params = self.params + float(delta_t) * dtheta
        self.params_list.append(new_params)
        state = self.ansatz_state_fn(new_params)
        self.state_list.append(np.asarray(state))
        self.t_list.append(self.t + float(delta_t))
        return new_params

    def step_pvqd(self, delta_t: float) -> np.ndarray:
        loss = pvqd_loss_function(self.ansatz_state_fn)
        try:
            opt_res = loss(np.zeros_like(self.params), self.params, self.h, float(delta_t))
            new_params = self.params + opt_res.x
        except Exception:
            # fallback: single VQD step if optimizer not available
            new_params = self.step_vqd(delta_t)
            return new_params
        self.params_list.append(new_params)
        state = self.ansatz_state_fn(new_params)
        self.state_list.append(np.asarray(state))
        self.t_list.append(self.t + float(delta_t))
        return new_params

