"""
VQNHE application
"""

from functools import lru_cache
from itertools import product
import time
from typing import (
    List,
    Any,
    Tuple,
    Callable,
    Optional,
    Dict,
)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..circuit import Circuit
from .. import gates as G
from ..cons import backend


Tensor = Any
Array = Any
Model = Any
dtype = np.complex128
# only guarantee compatible with float64 mode
# support all backends through tyxonq backend interface
# i.e. use the following setup in the code
# tq.set_backend("pytorch")
# tq.set_dtype("complex128")

x = np.array([[0, 1.0], [1.0, 0]], dtype=dtype)
y = np.array([[0, -1j], [1j, 0]], dtype=dtype)
z = np.array([[1, 0], [0, -1]], dtype=dtype)
xx = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=dtype)
yy = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]], dtype=dtype)
zz = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=dtype)
swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=dtype)

pauli = [np.eye(2), x, y, z]


@lru_cache()
def paulistring(term: Tuple[int, ...]) -> Array:
    n = len(term)
    if n == 1:
        return pauli[term[0]]
    ps = np.kron(pauli[term[0]], paulistring(tuple(list(term)[1:])))
    return ps


def construct_matrix(ham: List[List[float]]) -> Array:
    # deprecated
    # only suitable for matrix of small Hilbert dimensions, say <14 qubits
    h = 0.0j
    for term in ham:
        term = list(term)
        for i, t in enumerate(term):
            if i > 0:
                term[i] = int(t)
        h += term[0] * paulistring(tuple(term[1:]))
    return h


# replace with QuOperator tensor_product approach for Hamiltonian construction
# i.e. using ``construct_matrix_v2``


def construct_matrix_tf(ham: List[List[float]], dtype: Any = None) -> Tensor:
    # deprecated
    if dtype is None:
        dtype = backend.dtype
    h = 0.0j
    for term in ham:
        term = list(term)
        for i, t in enumerate(term):
            if i > 0:
                term[i] = int(t)
        h += backend.cast(term[0], dtype=dtype) * backend.convert_to_tensor(
            paulistring(tuple(term[1:])), dtype=dtype
        )
    return h


# Local replacement for legacy generate_local_hamiltonian: tensor product of local terms
def _generate_local_hamiltonian(*hlist: Any) -> Any:
    mats = [backend.numpy(h) for h in hlist]
    if not mats:
        return np.array([[1.0 + 0.0j]])
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def construct_matrix_v2(ham: List[List[float]], dtype: Any = None) -> Tensor:
    # deprecated
    if dtype is None:
        dtype = backend.dtype
    s = len(ham[0]) - 1
    h = backend.zeros([2**s, 2**s], dtype=dtype)
    for term in tqdm(ham, desc="Hamiltonian building"):
        term = list(term)
        for i, t in enumerate(term):
            if i > 0:
                term[i] = int(t)
        local_matrix_list = [backend.convert_to_tensor(pauli[t], dtype=dtype) for t in term[1:]]
        h += backend.cast(term[0], dtype=dtype) * backend.cast(
            _generate_local_hamiltonian(*local_matrix_list), dtype=dtype
        )
    return h


def construct_matrix_v3(ham: List[List[float]], dtype: Any = None) -> Tensor:
    # Dense fallback using pauli helpers
    from ..core.operations.pauli import pauli_string_sum_dense

    H = pauli_string_sum_dense([h[1:] for h in ham], [h[0] for h in ham])
    if dtype is None:
        dtype = backend.dtype
    return backend.convert_to_tensor(H, dtype=dtype)


def vqe_energy(c: Circuit, h: List[List[float]], reuse: bool = True) -> Tensor:
    loss = 0.0
    for term in h:
        ep = []
        for i, t in enumerate(term[1:]):
            if t == 3:
                ep.append((G.z(), [i]))  # type: ignore
            elif t == 1:
                ep.append((G.x(), [i]))  # type: ignore
            elif t == 2:
                ep.append((G.y(), [i]))  # type: ignore
        if ep:
            loss += term[0] * c.expectation(*ep, reuse=reuse)
        else:
            loss += term[0]

    return loss


def vqe_energy_shortcut(c: Circuit, h: Tensor) -> Tensor:
    from ..templates.measurements import operator_expectation

    return operator_expectation(c, h)


class Linear(nn.Module):
    """
    Dense layer but with complex weights, used for building complex RBM
    """

    def __init__(self, units: int, input_dim: int, stddev: float = 0.1) -> None:
        super().__init__()
        self.wr = nn.Parameter(torch.randn(input_dim, units, dtype=torch.float64) * stddev)
        self.wi = nn.Parameter(torch.randn(input_dim, units, dtype=torch.float64) * stddev)
        self.br = nn.Parameter(torch.randn(units, dtype=torch.float64) * stddev)
        self.bi = nn.Parameter(torch.randn(units, dtype=torch.float64) * stddev)

    def forward(self, inputs: Tensor) -> Tensor:
        # Convert to complex
        inputs = torch.as_tensor(inputs, dtype=torch.complex128)
        weight = self.wr.to(torch.complex128) + 1.0j * self.wi.to(torch.complex128)
        bias = self.br.to(torch.complex128) + 1.0j * self.bi.to(torch.complex128)
        return torch.matmul(inputs, weight) + bias


class JointSchedule:
    def __init__(
        self,
        steps: int = 300,
        pre_rate: float = 0.1,
        pre_decay: int = 400,
        post_rate: float = 0.001,
        post_decay: int = 4000,
        dtype: Any = None,
    ) -> None:
        self.steps = steps
        self.pre_rate = pre_rate
        self.pre_decay = pre_decay
        self.post_rate = post_rate
        self.post_decay = post_decay
        self.dtype = dtype

    def __call__(self, step: int) -> float:
        if step < self.steps:
            return self.pre_rate * (0.5) ** (step / self.pre_decay)
        else:
            return self.post_rate * (0.5) ** ((step - self.steps) / self.post_decay)


class VQNHE:
    def __init__(
        self,
        n: int,
        hamiltonian: List[List[float]],
        model_params: Optional[Dict[str, Any]] = None,
        circuit_params: Optional[Dict[str, Any]] = None,
        shortcut: bool = False,
    ) -> None:
        self.n = n
        if not model_params:
            model_params = {}
        self.model = self.create_model(**model_params)
        self.model_params = model_params
        self.cut = model_params.get("cut", 20)
        if not circuit_params:
            circuit_params = {"epochs": 2, "stddev": 0.1}
        self.epochs = circuit_params.get("epochs", 2)
        self.circuit_stddev = circuit_params.get("stddev", 0.1)
        self.channel = circuit_params.get("channel", 2)

        self.circuit_variable = torch.nn.Parameter(
            torch.randn(
                self.epochs, self.n, self.channel,
                dtype=torch.float64,
            ) * self.circuit_stddev
        )
        self.circuit = self.create_circuit(**circuit_params)
        self.base = torch.tensor(list(product(*[[0, 1] for _ in range(n)])), dtype=torch.float64)
        self.hamiltonian = hamiltonian
        self.shortcut = shortcut
        self.history: List[float] = []

    def assign(
        self, c: Optional[List[Tensor]] = None, q: Optional[Tensor] = None
    ) -> None:
        if c is not None:
            # Load state dict for PyTorch model
            self.model.load_state_dict(c)
        if q is not None:
            self.circuit_variable.data = torch.as_tensor(q, dtype=torch.float64)

    def recover(self) -> None:
        try:
            self.assign(self.ccache, self.qcache)
        except AttributeError:
            pass

    def load(self, path: str) -> None:
        w = np.load(path, allow_pickle=True)
        self.assign(w["c"], w["q"])
        self.history = list(w["h"])

    def save(self, path: str) -> None:
        np.savez(path, c=self.ccache, q=self.qcache, h=np.array(self.history))

    def create_model(self, choose: str = "real", **kws: Any) -> Model:
        if choose == "real":
            return self.create_real_model(**kws)
        if choose == "complex":
            return self.create_complex_model(**kws)
        if choose == "real-rbm":
            return self.create_real_rbm_model(**kws)
        if choose == "complex-rbm":
            return self.create_complex_rbm_model(**kws)

    def create_real_model(
        self,
        init_value: float = 0.0,
        max_value: float = 1.0,
        min_value: float = 0,
        depth: int = 2,
        width: int = 2,
        **kws: Any,
    ) -> Model:
        model = tf.keras.Sequential()
        for _ in range(depth):
            model.add(tf.keras.layers.Dense(width * self.n, activation="relu"))

        model.add(tf.keras.layers.Dense(1, activation="tanh"))
        model.add(
            tf.keras.layers.Dense(
                1,
                activation=None,
                use_bias=False,
                kernel_initializer=tf.keras.initializers.Constant(value=init_value),
                kernel_constraint=tf.keras.constraints.MinMaxNorm(
                    min_value=min_value,
                    max_value=max_value,
                ),
            )
        )
        model.build([None, self.n])
        return model

    def create_complex_model(
        self,
        init_value: float = 0.0,
        max_value: float = 1.0,
        min_value: float = 0,
        **kws: Any,
    ) -> Model:
        inputs = tf.keras.layers.Input(shape=[self.n])
        x = tf.keras.layers.Dense(2 * self.n, activation="relu")(inputs)
        x = tf.keras.layers.Dense(4 * self.n, activation="relu")(x)

        lnr = tf.keras.layers.Dense(2 * self.n, activation="relu")(x)
        lnr = tf.keras.layers.Dense(self.n, activation=None)(lnr)
        lnr = tf.math.log(tf.math.cosh(lnr))
        # lnr = lnr + inputs

        phi = tf.keras.layers.Dense(2 * self.n, activation="relu")(x)
        phi = tf.keras.layers.Dense(self.n, activation="relu")(phi)
        # phi = phi + inputs

        lnr = tf.keras.layers.Dense(1, activation=None)(lnr)
        # lnr = tf.keras.layers.Dense(
        #     1,
        #     activation=None,
        #     use_bias=False,
        #     kernel_initializer=tf.keras.initializers.Constant(value=init_value),
        #     kernel_constraint=tf.keras.constraints.MinMaxNorm(
        #         min_value=min_value,
        #         max_value=max_value,
        #     ),
        # )(lnr)
        phi = tf.keras.layers.Dense(1, activation="tanh")(phi)
        lnr = tf.cast(lnr, dtype=dtype)
        phi = tf.cast(phi, dtype=dtype)
        phi = 3.14159j * phi
        outputs = lnr + phi
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def create_real_rbm_model(
        self, stddev: float = 0.1, width: int = 2, **kws: Any
    ) -> Model:
        inputs = tf.keras.layers.Input(shape=[self.n])
        x = tf.keras.layers.Dense(width * self.n, activation=None)(inputs)
        x = tf.math.log(2 * tf.math.cosh(x))
        x = tf.reduce_sum(x, axis=-1)
        x = tf.reshape(x, [-1, 1])
        y = tf.keras.layers.Dense(1, activation=None)(inputs)
        outputs = y + x
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def create_complex_rbm_model(
        self, stddev: float = 0.1, width: int = 2, **kws: Any
    ) -> Model:
        class ComplexRBMModel(nn.Module):
            def __init__(self, n, stddev, width):
                super().__init__()
                self.linear1 = Linear(width * n, n, stddev=stddev)
                self.linear2 = Linear(1, n, stddev=stddev)
            
            def forward(self, inputs):
                x = self.linear1(inputs)
                x = torch.log(2 * torch.cosh(x))
                x = torch.sum(x, dim=-1, keepdim=True)
                y = self.linear2(inputs)
                return y + x
            
            def get_weights(self):
                """Get model weights (PyTorch compatibility)"""
                return self.state_dict()
            
            def set_weights(self, weights):
                """Set model weights (PyTorch compatibility)"""
                self.load_state_dict(weights)
        
        return ComplexRBMModel(self.n, stddev, width)

    def create_circuit(
        self, choose: str = "hea", **kws: Any
    ) -> Callable[[Tensor], Tensor]:
        if choose == "hea":
            return self.create_hea_circuit(**kws)
        if choose == "hea2":
            return self.create_hea2_circuit(**kws)
        if choose == "hn":
            return self.create_hn_circuit(**kws)
        return self.create_functional_circuit(**kws)  # type: ignore

    def create_functional_circuit(self, **kws: Any) -> Callable[[Tensor], Tensor]:
        func = kws.get("func")
        return func  # type: ignore

    def create_hn_circuit(self, **kws: Any) -> Callable[[Tensor], Tensor]:
        def circuit(a: Tensor) -> Tensor:
            c = Circuit(self.n)
            for i in range(self.n):
                c.H(i)  # type: ignore
            return c

        return circuit

    def create_hea_circuit(
        self, epochs: int = 2, filled_qubit: Optional[List[int]] = None, **kws: Any
    ) -> Callable[[Tensor], Tensor]:
        if filled_qubit is None:
            filled_qubit = [0]

        def circuit(a: Tensor) -> Tensor:
            c = Circuit(self.n)
            if filled_qubit:
                for i in filled_qubit:
                    c.X(i)  # type: ignore
            for epoch in range(epochs):
                for i in range(self.n):
                    c.rx(i, theta=a[epoch, i, 0])  # type: ignore
                for i in range(self.n):
                    c.rz(i, theta=a[epoch, i, 1])  # type: ignore
                for i in range(self.n - 1):
                    c.cnot(i, (i + 1))  # type: ignore
            return c

        return circuit

    def create_hea2_circuit(
        self, epochs: int = 2, filled_qubit: Optional[List[int]] = None, **kws: Any
    ) -> Callable[[Tensor], Tensor]:
        if filled_qubit is None:
            filled_qubit = [0]

        def circuit(a: Tensor) -> Tensor:
            c = Circuit(self.n)
            if filled_qubit:
                for i in filled_qubit:
                    c.X(i)  # type: ignore
            for epoch in range(epochs):
                for i in range(self.n):
                    c.rx(i, theta=a[epoch, i, 0])  # type: ignore
                for i in range(self.n):
                    c.rz(i, theta=a[epoch, i, 1])  # type: ignore
                for i in range(self.n):
                    c.rx(i, theta=a[epoch, i, 2])  # type: ignore
                for i in range(self.n - 1):
                    c.exp1(i, (i + 1), theta=a[epoch, i, 3], unitary=G._zz_matrix)  # type: ignore
            return c

        return circuit

    def evaluation(self, cv: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        VQNHE

        :param cv: [description]
        :type cv: Tensor
        :return: [description]
        :rtype: Tuple[Tensor, Tensor, Tensor]
        """
        def forward(cv):
            w = self.circuit(cv).wavefunction()
            f2 = backend.reshape(self.model(self.base), [-1])
            f2 -= backend.mean(f2)
            f2 = (
                torch.as_tensor(
                    torch.clamp(backend.real(f2), min=-self.cut, max=self.cut),
                    dtype=torch.complex128,
                )
                + torch.as_tensor(backend.imag(f2), dtype=torch.complex128) * 1.0j
            )
            f2 = torch.exp(f2)
            w1 = f2 * w
            nm = backend.norm(w1)
            w1 = w1 / nm
            c2 = Circuit(self.n, inputs=w1)
            if not self.shortcut:
                loss = backend.real(vqe_energy(c2, self.hamiltonian))
            else:
                loss = backend.real(vqe_energy_shortcut(c2, self.hamiltonian))
            return loss, nm
        
        # Differentiate loss while carrying normalization as auxiliary
        gav = backend.value_and_grad(forward, has_aux=True)
        (loss, nm), grad = gav(cv)
        return loss, grad, nm

    def plain_evaluation(self, cv: Tensor) -> Tensor:
        """
        VQE

        :param cv: [description]
        :type cv: Tensor
        :return: [description]
        :rtype: Tensor
        """
        def forward(cv):
            c = self.circuit(cv)
            if not self.shortcut:
                loss = backend.real(vqe_energy(c, self.hamiltonian))
            else:
                loss = backend.real(vqe_energy_shortcut(c, self.hamiltonian))
            return loss
        
        loss = forward(cv)
        grad_fn = backend.grad(forward)
        grad = grad_fn(cv)
        return loss, grad

    def training(
        self,
        maxiter: int = 5000,
        optq: Optional[Callable[[int], Any]] = None,
        optc: Optional[Callable[[int], Any]] = None,
        threshold: float = 1e-8,
        debug: int = 100,
        onlyq: int = 0,
        checkpoints: Optional[List[Tuple[int, float]]] = None,
    ) -> Tuple[Array, Array, Array, int, List[float]]:
        loss_prev = None
        loss_opt = 1e8
        j_opt = 0
        nm_opt = 1
        ccount = 0
        self.qcache = self.circuit_variable.detach().numpy()
        self.ccache = self.model.get_weights()

        if not optc:
            optc = 0.001  # type: ignore
        # Simplified optimizer handling for PyTorch backend (wrap torch Adam)
        if isinstance(optc, float):
            lr_c = optc
            optc = backend.optimizer(lambda params: optim.Adam(params, lr=lr_c))
        optcf = lambda _: optc
        if not optq:
            optq = 0.01  # type: ignore
        if isinstance(optq, float):
            lr_q = optq
            optq = backend.optimizer(lambda params: optim.Adam(params, lr=lr_q))
        optqf = lambda _: optq

        nm = backend.ones([])
        times = []
        history = []

        try:
            for j in range(maxiter):
                if j < onlyq:
                    loss, grad = self.plain_evaluation(self.circuit_variable)
                else:
                    loss, grad, nm = self.evaluation(self.circuit_variable)
                if not backend.is_finite(loss):
                    print("failed optimization since nan in %s tries" % j)
                    # in case numerical instability
                    break
                if (loss_prev is not None) and (torch.abs(loss.detach() - loss_prev) < threshold):  # 0.3e-7
                    ccount += 1
                    if ccount >= 100:
                        print("finished in %s round" % j)
                        break
                history.append(backend.numpy(loss.detach()))
                if checkpoints is not None:
                    bk = False
                    for rd, tv in checkpoints:
                        if j > rd and backend.numpy(loss) > tv:
                            print("failed optimization, as checkpoint is not met")
                            bk = True
                            break
                    if bk:
                        break
                if debug > 0:
                    if j % debug == 0:
                        times.append(time.time())
                        print("energy estimation:", backend.numpy(loss.detach()))

                        quantume, _ = self.plain_evaluation(self.circuit_variable)
                        print("quantum part energy: ", backend.numpy(quantume))
                        if len(times) > 1:
                            print("running time: ", (times[-1] - times[-2]) / debug)
                loss_prev = loss.detach()
                # gradient for circuit variables
                gradofc = [grad]
                if loss < loss_opt:
                    loss_opt = loss.detach()
                    nm_opt = nm
                    j_opt = j
                    self.qcache = self.circuit_variable.detach().numpy()
                    self.ccache = self.model.get_weights()
                # Apply gradient
                optq_cur = optqf(j)
                if hasattr(optq_cur, 'update'):
                    # backend optimizer wrapper
                    updated = optq_cur.update(gradofc, [self.circuit_variable])
                    self.circuit_variable = updated[0]
                else:
                    # Treat optq as a learning-rate schedule (callable) or scalar
                    def _resolve_lr(fn_or_val, step):
                        # scalar
                        if isinstance(fn_or_val, (int, float)):
                            return float(fn_or_val)
                        # callable returning a value or another callable (schedule)
                        if callable(fn_or_val):
                            try:
                                val = fn_or_val(step)
                            except TypeError:
                                val = fn_or_val()
                            if callable(val):
                                try:
                                    val = val(step)
                                except TypeError:
                                    val = val()
                            try:
                                return float(val)
                            except Exception:
                                return 1e-2
                        # fallback
                        return 1e-2

                    lr_now = _resolve_lr(optq, j)
                    if not hasattr(self, '_torch_opt_q') or (self._torch_opt_q.param_groups[0]['params'][0] is not self.circuit_variable):
                        self._torch_opt_q = optim.Adam([self.circuit_variable], lr=lr_now)
                    else:
                        for g in self._torch_opt_q.param_groups:
                            g['lr'] = lr_now
                    self._torch_opt_q.zero_grad()
                    # assign gradient
                    self.circuit_variable.grad = grad
                    self._torch_opt_q.step()
                # TODO: model parameter updates when model is fully ported to torch

        except KeyboardInterrupt:
            pass

        quantume, _ = self.plain_evaluation(self.circuit_variable)
        print(
            "vqnhe prediction: ",
            backend.numpy(loss_prev if loss_prev is not None else loss_opt),  # type: ignore
            "quantum part energy: ",
            backend.numpy(quantume),
            "classical model scale: ",
            0.0,
        )
        print("----------END TRAINING------------")
        self.history += history[:j_opt]
        return (
            backend.numpy(loss_opt),  # type: ignore
            np.real(backend.numpy(nm_opt)),  # type: ignore
            backend.numpy(quantume),
            j_opt,
            history[:j_opt],
        )

    def multi_training(
        self,
        maxiter: int = 5000,
        optq: Optional[Callable[[int], Any]] = None,
        optc: Optional[Callable[[int], Any]] = None,
        threshold: float = 1e-8,
        debug: int = 150,
        onlyq: int = 0,
        checkpoints: Optional[List[Tuple[int, float]]] = None,
        tries: int = 10,
        initialization_func: Optional[Callable[[int], Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        # TODO(@refraction-ray): old multiple training implementation, we can use vmap infra for batched training
        rs = []
        try:
            for t in range(tries):
                print("---------- %s training loop ----------" % t)
                if initialization_func is not None:
                    weights = initialization_func(t)
                else:
                    weights = {}
                qweights = weights.get("q", None)
                cweights = weights.get("c", None)
                if cweights is None:
                    cweights = self.create_model(**self.model_params).get_weights()
                self.model.set_weights(cweights)
                if qweights is None:
                    self.circuit_variable = torch.nn.Parameter(
                        torch.randn(
                            self.epochs, self.n, self.channel,
                            dtype=torch.float64,
                        ) * self.circuit_stddev
                    )
                else:
                    self.circuit_variable = qweights
                self.history = []  # refresh the history
                r = self.training(
                    maxiter, optq, optc, threshold, debug, onlyq, checkpoints
                )
                rs.append(
                    {
                        "energy": r[0],
                        "norm": r[1],
                        "quantum_energy": r[2],
                        "iterations": r[-2],
                        "history": r[-1],
                        "model_weights": self.ccache,
                        "circuit_weights": self.qcache,
                    }
                )
        except KeyboardInterrupt:
            pass

        if rs:
            es = [r["energy"] for r in rs]
            ind = np.argmin(es)
            self.assign(rs[ind]["model_weights"], rs[ind]["circuit_weights"])
            self.history = rs[ind]["history"]
        return rs
