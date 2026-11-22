"""UCC 数值引擎的薄分派；数学实现均由既有 wavefunction/libs 提供。"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from tyxonq.applications.chem.hamiltonian_builders import apply_op
from tyxonq.core.ir.circuit import Circuit
from tyxonq.numerics import NumericBackend as nb

from .ci_state_mapping import (
    civector_to_statevector,
    get_ci_strings,
    statevector_to_civector,
)
from .civector_ops import (
    apply_excitation_civector,
    apply_excitation_civector_nocache,
    energy_and_grad_civector,
    energy_and_grad_civector_nocache,
    get_civector,
    get_civector_nocache,
)
from .pyscf_civector import (
    apply_excitation_pyscf,
    get_civector_pyscf,
    get_energy_and_grad_pyscf,
)
from .statevector_ops import (
    apply_excitation_statevector,
    energy_and_grad_statevector,
    get_statevector,
)


def _engine_name(numeric_engine: str | None) -> str:
    engine = "statevector" if numeric_engine is None else str(numeric_engine)
    if engine not in {"statevector", "civector", "civector-large", "pyscf"}:
        raise ValueError(f"Unsupported numeric_engine: {engine}")
    return engine


def translate_init_state(init_state, n_qubits: int, n_elec_s, mode: str, numeric_engine: str | None):
    """仅适配 CI 系引擎的输入表示；statevector 保留原始初态与 autograd 链。"""
    engine = _engine_name(numeric_engine)
    if engine == "statevector" or init_state is None:
        return init_state
    if isinstance(init_state, Circuit):
        init_state = init_state.state(form="numpy")
    if not isinstance(init_state, np.ndarray):
        init_state = np.asarray(init_state)
    if init_state.size == (1 << int(n_qubits)):
        ci_strings = get_ci_strings(n_qubits, n_elec_s, mode)
        return statevector_to_civector(init_state, ci_strings)
    ci_size = len(get_ci_strings(n_qubits, n_elec_s, mode))
    if init_state.size == ci_size:
        return np.asarray(init_state, dtype=np.float64)
    raise ValueError(
        f"init_state size {init_state.size} incompatible with statevector size "
        f"{1 << int(n_qubits)} and civector size {ci_size}"
    )


def ucc_civector(params, n_qubits, n_elec_s, ex_ops, param_ids, *, mode="fermion",
                 init_state=None, numeric_engine=None):
    engine = _engine_name(numeric_engine)
    if engine == "statevector":
        full = ucc_statevector(
            params, n_qubits, n_elec_s, ex_ops, param_ids,
            mode=mode, init_state=init_state,
        )
        ci_strings = get_ci_strings(n_qubits, n_elec_s, mode)
        return np.asarray(np.real(np.asarray(full)[ci_strings]), dtype=np.float64)
    init = translate_init_state(init_state, n_qubits, n_elec_s, mode, engine)
    kwargs = dict(params=params, n_qubits=n_qubits, n_elec_s=n_elec_s,
                  ex_ops=ex_ops, param_ids=param_ids, mode=mode, init_state=init)
    if engine == "civector":
        return get_civector(**kwargs)
    if engine == "civector-large":
        return get_civector_nocache(**kwargs)
    return get_civector_pyscf(**kwargs)


def ucc_statevector(params, n_qubits, n_elec_s, ex_ops, param_ids, *, mode="fermion",
                    init_state=None):
    return get_statevector(
        params, n_qubits, n_elec_s, ex_ops, param_ids, mode=mode, init_state=init_state
    )


def numeric_energy_and_grad(params, hamiltonian, n_qubits, n_elec_s, ex_ops, param_ids, *,
                        mode="fermion", init_state=None, numeric_engine=None,
                        e_core=0.0):
    engine = _engine_name(numeric_engine)
    init = translate_init_state(init_state, n_qubits, n_elec_s, mode, engine)
    kwargs = dict(params=params, hamiltonian=hamiltonian, n_qubits=n_qubits,
                  n_elec_s=n_elec_s, ex_ops=ex_ops, param_ids=param_ids,
                  mode=mode, init_state=init)
    if engine == "statevector":
        energy, grad = energy_and_grad_statevector(**kwargs)
    elif engine == "civector":
        energy, grad = energy_and_grad_civector(**kwargs)
    elif engine == "civector-large":
        energy, grad = energy_and_grad_civector_nocache(**kwargs)
    else:
        energy, grad = get_energy_and_grad_pyscf(**kwargs)
    return float(energy + float(e_core)), np.asarray(grad, dtype=np.float64)


def numeric_energy(params, hamiltonian, n_qubits, n_elec_s, ex_ops, param_ids, *,
               mode="fermion", init_state=None, numeric_engine=None, e_core=0.0):
    engine = _engine_name(numeric_engine)
    init = translate_init_state(init_state, n_qubits, n_elec_s, mode, engine)
    kwargs = dict(params=params, n_qubits=n_qubits, n_elec_s=n_elec_s,
                  ex_ops=ex_ops, param_ids=param_ids, mode=mode, init_state=init)
    if engine == "statevector":
        ket = get_statevector(**kwargs)
        energy = nb.real(nb.sum(nb.conj(ket) * apply_op(hamiltonian, ket)))
        return float(energy) + float(e_core)
    if engine == "civector":
        ket = get_civector(**kwargs)
    elif engine == "civector-large":
        ket = get_civector_nocache(**kwargs)
    else:
        ket = get_civector_pyscf(**kwargs)
    ket = np.asarray(ket, dtype=np.float64)
    energy = float(np.dot(np.asarray(apply_op(hamiltonian, ket), dtype=np.float64), ket))
    return energy + float(e_core)


def apply_excitation(statevector, f_idx: Sequence[int], n_qubits, n_elec_s, *,
                     mode="fermion", numeric_engine=None):
    """按输入表示返回 CI 或全态结果，底层激发由现有四引擎函数完成。"""
    engine = _engine_name(numeric_engine)
    f_idx = tuple(int(i) for i in f_idx)
    ci_strings = get_ci_strings(n_qubits, n_elec_s, mode)
    full_input = np.asarray(statevector).size == (1 << int(n_qubits))
    if engine == "statevector":
        full = statevector if full_input else civector_to_statevector(
            statevector, n_qubits, ci_strings
        )
        out = apply_excitation_statevector(full, n_qubits, f_idx, mode)
        return out if full_input else statevector_to_civector(out, ci_strings)
    if full_input:
        civector = statevector_to_civector(statevector, ci_strings)
    else:
        civector = statevector
    if engine == "civector":
        out = apply_excitation_civector(civector, n_qubits, n_elec_s, f_idx, mode)
    elif engine == "civector-large":
        out = apply_excitation_civector_nocache(civector, n_qubits, n_elec_s, f_idx, mode)
    else:
        out = apply_excitation_pyscf(civector, n_qubits, n_elec_s, f_idx, mode)
    if full_input:
        return civector_to_statevector(out, n_qubits, ci_strings)
    return out


__all__ = [
    "apply_excitation",
    "translate_init_state",
    "ucc_civector",
    "numeric_energy",
    "numeric_energy_and_grad",
    "ucc_statevector",
]
