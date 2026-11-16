"""Shared, stateless execution utilities for HEA and UCC algorithms.

Algorithm-specific energy and gradient formulas live in ``hea.py`` and
``ucc_base.py``. This module only handles execution options, Pauli measurement,
parameter-shift bookkeeping, and SciPy optimization.
"""

from __future__ import annotations

from functools import lru_cache
import logging
from typing import Callable, Sequence
import warnings

import numpy as np

from tyxonq.core.ir.circuit import Circuit
from tyxonq.postprocessing import apply_postprocessing

_SIMULATORS = {"simulator", "local", "default"}
_DEVICE_ONLY_OPTIONS = {
    "noise", "readout_correction", "wait_async_result", "postprocessing",
    "use_noise", "device_dialect",
}


def reject_runtime_option(options: dict) -> None:
    if "runtime" in options:
        raise TypeError(
            "runtime= was removed; use shots=0 for direct numeric evaluation or "
            "shots>0 with provider/device for circuit measurement"
        )


def resolve_execution(*, shots, provider="simulator", device="statevector",
                      options: dict | None = None) -> tuple[int, str, str, dict]:
    """Validate options and keep zero-shot hardware configurations local."""
    opts = dict(options or {})
    reject_runtime_option(opts)
    provider_name = "simulator" if provider is None else str(provider)
    device_name = "statevector" if device is None else str(device)
    if shots is None:
        shots = 0 if provider_name in _SIMULATORS else 2048
    if isinstance(shots, (bool, np.bool_)) or not isinstance(shots, (int, np.integer)):
        raise TypeError(f"shots must be an integer, got {type(shots).__name__}")
    shots = int(shots)
    if shots < 0:
        raise ValueError(f"shots must be >= 0, got {shots}")
    if shots == 0:
        if provider_name not in _SIMULATORS:
            warnings.warn(
                f"shots=0 selects the VQE algorithm-level numeric_engine, not "
                f"Circuit.run(shots=0); provider={provider_name!r} and "
                f"device={device_name!r} are ignored.",
                UserWarning,
                stacklevel=2,
            )
            provider_name = "simulator"
            device_name = "statevector"
        ambiguous = sorted(
            key for key in opts if key in _DEVICE_ONLY_OPTIONS and opts[key] is not None
        )
        if ambiguous:
            raise ValueError(
                f"Circuit-only options {ambiguous} are ambiguous with shots=0 numeric "
                "evaluation; use shots>0 or remove these options"
            )
    return shots, provider_name, device_name, opts


@lru_cache(maxsize=256)
def measurement_ops(n_qubits: int, bases: tuple[str, ...]) -> tuple[tuple, ...]:
    """Rotate X/Y/Z/I bases to Z measurements using the IR little-endian convention."""
    if len(bases) != int(n_qubits):
        raise ValueError("measurement basis length does not match n_qubits")
    ops: list[tuple] = []
    for q, pauli in enumerate(bases):
        if pauli == "X":
            ops.append(("h", q))
        elif pauli == "Y":
            ops.append(("sdg", q))
            ops.append(("h", q))
        elif pauli not in {"Z", "I"}:
            raise ValueError(f"Unsupported measurement basis {pauli!r}")
    for q in range(int(n_qubits)):
        ops.append(("measure_z", q))
    return tuple(ops)


def _execute_pauli_circuits(circuits: Sequence[Circuit], items_by_circuit: Sequence[list], *,
                            shots: int, provider: str, device: str,
                            postprocessing: dict | None, noise: dict | None,
                            device_options: dict) -> list[float]:
    if shots <= 0:
        raise ValueError("circuit measurement requires shots>0; shots=0 is numeric evaluation")
    run_options = dict(device_options)
    run_options.setdefault("wait_async_result", provider not in _SIMULATORS)
    payloads = []
    for circuit in circuits:
        result = circuit.run(
            provider=provider, device=device, shots=shots, noise=noise, **run_options
        )
        payloads.extend(result if isinstance(result, (list, tuple)) else [result])
    if len(payloads) != len(circuits):
        raise RuntimeError(
            f"Circuit execution returned {len(payloads)} payloads for {len(circuits)} circuits"
        )

    energies = []
    for index, payload in enumerate(payloads):
        pp = dict(postprocessing or {})
        pp.update({
            "method": "expval_pauli_sum",
            "identity_const": 0.0,
            "items": items_by_circuit[index],
        })
        post = apply_postprocessing(payload, pp)
        result = post.get("result") if isinstance(post, dict) else None
        if not isinstance(result, dict) or "energy" not in result:
            raise RuntimeError(f"Missing Pauli energy in postprocessing result at index {index}")
        energy = float(result["energy"])
        if not np.isfinite(energy):
            raise RuntimeError(f"Non-finite Pauli energy at circuit index {index}: {energy}")
        energies.append(energy)
    return energies


def evaluate_grouped_circuits(builders: Sequence[Callable[[], Circuit]], groups: dict, *,
                              identity_const: float, n_qubits: int, shots: int,
                              provider: str, device: str, postprocessing: dict | None = None,
                              noise: dict | None = None,
                              device_options: dict | None = None) -> list[float]:
    """Submit all parameter variants and measurement groups, then aggregate energies."""
    circuits: list[Circuit] = []
    items_by_circuit: list[list] = []
    group_sequence = list(groups.items())
    for builder in builders:
        base = builder()
        for bases, items in group_sequence:
            circuits.append(base.extended(list(measurement_ops(n_qubits, bases))))
            items_by_circuit.append(items)
    contributions = _execute_pauli_circuits(
        circuits, items_by_circuit, shots=shots, provider=provider, device=device,
        postprocessing=postprocessing, noise=noise,
        device_options=dict(device_options or {}),
    )
    energies = []
    n_groups = len(group_sequence)
    for variant in range(len(builders)):
        start = variant * n_groups
        energies.append(float(identity_const + sum(contributions[start:start + n_groups])))
    return energies


def shifted_parameter_vectors(base: Sequence[float], shifts: Sequence[float]) -> list[np.ndarray]:
    """Return the base vector followed by each parameter's plus/minus variants."""
    vector = np.asarray(base, dtype=np.float64)
    vectors = [vector.copy()]
    for param_index in range(len(vector)):
        for shift in shifts:
            plus = vector.copy()
            minus = vector.copy()
            plus[param_index] += float(shift)
            minus[param_index] -= float(shift)
            vectors.extend((plus, minus))
    return vectors


def parameter_shift_gradient(energies: Sequence[float], n_params: int, *,
                             shifts: Sequence[float],
                             weights: Sequence[float]) -> tuple[float, np.ndarray]:
    if len(shifts) != len(weights):
        raise ValueError("shifts and weights must have the same length")
    n_variants = 1 + 2 * n_params * len(shifts)
    if len(energies) != n_variants:
        raise RuntimeError(f"Expected {n_variants} shifted energies, got {len(energies)}")
    gradient = np.zeros(int(n_params), dtype=np.float64)
    cursor = 1
    for param_index in range(int(n_params)):
        for weight in weights:
            plus = float(energies[cursor])
            minus = float(energies[cursor + 1])
            gradient[param_index] += float(weight) * (plus - minus)
            cursor += 2
    return float(energies[0]), gradient


def prepare_call(shots, provider, device, postprocessing, noise,
                 numeric_engine, options):
    opts = dict(options)
    if postprocessing is not None:
        opts["postprocessing"] = postprocessing
    if noise is not None:
        opts["noise"] = noise
    explicit_engine = opts.pop("numeric_engine", numeric_engine)
    shots, provider, device, opts = resolve_execution(
        shots=shots, provider=provider, device=device, options=opts
    )
    if shots > 0 and explicit_engine is not None:
        raise ValueError("numeric_engine is only valid with shots=0")
    postprocessing = opts.pop("postprocessing", None)
    noise = opts.pop("noise", None)
    return shots, provider, device, postprocessing, noise, explicit_engine, opts


def optimize_vqe(model, options: dict, *, ucc: bool):
    from scipy.optimize import minimize

    opts = dict(options)
    reject_runtime_option(opts)
    shots, provider, device, opts = resolve_execution(
        shots=opts.get("shots"), provider=opts.get("provider", "simulator"),
        device=opts.get("device", "statevector"), options=opts,
    )
    opts.update({"shots": shots, "provider": provider, "device": device})
    if model.init_guess is None:
        model.init_guess = np.zeros(model.n_params)

    cache = {"x": None, "eg": None}

    def with_grad(x):
        xa = np.asarray(x, dtype=np.float64)
        if cache["x"] is not None and np.array_equal(xa, cache["x"]):
            return cache["eg"]
        energy, gradient = model.energy_and_grad(xa, **opts)
        cache["x"] = xa.copy()
        cache["eg"] = (float(energy), np.asarray(gradient, dtype=np.float64))
        return cache["eg"]

    if model.scipy_minimize_options is None:
        minimize_options = {
            "ftol": 1e-9, "gtol": 1e-6, "maxiter": 200 if shots == 0 else 100,
        }
    else:
        minimize_options = model.scipy_minimize_options
    if model.grad == "free":
        result = minimize(
            lambda x: float(model.energy(x, **opts)), x0=model.init_guess,
            jac=False, method="COBYLA", options=minimize_options,
        )
    else:
        result = minimize(
            lambda x: with_grad(x)[0], x0=model.init_guess,
            jac=lambda x: with_grad(x)[1], method="L-BFGS-B", options=minimize_options,
        )
    if not result.success:
        logging.getLogger(__name__).warning("Optimization failed. See `.opt_res` for details.")

    params = np.asarray(getattr(result, "x", model.init_guess), dtype=np.float64)
    energy = float(result.fun)
    if ucc:
        result["init_guess"] = model.init_guess
        result["e"] = energy
        model.opt_res = result
    else:
        model.opt_res = {
            "x": params, "e": energy, "fun": energy,
            "success": bool(result.success), "message": str(result.message),
        }
    model._params = params
    return energy


__all__ = [
    "evaluate_grouped_circuits",
    "measurement_ops",
    "optimize_vqe",
    "parameter_shift_gradient",
    "prepare_call",
    "reject_runtime_option",
    "resolve_execution",
    "shifted_parameter_vectors",
]
