from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union


@dataclass
class SimTask:
    def __init__(self, id: str, device: str, result: Dict[str, Any]):
        self.id = id
        self.device = device
        self._result = result
        self.task_info = None
        self.async_result = False

    def get_result(self, *, wait: bool = False, poll_interval: float = 0.0, timeout: float = 0.0) -> Dict[str, Any]:
        # Simulator tasks are always ready; ignore wait params
        return self._result


def _select_engine(device: str):
    name = device.split("::")[-1] if "::" in device else device
    if name in ("simulator:mps", "mps", "matrix_product_state"):
        from .matrix_product_state.engine import MatrixProductStateEngine as Engine
    elif name in ("simulator:statevector", "statevector"):
        from .statevector.engine import StatevectorEngine as Engine
    elif name in ("simulator:density_matrix", "density_matrix"):
        from .density_matrix.engine import DensityMatrixEngine as Engine
    else:
        raise ValueError(f"Unsupported simulator device: {device}")
    return Engine


def list_devices(token: Optional[str] = None, **kws: Any) -> List[str]:
    return [
        "simulator::matrix_product_state",
        "simulator::statevector",
        "simulator::density_matrix",
    ]


def _qasm_to_ir_if_needed(circuit: Any, source: Any) -> Any:
    if source is None:
        return circuit
    try:
        from ...compiler.compile_engine.qiskit.dialect import qasm_to_ir  # type: ignore

        if isinstance(source, (list, tuple)):
            return [qasm_to_ir(s) for s in source]
        return qasm_to_ir(source)
    except Exception as exc:
        raise ValueError(
            "OpenQASM support requires qiskit; please install qiskit or pass an IR circuit"
        ) from exc


def submit_task(
    device: str,
    token: Optional[str] = None,
    *,
    circuit: Optional[Union[Any, Sequence[Any]]] = None,
    source: Optional[Union[str, Sequence[str]]] = None,
    shots: Union[int, Sequence[int]] = 1024,
    **opts: Any,
) -> List[Any]:
    return run(device, token, circuit=circuit, source=source, shots=shots, **opts)

def run(
    device: str,
    token: Optional[str] = None,
    *,
    circuit: Optional[Union[Any, Sequence[Any]]] = None,
    source: Optional[Union[str, Sequence[str]]] = None,
    shots: Union[int, Sequence[int]] = 1024,
    **opts: Any,
) -> List[Any]:
    circuit = _qasm_to_ir_if_needed(circuit, source)
    Engine = _select_engine(device)
    eng = Engine()

    from uuid import uuid4

    def _one(c: Any) -> Any:
        error = ''
        try:
            out = eng.run(c, shots=shots, **opts)
        except Exception as e:
            error = str(e)  
        # Normalize simulator outputs:
        # - shots>0: counts in 'result'
        # - shots==0: analytic expectations in 'expectations'; also provide probabilities for exact multi-Z
        counts = out.get("result") or {}
        expectations = out.get("expectations") or {}
        meta = dict(out.get("metadata", {}))
        prob = None
        statevec = None
        if int(shots) == 0:
            try:
                import numpy as _np
                # Compute probabilities from exact state without changing engine API
                psi = eng.state(c)
                prob = _np.abs(_np.asarray(psi)) ** 2
                meta.setdefault("num_qubits", int(getattr(c, "num_qubits", 0)))
                statevec = _np.asarray(psi)
            except Exception as e:
                error = str(e)
                prob = None
                statevec = None
        result = {
            'result': counts,
            'expectations': expectations,
            'probabilities': prob,
            'statevector': statevec,
            'result_meta': meta,
            'uni_status': 'completed',
            'error': error
        }
        return SimTask(id=str(uuid4()), device=device, result=result)

    if isinstance(circuit, (list, tuple)):
        return [_one(c) for c in circuit]  # type: ignore
    return [_one(circuit)]


def get_task_details(task: SimTask, token: Optional[str] = None, prettify: bool = False) -> Dict[str, Any]:
    return task.get_result()

def remove_task(task: Any, token: Optional[str] = None) -> Any:
    return {"state": "cancelled"}


# --- Analytic expectation (shots==0) unified entry ---
def expval(
    device: str,
    token: Optional[str] = None,
    *,
    circuit: Any,
    observable: Any,
    **opts: Any,
) -> float:
    """Route to specific simulator engine and compute analytic expectation.

    This API is intended for provider in {simulator, local} and shots==0.
    """
    Engine = _select_engine(device)
    eng = Engine()
    return float(eng.expval(circuit, observable, **opts))


