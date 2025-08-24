from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union


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
        from ...compiler.providers.qiskit.dialect import qasm_to_ir  # type: ignore

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
    circuit = _qasm_to_ir_if_needed(circuit, source)
    Engine = _select_engine(device)
    eng = Engine()

    from uuid import uuid4

    class _SimTask:
        def __init__(self, id_: str, device: str, results: Dict[str, Any]):
            self.id_ = id_
            self.device = device
            self._results = {"results": results}

        def results(self) -> Dict[str, Any]:
            return dict(self._results)

    def _one(c: Any) -> Any:
        out = eng.run(c, shots=shots)
        results = out.get("results") or out.get("expectations") or {}
        return _SimTask(id_=str(uuid4()), device=device, results=results)

    if isinstance(circuit, (list, tuple)):
        return [_one(c) for c in circuit]  # type: ignore
    return [_one(circuit)]


def get_task_details(task: Any, token: Optional[str] = None, prettify: bool = False) -> Dict[str, Any]:
    return getattr(task, "results", lambda: {})()


def remove_task(task: Any, token: Optional[str] = None) -> Any:
    return {"state": "cancelled"}


