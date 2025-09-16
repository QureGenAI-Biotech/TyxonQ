from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Set, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from tyxonq.core.ir import Circuit
    from tyxonq.devices import DeviceRule


class LightconeSimplifyPass:
    name = "simplify/lightcone"

    def execute_plan(self, circuit: "Circuit", device_rule: "DeviceRule" = None, **opts: Any) -> "Circuit":
        # Options:
        # - assume_measure_all: if no explicit measure ops, treat all qubits as measured
        assume_measure_all: bool = bool(opts.get("assume_measure_all", False))

        ops: List[Tuple] = list(getattr(circuit, "ops", []))
        if not ops:
            return circuit

        # 1) Collect measured qubits
        measured: List[int] = []
        for op in ops:
            if isinstance(op, (list, tuple)) and op:
                if op[0] == "measure_z":
                    measured.append(int(op[1]))

        # If no measures and not assuming measure-all, return as-is
        if not measured and not assume_measure_all:
            return circuit

        if not measured and assume_measure_all:
            measured = list(range(int(getattr(circuit, "num_qubits", 0))))

        # 2) Backward slice to find lightcone ops
        active: Set[int] = set(measured)
        keep_indices: List[int] = []

        # Define how each op affects qubits
        def op_qubits(op: Tuple) -> Tuple[List[int], bool]:
            name = op[0]
            if name in ("h", "rx", "ry", "rz", "phase", "s", "t", "sd", "td"):
                return [int(op[1])], False
            if name in ("x", "y", "z", "i"):
                return [int(op[1])], False
            if name in ("cx", "cz", "swap"):
                return [int(op[1]), int(op[2])], True
            if name in ("rxx", "ryy", "rzz"):
                return [int(op[1]), int(op[2])], True
            if name in ("project_z", "reset"):
                return [int(op[1])], False
            if name in ("barrier",):
                # Barrier doesn't affect dependencies; keep only if already active later
                # Treat as no-qubit op here
                return [], False
            if name in ("measure_z",):
                return [int(op[1])], False
            # Fallback: collect any int operands as qubits (defensive)
            qs: List[int] = []
            for a in op[1:]:
                if isinstance(a, int):
                    qs.append(int(a))
            return qs, len(qs) >= 2

        for i in range(len(ops) - 1, -1, -1):
            op = ops[i]
            if not isinstance(op, (list, tuple)) or not op:
                continue
            name = op[0]
            qs, is_two_q = op_qubits(op)

            if name == "measure_z":
                # Always keep explicit measurements
                keep_indices.append(i)
                continue

            # If this op acts on any active qubit, keep it and update active set
            if any(q in active for q in qs):
                keep_indices.append(i)
                if is_two_q and len(qs) == 2:
                    # propagate dependency across entangling gates
                    active.update(qs)
                else:
                    # single-qubit ops keep the same active set implicitly
                    active.update(qs)

        keep_indices.sort()
        if len(keep_indices) == len(ops):
            return circuit

        new_ops = [ops[i] for i in keep_indices]
        return circuit.with_ops(new_ops) if hasattr(circuit, "with_ops") else type(circuit)(
            num_qubits=circuit.num_qubits,
            ops=new_ops,
        )


