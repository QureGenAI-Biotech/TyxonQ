from __future__ import annotations

from typing import List

from ..core.ir.circuit import Circuit


def _escape(s: str) -> str:
    return s.replace("\"", "\\\"")


def circuit_to_dot(circuit: Circuit, *, graph_name: str = "Circuit") -> str:
    """Generate a simple DOT graph for an IR Circuit.

    - Qubits are rendered as horizontal tracks (rank=same) with labels q0..q{n-1}.
    - Ops appear as boxes on the corresponding qubit nodes in order.
    - Two-qubit ops connect control/target boxes with an edge.

    This produces DOT text compatible with GraphViz.
    """
    n = int(circuit.num_qubits)
    ops = list(circuit.ops)

    lines: List[str] = []
    lines.append(f"digraph \"{_escape(graph_name)}\" {{")
    lines.append("  rankdir=LR;")
    # Create qubit tracks
    for q in range(n):
        lines.append(f"  q{q} [label=\"q{q}\", shape=plaintext];")

    # Sequentially place ops with unique ids
    box_id = 0
    for op in ops:
        name = str(op[0])
        qubits = [int(a) for a in op[1:] if isinstance(a, int)]
        if not qubits:
            # global op (barrier) — attach to all
            label = _escape(name)
            for q in range(n):
                node = f"op{box_id}_{q}"
                lines.append(f"  {node} [label=\"{label}\", shape=box];")
                lines.append(f"  q{q} -> {node} -> q{q};")
            box_id += 1
            continue

        # place single or multi-qubit gate
        label = _escape(name)
        nodes = []
        for q in qubits:
            node = f"op{box_id}_q{q}"
            nodes.append((q, node))
            lines.append(f"  {node} [label=\"{label}\", shape=box];")
        # connect each involved qubit track through the node and back
        for q, node in nodes:
            lines.append(f"  q{q} -> {node} -> q{q};")
        # if two-qubit or more, add edges among nodes for visual link
        if len(nodes) >= 2:
            for i in range(len(nodes) - 1):
                lines.append(f"  {nodes[i][1]} -> {nodes[i+1][1]} [style=dashed, arrowsize=0.5];")
        box_id += 1

    lines.append("}")
    return "\n".join(lines)


