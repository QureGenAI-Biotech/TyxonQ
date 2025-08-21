from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Set

if TYPE_CHECKING:  # pragma: no cover
    from tyxonq.core.ir import Circuit
    from tyxonq.devices import DeviceCapabilities


class MeasurementRewritePass:
    """Rewrite measurement-related constructs.

    Advantages of this approach compared to ad-hoc handling:
    - Explicit grouping metadata (basis, basis_map, wires) is stored in
      `Circuit.metadata`, improving observability and downstream scheduling.
    - Greedy, product-basis-safe grouping enables safe shot reuse without
      changing circuit semantics and keeps complexity linear in the number of
      measurement items.
    - Extensible: grouping policy can evolve (e.g., commuting sets, cost-aware
      packing) without touching device code.
    """

    # TODO(next): Commuting-set weighted cost model (variance-aware) for grouping.
    #   - Incorporate observable commutation relations and target variance to
    #     decide grouping and shot allocation.
    # TODO(next): Integrate grouping metadata into shot scheduler for settings reuse.
    #   - Propagate `basis_map`, `estimated_settings`, and `estimated_shots_per_group`
    #     to the scheduling stage to minimize total measurement settings.

    def run(self, circuit: "Circuit", caps: "DeviceCapabilities", **opts) -> "Circuit":
        measurements = opts.get("measurements") or []
        groups = _group_measurements(measurements)
        circuit.metadata["measurement_groups"] = groups
        return circuit


def _parse_pauli_support(obs: Any, wires: Tuple[int, ...]) -> Tuple[str, Dict[int, str]]:
    """Parse a simple Pauli observable string into a per-wire basis map.

    Examples:
        obs='Z', wires=(0,) -> {'0': 'Z'}
        obs='ZX', wires=(0,1) -> {0:'Z', 1:'X'}
    Fallback: empty map (unknown), basis='pauli'.
    """
    basis = "pauli"
    if not isinstance(obs, str):
        return basis, {}
    # Normalize to per-wire letters, honoring identity 'I'
    if len(obs) == len(wires):
        mapping: Dict[int, str] = {}
        for i, w in enumerate(wires):
            c = obs[i]
            if c in {"X", "Y", "Z"}:
                mapping[int(w)] = c
            # ignore 'I' or other characters
        return basis, mapping
    # Fallback: single-letter observable on a single wire
    if len(obs) == 1 and len(wires) == 1 and obs in {"X", "Y", "Z"}:
        return basis, {int(wires[0]): obs}
    return basis, {}


def _group_measurements(measurements: List[Any]) -> List[Dict[str, Any]]:
    """Group measurements by non-overlapping wires with a basis tag.

    Greedy strategy that packs items if their wires do not overlap with the
    group's used wires. Basis currently fixed as 'pauli' for simplicity.
    """

    groups: List[Dict[str, Any]] = []
    for m in measurements:
        wires = tuple(sorted(getattr(m, "wires", ())))
        basis, basis_map = _parse_pauli_support(getattr(m, "obs", None), wires)
        placed = False
        for g in groups:
            if g["basis"] != basis:
                continue
            # product-basis-safe merge: allow overlapping wires if basis agrees
            conflict = False
            for w, b in basis_map.items():
                if w in g["basis_map"] and g["basis_map"][w] != b:
                    conflict = True
                    break
            if conflict:
                continue
            # merge
            g["items"].append(m)
            g["wires"].update(wires)
            g["basis_map"].update(basis_map)
            placed = True
            break
        if not placed:
            groups.append({
                "items": [m],
                "wires": set(wires),
                "basis": basis,
                "basis_map": dict(basis_map) if basis_map else {},
            })
    for g in groups:
        g["wires"] = tuple(sorted(g["wires"]))
        # Heuristic cost model:
        # - settings = 1 if a consistent product basis exists (by construction)
        # - estimated_shots_per_group: proportional to number of items and wires
        num_items = len(g["items"])
        num_wires = len(g["wires"])
        g["estimated_settings"] = 1
        g["estimated_shots_per_group"] = max(1, num_items) * max(1, num_wires)
    return groups


