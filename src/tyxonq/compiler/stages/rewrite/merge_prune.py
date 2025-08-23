from __future__ import annotations

from typing import Any, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from tyxonq.core.ir import Circuit
    from tyxonq.devices import DeviceCapabilities


DEFAULT_MERGE_RULES: Dict[Tuple[str, str], str] = {
    ("s", "s"): "z",
    ("sd", "sd"): "z",
    ("t", "t"): "s",
    ("td", "td"): "sd",
    ("x", "x"): "i",
    ("y", "y"): "i",
    ("z", "z"): "i",
    ("h", "h"): "i",
    ("rz", "rz"): "rz",
    ("rx", "rx"): "rx",
    ("ry", "ry"): "rx",
}


class MergePrunePass:
    """Merge adjacent gates by simple identities and prune identities.

    This is a lightweight variant inspired by legacy `simple_compiler.merge/prune`.
    """

    def __init__(self, rules: Dict[Tuple[str, str], str] | None = None) -> None:
        self.rules = dict(DEFAULT_MERGE_RULES)
        if rules:
            self.rules.update(rules)

    def run(self, circuit: "Circuit", caps: "DeviceCapabilities", **opts: Any) -> "Circuit":
        ops = list(circuit.ops)
        changed = True
        while changed:
            ops, changed = self._merge_once(ops)
            ops = self._prune_identities(ops)
        from dataclasses import replace

        return replace(circuit, ops=ops)

    def _merge_once(self, ops: list[Any]) -> tuple[list[Any], bool]:
        i = 0
        changed = False
        out: list[Any] = []
        while i < len(ops):
            cur = ops[i]
            if i + 1 < len(ops):
                nxt = ops[i + 1]
                if self._same_wire(cur, nxt):
                    nm = (str(cur[0]).lower(), str(nxt[0]).lower())
                    if nm in self.rules:
                        nn = self.rules[nm]
                        if nn == "i":
                            i += 2
                            changed = True
                            continue
                        if nn in ("rz", "rx"):
                            theta = float(cur[2]) + float(nxt[2])
                            out.append((nn, int(cur[1]), theta))
                            i += 2
                            changed = True
                            continue
                        out.append((nn, int(cur[1])))
                        i += 2
                        changed = True
                        continue
            out.append(cur)
            i += 1
        return out, changed

    def _same_wire(self, a: Any, b: Any) -> bool:
        try:
            return int(a[1]) == int(b[1]) and isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))
        except Exception:
            return False

    def _prune_identities(self, ops: list[Any]) -> list[Any]:
        return [op for op in ops if not (isinstance(op, (list, tuple)) and op and op[0] == "i")]


