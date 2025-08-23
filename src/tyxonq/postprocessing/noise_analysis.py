from __future__ import annotations

from typing import Dict

"""Lightweight noise adapters for classical postprocessing on counts.

These utilities approximate certain noise channels directly on classical
counts dictionaries. They are not substitutes for state-level noise
simulation but are useful for quick what-if analyses and baseline tests.
"""


__all__ = ["apply_bitflip_counts", "apply_depolarizing_counts"]



def _num_qubits_from_keys(counts: Dict[str, int]) -> int:
    if not counts:
        return 0
    key = next(iter(counts))
    return len(key)


def apply_bitflip_counts(counts: Dict[str, int], p: float) -> Dict[str, int]:
    """Apply independent bit-flip noise with probability p to bitstring counts.

    Parameters
    ----------
    counts:
        Mapping from bitstring (e.g., "010") to counts.
    p:
        Bit-flip probability for each qubit independently, 0 <= p <= 1.

    Returns
    -------
    Dict[str, int]
        New counts after flipping each bit with probability p in expectation.
    """
    if p <= 0:
        return dict(counts)
    n = _num_qubits_from_keys(counts)
    total = sum(counts.values())
    if n == 0 or total == 0:
        return dict(counts)

    out: Dict[str, float] = {k: 0.0 for k in counts}
    # For small adapters, approximate by distributing mass to single-bit flips only
    # Higher-order flips contribute O(p^k); here we keep first-order for simplicity.
    for bitstr, c in counts.items():
        keep_mass = (1 - p) ** n
        out[bitstr] = out.get(bitstr, 0.0) + c * keep_mass
        for i in range(n):
            flipped = list(bitstr)
            flipped[i] = "1" if flipped[i] == "0" else "0"
            flipped_key = "".join(flipped)
            out[flipped_key] = out.get(flipped_key, 0.0) + c * (p * (1 - p) ** (n - 1))
    # Normalize rounding to integers while conserving total
    scale = total / max(1.0, sum(out.values()))
    rounded = {k: int(round(v * scale)) for k, v in out.items()}
    # Fix rounding drift
    drift = total - sum(rounded.values())
    if drift != 0:
        # Adjust the lexicographically smallest key deterministically
        key0 = sorted(rounded.keys())[0]
        rounded[key0] += drift
    return rounded


def apply_depolarizing_counts(counts: Dict[str, int], p: float) -> Dict[str, int]:
    """Apply an n-qubit depolarizing channel at the measurement level.

    This approximates by convex-combining with the uniform distribution over
    all bitstrings: (1-p) * counts + p * uniform.
    """
    if p <= 0:
        return dict(counts)
    n = _num_qubits_from_keys(counts)
    total = sum(counts.values())
    if n == 0 or total == 0:
        return dict(counts)
    num_outcomes = 2 ** n
    uniform = total / num_outcomes
    out: Dict[str, int] = {}
    for bitstr, c in counts.items():
        out[bitstr] = int(round((1 - p) * c + p * uniform))
    # Ensure all outcomes exist
    if len(out) < num_outcomes:
        # add missing keys with uniform mass
        from itertools import product

        for bits in product("01", repeat=n):
            key = "".join(bits)
            if key not in out:
                out[key] = int(round(p * uniform))
    # Fix drift
    drift = total - sum(out.values())
    if drift != 0:
        key0 = sorted(out.keys())[0]
        out[key0] += drift
    return out





