from __future__ import annotations

"""Postprocessing IO helpers for counts and probabilities."""

from typing import Dict


def counts_to_csv(counts: Dict[str, int]) -> str:
    lines = ["bitstring,count"]
    for k, v in counts.items():
        lines.append(f"{k},{v}")
    return "\n".join(lines)


def csv_to_counts(text: str) -> Dict[str, int]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines or lines[0].lower() != "bitstring,count":
        raise ValueError("invalid header for counts csv")
    out: Dict[str, int] = {}
    for line in lines[1:]:
        bits, num = line.split(",", 1)
        out[bits] = int(num)
    return out


__all__ = ["counts_to_csv", "csv_to_counts"]


