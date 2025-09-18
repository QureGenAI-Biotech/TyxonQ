from __future__ import annotations

"""Postprocessing IO helpers for counts and probabilities."""

from typing import Dict, Any, Sequence, Tuple, List
import numpy as np


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


def reverse_count(count: Dict[str, int]) -> Dict[str, int]:
    return {k[::-1]: v for k, v in count.items()}


def sort_count(count: Dict[str, int]) -> Dict[str, int]:
    return {k: v for k, v in sorted(count.items(), key=lambda kv: -kv[1])}


def normalized_count(count: Dict[str, int]) -> Dict[str, float]:
    shots = max(1, int(sum(count.values())))
    return {k: v / shots for k, v in count.items()}


def count2vec(count: Dict[str, int], normalization: bool = True) -> "np.ndarray":
    nqubit = len(next(iter(count.keys())))
    size = 1 << nqubit
    prob: Any = np.zeros(size, dtype=float)
    shots = float(sum(count.values()))
    for k, v in count.items():
        idx = int(k, 2)
        prob[idx] = float(v)
    if normalization and shots > 0:
        prob = prob / shots
    return prob


def vec2count(vec: "np.ndarray", prune: bool = False) -> Dict[str, int]:
    if isinstance(vec, list):
        vec = np.array(vec)
    n = int(np.round(np.log2(vec.shape[0])))
    counts: Dict[str, int] = {}
    for idx, p in enumerate(vec):
        v = int(round(float(p))) if p <= 1.0 else int(round(float(p)))
        if prune and abs(v) < 1:
            continue
        if v <= 0:
            continue
        bit = format(idx, f"0{n}b")
        counts[bit] = v
    return counts


def marginal_count(count: Dict[str, int], keep_list: Sequence[int]) -> Dict[str, int]:
    try:
        import qiskit  # type: ignore

        c = reverse_count(count)
        ncount = qiskit.result.utils.marginal_distribution(c, keep_list)  # type: ignore
        return reverse_count(ncount)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("marginal_count requires qiskit") from exc


def plot_histogram(data: Any, **kws: Any) -> Any:
    try:
        from qiskit.visualization import plot_histogram  # type: ignore

        return plot_histogram(data, **kws)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("plot_histogram requires qiskit") from exc


__all__ = [
    "counts_to_csv",
    "csv_to_counts",
    "reverse_count",
    "sort_count",
    "normalized_count",
    "count2vec",
    "vec2count",
    "marginal_count",
    "plot_histogram",
]


# ---- Sampling and count transformations (ported from legacy quantum) ----

def count_s2d(srepr: Tuple[np.ndarray, np.ndarray], n: int) -> np.ndarray:
    """Sparse tuple (indices, counts) to dense vector of length 2**n."""
    idx, val = srepr
    out = np.zeros(1 << n, dtype=val.dtype)
    out[np.asarray(idx, dtype=int)] = val
    return out


counts_v2t = count_s2d


def count_d2s(drepr: np.ndarray, eps: float = 1e-7) -> Tuple[np.ndarray, np.ndarray]:
    """Dense vector to sparse tuple (indices, values) with cutoff eps."""
    idx = np.nonzero(np.abs(drepr) > eps)[0]
    vals = drepr[idx]
    return idx.astype(int), np.asarray(vals)


count_t2v = count_d2s


def sample_int2bin(sample: np.ndarray, n: int) -> np.ndarray:
    """Convert integer samples to binary matrix shape [trials, n]."""
    sample = np.asarray(sample, dtype=int)
    bits = ((sample[:, None] >> np.arange(n)[::-1]) & 1).astype(int)
    return bits


def sample_bin2int(sample: np.ndarray, n: int) -> np.ndarray:
    """Convert binary samples [trials, n] to integers [trials]."""
    sample = np.asarray(sample, dtype=int)
    power = 2 ** np.arange(n)[::-1]
    return (sample * power).sum(axis=-1)


def sample2count(sample: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert integer samples to (indices, counts) sparse tuple."""
    sample = np.asarray(sample, dtype=int)
    idx, counts = np.unique(sample, return_counts=True)
    return idx.astype(int), counts.astype(int)


def count_vector2dict(count: np.ndarray, n: int, key: str = "bin") -> Dict[Any, int]:
    """Convert dense count vector to dict keyed by 'bin' or 'int'."""
    d = {i: int(count[i]) for i in range(1 << n)}
    if key == "int":
        return d
    return {format(k, f"0{n}b"): v for k, v in d.items() if v}


def count_tuple2dict(count: Tuple[np.ndarray, np.ndarray], n: int, key: str = "bin") -> Dict[Any, int]:
    """Convert sparse (indices, counts) to dict keyed by 'bin' or 'int'."""
    idx, vals = count
    d = {int(i): int(v) for i, v in zip(idx, vals) if int(i) >= 0 and int(v) != 0}
    if key == "int":
        return d
    return {format(k, f"0{n}b"): v for k, v in d.items()}


def sample2all(sample: np.ndarray, n: int, format: str = "count_vector") -> Any:
    """Transform integer/bin samples to requested representation.

    Supported: 'sample_int', 'sample_bin', 'count_tuple', 'count_vector', 'count_dict_bin', 'count_dict_int'.
    """
    sample = np.asarray(sample)
    if sample.ndim == 1:
        sample_int = sample
        sample_bin = sample_int2bin(sample_int, n)
    elif sample.ndim == 2:
        sample_bin = sample
        sample_int = sample_bin2int(sample_bin, n)
    else:
        raise ValueError("unrecognized sample shape")
    if format == "sample_int":
        return sample_int
    if format == "sample_bin":
        return sample_bin
    idx, cnts = sample2count(sample_int, n)
    if format == "count_tuple":
        return (idx, cnts)
    if format == "count_vector":
        return count_s2d((idx, cnts), n)
    if format == "count_dict_bin":
        return count_tuple2dict((idx, cnts), n, key="bin")
    if format == "count_dict_int":
        return count_tuple2dict((idx, cnts), n, key="int")
    raise ValueError(f"unsupported format {format}")


def spin_by_basis(n: int, m: int, elements: Tuple[int, int] = (1, -1)) -> np.ndarray:
    """Generate column m of all n-bitstrings mapped to elements (1,-1)."""
    col = np.tile(np.array([[elements[0]], [elements[1]]], dtype=int), (2**m, 2 ** (n - m - 1)))
    return col.reshape(-1)


def correlation_from_samples(index: Sequence[int], results: np.ndarray, n: int) -> float:
    """Estimate product of spins at positions `index` from samples (int or bin)."""
    if results.ndim == 1:
        results = sample_int2bin(results, n)
    spins = 1 - 2 * results  # 0 -> +1, 1 -> -1
    r = spins[:, index[0]].astype(float)
    for i in index[1:]:
        r *= spins[:, i]
    return float(r.mean())


def correlation_from_counts(index: Sequence[int], prob_vec: np.ndarray) -> float:
    """Compute correlation from probability vector over bitstrings."""
    prob = np.asarray(prob_vec, dtype=float).reshape(-1)
    prob = prob / max(1e-12, prob.sum())
    n = int(np.round(np.log2(prob.shape[0])))
    spin = prob.copy()
    for i in index:
        spin *= spin_by_basis(n, i).astype(float)
    return float(spin.sum())



