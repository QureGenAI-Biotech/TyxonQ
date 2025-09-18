import pytest
import numpy as np

from tyxonq.postprocessing.io import (
    reverse_count,
    sort_count,
    normalized_count,
    count2vec,
    vec2count,
    marginal_count,
    plot_histogram,
)


def test_reverse_sort_normalize_and_roundtrip():
    counts = {"00": 1, "01": 3, "10": 2, "11": 0}
    # reverse twice equals original (ignoring zero entries)
    assert reverse_count(reverse_count(counts)) == counts
    # sort desc keeps highest first
    sorted_counts = sort_count(counts)
    assert list(sorted_counts.items())[0][0] == "01"
    # normalize sums to 1
    norm = normalized_count(counts)
    assert abs(sum(norm.values()) - 1.0) < 1e-12
    # count2vec and vec2count roundtrip (without prune keeps zeros out)
    vec = count2vec(counts, normalization=True)
    assert np.isclose(vec.sum(), 1.0)
    back = vec2count((vec * 6).astype(int), prune=True)
    # compare non-zero entries only
    for k, v in counts.items():
        if v > 0:
            assert k in back


@pytest.mark.skipif(pytest.importorskip("qiskit", reason="qiskit not installed") is None, reason="qiskit not installed")
def test_marginal_and_plot_histogram_qiskit():
    import qiskit  # noqa: F401

    counts = {"000": 5, "001": 3, "010": 2, "100": 1}
    # marginal over qubits [0,2]
    m = marginal_count(counts, keep_list=[0, 2])
    # keys should be length 2
    assert all(len(k) == 2 for k in m.keys())
    # histogram plots without raising
    fig = plot_histogram(counts)
    assert fig is not None


