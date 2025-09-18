from __future__ import annotations

from tyxonq.postprocessing.noise_analysis import apply_bitflip_counts, apply_depolarizing_counts


def test_apply_bitflip_counts_simple():
    counts = {"0": 80, "1": 20}
    out = apply_bitflip_counts(counts, p=0.1)
    # Expect probability mass move between 0 and 1
    assert abs(sum(out.values()) - sum(counts.values())) < 1e-9
    assert out["0"] > out["1"]


def test_apply_depolarizing_counts_simple():
    counts = {"00": 50, "01": 0, "10": 0, "11": 50}
    out = apply_depolarizing_counts(counts, p=0.2)
    assert abs(sum(out.values()) - sum(counts.values())) < 1e-9
    # With depolarizing, probability mass spreads more uniformly
    assert out["01"] > 0 and out["10"] > 0


