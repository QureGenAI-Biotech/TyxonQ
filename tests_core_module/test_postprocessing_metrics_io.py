from tyxonq.postprocessing.metrics import normalized_count, kl_divergence, expectation
from tyxonq.postprocessing.io import counts_to_csv, csv_to_counts


def test_metrics_and_io_roundtrip():
    counts = {"00": 60, "01": 20, "10": 15, "11": 5}
    norm = normalized_count(counts)
    assert abs(sum(norm.values()) - 1.0) < 1e-12
    kl = kl_divergence(counts, counts)
    assert abs(kl) < 1e-12
    expz0 = expectation(counts, z=[0])
    assert -1.0 <= expz0 <= 1.0

    csv = counts_to_csv(counts)
    rec = csv_to_counts(csv)
    assert rec == counts


