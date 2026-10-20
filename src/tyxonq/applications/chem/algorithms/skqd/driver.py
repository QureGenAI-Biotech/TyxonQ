"""将演化样本接入既有 SQD，能量常数只在最外层加一次。"""

from dataclasses import dataclass

import numpy as np

from ..sqd import SCIResult, run_sqd_fermion
from ...chem_libs.fermionic_operators import validate_integrals
from .ensemble import SKQDEnsemble, merge_skqd_samples


@dataclass
class SKQDResult:
    """sci_result.energy 为电子能；energy_offset 包含核排斥和冻结核贡献。"""

    sci_result: SCIResult
    energy_offset: float
    history: list
    statistics: dict

    @property
    def energy(self):
        return self.sci_result.energy

    @property
    def total_energy(self):
        return self.energy + self.energy_offset


def postprocess_skqd(h1e, eri, samples, ensemble: SKQDEnsemble, *,
                     samples_per_batch=100, energy_offset=0.0, **sqd_options):
    """samples 必须按 circuit_id 保存原始 counts；函数只进行本地经典求解。"""
    h, g, n, nelec = validate_integrals(h1e, eri, ensemble.norb, ensemble.nelec)
    if not np.isfinite(energy_offset):
        raise ValueError("energy_offset must be finite")
    if "nuclear_repulsion_energy" in sqd_options or "sample_order" in sqd_options:
        raise ValueError("use energy_offset; sample_order is determined by ensemble normalization")
    counts = merge_skqd_samples(samples, ensemble)
    history = []
    callback = sqd_options.pop("callback", None)

    def capture(results):
        history.append([{"energy": r.energy,
                         "dimension": int(r.sci_state.amplitudes.size)} for r in results])
        if callback is not None:
            callback(results)

    sqd_options.setdefault("symmetrize_spin", False)
    sqd_options.setdefault("seed", 24)
    result = run_sqd_fermion(h, g, counts, samples_per_batch, n, nelec,
                            sample_order="beta_alpha", nuclear_repulsion_energy=0.0,
                            callback=capture, **sqd_options)
    valid = sum(v for b, v in counts.items()
                if b[:n].count("1") == nelec[1] and b[n:].count("1") == nelec[0])
    stats = {"circuits": len(ensemble.circuits), "shots": sum(counts.values()),
             "unique_bitstrings": len(counts), "valid_shots": valid,
             "ci_dimension": int(result.sci_state.amplitudes.size),
             "spin_pool_sizes": list(result.sci_state.amplitudes.shape),
             "basis_id": ensemble.records[0].basis_id, "method": ensemble.method}
    return SKQDResult(result, float(energy_offset), history, stats)
