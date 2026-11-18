"""Functional wrapper for native Selected Basis Diagonalization (SBD).

SBD is a general selected-basis eigensolver, so it lives in numerics rather than
in the SQD application layer. This module owns CUDA extension loading, determinant
signs and the native solve call; SQD supplies the sampled basis and validation.
"""

from __future__ import annotations

import importlib
from time import perf_counter
from typing import Any

import numpy as np


def load_extension() -> Any:
    """Load the explicitly built optional CUDA extension (never on import)."""
    name = "tyxonq.numerics.sbd.cuda._sbd_tpb_cuda"
    try:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name:
                raise
            return importlib.import_module("_sbd_tpb_cuda")
    except ImportError as error:
        raise ImportError(
            "SBD CUDA extension unavailable. Build it with "
            "python -m tyxonq.numerics.sbd.cuda.build --output-dir DIR; "
            "CPU and PyTorch paths do not require this extension."
        ) from error


def is_available() -> bool:
    try:
        return hasattr(load_extension(), "TPBContext")
    except Exception:
        return False


def normalize_problem(
    ci_strings: tuple[np.ndarray, np.ndarray],
    h1e: np.ndarray,
    eri: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray, int, tuple[int, int]]:
    if len(ci_strings) != 2:
        raise ValueError("ci_strings must contain alpha and beta determinant arrays")
    alpha = np.asarray(ci_strings[0], dtype=np.int64)
    beta = np.asarray(ci_strings[1], dtype=np.int64)
    h1 = np.asarray(h1e, dtype=np.float64)
    g2 = np.asarray(eri, dtype=np.float64)
    if h1.shape != (norb, norb) or g2.shape != (norb,) * 4:
        raise ValueError(f"integral shapes mismatch: h1e={h1.shape}, eri={g2.shape}, norb={norb}")
    if len(nelec) != 2 or min(nelec) < 0:
        raise ValueError("nelec must be a pair of non-negative electron counts")
    return (alpha, beta), h1, g2, norb, (int(nelec[0]), int(nelec[1]))


def interleaved_phase(ci_strings: tuple[np.ndarray, np.ndarray], norb: int, *, device: Any) -> Any:
    """Fermionic sign for interleaved orbitals to alpha-then-beta ordering."""
    import torch

    alpha = torch.as_tensor(ci_strings[0], dtype=torch.int64, device=device)
    beta = torch.as_tensor(ci_strings[1], dtype=torch.int64, device=device)
    parity = torch.zeros((alpha.numel(), beta.numel()), dtype=torch.int8, device=device)
    lower_beta = torch.zeros(beta.numel(), dtype=torch.int8, device=device)
    for orbital in range(norb):
        occupied_alpha = ((alpha >> orbital) & 1).to(torch.int8)
        parity.bitwise_xor_(occupied_alpha[:, None] * lower_beta[None, :])
        lower_beta.bitwise_xor_(((beta >> orbital) & 1).to(torch.int8))
    return (1.0 - 2.0 * parity.to(torch.float64)).contiguous()


def convert_raw_rdms(raw: dict[str, Any], norb: int) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Convert SBD spin blocks/Fortran order to PySCF RDM convention."""
    one, two = raw["one_p_rdm"], raw["two_p_rdm"]
    if len(one) != 2 or len(two) != 4:
        raise ValueError("SBD must return two 1-RDM and four 2-RDM spin blocks")
    dm1 = tuple(
        np.asarray(block, dtype=np.float64).reshape((norb, norb), order="F").copy()
        for block in one
    )
    dm2 = tuple(
        np.asarray(block, dtype=np.float64).reshape((norb,) * 4, order="F")
        .transpose(0, 2, 1, 3).copy()
        for block in two
    )
    if not np.allclose(dm2[1], dm2[2].transpose(2, 3, 0, 1), atol=1e-8, rtol=1e-8):
        raise RuntimeError("SBD alpha-beta and beta-alpha RDM blocks are inconsistent")
    return dm1, (dm2[0], dm2[2], dm2[3])


class SBDContractor:
    """Owns a determinant connection table and native CUDA integrals."""

    def __init__(
        self,
        h1e: np.ndarray,
        eri: np.ndarray,
        ci_strings: tuple[np.ndarray, np.ndarray],
        norb: int,
        nelec: tuple[int, int],
        *,
        device: Any,
        memory_fraction: float = 0.8,
    ) -> None:
        import torch

        strings, h1e, eri, norb, nelec = normalize_problem(ci_strings, h1e, eri, norb, nelec)
        if not 0 < memory_fraction <= 0.8:
            raise ValueError("memory_fraction must be in (0, 0.8]")
        self.device = torch.device(device)
        self.shape = tuple(map(len, strings))
        self.norb, self.nelec = norb, nelec
        self.memory_fraction = memory_fraction
        self.ci_strings = strings
        self.phase = interleaved_phase(strings, norb, device=self.device)
        self.device = self.phase.device
        device_id = torch.cuda.current_device() if self.device.index is None else self.device.index
        self.context = load_extension().TPBContext(
            h1e, eri, *strings, norb, nelec, device_id=device_id,
            memory_fraction=memory_fraction,
        )

    def update_integrals(self, h1e: np.ndarray, eri: np.ndarray) -> None:
        _, h1e, eri, _, _ = normalize_problem(self.ci_strings, h1e, eri, self.norb, self.nelec)
        self.context.update_integrals(h1e, eri)

    def assert_compatible(
        self,
        ci_strings: tuple[np.ndarray, np.ndarray],
        norb: int,
        nelec: tuple[int, int],
        device: Any,
        *,
        memory_fraction: float | None = None,
    ) -> None:
        import torch

        strings, _, _, norb, nelec = normalize_problem(
            ci_strings, np.zeros((norb, norb)), np.zeros((norb,) * 4), norb, nelec
        )
        requested = torch.device(device)
        index = torch.cuda.current_device() if requested.index is None else requested.index
        if (
            norb != self.norb
            or tuple(nelec) != tuple(self.nelec)
            or requested.type != self.phase.device.type
            or index != self.phase.device.index
            or any(not np.array_equal(x, y) for x, y in zip(strings, self.ci_strings))
        ):
            raise ValueError("SBD context does not match determinants, norb, nelec or device")
        if memory_fraction is not None and memory_fraction != self.memory_fraction:
            raise ValueError("SBD context has a different memory_fraction")

    def diagonal(self) -> Any:
        import torch

        raw = np.asarray(self.context.hdiag()).reshape(self.shape)
        return torch.as_tensor(raw, dtype=torch.float64, device=self.device)

    def _raw(self, coefficients: Any) -> Any:
        import torch

        if (
            coefficients.dtype != torch.float64
            or coefficients.device != self.phase.device
            or coefficients.numel() != self.phase.numel()
        ):
            raise ValueError("SBD coefficients must be FP64 on the context device with matching size")
        return (coefficients.reshape(self.shape) * self.phase).contiguous()

    def contract(self, coefficients: Any) -> Any:
        import torch

        raw = self._raw(coefficients)
        output = torch.empty_like(raw)
        torch.cuda.synchronize(self.device)
        self.context.matvec_device(raw.data_ptr(), output.data_ptr(), raw.numel())
        torch.cuda.synchronize(self.device)
        return output * self.phase

    def rdms(self, coefficients: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        import torch

        raw = self._raw(coefficients)
        torch.cuda.synchronize(self.device)
        result = self.context.rdms_device(raw.data_ptr(), raw.numel())
        torch.cuda.synchronize(self.device)
        return convert_raw_rdms(result, self.norb)


def solve_selected_basis(
    contractor: SBDContractor,
    initial_coefficients: np.ndarray,
    *,
    tolerance: float = 1e-8,
    max_cycles: int = 100,
    max_space: int = 20,
    compute_rdms: bool = True,
) -> tuple[dict[str, Any], float]:
    """Run the native Davidson solve and return its raw result plus elapsed seconds."""
    import torch

    started = perf_counter()
    raw = contractor.context.solve(
        tolerance=tolerance,
        max_cycles=max_cycles,
        max_space=max_space,
        initial_coefficients=np.asarray(initial_coefficients, dtype=np.float64),
        compute_rdms=compute_rdms,
    )
    torch.cuda.synchronize(contractor.device)
    return raw, perf_counter() - started


__all__ = [
    "SBDContractor",
    "convert_raw_rdms",
    "interleaved_phase",
    "is_available",
    "load_extension",
    "normalize_problem",
    "solve_selected_basis",
]
