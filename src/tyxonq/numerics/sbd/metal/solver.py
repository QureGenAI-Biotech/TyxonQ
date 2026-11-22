"""Single-device native MLX/Metal Selected Basis Diagonalization.

The host builds PySCF connection tables and solves only the small Davidson
projected eigenproblem. Long vectors, all three H@c spin contributions, S²,
orthogonalization, and residual vectors remain MLX arrays on the Metal GPU.

The row-wise kernel visits only connected determinants. The pair-table path
bounds setup cost when the same-spin connection matrix would be too large.
Hardware crossover and final accuracy are measured separately for each workload.

IMPORTANT — FP32 MLX GEMM precision switch:
    MLX 0.32.4 may silently use reduced-precision NAX arithmetic for FP32
    ``mx.matmul`` on supported Apple hardware. Set ``MLX_ENABLE_TF32=0`` before
    process startup when evaluating full-FP32 MLX GEMM. The production path below
    deliberately uses deterministic FP32 kernels and neither reads nor changes
    this process-global setting. Full-FP32 MLX GEMM passed the SBD golden tests,
    but showed no reproducible performance advantage and is not the default.
    The connected-rows, dot-tiles and projected-reduction kernels are still FP32,
    not FP64; each new path requires CPU FP64 golden, fresh-residual and normal-
    convergence validation before its performance can be adopted.

Historical pre-PR21 CPU FP32 break-even (fixed 40 cycles, not converged energies):
    Davidson was stably faster from 14,400 determinants; H@c was faster at 18,496
    and 23,409 determinants. Smaller cases remained slower than CPU, and the MPS
    ranking changed at large sizes. These are pre-connected-rows/pre-dot-tiles
    results; see docs/SBD_METAL_OPTIMIZATION.md for post-merge measurements.

TODO — multi-Apple-GPU SBD (planned; not active in this single-device module):
    Use MLX-native distribution only: ``mx.distributed`` plus ``mlx.launch``,
    with Ring/MPI as portable baselines and JACCL for Thunderbolt RDMA. Keep the
    validated per-rank MLX/MSL kernels. Partition determinant/link ownership and
    exchange H@c halo contributions; use distributed all-sum for Davidson norms,
    dot products and projected matrices. Require the requested world size rather
    than accepting a silent singleton fallback. Report rank/world size, collective
    backend, communication time and scaling alongside golden/residual/RDM checks.
    Do not claim multi-host support until multi-process and multi-Mac runs pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, wraps
from importlib import resources
from typing import Any, Callable, TypedDict

import numpy as np
from pyscf.fci import direct_spin1, selected_ci

from tyxonq.numerics.backends.base import BackendUnavailableError


class MetalExecution(TypedDict):
    device: str | None
    stream: str | None
    completed_evaluations: int


def _mlx() -> Any:
    try:
        import mlx.core as mx
    except ImportError as error:
        raise BackendUnavailableError(
            "Metal SBD requires MLX on Apple Silicon. Install it with "
            "`python -m pip install mlx` or `pip install 'tyxonq[metal]'`."
        ) from error
    if not mx.metal.is_available():
        raise BackendUnavailableError("Metal SBD requires an available Metal GPU")
    return mx


def metal_gpu_execution(function: Callable[..., Any]) -> Callable[..., Any]:
    """Run one native operation on the Apple GPU without changing global state."""
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        mx = _mlx()
        with mx.stream(mx.gpu):
            return function(*args, **kwargs)
    return wrapped


def _record_execution(owner: Any, mx: Any) -> None:
    """Capture the actual MLX launch context inside a native operator."""
    device = mx.default_device()
    owner.execution.update(device=str(device), stream=str(mx.default_stream(device)))


@lru_cache(maxsize=1)
def _kernels() -> dict[str, Any]:
    mx = _mlx()
    text = resources.files("tyxonq.numerics.sbd.metal").joinpath("kernels.metal").read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    name: str | None = None
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("// @kernel "):
            if name is not None:
                sections[name] = "\n".join(body).strip()
            name = line.removeprefix("// @kernel ").strip()
            body = []
        elif name is not None:
            body.append(line)
    if name is not None:
        sections[name] = "\n".join(body).strip()
    return {
        "mixed_rdm_q60": mx.fast.metal_kernel(
            name="tyxonq_sbd_mixed_rdm_q60", input_names=["hi", "lo", "al", "ao", "bl", "bo", "shape"],
            output_names=["out"], source=sections["mixed_rdm_q60"],
        ),
        "connected_rows": mx.fast.metal_kernel(
            name="tyxonq_sbd_connected_rows",
            input_names=["c", "diag", "a_offset", "a_single", "b_offset", "b_single",
                         "aa_offset", "aa_col", "aa_val", "bb_offset", "bb_col", "bb_val",
                         "a_field", "b_field", "mixed", "dimensions"],
            output_names=["out"], source=sections["connected_rows"],
        ),
        "gather_slots": mx.fast.metal_kernel(
            name="tyxonq_sbd_gather_slots",
            input_names=["coefficients", "address", "signs", "width"],
            output_names=["out"],
            source=sections["gather_slots"],
        ),
        "pair_contract_links": mx.fast.metal_kernel(
            name="tyxonq_sbd_pair_contract_links",
            input_names=["matrix", "temporary", "pair", "outer", "signs", "inner_dimension", "width", "temporary_width"],
            output_names=["out"],
            source=sections["pair_contract_links"],
        ),
        "pair_contract_mixed": mx.fast.metal_kernel(
            name="tyxonq_sbd_pair_contract_mixed",
            input_names=["matrix", "temporary", "pair", "outer", "signs", "inner_dimension", "width", "temporary_width", "alpha_count"],
            output_names=["out"],
            source=sections["pair_contract_mixed"],
        ),
        "pair_contract": mx.fast.metal_kernel(
            name="tyxonq_sbd_pair_contract",
            input_names=["matrix", "temporary", "inner_dimension", "width"],
            output_names=["out"],
            source=sections["pair_contract"],
        ),
        "dot_tiles": mx.fast.metal_kernel(
            name="tyxonq_sbd_dot_tiles", input_names=["left", "right_t", "shape"],
            output_names=["out"], source=sections["dot_tiles"],
        ),
        "projected_reduce": mx.fast.metal_kernel(
            name="tyxonq_sbd_projected_reduce",
            input_names=["matrix", "temporary", "inner_dimension", "width"],
            output_names=["out"],
            source=sections["projected_reduce"],
        ),
        "spin_exchange": mx.fast.metal_kernel(
            name="tyxonq_sbd_spin_exchange",
            input_names=["c", "alpha_bits", "beta_bits", "offsets", "alpha_moves",
                         "beta_lookup", "dimensions", "diagonal_constant"],
            output_names=["out"], source=sections["spin_exchange"],
        ),
    }


def _eval_scalar(value: Any) -> float:
    mx = _mlx()
    mx.eval(value)
    return float(value.item())


def _to_numpy(value: Any) -> np.ndarray:
    mx = _mlx()
    mx.eval(value)
    return np.asarray(value)


@metal_gpu_execution
def _metal_matmul(left: Any, right: Any) -> Any:
    """Accurate FP32 row-major GEMM for SBD long-vector operations.

    Default MLX 0.32.4 FP32 GEMM may use reduced-precision NAX on this host.
    SBD uses deterministic accumulation so correctness does not depend on the
    process-global MLX_ENABLE_TF32 setting.
    """
    mx = _mlx()
    rows, inner = map(int, left.shape)
    if int(right.shape[0]) != inner or right.ndim != 2:
        raise ValueError("incompatible matrix shapes for Metal contraction")
    width = int(right.shape[1])
    total = rows * width
    inputs = [left, right, mx.array([inner], dtype=mx.int32), mx.array([width], dtype=mx.int32)]
    if inner >= 16384 and total <= 1024:
        # 将长向量内积分配给多个线程组，再归约各个等长块，提高带宽利用率。
        tiles = (inner + 1023) // 1024
        tile_sums = _kernels()["dot_tiles"](
            inputs=[left, right.T, mx.array([inner, width, tiles], dtype=mx.uint32)],
            grid=(total * tiles * 256, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(rows, width, tiles)], output_dtypes=[mx.float32],
        )[0]
        return mx.sum(tile_sums, axis=2)
    if inner >= 128 and total <= 1024:
        return _kernels()["projected_reduce"](
            inputs=inputs, template=[("T", mx.float32)],
            grid=(total * 256, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(rows, width)], output_dtypes=[mx.float32],
        )[0]
    return _kernels()["pair_contract"](
        inputs=[left, right, mx.array([inner], dtype=mx.int32), mx.array([width], dtype=mx.int32)],
        template=[("T", mx.float32)], grid=(total, 1, 1), threadgroup=(min(256, total), 1, 1),
        output_shapes=[(rows, width)], output_dtypes=[mx.float32],
    )[0]


def absorb_h1e(h1e: np.ndarray, eri: np.ndarray, nelec: tuple[int, int], fac: float = 0.5) -> np.ndarray:
    """PySCF-compatible one-body absorption without a Torch dependency."""
    h2e = np.array(eri, dtype=np.float64, copy=True)
    f1e = (h1e - 0.5 * np.einsum("jiik->jk", h2e)) / float(sum(nelec))
    for orbital in range(h1e.shape[0]):
        h2e[orbital, orbital, :, :] += f1e
        h2e[:, :, orbital, orbital] += f1e
    return np.ascontiguousarray(h2e * fac)


@dataclass(frozen=True)
class MetalLinkTable:
    pair: Any
    outer: Any
    address: Any
    sign: Any
    n_outer: int
    gather_slot: Any


@dataclass(frozen=True)
class MetalSlotMap:
    address: Any
    sign: Any


@dataclass(frozen=True)
class MetalMixedChunk:
    pair: Any
    outer: Any
    address: Any
    sign: Any


def _convert_link_table(link_index: np.ndarray | None, n_rows: int) -> MetalLinkTable:
    mx = _mlx()
    if link_index is None:
        empty = mx.array([], dtype=mx.int32)
        return MetalLinkTable(empty, empty, empty, empty.astype(mx.float32), 0, empty)
    raw = np.asarray(link_index)
    valid = raw[..., 3] != 0
    n_outer, n_link = raw.shape[:2]
    outer_grid = np.broadcast_to(np.arange(n_outer, dtype=np.int32)[:, None], (n_outer, n_link))
    pair = np.ascontiguousarray(raw[..., 0][valid], dtype=np.int32)
    outer = np.ascontiguousarray(outer_grid[valid], dtype=np.int32)
    address = np.ascontiguousarray(raw[..., 2][valid], dtype=np.int32)
    sign = np.ascontiguousarray(raw[..., 3][valid], dtype=np.float32)
    pair_mx, outer_mx = mx.array(pair), mx.array(outer)
    return MetalLinkTable(
        pair=pair_mx,
        outer=outer_mx,
        address=mx.array(address),
        sign=mx.array(sign),
        n_outer=n_outer,
        gather_slot=pair_mx * n_rows + outer_mx,
    )


def _make_slot_map(links: MetalLinkTable, total_rows: int) -> MetalSlotMap:
    """Build the complete gather layout once; unused pair/outer slots are zero."""
    mx = _mlx()
    address = np.zeros(total_rows, dtype=np.int32)
    sign = np.zeros(total_rows, dtype=np.float32)
    if total_rows and int(links.address.size):
        slots = _to_numpy(links.gather_slot).astype(np.int64, copy=False)
        address[slots] = _to_numpy(links.address)
        sign[slots] = _to_numpy(links.sign)
    result = MetalSlotMap(mx.array(address), mx.array(sign))
    mx.eval(result.address, result.sign)
    return result


def _occupation(strings: np.ndarray, norb: int) -> Any:
    mx = _mlx()
    encoded = mx.array(np.asarray(strings, dtype=np.int64))
    orbitals = mx.arange(norb, dtype=mx.int64)
    return ((encoded[:, None] >> orbitals[None, :]) & 1).astype(mx.float32)


@metal_gpu_execution
def make_hdiag(h1e: Any, eri: Any, ci_strings: tuple[np.ndarray, np.ndarray]) -> Any:
    """Construct selected-CI diagonal elements with MLX array operations."""
    mx = _mlx()
    norb = h1e.shape[0]
    occ_a, occ_b = _occupation(ci_strings[0], norb), _occupation(ci_strings[1], norb)
    one_body = mx.diagonal(h1e)
    coulomb = mx.einsum("iijj->ij", eri)
    exchange = mx.einsum("ijji->ij", eri)
    same_spin = coulomb - exchange
    energy_a = _metal_matmul(occ_a, one_body[:, None]).reshape(-1)
    energy_a = energy_a + 0.5 * mx.sum(occ_a * _metal_matmul(occ_a, same_spin), axis=1)
    energy_b = _metal_matmul(occ_b, one_body[:, None]).reshape(-1)
    energy_b = energy_b + 0.5 * mx.sum(occ_b * _metal_matmul(occ_b, same_spin), axis=1)
    opposite = _metal_matmul(_metal_matmul(occ_a, coulomb), occ_b.T)
    return energy_a[:, None] + energy_b[None, :] + opposite


def _single_connections(cd: np.ndarray, norb: int):
    """将单激发按输出串压缩。对角连接另存为轨道占据，用于预收缩。"""
    mx = _mlx()
    row, slot = np.nonzero(cd[..., 3])
    entries = cd[row, slot]
    same = row == entries[:, 2]
    orbitals = np.arange(norb)
    diag_pairs = orbitals * (orbitals + 3) // 2
    occ = np.zeros((len(cd), norb))
    occ[row[same], np.searchsorted(diag_pairs, entries[same, 0])] = 1
    off = entries[~same]
    degree = np.bincount(row[~same], minlength=len(cd))
    offsets = np.r_[0, np.cumsum(degree)].astype(np.int32)
    links = np.column_stack([off[:, 2], off[:, 3] * (off[:, 0] + 1)]).astype(np.int32)
    # Metal 要求每个输入有可绑定的非空缓冲；空连接的 offset 均为零。
    return occ, (mx.array(offsets), mx.array(links if links.size else np.zeros((1, 2), np.int32)))


def _compress_double_links(dd):
    """只保存非空的双湮灭连接，避免在每次更新积分时复制补零的大表。"""
    if dd is None:
        return None
    valid = dd[..., 3] != 0
    degree = np.count_nonzero(valid, axis=1)
    offsets = np.r_[0, np.cumsum(degree)]
    return degree, offsets, dd[valid]


def _same_spin_csr(dd, pairs: np.ndarray, size: int):
    """对具有相同双湮灭中间态的连接求积并累加，得到单自旋 CSR 矩阵。"""
    from scipy.sparse import coo_matrix, csr_matrix

    mx = _mlx()
    matrix = csr_matrix((size, size), dtype=np.float64)
    if dd is not None:
        degree, offsets, entries = dd
        # 按行长度分组并分块，避免将所有中间态补齐到最大连接数。
        for width in np.unique(degree):
            if width == 0:
                continue
            rows = np.flatnonzero(degree == width)
            step = max(1, 262144 // int(width * width))
            for start in range(0, len(rows), step):
                starts = offsets[rows[start:start + step]]
                links = entries[starts[:, None] + np.arange(width)]
                address, pair, sign = links[:, :, 2], links[:, :, 0], links[:, :, 3]
                values = pairs[pair[:, :, None], pair[:, None, :]] * sign[:, :, None] * sign[:, None, :]
                left = np.broadcast_to(address[:, :, None], values.shape)
                right = np.broadcast_to(address[:, None, :], values.shape)
                matrix += coo_matrix((values.ravel(), (left.ravel(), right.ravel())), shape=matrix.shape).tocsr()
    diagonal = matrix.diagonal()
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    columns = matrix.indices if matrix.nnz else np.zeros(1, np.int32)
    values = matrix.data if matrix.nnz else np.zeros(1)
    return diagonal, (mx.array(matrix.indptr.astype(np.int32)), mx.array(columns.astype(np.int32)),
                      mx.array(values.astype(np.float32)))


class MetalSelectedCIContractor:
    """Restricted selected-CI H@c with alpha, beta, and mixed-spin excitations."""

    @metal_gpu_execution
    def __init__(self, h2e: np.ndarray, link_index: Any, nelec: tuple[int, int], *, chunk_size: int = 32, strategy: str = "auto") -> None:
        mx = _mlx()
        if h2e.ndim != 4 or len(set(h2e.shape)) != 1:
            raise ValueError("h2e must be a full (norb, norb, norb, norb) tensor")
        if min(nelec) < 1:
            raise ValueError("MetalSelectedCIContractor requires nonempty spin sectors")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.norb = int(h2e.shape[0])
        self.nelec = tuple(nelec)
        self.chunk_size = int(chunk_size)
        self.native_calls = {"h_matvec": 0, "metal_gather": 0, "same_spin": 0, "mixed_spin": 0}
        self.execution: MetalExecution = {"device": None, "stream": None, "completed_evaluations": 0}
        cd_a, dd_a, cd_b, dd_b = link_index
        self.n_alpha, self.n_beta = int(cd_a.shape[0]), int(cd_b.shape[0])
        if strategy not in ("auto", "connected", "pair"):
            raise ValueError("Metal contraction strategy must be auto, connected, or pair")
        # 每项对应一对经过同一双湮灭中间态的连接。限制 CPU 建表开销。
        self.strategy = strategy
        if strategy == "auto":
            terms = sum(int(np.square(np.count_nonzero(dd[..., 3], axis=1)).sum())
                        for dd in (dd_a, dd_b) if dd is not None)
            pair_count = self.norb * (self.norb - 1) // 2
            max_buffer_bytes = max(
                (0 if dd is None else len(dd)) * min(width, self.chunk_size) * pair_count * 4
                for dd, width in ((dd_a, self.n_beta), (dd_b, self.n_alpha)))
            # 若单个补零缓冲就超过 256 MiB，即使建 CSR 较慢仍采用压缩连接。
            self.strategy = "connected" if terms <= 4_000_000 or max_buffer_bytes > 256 * 2**20 else "pair"
        if self.strategy == "connected":
            self._occup_a, self._single_a = _single_connections(cd_a, self.norb)
            self._occup_b, self._single_b = _single_connections(cd_b, self.norb)
            self._dd = _compress_double_links(dd_a), _compress_double_links(dd_b)
            self.update_integrals(h2e)
            return
        self.cd_a = _convert_link_table(cd_a, self.n_alpha)
        self.dd_a = _convert_link_table(dd_a, 0 if dd_a is None else int(dd_a.shape[0]))
        self.cd_b = _convert_link_table(cd_b, self.n_beta)
        self.dd_b = _convert_link_table(dd_b, 0 if dd_b is None else int(dd_b.shape[0]))
        self.mixed_chunks = self._build_mixed_chunks()
        self.update_integrals(h2e)
        same_pairs = int(self.same_spin_matrix.shape[0])
        mixed_pairs = int(self.mixed_spin_matrix.shape[0])
        self._alpha_slots = _make_slot_map(self.dd_a, same_pairs * self.dd_a.n_outer)
        self._beta_slots = _make_slot_map(self.dd_b, same_pairs * self.dd_b.n_outer)
        self._mixed_slots = _make_slot_map(self.cd_a, mixed_pairs * self.n_alpha)


    @metal_gpu_execution
    def update_integrals(self, h2e: np.ndarray) -> None:
        mx = _mlx()
        if self.strategy == "connected":
            self._prepare_connected(np.asarray(h2e, dtype=np.float64))
            return
        h2e_mx = mx.array(np.asarray(h2e, dtype=np.float32))
        n2 = self.norb * self.norb
        rows, columns = np.tril_indices(self.norb, -1)
        offdiag = mx.array(np.asarray(rows * self.norb + columns, dtype=np.int32))
        antisym = mx.transpose(h2e_mx, (0, 2, 1, 3)) - mx.transpose(h2e_mx, (0, 2, 3, 1))
        antisym_2d = mx.reshape(antisym, (n2, n2))
        self.same_spin_matrix = 2.0 * antisym_2d[offdiag[:, None], offdiag[None, :]]

        h_ps = mx.einsum("pqqs->ps", h2e_mx)
        mixed_full = 2.0 * h2e_mx
        for orbital in range(self.norb):
            mixed_full = mixed_full.at[orbital, orbital, :, :].add(h_ps / float(self.nelec[1]))
            mixed_full = mixed_full.at[:, :, orbital, orbital].add(h_ps / float(self.nelec[0]))
        lower_rows, lower_columns = np.tril_indices(self.norb, 0)
        lower = mx.array(np.asarray(lower_rows * self.norb + lower_columns, dtype=np.int32))
        mixed_2d = mx.reshape(mixed_full, (n2, n2))
        self.mixed_spin_matrix = mixed_2d[lower[:, None], lower[None, :]]
        mx.eval(self.same_spin_matrix, self.mixed_spin_matrix)

    def _prepare_connected(self, h2e: np.ndarray) -> None:
        """将同自旋双激发、对角占据的贡献预收缩；不构造乘积空间矩阵。"""
        mx = _mlx()
        p, q = np.tril_indices(self.norb, -1)
        antisym = 2 * (h2e[p[:, None], p, q[:, None], q] - h2e[p[:, None], q, q[:, None], p])
        da, self._same_a = _same_spin_csr(self._dd[0], antisym, self.n_alpha)
        db, self._same_b = _same_spin_csr(self._dd[1], antisym, self.n_beta)
        p, q = np.tril_indices(self.norb)
        diagonal_pairs = np.flatnonzero(p == q)
        mixed = 2 * h2e[p[:, None], q[:, None], p, q]
        trace = np.einsum("pqqs->ps", h2e)[p, q]
        mixed[diagonal_pairs, :] += trace[None, :] / self.nelec[1]
        mixed[:, diagonal_pairs] += trace[:, None] / self.nelec[0]
        # 旧核的左乘行属于 beta、列属于 alpha。在此转为先 alpha 后 beta。
        mixed = np.ascontiguousarray(mixed.T)
        field_a = self._occup_a @ mixed[diagonal_pairs, :]
        field_b = self._occup_b @ mixed[:, diagonal_pairs].T
        diagonal = da[:, None] + db[None, :] + field_a[:, diagonal_pairs] @ self._occup_b.T
        self._diagonal = mx.array(diagonal.astype(np.float32))
        self._field_a = mx.array(field_a.astype(np.float32))
        self._field_b = mx.array(field_b.astype(np.float32))
        self._mixed = mx.array(mixed.astype(np.float32))
        self._dimensions = mx.array([self.n_beta, len(p)], dtype=mx.uint32)
        mx.eval(self._diagonal, self._field_a, self._field_b, self._mixed, *self._same_a, *self._same_b)

    def _connected_product(self, coefficients: Any) -> Any:
        """每个 SIMD 组负责一个输出行列式，组内归约，不需要原子写回。"""
        mx = _mlx()
        self.native_calls["metal_connected_rows"] = self.native_calls.get("metal_connected_rows", 0) + 1
        return _kernels()["connected_rows"](
            inputs=[coefficients, self._diagonal, *self._single_a, *self._single_b,
                    *self._same_a, *self._same_b, self._field_a, self._field_b, self._mixed, self._dimensions],
            grid=(32 * self.n_beta, self.n_alpha, 1), threadgroup=(32 * min(8, self.n_beta), 1, 1),
            output_shapes=[(self.n_alpha, self.n_beta)], output_dtypes=[mx.float32],
        )[0]

    def _build_mixed_chunks(self) -> tuple[MetalMixedChunk, ...]:
        mx = _mlx()
        cd_b_address = _to_numpy(self.cd_b.address)
        cd_b_pair = _to_numpy(self.cd_b.pair)
        cd_b_outer = _to_numpy(self.cd_b.outer)
        cd_b_sign = _to_numpy(self.cd_b.sign)
        chunks = []
        for start in range(0, self.n_beta, self.chunk_size):
            stop = min(start + self.chunk_size, self.n_beta)
            width = stop - start
            mask = (cd_b_outer >= start) & (cd_b_outer < stop)
            pair = cd_b_pair[mask]
            outer = cd_b_outer[mask] - start
            chunks.append(MetalMixedChunk(
                pair=mx.array(pair), outer=mx.array(outer),
                address=mx.array(cd_b_address[mask]), sign=mx.array(cd_b_sign[mask]),
            ))
        return tuple(chunks)

    def _gather_rows(self, coefficients: Any, slots: MetalSlotMap) -> Any:
        mx = _mlx()
        width = int(coefficients.shape[1])
        rows = int(slots.address.size)
        if rows == 0 or width == 0:
            return mx.zeros((rows, width), dtype=mx.float32)
        total = rows * width
        self.native_calls["metal_gather"] += 1
        return _kernels()["gather_slots"](
            inputs=[coefficients, slots.address, slots.sign, mx.array([width], dtype=mx.int32)],
            template=[("T", mx.float32)],
            grid=(total, 1, 1), threadgroup=(min(256, total), 1, 1),
            output_shapes=[(rows, width)], output_dtypes=[mx.float32],
        )[0]

    def _pair_contract(self, matrix: Any, temporary: Any, pair: Any, outer: Any,
                       sign: Any, width: int, alpha_count: int | None = None) -> Any:
        mx = _mlx()
        inner = int(matrix.shape[1])
        temporary_width = int(temporary.shape[1])
        self.native_calls["metal_pair_contract"] = self.native_calls.get("metal_pair_contract", 0) + 1
        if alpha_count is None:
            total = int(pair.size) * width
            return _kernels()["pair_contract_links"](
                inputs=[matrix, temporary, pair, outer, sign, mx.array([inner], dtype=mx.int32),
                        mx.array([width], dtype=mx.int32), mx.array([temporary_width], dtype=mx.int32)],
                template=[("T", mx.float32)], grid=(total, 1, 1), threadgroup=(min(256, total), 1, 1),
                output_shapes=[(int(pair.size), width)], output_dtypes=[mx.float32],
            )[0]
        total = int(pair.size) * alpha_count
        return _kernels()["pair_contract_mixed"](
            inputs=[matrix, temporary, pair, outer, sign, mx.array([inner], dtype=mx.int32),
                    mx.array([width], dtype=mx.int32), mx.array([temporary_width], dtype=mx.int32),
                    mx.array([alpha_count], dtype=mx.int32)],
            template=[("T", mx.float32)], grid=(total, 1, 1), threadgroup=(min(256, total), 1, 1),
            output_shapes=[(int(pair.size), alpha_count)], output_dtypes=[mx.float32],
        )[0]

    def _same_spin(self, coefficients: Any, links: MetalLinkTable, slots: MetalSlotMap) -> Any:
        mx = _mlx()
        if links.n_outer == 0:
            return mx.zeros_like(coefficients)
        self.native_calls["same_spin"] += 1
        n_pairs = int(self.same_spin_matrix.shape[0])
        output = mx.zeros_like(coefficients)
        for start in range(0, coefficients.shape[1], self.chunk_size):
            stop = min(start + self.chunk_size, coefficients.shape[1])
            temporary = self._gather_rows(coefficients[:, start:stop], slots)
            contributions = self._pair_contract(
                self.same_spin_matrix, temporary.reshape(n_pairs, -1),
                links.pair, links.outer, links.sign, stop - start,
            )
            block = output[:, start:stop].at[links.address].add(contributions)
            output = mx.concatenate([output[:, :start], block, output[:, stop:]], axis=1) if start or stop < output.shape[1] else block
        return output

    def _mixed_spin(self, coefficients: Any) -> Any:
        mx = _mlx()
        self.native_calls["mixed_spin"] += 1
        n_pairs = int(self.mixed_spin_matrix.shape[0])
        output_t = mx.zeros((self.n_beta, self.n_alpha), dtype=mx.float32)
        for index, start in enumerate(range(0, self.n_beta, self.chunk_size)):
            stop = min(start + self.chunk_size, self.n_beta)
            width = stop - start
            temporary = self._gather_rows(coefficients[:, start:stop], self._mixed_slots)
            chunk = self.mixed_chunks[index]
            contributions = self._pair_contract(
                self.mixed_spin_matrix, temporary.reshape(n_pairs, -1),
                chunk.pair, chunk.outer, chunk.sign, width, alpha_count=self.n_alpha,
            )
            output_t = output_t.at[chunk.address].add(contributions)
        return output_t.T

    @metal_gpu_execution
    def contract(self, coefficients: Any) -> Any:
        mx = _mlx()
        if tuple(coefficients.shape) != (self.n_alpha, self.n_beta):
            raise ValueError(f"coefficients must have shape {(self.n_alpha, self.n_beta)}")
        self.native_calls["h_matvec"] += 1
        _record_execution(self, mx)
        if self.strategy == "connected":
            result = self._connected_product(coefficients)
        else:
            result = self._mixed_spin(coefficients)
            if self.nelec[0] > 1:
                result = result + self._same_spin(coefficients, self.dd_a, self._alpha_slots)
            if self.nelec[1] > 1:
                result = result + self._same_spin(coefficients.T, self.dd_b, self._beta_slots).T
        mx.eval(result)
        self.execution["completed_evaluations"] += 1
        return result


class MetalSelectedS2Contractor:
    """只遍历在选定基内交换两个轨道自旋产生的连接，计算投影 S²c。"""

    @metal_gpu_execution
    def __init__(self, ci_strings: tuple[np.ndarray, np.ndarray], norb: int, nelec: tuple[int, int]) -> None:
        mx = _mlx()
        alpha, beta = (np.ascontiguousarray(x, dtype=np.int64) for x in ci_strings)
        self.shape = len(alpha), len(beta)
        self.native_calls = {"spin_matvec": 0, "metal_spin_exchange": 0}
        self.execution: MetalExecution = {"device": None, "stream": None, "completed_evaluations": 0}
        self._alpha_bits = mx.array(alpha.view(np.uint32))
        self._beta_bits = mx.array(beta.view(np.uint32))
        # 对 alpha 的 q→p 单激发，beta 必须恰好有反向 p→q 连接。
        alpha_table = selected_ci.cre_des_linkstr(alpha, norb, nelec[0])
        beta_table = selected_ci.cre_des_linkstr(beta, norb, nelec[1])
        row, slot = np.nonzero((alpha_table[..., 3] != 0) & (alpha_table[..., 0] != alpha_table[..., 1]))
        moves = alpha_table[row, slot]
        reverse = moves[:, 1] * norb + moves[:, 0]
        packed = np.column_stack(((moves[:, 2] + 1) * moves[:, 3], reverse))
        self._moves = mx.array(packed.astype(np.int32) if moves.size else np.zeros((1, 2), np.int32))
        self._offsets = mx.array(np.r_[0, np.cumsum(np.bincount(row, minlength=len(alpha)))].astype(np.int32))
        row, slot = np.nonzero((beta_table[..., 3] != 0) & (beta_table[..., 0] != beta_table[..., 1]))
        moves = beta_table[row, slot]
        lookup = np.zeros((len(beta), norb * norb), dtype=np.int32)
        lookup[row, moves[:, 0] * norb + moves[:, 1]] = (moves[:, 2] + 1) * moves[:, 3]
        self._lookup = mx.array(lookup)
        self._dimensions = mx.array([len(beta), norb * norb], dtype=mx.uint32)
        self._constant = mx.array([.25 * (nelec[0] - nelec[1]) ** 2 + .5 * sum(nelec)], dtype=mx.float32)

    @metal_gpu_execution
    def contract(self, coefficients: Any) -> Any:
        mx = _mlx()
        matrix = coefficients.reshape(self.shape)
        _record_execution(self, mx)
        total = int(matrix.size)
        result = _kernels()["spin_exchange"](
            inputs=[matrix, self._alpha_bits, self._beta_bits, self._offsets, self._moves,
                    self._lookup, self._dimensions, self._constant],
            grid=(total, 1, 1), threadgroup=(min(total, 256), 1, 1),
            output_shapes=[self.shape], output_dtypes=[mx.float32],
        )[0]
        mx.eval(result)
        self.native_calls["spin_matvec"] += 1
        self.native_calls["metal_spin_exchange"] += 1
        self.execution["completed_evaluations"] += 1
        return result


@metal_gpu_execution
def make_spin_square_diagonal(ci_strings: tuple[np.ndarray, np.ndarray], norb: int, nelec: tuple[int, int]) -> Any:
    mx = _mlx()
    occ_a, occ_b = _occupation(ci_strings[0], norb), _occupation(ci_strings[1], norb)
    double = _metal_matmul(occ_a, occ_b.T)
    sz_square = 0.25 * float((nelec[0] - nelec[1]) ** 2)
    return sz_square + 0.5 * (float(sum(nelec)) - 2.0 * double)


@dataclass
class MetalDavidsonResult:
    energy: float
    coefficients: Any
    residual_norm: float
    converged: bool
    iterations: int
    h_matvecs: int
    history: list[dict[str, float | int]]


def _orthogonalize(vector: Any, basis: Any) -> Any:
    result = vector
    for _ in range(2):
        overlaps = _metal_matmul(basis.T, result[:, None])
        result = result - _metal_matmul(basis, overlaps).reshape(-1)
    return result


@metal_gpu_execution
def davidson_lowest(
    matvec: Callable[[Any], Any], diagonal: Any, *, initial_vector: Any | None = None,
    tolerance: float = 1e-6, max_cycles: int = 100, max_space: int = 20,
    denominator_floor: float = 1e-4,
    callback: Callable[[dict[str, float | int]], None] | None = None,
) -> MetalDavidsonResult:
    """MLX Davidson with device vectors and host FP64 small Hermitian eigh."""
    mx = _mlx()
    if tolerance <= 0 or max_cycles < 1 or max_space < 2:
        raise ValueError("invalid Davidson limits")
    if initial_vector is None:
        initial = mx.zeros_like(diagonal).at[int(mx.argmin(diagonal).item())].add(1.0)
    else:
        norm = mx.linalg.norm(initial_vector)
        if _eval_scalar(norm) <= 0:
            raise ValueError("initial_vector must be nonzero")
        initial = initial_vector / norm
    capacity = min(max_space, int(diagonal.size))
    basis, images = [initial], []
    projection_cache = np.empty((capacity, capacity))
    cached = 0
    h_matvecs = 0
    coefficients = initial
    energy, residual_norm = float("nan"), float("inf")
    history: list[dict[str, float | int]] = []
    for cycle in range(1, max_cycles + 1):
        while len(images) < len(basis):
            images.append(matvec(basis[len(images)]).reshape(-1))
            h_matvecs += 1
        basis_matrix = mx.stack(basis, axis=1)
        image_matrix = mx.stack(images, axis=1)
        nvec = len(basis)
        if cached and nvec == cached + 1:
            # 原有 Ritz 子矩阵不变；只计算新列及其厄米共轭，保持显式对称化。
            last_col = _metal_matmul(basis_matrix.T, images[-1][:, None])
            last_row = _metal_matmul(image_matrix.T, basis[-1][:, None])
            edge = _to_numpy(0.5 * (last_col + last_row)).reshape(-1)
            projection_cache[:nvec, cached] = edge
            projection_cache[cached, :nvec] = edge
        else:
            projected = _metal_matmul(basis_matrix.T, image_matrix)
            projection_cache[:nvec, :nvec] = _to_numpy(0.5 * (projected + projected.T))
        cached = nvec
        eigenvalues, eigenvectors = np.linalg.eigh(projection_cache[:nvec, :nvec])
        energy = float(eigenvalues[0])
        weights = mx.array(eigenvectors[:, 0].astype(np.float32))
        coefficients = _metal_matmul(basis_matrix, weights[:, None]).reshape(-1)
        image = _metal_matmul(image_matrix, weights[:, None]).reshape(-1)
        residual = image - float(energy) * coefficients
        residual_norm_value = mx.linalg.norm(residual)
        subspace_dimension = len(basis)
        residual_norm = _eval_scalar(residual_norm_value)
        record: dict[str, float | int] = {
            "cycle": cycle, "subspace_dimension": subspace_dimension,
            "energy": energy, "residual_norm": residual_norm,
        }
        history.append(record)
        if callback is not None:
            callback(dict(record))
        if residual_norm <= tolerance:
            return MetalDavidsonResult(energy, coefficients / mx.linalg.norm(coefficients), residual_norm,
                                       True, cycle, h_matvecs, history)
        denominator = float(energy) - diagonal
        signs = mx.where(denominator >= 0, mx.ones_like(denominator), -mx.ones_like(denominator))
        denominator = mx.where(mx.abs(denominator) < denominator_floor, signs * denominator_floor, denominator)
        correction = residual / denominator
        if len(basis) >= capacity:
            retained_count = min(max(2, capacity // 4), 8, capacity - 1)
            retained = mx.array(eigenvectors[:, :retained_count].astype(np.float32))
            retained_basis = _metal_matmul(basis_matrix, retained)
            retained_images = _metal_matmul(image_matrix, retained)
            cached = 0
            basis = [retained_basis[:, i] for i in range(retained_count)]
            images = [retained_images[:, i] for i in range(retained_count)]
            basis_matrix = mx.stack(basis, axis=1)
        correction = _orthogonalize(correction, basis_matrix)
        correction_norm = mx.linalg.norm(correction)
        if float(correction_norm.item()) <= 1e-12:
            correction = _orthogonalize(residual, basis_matrix)
            correction_norm = mx.linalg.norm(correction)
            if _eval_scalar(correction_norm) <= 1e-12:
                break
        basis.append(correction / correction_norm)
    return MetalDavidsonResult(energy, coefficients / mx.linalg.norm(coefficients), residual_norm,
                               False, len(history), h_matvecs, history)


__all__ = [
    "MetalDavidsonResult", "MetalSelectedCIContractor", "MetalSelectedS2Contractor",
    "absorb_h1e", "davidson_lowest", "make_hdiag", "make_spin_square_diagonal",
    "metal_gpu_execution",
]
