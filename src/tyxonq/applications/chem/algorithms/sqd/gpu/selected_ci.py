"""使用 PyTorch 计算实数 restricted selected-CI 的 Hamiltonian 算符像 Hc。

按 aa、ab、bb 自旋块收缩，一体项先以 fac=0.5 吸收到二体张量中。
行列式连接表在 CPU 上预计算，大数组运算可在 CPU 或 CUDA 设备执行。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_SAME_SPIN_BUFFER_BYTES = 3 * 1024**3


@dataclass(frozen=True)
class TorchLinkTable:
    """把 PySCF 的压缩 link table 展平成适合 gather/scatter 的张量。"""

    pair: torch.Tensor
    outer: torch.Tensor
    address: torch.Tensor
    sign: torch.Tensor
    n_outer: int


@dataclass(frozen=True)
class TorchMixedLinkChunk:
    """一次 beta 分块所需的恒定 link 元数据。"""

    pair: torch.Tensor
    outer: torch.Tensor
    address: torch.Tensor
    sign: torch.Tensor


def _occupation_matrix(
    strings,
    norb: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """把整数 determinant 编码转换为 ``(n_strings, norb)`` 占据矩阵。"""

    encoded = torch.as_tensor(strings, dtype=torch.int64, device=device)
    orbitals = torch.arange(norb, dtype=torch.int64, device=device)
    return ((encoded[:, None] >> orbitals[None, :]) & 1).to(dtype=dtype)


def make_hdiag_torch(
    h1e: torch.Tensor,
    eri: torch.Tensor,
    ci_strings,
) -> torch.Tensor:
    """用 PyTorch 构造 restricted selected-CI Hamiltonian 对角元。

    返回 shape 为 ``(n_alpha_strings, n_beta_strings)`` 的设备张量。
    determinant 整数只作为离散元数据传入；占据矩阵、Coulomb/exchange
    收缩和 alpha-beta 外积都在当前 PyTorch device 上完成。
    """

    if h1e.is_complex() or eri.is_complex():
        raise NotImplementedError("当前 benchmark 只支持实数 Hamiltonian")
    norb = h1e.shape[0]
    if h1e.shape != (norb, norb) or eri.shape != (norb, norb, norb, norb):
        raise ValueError("h1e/eri shape 与 norb 不一致")

    occ_a = _occupation_matrix(
        ci_strings[0], norb, device=h1e.device, dtype=h1e.dtype
    )
    occ_b = _occupation_matrix(
        ci_strings[1], norb, device=h1e.device, dtype=h1e.dtype
    )
    one_body_diagonal = torch.diagonal(h1e)
    coulomb = torch.einsum("iijj->ij", eri)
    exchange = torch.einsum("ijji->ij", eri)
    same_spin = coulomb - exchange

    energy_a = occ_a @ one_body_diagonal
    energy_a = energy_a + 0.5 * torch.einsum(
        "pi,ij,pj->p", occ_a, same_spin, occ_a
    )
    energy_b = occ_b @ one_body_diagonal
    energy_b = energy_b + 0.5 * torch.einsum(
        "pi,ij,pj->p", occ_b, same_spin, occ_b
    )
    opposite_spin = (occ_a @ coulomb) @ occ_b.transpose(0, 1)
    return energy_a[:, None] + energy_b[None, :] + opposite_spin


def _convert_link_table(
    link_index,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> TorchLinkTable:
    """把 ``[..., (pair, unused, address, sign)]`` 转为设备张量。"""

    if link_index is None:
        empty_index = torch.empty(0, dtype=torch.int64, device=device)
        empty_value = torch.empty(0, dtype=dtype, device=device)
        return TorchLinkTable(empty_index, empty_index, empty_index, empty_value, 0)

    raw = torch.as_tensor(link_index).to(device=device)
    valid = raw[..., 3] != 0
    n_outer, n_link = raw.shape[:2]
    outer_grid = (
        torch.arange(n_outer, dtype=torch.int64, device=device)
        .view(-1, 1)
        .expand(n_outer, n_link)
    )
    return TorchLinkTable(
        pair=raw[..., 0][valid].to(torch.int64),
        outer=outer_grid[valid],
        address=raw[..., 2][valid].to(torch.int64),
        sign=raw[..., 3][valid].to(dtype=dtype),
        n_outer=n_outer,
    )


def absorb_h1e_torch(
    h1e: torch.Tensor,
    eri: torch.Tensor,
    nelec: tuple[int, int],
    *,
    fac: float = 0.5,
) -> torch.Tensor:
    """用 PyTorch 复现 PySCF ``direct_spin1.absorb_h1e`` 的完整张量形式。"""

    if h1e.is_complex() or eri.is_complex():
        raise NotImplementedError("当前 benchmark 只支持实数 Hamiltonian")
    norb = h1e.shape[0]
    if h1e.shape != (norb, norb) or eri.shape != (norb, norb, norb, norb):
        raise ValueError("h1e/eri shape 与 norb 不一致")

    h2e = eri.clone()
    f1e = h1e - 0.5 * torch.einsum("jiik->jk", h2e)
    f1e = f1e / float(sum(nelec))
    for orbital in range(norb):
        h2e[orbital, orbital, :, :] += f1e
        h2e[:, :, orbital, orbital] += f1e
    return h2e * fac


class TorchSelectedCIContractor:
    """在 PyTorch CPU/CUDA 上执行 restricted selected-CI ``H @ c``。"""

    def __init__(
        self,
        h2e: torch.Tensor,
        link_index,
        nelec: tuple[int, int],
        *,
        chunk_size: int = 32,
    ) -> None:
        if h2e.ndim != 4 or len(set(h2e.shape)) != 1:
            raise ValueError("h2e 必须是 shape=(norb,norb,norb,norb) 的完整张量")
        if h2e.dtype not in (torch.float32, torch.float64):
            raise ValueError("benchmark 要求 h2e 使用 torch.float32 或 torch.float64")
        if chunk_size < 1:
            raise ValueError("chunk_size 必须为正整数")
        if min(nelec) < 1:
            raise ValueError("TorchSelectedCIContractor requires nonempty spin sectors; "
                             "use solve_sci_cuda for zero-sector spectator handling.")

        self.device = h2e.device
        self.dtype = h2e.dtype
        self.norb = h2e.shape[0]
        self.nelec = nelec
        self.chunk_size = chunk_size

        cd_indexa, dd_indexa, cd_indexb, dd_indexb = link_index
        self.cd_a = _convert_link_table(
            cd_indexa, device=self.device, dtype=self.dtype
        )
        self.dd_a = _convert_link_table(
            dd_indexa, device=self.device, dtype=self.dtype
        )
        self.cd_b = _convert_link_table(
            cd_indexb, device=self.device, dtype=self.dtype
        )
        self.dd_b = _convert_link_table(
            dd_indexb, device=self.device, dtype=self.dtype
        )

        self.n_alpha = int(cd_indexa.shape[0])
        self.n_beta = int(cd_indexb.shape[0])
        # link table 与 chunk_size 在整个 Davidson 求解期间都不会变化。
        # 这些索引原先在每次 H @ c 中重复构造，会触发不必要的设备布尔索引
        # 和 D2H/H2D 同步；现在统一移到一次性的 setup 阶段。
        self.cd_a_gather_slot = self.cd_a.pair * self.n_alpha + self.cd_a.outer
        self.dd_a_gather_slot = self.dd_a.pair * self.dd_a.n_outer + self.dd_a.outer
        self.dd_b_gather_slot = self.dd_b.pair * self.dd_b.n_outer + self.dd_b.outer
        self.mixed_beta_chunks = self._build_mixed_beta_chunks()
        self.same_spin_matrix, self.mixed_spin_matrix = self._build_pair_matrices(h2e)

    def _build_mixed_beta_chunks(self) -> tuple[TorchMixedLinkChunk, ...]:
        """按固定 chunk_size 预计算 beta link 切片。"""

        chunks: list[TorchMixedLinkChunk] = []
        for start in range(0, self.n_beta, self.chunk_size):
            stop = min(start + self.chunk_size, self.n_beta)
            in_chunk = (self.cd_b.outer >= start) & (self.cd_b.outer < stop)
            chunks.append(
                TorchMixedLinkChunk(
                    pair=self.cd_b.pair[in_chunk],
                    outer=self.cd_b.outer[in_chunk] - start,
                    address=self.cd_b.address[in_chunk],
                    sign=self.cd_b.sign[in_chunk],
                )
            )
        return tuple(chunks)

    def _build_pair_matrices(
        self, h2e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """构造与 PySCF 三角 pair 顺序一致的两个积分矩阵。"""

        norb = self.norb
        n2 = norb * norb

        # 同自旋部分只需要严格下三角 orbital pairs：i > j。
        lower_offdiag = torch.tril_indices(
            norb, norb, offset=-1, device=self.device
        )
        flat_offdiag = lower_offdiag[0] * norb + lower_offdiag[1]
        antisym = h2e.permute(0, 2, 1, 3) - h2e.permute(0, 2, 3, 1)
        antisym_2d = antisym.reshape(n2, n2)
        same_spin = (
            antisym_2d.index_select(0, flat_offdiag)
            .index_select(1, flat_offdiag)
            .mul(2.0)
            .contiguous()
        )

        # 异自旋部分使用包含对角线的下三角 orbital pairs：i >= j。
        h_ps = torch.einsum("pqqs->ps", h2e)
        mixed_full = h2e.mul(2.0)
        for orbital in range(norb):
            # 与 PySCF selected_ci.contract_2e 一致：第一对角 pair 用 N_beta，
            # 第二对角 pair 用 N_alpha。闭壳层相等时看不出这两个分母的顺序。
            mixed_full[orbital, orbital, :, :] += h_ps / float(self.nelec[1])
            mixed_full[:, :, orbital, orbital] += h_ps / float(self.nelec[0])
        lower = torch.tril_indices(norb, norb, offset=0, device=self.device)
        flat_lower = lower[0] * norb + lower[1]
        mixed_2d = mixed_full.reshape(n2, n2)
        mixed_spin = (
            mixed_2d.index_select(0, flat_lower)
            .index_select(1, flat_lower)
            .contiguous()
        )
        return same_spin, mixed_spin

    def _contract_same_spin_rows(
        self,
        coefficients: torch.Tensor,
        links: TorchLinkTable,
        gather_slot: torch.Tensor,
    ) -> torch.Tensor:
        """收缩一个自旋块；行是目标自旋，列是旁观自旋。"""

        if links.n_outer == 0:
            return torch.zeros_like(coefficients)

        n_rows, n_columns = coefficients.shape
        n_pairs = self.same_spin_matrix.shape[0]
        output = torch.zeros_like(coefficients)

        # 更大活性空间的两电子 link 池可能增长；单个稠密工作张量限制为 3 GiB。
        # 只细分旁观自旋列，不删 determinant，也不改变任何收缩项。
        bytes_per_column = n_pairs * links.n_outer * coefficients.element_size()
        width_limit = min(self.chunk_size, max(1, _SAME_SPIN_BUFFER_BYTES // bytes_per_column))
        for start in range(0, n_columns, width_limit):
            stop = min(start + width_limit, n_columns)
            width = stop - start
            temporary = torch.zeros(
                (n_pairs * links.n_outer, width),
                dtype=self.dtype,
                device=self.device,
            )
            # 必须先截取当前列块，再按 link address 选行。反过来会在每个
            # chunk 都物化一次全宽大张量，既浪费带宽，也会额外占用数 GiB。
            source = coefficients[:, start:stop].index_select(0, links.address)
            source.mul_(links.sign[:, None])
            temporary.index_add_(
                0,
                gather_slot,
                source,
            )
            # gather 结果已累加完，及时释放，避免与下一步大矩阵同时驻留。
            del source
            transformed = torch.mm(
                self.same_spin_matrix,
                temporary.reshape(n_pairs, -1),
            ).reshape(n_pairs, links.n_outer, width)
            del temporary
            contributions = transformed[links.pair, links.outer, :]
            del transformed
            contributions.mul_(links.sign[:, None])
            output[:, start:stop].index_add_(0, links.address, contributions)
            # 下一分块重新分配之前，结束本分块所有临时张量的生命周期。
            del contributions

        if output.shape != (n_rows, n_columns):
            raise RuntimeError("同自旋收缩产生了意外 shape")
        return output

    def _contract_mixed_spin(self, coefficients: torch.Tensor) -> torch.Tensor:
        """计算 PySCF ``SCIcontract_2e_bbaa`` 对应的异自旋部分。"""

        n_pairs = self.mixed_spin_matrix.shape[0]
        output_transposed = torch.zeros(
            (self.n_beta, self.n_alpha),
            dtype=self.dtype,
            device=self.device,
        )

        for chunk_index, start in enumerate(range(0, self.n_beta, self.chunk_size)):
            stop = min(start + self.chunk_size, self.n_beta)
            width = stop - start
            temporary = torch.zeros(
                (n_pairs * self.n_alpha, width),
                dtype=self.dtype,
                device=self.device,
            )
            source = coefficients[:, start:stop].index_select(0, self.cd_a.address)
            source.mul_(self.cd_a.sign[:, None])
            temporary.index_add_(
                0,
                self.cd_a_gather_slot,
                source,
            )
            del source
            # beta link 的 outer 就是当前被展开的 beta determinant。
            beta_chunk = self.mixed_beta_chunks[chunk_index]
            transformed = torch.mm(
                self.mixed_spin_matrix,
                temporary.reshape(n_pairs, -1),
            ).reshape(n_pairs, self.n_alpha, width)
            del temporary
            contributions = transformed[
                beta_chunk.pair,
                :,
                beta_chunk.outer,
            ]
            del transformed
            contributions.mul_(beta_chunk.sign[:, None])
            output_transposed.index_add_(0, beta_chunk.address, contributions)
            del contributions

        return output_transposed.transpose(0, 1)

    def contract(self, coefficients: torch.Tensor) -> torch.Tensor:
        """返回与 PySCF ``selected_ci.contract_2e`` 等价的 ``H @ c``。"""

        if coefficients.shape != (self.n_alpha, self.n_beta):
            raise ValueError(
                "coefficients shape 应为 "
                f"({self.n_alpha}, {self.n_beta})，实际为 {tuple(coefficients.shape)}"
            )
        if coefficients.device != self.device or coefficients.dtype != self.dtype:
            raise ValueError("coefficients 的 device/dtype 必须与 contractor 一致")

        result = self._contract_mixed_spin(coefficients)
        if self.nelec[0] > 1:
            result = result + self._contract_same_spin_rows(
                coefficients,
                self.dd_a,
                self.dd_a_gather_slot,
            )
        if self.nelec[1] > 1:
            result = result + self._contract_same_spin_rows(
                coefficients.transpose(0, 1),
                self.dd_b,
                self.dd_b_gather_slot,
            ).transpose(0, 1)
        return result
