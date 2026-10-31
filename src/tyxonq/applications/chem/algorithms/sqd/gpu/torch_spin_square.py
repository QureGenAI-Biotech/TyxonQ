"""使用 PyTorch 或可选 CUDA 融合内核计算 selected-CI 的 S² 算符像。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf.fci import selected_ci
import torch


@dataclass(frozen=True)
class _SpinTransition:
    """保存一次自旋升降算符需要的稀疏地址与费米子符号。"""

    source_rows: torch.Tensor
    source_columns: torch.Tensor
    target_rows: torch.Tensor
    target_columns: torch.Tensor
    row_signs: torch.Tensor
    column_signs: torch.Tensor


def _link_map(
    generator,
    strings: np.ndarray,
    norb: int,
    nelec: int,
    *,
    destroy: bool,
) -> np.ndarray:
    """复现 PySCF selected_ci.contract_ss 使用的地址表。"""

    links = generator(strings, norb, nelec)
    mapping = np.zeros((links.shape[0], norb, 2), dtype=np.int32)
    orbital_column = 1 if destroy else 0
    for string_index, table in enumerate(links):
        nonzero = table[:, 3] != 0
        table = table[nonzero]
        mapping[string_index, table[:, orbital_column]] = table[:, 2:]
    return mapping


def _transitions(
    row_map: np.ndarray | None,
    column_map: np.ndarray | None,
    norb: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[_SpinTransition | None, ...]:
    """把每个轨道的非零 selected-CI 地址移动到计算设备。"""

    if row_map is None or column_map is None:
        return (None,) * norb
    output: list[_SpinTransition | None] = []
    for orbital in range(norb):
        row_mask = np.flatnonzero(row_map[:, orbital, 1])
        column_mask = np.flatnonzero(column_map[:, orbital, 1])
        if not row_mask.size or not column_mask.size:
            output.append(None)
            continue
        output.append(
            _SpinTransition(
                source_rows=torch.as_tensor(row_mask, dtype=torch.long, device=device),
                source_columns=torch.as_tensor(
                    column_mask, dtype=torch.long, device=device
                ),
                target_rows=torch.as_tensor(
                    row_map[row_mask, orbital, 0], dtype=torch.long, device=device
                ),
                target_columns=torch.as_tensor(
                    column_map[column_mask, orbital, 0],
                    dtype=torch.long,
                    device=device,
                ),
                row_signs=torch.as_tensor(
                    row_map[row_mask, orbital, 1], dtype=dtype, device=device
                ),
                column_signs=torch.as_tensor(
                    column_map[column_mask, orbital, 1], dtype=dtype, device=device
                ),
            )
        )
    return tuple(output)


class TorchSelectedS2Contractor:
    """在 CPU 或 GPU 上计算 selected-CI 子空间内的 ``S^2 @ c``。"""

    def __init__(
        self,
        ci_strings: tuple[np.ndarray, np.ndarray],
        norb: int,
        nelec: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
        backend: str = "torch",
        max_memory_fraction: float = 0.8,
    ) -> None:
        if backend not in {"torch", "cuda_fused"}:
            raise ValueError("spin backend must be 'torch' or 'cuda_fused'")
        device = torch.device(device)
        if backend == "cuda_fused" and (device.type != "cuda" or dtype != torch.float64):
            raise ValueError("cuda_fused spin requires CUDA float64")
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.backend = backend
        self._device = device
        self._dtype = dtype
        self._fused_context = None
        strings_a = np.asarray(ci_strings[0], dtype=np.int64)
        strings_b = np.asarray(ci_strings[1], dtype=np.int64)
        nalpha, nbeta = nelec

        # PySCF 的 selected-CI 地址表只保留仍落在当前子空间内的跃迁。
        destroy_a = (
            _link_map(
                selected_ci.gen_des_linkstr,
                strings_a,
                norb,
                nalpha,
                destroy=True,
            )
            if nalpha > 0
            else None
        )
        destroy_b = (
            _link_map(
                selected_ci.gen_des_linkstr,
                strings_b,
                norb,
                nbeta,
                destroy=True,
            )
            if nbeta > 0
            else None
        )
        create_a = (
            _link_map(
                selected_ci.gen_cre_linkstr,
                strings_a,
                norb,
                nalpha,
                destroy=False,
            )
            if nalpha < norb
            else None
        )
        create_b = (
            _link_map(
                selected_ci.gen_cre_linkstr,
                strings_b,
                norb,
                nbeta,
                destroy=False,
            )
            if nbeta < norb
            else None
        )
        self._shape = (len(strings_a), len(strings_b))
        self._sz_square = 0.25 * float((nalpha - nbeta) ** 2)
        self._plus_minus_shape = (
            None
            if destroy_a is None or create_b is None
            else (len(destroy_a), len(create_b))
        )
        self._minus_plus_shape = (
            None
            if create_a is None or destroy_b is None
            else (len(create_a), len(destroy_b))
        )
        self._plus_minus = _transitions(
            destroy_a, create_b, norb, device, dtype
        )
        self._minus_plus = _transitions(
            create_a, destroy_b, norb, device, dtype
        )
        if backend == "cuda_fused":
            from .sbd_solver import _load_extension

            try:
                extension = _load_extension()
            except ImportError as error:
                raise ImportError("cuda_fused spin requires the optional SBD CUDA extension") from error
            if not hasattr(extension, "FusedSpinContext"):
                raise ImportError("rebuild the SBD CUDA extension to enable fused spin kernels")

            def metadata(transitions):
                # self 持有原 Tensor；C++ 仅借用其 GPU 地址，不允许提前回收或 resize。
                return [
                    (
                        transition.source_rows.data_ptr(), transition.source_columns.data_ptr(),
                        transition.target_rows.data_ptr(), transition.target_columns.data_ptr(),
                        transition.row_signs.data_ptr(), transition.column_signs.data_ptr(),
                        transition.source_rows.numel(), transition.source_columns.numel(),
                    )
                    for transition in transitions if transition is not None
                ]

            self._fused_context = extension.FusedSpinContext(
                metadata(self._plus_minus), self._plus_minus_shape or (0, 0),
                metadata(self._minus_plus), self._minus_plus_shape or (0, 0),
                self._shape, self._sz_square, device.index, max_memory_fraction,
            )

    def external_memory_bytes(self) -> int:
        """返回不属于 PyTorch 分配器的融合内核工作区容量。"""
        return 0 if self._fused_context is None else int(self._fused_context.memory_bytes())

    @staticmethod
    def _scatter_add(
        destination: torch.Tensor,
        rows: torch.Tensor,
        columns: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """以二维外积索引执行可重复地址的原位累加。"""

        linear = rows[:, None] * destination.shape[1] + columns[None, :]
        destination.reshape(-1).index_add_(0, linear.reshape(-1), values.reshape(-1))

    def _apply_transition(
        self,
        coefficients: torch.Tensor,
        transitions: tuple[_SpinTransition | None, ...],
        intermediate_shape: tuple[int, int] | None,
    ) -> torch.Tensor:
        """严格按 PySCF 的两步中间态收集/回写执行一次升降算符。"""

        if intermediate_shape is None:
            return torch.zeros_like(coefficients)
        intermediate = coefficients.new_zeros(intermediate_shape)
        for transition in transitions:
            if transition is None:
                continue
            phases = transition.row_signs[:, None] * transition.column_signs[None, :]
            values = coefficients[
                transition.target_rows[:, None], transition.target_columns[None, :]
            ] * phases
            self._scatter_add(
                intermediate,
                transition.source_rows,
                transition.source_columns,
                values,
            )

        output = torch.zeros_like(coefficients)
        for transition in transitions:
            if transition is None:
                continue
            phases = transition.row_signs[:, None] * transition.column_signs[None, :]
            values = intermediate[
                transition.source_rows[:, None], transition.source_columns[None, :]
            ] * phases
            self._scatter_add(
                output,
                transition.target_rows,
                transition.target_columns,
                values,
            )
        return output

    def contract(self, coefficients: torch.Tensor) -> torch.Tensor:
        """返回与 ``pyscf.fci.selected_ci.contract_ss`` 相同的结果。"""

        matrix = coefficients.reshape(self._shape)
        if self._fused_context is not None:
            if matrix.requires_grad:
                raise NotImplementedError("cuda_fused spin does not support autograd")
            if matrix.device != self._device or matrix.dtype != self._dtype:
                raise ValueError("coefficients device/dtype does not match fused spin context")
            matrix = matrix.contiguous()
            output = torch.empty_like(matrix)
            stream = torch.cuda.current_stream(self._device)
            self._fused_context.contract(matrix.data_ptr(), output.data_ptr(), matrix.numel(), stream.cuda_stream)
            return output
        spin_flips = self._apply_transition(
            matrix, self._plus_minus, self._plus_minus_shape
        )
        spin_flips += self._apply_transition(
            matrix, self._minus_plus, self._minus_plus_shape
        )
        return 0.5 * spin_flips + self._sz_square * matrix


def make_spin_square_diagonal_torch(
    ci_strings: tuple[np.ndarray, np.ndarray],
    norb: int,
    nelec: tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """返回 selected-CI 行列式基中 ``S^2`` 的精确对角元。"""

    strings_a = np.asarray(ci_strings[0], dtype=np.int64)
    strings_b = np.asarray(ci_strings[1], dtype=np.int64)
    orbitals = np.arange(norb, dtype=np.int64)

    # 一个行列式的 <S^2> 等于 Sz^2 加上未成对电子数的一半；
    # alpha/beta 占据矩阵的乘积给出每对行列式的双占轨道数。
    occupied_a = ((strings_a[:, None] >> orbitals) & 1).astype(np.float64)
    occupied_b = ((strings_b[:, None] >> orbitals) & 1).astype(np.float64)
    double_occupancies = occupied_a @ occupied_b.T
    nalpha, nbeta = nelec
    sz_square = 0.25 * float((nalpha - nbeta) ** 2)
    diagonal = sz_square + 0.5 * (
        float(nalpha + nbeta) - 2.0 * double_occupancies
    )
    return torch.as_tensor(diagonal, dtype=dtype, device=device)
