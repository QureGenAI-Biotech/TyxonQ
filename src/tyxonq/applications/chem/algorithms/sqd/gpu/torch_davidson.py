"""在 PyTorch 设备上求解实对称算符的最低本征态。

使用容量受控的 Davidson 工作区、两遍正交化和厚重启，
在同一设备保存基向量、算符像、预条件方向并复算最终残差。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


_SDAA_PROJECTION_CHUNK_ROWS = 3_000_000


@dataclass
class TorchDavidsonResult:
    """保存最低根 Davidson 的结果和可复核收敛记录。"""

    energy: float
    coefficients: torch.Tensor
    residual_norm: float
    converged: bool
    iterations: int
    h_matvecs: int
    history: list[dict[str, float | int]]


def _project_small_matrix(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """计算 ``left.T @ right``，并绕开 SDAA 超长约化维度缺陷。

    Davidson 的投影结果最多只有 ``max_space × max_space``，但约化维度等于
    CI 系数总数。Torch-SDAA 3.2.1 在 900 万维处会产生明显错误的超大内存
    申请；沿 determinant 行分块累加在数学上等价，且不会构造大中间张量。
    已验证可直接运行的 300 万行以下路径、CPU 和 CUDA 均保持原实现。
    """

    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("投影输入必须是二维张量")
    if left.shape[0] != right.shape[0]:
        raise ValueError("投影输入的行数必须一致")
    if left.device != right.device or left.dtype != right.dtype:
        raise ValueError("投影输入的 device/dtype 必须一致")

    n_rows = left.shape[0]
    if left.device.type != "sdaa" or n_rows <= _SDAA_PROJECTION_CHUNK_ROWS:
        return left.transpose(0, 1) @ right

    projected = torch.zeros(
        (left.shape[1], right.shape[1]),
        dtype=left.dtype,
        device=left.device,
    )
    for start in range(0, n_rows, _SDAA_PROJECTION_CHUNK_ROWS):
        stop = min(start + _SDAA_PROJECTION_CHUNK_ROWS, n_rows)
        projected.add_(left[start:stop].transpose(0, 1) @ right[start:stop])
    return projected


def _combine_columns(
    matrix: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """计算 ``matrix @ weights``，分块生成 SDAA 超长二维输出。"""

    if matrix.ndim != 2 or weights.ndim != 2:
        raise ValueError("列组合输入必须是二维张量")
    if matrix.shape[1] != weights.shape[0]:
        raise ValueError("列组合输入的收缩维度必须一致")
    if matrix.device != weights.device or matrix.dtype != weights.dtype:
        raise ValueError("列组合输入的 device/dtype 必须一致")

    n_rows = matrix.shape[0]
    if matrix.device.type != "sdaa" or n_rows <= _SDAA_PROJECTION_CHUNK_ROWS:
        return matrix @ weights

    # ``eigenvectors[:, :retained_count]`` 是很小但非连续的列切片。
    # Torch-SDAA 3.2.1 会为该视图错误申请数 GiB workspace；先连续化只需
    # 复制至多 ``max_space × retained_count`` 个元素。
    weights = weights.contiguous()

    combined = torch.empty(
        (n_rows, weights.shape[1]),
        dtype=matrix.dtype,
        device=matrix.device,
    )
    for start in range(0, n_rows, _SDAA_PROJECTION_CHUNK_ROWS):
        stop = min(start + _SDAA_PROJECTION_CHUNK_ROWS, n_rows)
        combined[start:stop] = matrix[start:stop] @ weights
    return combined


def _orthogonalize(
    vector: torch.Tensor,
    basis_matrix: torch.Tensor,
) -> torch.Tensor:
    """使用两遍 modified Gram-Schmidt 降低 float64 正交化误差。"""

    result = vector
    for _ in range(2):
        overlaps = _project_small_matrix(basis_matrix, result[:, None]).reshape(-1)
        result = result - basis_matrix @ overlaps
    return result


def davidson_lowest(
    matvec,
    diagonal: torch.Tensor,
    *,
    initial_vector: torch.Tensor | None = None,
    tolerance: float = 1.0e-8,
    max_cycles: int = 100,
    max_space: int = 12,
    denominator_floor: float = 1.0e-4,
    callback: Callable[[dict[str, float | int]], None] | None = None,
) -> TorchDavidsonResult:
    """求实对称算符最低本征对，并返回 GPU 常驻的归一化 CI 向量。"""

    if diagonal.ndim != 1 or diagonal.numel() == 0:
        raise ValueError("diagonal 必须是一维非空张量")
    if not diagonal.is_floating_point() or diagonal.is_complex():
        raise ValueError("diagonal 必须是实浮点张量")
    if tolerance <= 0 or max_cycles < 1 or max_space < 2:
        raise ValueError("tolerance/max_cycles/max_space 参数不合法")

    if initial_vector is None:
        initial_address = int(torch.argmin(diagonal).item())
        initial = torch.zeros_like(diagonal)
        initial[initial_address] = 1.0
    else:
        if initial_vector.shape != diagonal.shape:
            raise ValueError("initial_vector shape 必须与 diagonal 一致")
        if initial_vector.device != diagonal.device:
            raise ValueError("initial_vector 与 diagonal 必须位于同一 device")
        if initial_vector.dtype != diagonal.dtype:
            raise ValueError("initial_vector 与 diagonal 必须使用同一 dtype")
        initial_norm = torch.linalg.vector_norm(initial_vector)
        if not torch.isfinite(initial_norm) or float(initial_norm.item()) <= 0:
            raise ValueError("initial_vector 必须是有限的非零向量")
        initial = initial_vector / initial_norm
    history: list[dict[str, float | int]] = []
    h_matvecs = 0
    energy = float("nan")
    residual_norm = float("inf")
    coefficients = initial

    with torch.inference_mode():
        # 每个基向量占一行，使传给 H@c 的向量连续；投影使用转置视图。
        # 固定容量仅分配一次，避免每轮 stack 复制整套 V/AV，降低带宽与峰值显存。
        capacity = min(int(max_space), diagonal.numel())
        basis_storage = diagonal.new_empty((capacity, diagonal.numel()))
        image_storage = torch.empty_like(basis_storage)
        basis_storage[0].copy_(initial)
        basis_count, image_count = 1, 0
        for cycle in range(1, max_cycles + 1):
            while image_count < basis_count:
                image = matvec(basis_storage[image_count]).reshape(-1)
                if image.shape != diagonal.shape:
                    raise ValueError("matvec 返回 shape 与 diagonal 不一致")
                image_storage[image_count].copy_(image)
                image_count += 1
                h_matvecs += 1
                del image

            basis_matrix = basis_storage[:basis_count].transpose(0, 1)
            image_matrix = image_storage[:basis_count].transpose(0, 1)
            if diagonal.device.type == "sdaa":
                # SDAA 保留既有的连续列矩阵布局及分块投影路径；本轮优化针对 CUDA。
                basis_matrix = basis_matrix.contiguous()
                image_matrix = image_matrix.contiguous()
            projected = _project_small_matrix(basis_matrix, image_matrix)
            # 数值舍入可能产生极小非对称部分，显式对称化后再求本征对。
            projected = 0.5 * (projected + projected.transpose(0, 1))
            eigenvalues, eigenvectors = torch.linalg.eigh(projected)
            energy_tensor = eigenvalues[0]
            subspace_coefficients = eigenvectors[:, 0]
            coefficients = basis_matrix @ subspace_coefficients
            image = image_matrix @ subspace_coefficients
            residual = image - energy_tensor * coefficients
            residual_norm_tensor = torch.linalg.vector_norm(residual)
            energy = float(energy_tensor.item())
            residual_norm = float(residual_norm_tensor.item())
            record: dict[str, float | int] = {
                "cycle": cycle,
                "subspace_dimension": basis_count,
                "energy": energy,
                "residual_norm": residual_norm,
            }
            history.append(record)
            if callback is not None:
                callback(dict(record))
            if residual_norm <= tolerance:
                return TorchDavidsonResult(
                    energy=energy,
                    coefficients=coefficients / torch.linalg.vector_norm(coefficients),
                    residual_norm=residual_norm,
                    converged=True,
                    iterations=cycle,
                    h_matvecs=h_matvecs,
                    history=history,
                )

            denominator = energy_tensor - diagonal
            signs = torch.where(
                denominator >= 0,
                torch.ones_like(denominator),
                -torch.ones_like(denominator),
            )
            denominator = torch.where(
                torch.abs(denominator) < denominator_floor,
                signs * denominator_floor,
                denominator,
            )
            correction = residual / denominator

            # 达到子空间上限时保留若干最低 Ritz 向量做厚重启，再加入新方向。
            # 只保留最低一个向量会在近简并谱中反复丢掉已经学到的低能子空间，
            # 开壳层自旋惩罚算符尤其容易因此循环而不收敛。
            if basis_count >= capacity:
                # 至少空出一个位置给残差方向；max_space=2 时只保留最低根。
                retained_count = min(max(2, capacity // 4), 8, capacity - 1)
                if retained_count < 1:
                    break
                retained_weights = eigenvectors[:, :retained_count]
                retained_vectors = _combine_columns(basis_matrix, retained_weights)
                retained_images = _combine_columns(image_matrix, retained_weights)
                basis_storage[:retained_count].copy_(retained_vectors.transpose(0, 1))
                image_storage[:retained_count].copy_(retained_images.transpose(0, 1))
                basis_count = image_count = retained_count
                basis_matrix = basis_storage[:basis_count].transpose(0, 1)
                if diagonal.device.type == "sdaa":
                    basis_matrix = basis_matrix.contiguous()
                del retained_vectors, retained_images

            correction = _orthogonalize(correction, basis_matrix)
            correction_norm = torch.linalg.vector_norm(correction)
            if float(correction_norm.item()) <= 1.0e-12:
                # 对角算符的 r/(E-diag) 可能与当前 Ritz 向量共线；此时
                # 原残差仍能扩展子空间，不能把预条件方向退化当作无法继续。
                correction = _orthogonalize(residual, basis_matrix)
                correction_norm = torch.linalg.vector_norm(correction)
                if float(correction_norm.item()) <= 1.0e-12:
                    break
            basis_storage[basis_count].copy_(correction / correction_norm)
            basis_count += 1
            # H@c 的工作区可能很大，下一轮收缩前释放本轮长向量临时量。
            del basis_matrix, image_matrix, image, denominator, signs, residual, correction

    return TorchDavidsonResult(
        energy=energy,
        coefficients=coefficients / torch.linalg.vector_norm(coefficients),
        residual_norm=residual_norm,
        converged=False,
        iterations=len(history),
        h_matvecs=h_matvecs,
        history=history,
    )
