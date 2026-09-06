"""化学量子动力学包（时间演化），与 ``algorithms/``（电子结构）平级。

本包收纳数值时间演化运行时（``DynamicsNumericRuntime``）与基于 renormalizer
的模型哈密顿量（Pyrazine、SBM）。刻意与电子结构（VQE/SQD）代码隔离，使得
导入电子结构时不会被动拖入 ``renormalizer`` 依赖。
"""

from .evolution import DynamicsNumericRuntime  # noqa: F401

__all__ = ["DynamicsNumericRuntime"]
