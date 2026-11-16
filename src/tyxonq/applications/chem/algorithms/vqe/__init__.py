"""VQE 族电子结构算法（变分量子本征求解器）。

收纳 UCC 家族（``ucc_base.UCC`` 基类 + ``uccsd``/``kupccgsd``/``puccd``）与
``hea``。执行参数由具体算法入口控制；``execution.py`` 提供无状态公共执行辅助，
``ucc_base.py`` 的模块级函数编排 UCC 执行，``wavefunction/ucc_numeric.py`` 提供数值分派。

与采样族（``algorithms.sqd``、``algorithms.lucj``）平行：VQE 在设备上测量哈密顿量
期望值并经典优化参数；SQD 则采样 LUCJ 电路后做经典 selected-CI。

导入采用 try/except 守卫，沿用原 ``algorithms/__init__.py`` 的优雅降级风格。
"""

__all__ = []

try:
    from .ucc_base import UCC  # noqa: F401

    __all__.append("UCC")
except ImportError:
    pass

try:
    from .uccsd import UCCSD, ROUCCSD  # noqa: F401

    __all__.extend(["UCCSD", "ROUCCSD"])
except ImportError:
    pass

try:
    from .kupccgsd import KUPCCGSD  # noqa: F401

    __all__.append("KUPCCGSD")
except ImportError:
    pass

try:
    from .puccd import PUCCD  # noqa: F401

    __all__.append("PUCCD")
except ImportError:
    pass

try:
    from .hea import HEA  # noqa: F401

    __all__.append("HEA")
except ImportError:
    pass
