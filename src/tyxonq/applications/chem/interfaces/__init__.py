"""TyxonQ 与分子动力学生态（ASE / i-PI / OpenMM / LAMMPS / MDI）的接口层。

全部依赖惰性导入：本包自身只硬依赖 PySCF。
"""

from .scanner import qc_scanner

__all__ = ["qc_scanner"]
