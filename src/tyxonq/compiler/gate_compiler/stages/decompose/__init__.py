"""门分解步骤包（decompose）。

包含把高层 / 参数化门降到原生门集合的编译步骤：
  - ``matchgate.MatchgateDecompose``：2-qubit 数守恒 matchgate（two_mode_fock/fSim）→ {rz,rx,h,cx}；
  - ``rotations.RotationsDecompose``：rx/ry/rxx/ryy/rzz → {h,rz,cx}。

步骤由 ``gate_compiler.steps.build_compile_steps`` 按类型化 ``DecomposeSpec`` 装配；
未启用的分解**不追加占位步骤**（不再有 NoOp 占位类）。
"""

from __future__ import annotations
