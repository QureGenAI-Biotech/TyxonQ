"""核心编译步骤包（stages）。

按职责分子包：
  - ``decompose/``：门分解（matchgate / rotations）；
  - ``rewrite/``：规范化重写（auto_measure / gates_transform / merge_prune）；
  - ``layout/``：布局与路由（routing，委托 qiskit.transpile）；
  - ``simplify/``：化简（lightcone）。

每个子包里的类都是一个 ``CompileStep``（方法 ``apply``），由
``gate_compiler.steps.build_compile_steps`` 按类型化参数装配。
"""


