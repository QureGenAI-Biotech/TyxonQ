"""VQE 族私有的波函数（CI/statevector）数值库。

CI 向量 ↔ statevector 映射（``ci_state_mapping``）、CI 算符张量与激发应用
（``civector_ops``、``statevector_ops``）、pyscf CI 向量互操作（``pyscf_civector``）。
仅服务 UCC/HEA 及其运行时；采样族（sqd/lucj）不使用。

因依赖 pyscf/openfermion，属化学领域层，不下沉到领域无关的 ``tyxonq.libs``。
本子包 ``__init__`` 不主动导入子模块（避免包级触发 pyscf）；调用方按需从具体
子模块导入。
"""
