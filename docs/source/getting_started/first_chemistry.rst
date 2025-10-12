=======================
第一个量子化学计算
=======================

欢迎来到量子化学的世界！本指南将带您一步步学习如何使用TyxonQ的VQE（变分量子本征求解器）执行您的第一个量子化学计算。

.. contents:: 本页内容
   :depth: 2
   :local:

概述
====

量子化学是量子计算最重要的应用领域之一。TyxonQ提供了完整的量子化学模块，包括：

- **分子建模**：使用 :class:`Molecule` 类定义分子结构
- **哈密顿量构建**：自动生成量子哈密顿量
- **VQE算法**：包括UCCSD、HEA等多种拟设
- **变分优化**：集成多种优化器

**核心概念**

- **VQE算法**：通过变分原理求解分子基态能量
- **拟设电路**：参数化的量子电路，用于准备试探波函数
- **哈密顿量**：描述分子电子结构的量子算符
- **对称性破缺**：从有限对称群到可解的类群

分子设置与建模
==============

创建分子对象
------------

使用TyxonQ的 :class:`Molecule` 类定义分子：

.. code-block:: python

   from tyxonq.applications.chem import Molecule
   
   # 定义氢分子 (H2)
   h2_molecule = Molecule(
       atoms=[
           ["H", [0.0, 0.0, 0.0]],      # 第一个氢原子
           ["H", [0.0, 0.0, 0.74]]      # 第二个氢原子，间距0.74埃
       ],
       basis="sto-3g",                   # 使用STO-3G基组
       charge=0,                         # 中性分子
       spin=0                            # 单重态
   )
   
   print(f"分子电荷: {h2_molecule.charge}")
   print(f"自旋: {h2_molecule.spin}")
   print(f"电子数: {h2_molecule.n_electrons}")

复杂分子示例
------------

创建水分子（H2O）：

.. code-block:: python

   # 水分子的优化几何结构
   h2o_molecule = Molecule(
       atoms=[
           ["O", [0.0, 0.0, 0.0]],
           ["H", [0.757, 0.586, 0.0]],
           ["H", [-0.757, 0.586, 0.0]]
       ],
       basis="6-31g",                    # 使用更精确的基组
       charge=0,
       spin=0
   )
   
   print(f"H2O电子数: {h2o_molecule.n_electrons}")
   print(f"轨道数: {h2o_molecule.n_orbitals}")

哈密顿量构建
============

电子哈密顿量
-----------

在二次量子化表示中，分子哈密顿量的一般形式为：

.. math::

   H = \sum_{ij} h_{ij} a_i^\dagger a_j + \frac{1}{2}\sum_{ijkl} h_{ijkl} a_i^\dagger a_j^\dagger a_k a_l

其中：
- 第一项：单电子项（动能 + 核吸引）
- 第二项：双电子项（电子排斥）

费米-量子比特映射
----------------

将费米子哈密顿量转换为量子比特哈密顿量：

.. code-block:: python

   # 获取哈密顿量（自动使用Jordan-Wigner变换）
   hamiltonian = h2_molecule.get_hamiltonian()
   
   print(f"哈密顿量项数: {len(hamiltonian)}")
   print(f"需要的量子比特数: {h2_molecule.n_qubits}")
   
   # 查看哈密顿量的第一项
   if hamiltonian:
       coeff, pauli_string = hamiltonian[0]
       print(f"第一项: {coeff:.6f} * {pauli_string}")

VQE算法基础
===========

变分原理
--------

VQE基于变分原理，通过优化参数化量子态来求解基态能量：

.. math::

   E_0 = \min_{\theta} \langle \psi(\theta) | H | \psi(\theta) \rangle

其中 $|\psi(\theta)\rangle$ 是由拟设电路准备的试探波函数。

运行VQE计算
============

使用HEA拟设
-----------

硬件适配拟设（HEA）是近期量子设备的理想选择：

.. code-block:: python

   from tyxonq.applications.chem import HEA
   
   # 创建HEA算法实例
   hea = HEA(
       molecule=h2_molecule,
       layers=2,                      # 拟设层数
       runtime="numeric",             # 使用数值后端
       mapping="jordan_wigner"        # 映射方法
   )
   
   print(f"HEA参数数量: {hea.n_params}")
   print(f"使用的量子比特数: {hea.n_qubits}")

执行VQE优化
------------

.. code-block:: python

   # 运行VQE优化
   result = hea.kernel(
       method="COBYLA",               # 优化算法
       options={
           "maxiter": 100,            # 最大迭代次数
           "disp": True               # 显示优化过程
       }
   )
   
   print(f"VQE能量: {result:.6f} Hartree")
   print(f"HF能量: {h2_molecule.hf_energy:.6f} Hartree")
   print(f"相关能: {result - h2_molecule.hf_energy:.6f} Hartree")

完整示例：H2分子VQE计算
=======================

以下是一个完整的H2分子VQE计算示例：

.. code-block:: python

   import tyxonq as tq
   from tyxonq.applications.chem import Molecule, HEA
   import numpy as np
   
   def h2_vqe_calculation():
       """完整的H2分子VQE计算示例"""
       
       # 1. 定义分子
       print("步骤1: 创建氢分子...")
       h2 = Molecule(
           atoms=[
               ["H", [0.0, 0.0, 0.0]],
               ["H", [0.0, 0.0, 0.74]]
           ],
           basis="sto-3g",
           charge=0,
           spin=0
       )
       
       print(f"分子信息:")
       print(f"  电子数: {h2.n_electrons}")
       print(f"  量子比特数: {h2.n_qubits}")
       print(f"  HF能量: {h2.hf_energy:.6f} Hartree")
       
       # 2. 创建HEA算法
       print("\n步骤2: 设置HEA算法...")
       hea = HEA(
           molecule=h2,
           layers=2,
           runtime="numeric"
       )
       
       print(f"HEA配置:")
       print(f"  层数: {hea.layers}")
       print(f"  参数数: {hea.n_params}")
       
       # 3. 执行VQE优化
       print("\n步骤3: 执行VQE优化...")
       vqe_energy = hea.kernel(
           method="COBYLA",
           options={"maxiter": 100, "disp": False}
       )
       
       # 4. 结果分析
       print("\n结果分析:")
       print(f"VQE能量: {vqe_energy:.6f} Hartree")
       print(f"HF能量:  {h2.hf_energy:.6f} Hartree")
       print(f"相关能: {vqe_energy - h2.hf_energy:.6f} Hartree")
       
       return vqe_energy
   
   # 运行计算
   if __name__ == "__main__":
       result = h2_vqe_calculation()

下一步
======

恭喜！您已经成功完成了第一个量子化学计算。接下来推荐学习：

- :doc:`../quantum_chemistry/algorithms/index` - 学习更多算法
- :doc:`../quantum_chemistry/fundamentals/index` - 深入理论基础
- :doc:`../examples/chemistry_examples` - 查看更多化学示例
- :doc:`../user_guide/numerics/index` - 了解数值后端

常见问题
========

**Q: 如何选择合适的基组？**

A: 基组选择影响计算精度和效率：
- STO-3G: 最小基组，适合快速测试
- 6-31G: 平衡精度和效率
- cc-pVDZ: 高精度计算

**Q: VQE算法不收敛怎么办？**

A: 尝试以下方法：
- 增加最大迭代次数
- 更改优化器（如BFGS、SLSQP）
- 调整拟设层数
- 使用更好的初始参数

**Q: 如何估计计算时间？**

A: 计算时间主要取决于：
- 分子大小（量子比特数）
- 拟设复杂度（参数数量）
- 优化迭代次数
- 所选的后端（numeric vs device）

相关资源
========

- :doc:`/api/applications/index` - 应用API参考
- :doc:`/quantum_chemistry/index` - 量子化学模块
- :doc:`/examples/chemistry_examples` - 化学示例
- :doc:`/user_guide/postprocessing/index` - 结果分析
