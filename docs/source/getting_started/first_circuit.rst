===============
第一个量子电路
===============

欢迎来到TyxonQ的量子电路世界！本指南将带您从零开始，学习如何创建、操作和执行量子电路。

.. contents:: 本页内容
   :depth: 2
   :local:

概述
====

量子电路是量子计算的基础构建块。在TyxonQ中，电路通过 :class:`Circuit` 类表示，它提供了直观的API来构建和执行量子算法。

**核心概念**

- **量子比特（Qubit）**：量子信息的基本单位
- **量子门（Gates）**：对量子比特执行的操作
- **测量（Measurement）**：从量子态中提取经典信息
- **电路（Circuit）**：量子门和测量的有序序列

创建您的第一个电路
==================

基础电路构建
------------

让我们从最简单的例子开始 - 创建一个Bell态（贝尔态）：

.. code-block:: python

   import tyxonq as tq
   
   # 创建2个量子比特的电路
   circuit = tq.Circuit(2)
   
   # 添加Hadamard门到量子比特0
   circuit.h(0)
   
   # 添加CNOT门，控制位为0，目标位为1
   circuit.cnot(0, 1)
   
   print(f"电路包含 {circuit.num_qubits} 个量子比特")
   print(f"门操作数量: {len(circuit.ops)}")

**输出结果**::

   电路包含 2 个量子比特
   门操作数量: 2

链式API构建
-----------

TyxonQ支持链式调用，让代码更简洁：

.. code-block:: python

   # 链式构建相同的Bell态电路
   bell_circuit = tq.Circuit(2).h(0).cnot(0, 1)
   
   # 更复杂的例子：GHZ态
   ghz_circuit = (
       tq.Circuit(3)
       .h(0)
       .cnot(0, 1)
       .cnot(1, 2)
   )

常用量子门
==========

单量子比特门
------------

.. list-table:: TyxonQ支持的单量子比特门
   :header-rows: 1
   :widths: 15 25 60

   * - 门名称
     - 方法调用
     - 功能描述
   * - Pauli-X
     - ``circuit.x(qubit)``
     - 比特翻转门（量子非门）
   * - Pauli-Y
     - ``circuit.y(qubit)``
     - Y方向旋转门
   * - Pauli-Z
     - ``circuit.z(qubit)``
     - 相位翻转门
   * - Hadamard
     - ``circuit.h(qubit)``
     - 叠加态生成门
   * - S门
     - ``circuit.s(qubit)``
     - 相位门（π/2旋转）
   * - T门
     - ``circuit.t(qubit)``
     - π/4相位门

**示例：单量子比特门操作**

.. code-block:: python

   # 创建单量子比特电路并应用不同的门
   single_qubit = tq.Circuit(1)
   
   # 应用Hadamard门创建叠加态
   single_qubit.h(0)
   
   # 应用Pauli-X门
   single_qubit.x(0)
   
   # 查看电路摘要
   print(single_qubit.gate_summary())

参数化旋转门
------------

参数化门允许您指定旋转角度：

.. code-block:: python

   import numpy as np
   
   # 参数化旋转门
   param_circuit = tq.Circuit(2)
   
   # RX门：绕X轴旋转π/4弧度
   param_circuit.rx(0, np.pi/4)
   
   # RY门：绕Y轴旋转π/2弧度
   param_circuit.ry(1, np.pi/2)
   
   # RZ门：绕Z轴旋转任意角度
   theta = 0.5
   param_circuit.rz(0, theta)

双量子比特门
------------

.. list-table:: 双量子比特门
   :header-rows: 1
   :widths: 15 25 60

   * - 门名称
     - 方法调用
     - 功能描述
   * - CNOT
     - ``circuit.cnot(control, target)``
     - 受控非门
   * - CZ
     - ``circuit.cz(control, target)``
     - 受控Z门
   * - SWAP
     - ``circuit.swap(qubit1, qubit2)``
     - 交换门
   * - CX
     - ``circuit.cx(control, target)``
     - CNOT的别名

**示例：创建纠缠态**

.. code-block:: python

   # 创建最大纠缠态（Bell态）
   entangled = (
       tq.Circuit(2)
       .h(0)        # 创建叠加态
       .cnot(0, 1)  # 创建纠缠
   )
   
   # 创建三量子比特GHZ态
   ghz_state = (
       tq.Circuit(3)
       .h(0)
       .cnot(0, 1)
       .cnot(1, 2)
   )

测量操作
========

添加测量
--------

测量是从量子电路中提取经典信息的方式：

.. code-block:: python

   # 创建电路并添加测量
   measured_circuit = (
       tq.Circuit(2)
       .h(0)
       .cnot(0, 1)
       .measure_z(0)  # 测量量子比特0
       .measure_z(1)  # 测量量子比特1
   )
   
   # 或者测量所有量子比特
   all_measured = (
       tq.Circuit(2)
       .h(0)
       .cnot(0, 1)
   )
   # 添加全局测量
   for i in range(2):
       all_measured.measure_z(i)

执行电路
========

基本执行
--------

使用TyxonQ的链式API执行电路：

.. code-block:: python

   # 执行Bell态电路
   circuit = tq.Circuit(2).h(0).cnot(0, 1)
   
   # 添加测量
   for i in range(2):
       circuit.measure_z(i)
   
   # 执行电路（使用默认设置）
   result = circuit.run()
   
   print("测量结果:", result)

指定执行选项
------------

您可以自定义执行选项：

.. code-block:: python

   # 指定shots数和设备类型
   result = (
       circuit
       .device(provider="simulator", device="statevector", shots=1024)
       .run()
   )
   
   # 查看counts结果
   if isinstance(result, list) and result:
       counts = result[0].get("result", {})
       print("Counts:", counts)
       # 期望输出类似: {'00': 512, '11': 512}

不同设备类型
------------

.. code-block:: python

   # 状态矢量模拟器（精确模拟）
   sv_result = (
       circuit
       .device(provider="simulator", device="statevector")
       .run()
   )
   
   # 密度矩阵模拟器（支持噪声）
   dm_result = (
       circuit
       .device(provider="simulator", device="density_matrix")
       .run()
   )
   
   # 矩阵乘积态模拟器（大规模电路）
   mps_result = (
       circuit
       .device(provider="simulator", device="mps")
       .run()
   )

电路属性和方法
==============

查询电路信息
------------

.. code-block:: python

   circuit = tq.Circuit(3).h(0).cnot(0, 1).rx(2, 0.5)
   
   # 基本属性
   print(f"量子比特数: {circuit.num_qubits}")
   print(f"操作数量: {len(circuit.ops)}")
   
   # 门操作摘要
   print(f"门摘要: {circuit.gate_summary()}")
   
   # 查看指令列表
   print(f"指令: {circuit.instructions}")

元数据管理
----------

为电路添加描述信息：

.. code-block:: python

   # 添加元数据
   circuit_with_meta = (
       tq.Circuit(2)
       .h(0).cnot(0, 1)
       .with_metadata(description="Bell state preparation", author="Alice")
   )
   
   print("元数据:", circuit_with_meta.metadata)

进阶示例
========

量子傅里叶变换
--------------

实现量子傅里叶变换（QFT）：

.. code-block:: python

   def qft_circuit(n_qubits):
       """构建n量子比特的量子傅里叶变换电路"""
       circuit = tq.Circuit(n_qubits)
       
       for j in range(n_qubits):
           circuit.h(j)
           for k in range(j + 1, n_qubits):
               angle = np.pi / (2 ** (k - j))
               # 受控相位门（需要实现controlled RZ）
               # 这里简化为RZ门示例
               circuit.rz(k, angle / 2)
       
       return circuit
   
   # 创建3量子比特QFT电路
   qft_3 = qft_circuit(3)
   print(f"QFT电路门数: {len(qft_3.ops)}")

变分量子电路
------------

创建参数化的变分电路：

.. code-block:: python

   def variational_circuit(n_qubits, layers, parameters):
       """构建变分量子电路"""
       circuit = tq.Circuit(n_qubits)
       param_idx = 0
       
       for layer in range(layers):
           # 单量子比特旋转层
           for i in range(n_qubits):
               circuit.ry(i, parameters[param_idx])
               param_idx += 1
           
           # 纠缠层
           for i in range(n_qubits - 1):
               circuit.cnot(i, i + 1)
       
       return circuit
   
   # 创建变分电路
   n_qubits, layers = 4, 2
   n_params = n_qubits * layers
   params = np.random.uniform(0, 2*np.pi, n_params)
   
   var_circuit = variational_circuit(n_qubits, layers, params)
   print(f"变分电路参数数: {n_params}")

最佳实践
========

代码组织
--------

1. **使用链式API**：提高代码可读性

   .. code-block:: python

      # 推荐
      result = (
          tq.Circuit(2).h(0).cnot(0, 1)
          .device(shots=1024)
          .run()
      )

2. **合理命名**：使用描述性变量名

   .. code-block:: python

      bell_state = tq.Circuit(2).h(0).cnot(0, 1)
      ghz_state = tq.Circuit(3).h(0).cnot(0, 1).cnot(1, 2)

3. **添加注释**：解释复杂的量子算法

   .. code-block:: python

      # 准备Bell态 |00⟩ + |11⟩
      circuit = tq.Circuit(2)
      circuit.h(0)        # 创建叠加态
      circuit.cnot(0, 1)  # 创建纠缠

错误处理
--------

.. code-block:: python

   try:
       circuit = tq.Circuit(2).h(0).cnot(0, 3)  # 错误：量子比特3不存在
       result = circuit.run()
   except Exception as e:
       print(f"电路执行错误: {e}")

性能提示
--------

1. **合理选择设备类型**：
   
   - 小电路（<20量子比特）：使用``statevector``
   - 需要噪声模拟：使用``density_matrix``
   - 大电路：使用``mps``

2. **优化shots数量**：
   
   - 调试阶段：使用较少shots（100-1000）
   - 生产环境：使用足够的shots（1000-10000）

下一步
======

恭喜！您已经掌握了TyxonQ量子电路的基础知识。接下来推荐学习：

- :doc:`first_chemistry` - 学习量子化学应用
- :doc:`basic_concepts` - 深入了解量子计算概念
- :doc:`../user_guide/compiler/index` - 了解编译优化
- :doc:`../examples/basic_examples` - 查看更多示例

常见问题
========

**Q: 如何查看电路的量子态？**

A: 使用状态矢量模拟器可以获取精确的量子态：

.. code-block:: python

   circuit = tq.Circuit(2).h(0).cnot(0, 1)
   
   # 获取状态矢量（不添加测量）
   result = circuit.device(device="statevector").run()
   # 结果包含量子态信息

**Q: 如何重复使用电路？**

A: 电路对象可以重复执行：

.. code-block:: python

   circuit = tq.Circuit(2).h(0).cnot(0, 1)
   
   # 多次执行
   result1 = circuit.run()
   result2 = circuit.run()

**Q: 支持哪些量子门？**

A: TyxonQ支持完整的通用量子门集，包括Pauli门、旋转门、Hadamard门、CNOT门等。详见API文档 :doc:`../api/core/index`。

相关资源
========

- :doc:`/api/core/index` - Core API参考
- :doc:`/examples/basic_examples` - 基础示例
- :doc:`/user_guide/devices/index` - 设备和模拟器
- :doc:`/user_guide/compiler/index` - 编译器优化
