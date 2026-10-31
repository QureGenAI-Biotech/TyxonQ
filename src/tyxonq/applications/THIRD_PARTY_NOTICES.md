# 第三方来源、修改说明与许可证

本文统一保存 TyxonQ applications 相关迁移代码及 SKQD 示例的第三方归属说明。
原有版权声明、许可文本和迁移改编说明集中保存在本文；算法文件头只描述功能。
下列路径以仓库根目录为基准。原始英文归属说明与许可证按原文保留。

| 来源 | 对应代码 | 许可证 |
| --- | --- | --- |
| Vayesta 1.0.1 | `src/tyxonq/applications/chem/algorithms/ewf/` 中的分片、浴、簇和能量实现 | Apache-2.0 |
| MIQuLab ewf-geom-opt | EWF 的 `density.py`、`gradient.py` | MIT |
| ffsim | LUCJ 及 `chem/chem_libs/fermionic_circuits.py` 的相关迁移实现 | Apache-2.0 |
| qiskit-addon-sqd、PySCF | SQD 及 UHF selected-CI 适配 | Apache-2.0 |
| SBD | `src/tyxonq/applications/chem/algorithms/sqd/gpu/` 的可选 SBD 后端 | Apache-2.0 |
| IBM SKQD 示例 | `examples/SKQD/` 及相关费米子线路 | 具体来源与改写范围见 SKQD 部分 |
| Qiskit / IBM 自旋示例 | `src/tyxonq/applications/spin/circuits.py` | Apache-2.0 |

## Vayesta

This package contains a minimal, modified implementation of molecular RHF/UHF
Embedded Wavefunction Theory algorithms adapted from **Vayesta 1.0.1**.

- Upstream project: [Vayesta](https://github.com/BoothGroup/Vayesta), Booth Group
- Upstream release: 1.0.1
- Local source snapshot: `Vayesta-master/vayesta/` (`__version__ = "1.0.1"`;
  snapshot commit is not recorded locally)
- Upstream license: Apache License 2.0
- TyxonQ adaptation: real molecular RHF and UHF, with ROHF converted to UHF;
  MPI, periodic systems, GHF, complex orbitals, spin-orbit coupling, symmetry
  machinery, screening, HDF5 and self-consistency infrastructure were removed,
  and the public/result APIs were rewritten for TyxonQ.
- Open-shell migration: spin-resolved IAO/DMET/UMP2-BNO, UHF cluster
  Hamiltonians with frozen dummy virtuals, UCCSD/UFCI wavefunctions, and UHF
  projected-WF/DMET energy contractions were adapted from the corresponding
  unrestricted Vayesta implementations.
- Local extension: `Fragment.set_active_orbitals()` and the associated
  incomplete-owner-space diagnostics are TyxonQ additions, not Vayesta 1.0.1
  APIs.

Algorithmic source areas used by this adaptation:

- `vayesta/core/fragmentation/iao.py`
- `vayesta/core/fragmentation/fragmentation.py`
- `vayesta/core/fragmentation/ufragmentation.py`
- `vayesta/core/bath/dmet.py`
- `vayesta/core/bath/bno.py`
- `vayesta/core/types/cluster.py`
- `vayesta/core/qemb/fragment.py`
- `vayesta/core/qemb/ufragment.py`
- `vayesta/core/qemb/qemb.py`
- `vayesta/core/qemb/rdm.py`
- `vayesta/core/types/wf/ccsd.py`
- `vayesta/core/types/wf/fci.py`
- `vayesta/core/types/wf/rdm.py`
- `vayesta/core/util.py`
- `vayesta/solver/hamiltonian.py`
- `vayesta/solver/ccsd.py`
- `vayesta/solver/fci.py`
- `vayesta/ewf/ewf.py`
- `vayesta/ewf/uewf.py`
- `vayesta/ewf/ufragment.py`

根目录 `LICENSE` 和本文保留 Apache-2.0 许可文本。各文件的改编关系如下；
这些文件均已按 TyxonQ 的接口修改，具体改写范围见本节：

| TyxonQ 文件（相对 `src/tyxonq/applications/chem/algorithms/ewf/`） | 对应上游模块或约定 |
| --- | --- |
| `__init__.py`、`driver.py` | EWF 接口与 `vayesta.ewf.ewf` 控制流程 |
| `bath.py` | `core.bath.dmet`、`core.bath.bno`、`core.qemb.fragment` |
| `cluster.py` | `core.types.cluster`、`core.qemb.fragment`、`solver.hamiltonian` |
| `energy.py` | `ewf/fragment.py`、`ewf/ewf.py`、`core/qemb/qemb.py` |
| `fragmentation.py` | `core.fragmentation.iao`、`core.fragmentation.fragmentation`、`core.qemb.fragment` |
| `linalg.py` | `core.util`、`core.fragmentation`、`core.qemb.fragment` 中的辅助函数 |
| `solver.py` | `solver/hamiltonian.py`、`solver/ccsd.py`、`solver/fci.py` |
| `types.py` | restricted cluster 数据布局，按 TyxonQ 重新实现 |

## MIQuLab

来源：[MIQuLab/ewf-geom-opt](https://github.com/MIQuLab/ewf-geom-opt)。
`density.py` 改编自 `Source/embedding_lagrangian.py` 和 `Source/EWF-CI_Geom_Opt_HPC.py`；
`gradient.py` 改编自 `Source/isolated_casci_gradient.py` 的全空间 `build_ewf_grad` 路径。
本地适配包含 TyxonQ 的片段结果接口、密度约定和收敛检查；梯度仍为 frozen-bath 近似，
不包含完整的浴轨道和片段投影响应。

以下完整保留原 MIT 许可证：

```text
MIT License

Copyright (c) 2026 MIQuLab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## ffsim

本目录的 spin-resolved LUCJ 参数约定与 αβ 显式 double factorization，最小改编自本仓库
`ffsim-main` 中以下实现：

- `python/ffsim/variational/ucj_spin_unbalanced.py`
- `python/ffsim/linalg/double_factorized_decomposition.py`
- `src/linalg/givens.rs` 中的 Slater determinant Givens 分解
- `python/ffsim/qiskit/gates/slater_determinant.py`
- `python/ffsim/qiskit/transpiler_passes/merge_orbital_rotations.py`

ffsim 采用 Apache License 2.0。这里保留其 `(aa, ab, bb)`、α/β 独立旋转及“αβ 因子优先”
约定，并把 Slater determinant preparation 与相邻 orbital rotation 合并改写为 TyxonQ 本地
NumPy 实现，不形成 ffsim 或 Qiskit 运行时依赖。未迁移 pass manager、routing、layout、VF2
或 hardware-aware optimization。迁移时还修正了上游 `from_t_amplitudes()` 对 `pairs_ab`
漏校验、重复校验 `pairs_bb` 的笔误。

`src/tyxonq/applications/chem/algorithms/lucj/linalg.py` 和
`src/tyxonq/applications/chem/chem_libs/fermionic_circuits.py` 中 Slater/Givens
分解原有的版权与修改标记统一保留如下：

```text
(C) Copyright IBM 2026. Apache License 2.0。
已为 TyxonQ 修改；从原 LUCJ 模块提取，保留原公开入口。
```

## SQD

本目录的采样驱动 selected-CI 流程参考了
`qiskit-addon-sqd`（IBM 及其贡献者，Apache License 2.0）。TyxonQ
保留了项目现有的样本整理、恢复、子采样与 selected-CI 调用结构，并增加了
本地接口适配和中文说明。

`selected_ci_uhf.py` 使用 PySCF 公开 Python 模块及其 `libfci` 绑定实现
unrestricted selected-CI Hamiltonian 收缩。PySCF 采用 Apache License 2.0。
TyxonQ 的新增部分包括 UHF 张量分派、固定 determinant 子空间 Davidson
控制流、dummy 轨道排除、spin-resolved RDM/能量和交叉轨道重叠下的
`S^2` 计算。

参考源码仅用于必要功能的最小迁移；`qiskit-addon-sqd` 与本地参考快照均不
作为 TyxonQ 的运行时依赖。

## SKQD

- `examples/SKQD/models.py` 中 SIAM 参数、bath 变基和近费米面七门网络依据 IBM 的 `sample-based-krylov-quantum-diagonalization.ipynb` 重新实现；线路改为 TyxonQ 的模式及矩阵位序，无 Qiskit/ffsim 运行调用。
- `src/tyxonq/applications/chem/chem_libs/fermionic_circuits.py` 的 Slater/Givens 数学从既有 LUCJ 提取，保留原改编标记：`(C) Copyright IBM 2026. Apache License 2.0`。详细来源见 本文的 ffsim 部分，许可证全文见根目录 `LICENSE`；新增相位门、Pauli 演化为 TyxonQ 适配函数。
- SqDRIFT 分组、F2Q MILP 与严格 determinant 求解依据论文数学独立实现，没有复制或导入 qiskit-fermions 的运行实现。

## 自旋线路

`src/tyxonq/applications/spin/circuits.py` 的 Cartan 分解改编自 Qiskit 项目，
具体参考 `qiskit-addon-aqc-tensor` 的 `KAK._define`；singlet/HVA/二阶 Trotter
结构改写自 `qiskit/simulate-neutron-scattering.ipynb`。已修改为 TyxonQ 门格式和
独立参数表达式，参数模板与局域旋转合并为本地实现，不使用 Qiskit 对象。
原有版权与许可标记保留如下：

```text
(C) Copyright IBM 2024.
This code is licensed under the Apache License, Version 2.0.
See http://www.apache.org/licenses/LICENSE-2.0.
```

## SBD

- 核心源码：https://github.com/r-ccs-cms/sbd.git，固定提交 `93ebabec6c841d0ade706018b053795d8c1db9bd`。
- Python wrapper 参考源码：https://github.com/Qiskit/sbd-eigensolver-python，固定提交 `b49c22b54813be1e0344ba52f8ffd03e3825433e`。
- TyxonQ 提供 TPB 内存接口、位序/相位和 RDM 约定适配，以及原生和混合求解入口。
- CUDA 编译兼容与数值稳定性修改记录在 `chem/algorithms/sqd/gpu/sbd_ext/nvcc.patch`；
  具体版本、源码哈希与补丁说明保存在同目录 `SOURCES.json`。
- 外部源码由显式构建步骤放入构建缓存，不整体内置到 TyxonQ 源码中。

以下完整保留 SBD 的版权声明与 Apache-2.0 许可证文本；Apache-2.0 正文亦适用于本文
其他标注为 Apache-2.0 的来源，其各自版权与归属声明仍分别保留。

```text
         Copyright 2025 RIKEN and its contributors
         Copyright 2026 Advanced Micro Devices, Inc.

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright 2025 RIKEN and its contributors.
   Copyright 2026 Advanced Micro Devices, Inc.
   Copyright 2026 International Business Machines, Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
