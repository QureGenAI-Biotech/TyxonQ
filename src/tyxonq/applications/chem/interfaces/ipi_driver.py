"""``TyxonQDriver``：i-PI ``ASEDriver`` 薄壳（P2）。

i-PI 侧的全部协议工作（socket 循环、单位换算、virial、Voigt 展开、
extras、批处理、共享内存传输）都由上游 ``ipi.pes._ase.ASEDriver`` 完成，
本类只做一件事：在 ``check_parameters()`` 之后把 ``ase_calculator`` 装成
:class:`~tyxonq.applications.chem.interfaces.ase_calculator.TyxonQCalculator`。

命令行用法（i-PI 3.x 的 Python driver 入口是 ``i-pi-driver-py``，
``-m custom -P`` 从外部文件加载 PES，不需要改上游任何代码）::

    i-pi-driver-py -p 31415 -m custom \\
        -P <tyxonq>/src/tyxonq/applications/chem/interfaces/ipi_driver.py \\
        -o "water.xyz,basis=sto-3g,active_space=4 4,method=uccsd"

``-o`` 参数串由上游 ``read_args_kwargs`` 解析：**先按逗号切分**，因此
``active_space`` 不能写 ``(4,4)``（逗号会被切断），约定写成空格分隔的
``4 4``（或 ``4x4``）；第一个不带 ``=`` 的 token 是 ASE 可读的模板文件
（xyz），其余 ``key=value`` 透传给 ``TyxonQCalculator``。

QM/MM 区域划分模式（E8 阶段 A）：模板含全体系原子，另传空格分隔的
``qm_indices=0 1 2`` 与 ``atom_charges=0.0 0.0 0.0 -0.834 0.417 0.417``
（同样不能用逗号）；QM 子集走静电嵌入，补集原子作每步更新的 MM 点电荷。

周期性固体嵌入（E8 阶段 B）：区域划分参数之外再传空格分隔的 9 个数
``mm_lattice=20 0 0 0 20 0 0 0 20``（3×3 对角，单位同 scanner 的 ``unit``）
与 ``rcut_ewald=8.0``、``rcut_hcore=9.0``；嵌入层换成 ``pyscf.qmmm.pbc``
Ewald 求和，约束与已知近似见 ``scanner.py`` 与 ``md_lammps_qmmm_pbc/VALIDATION.md``。

模块顶层的 ``__DRIVER_NAME__`` / ``__DRIVER_CLASS__`` 是 i-PI 的注册约定
（``ipi.pes.load_pes`` 用 AST 扫描这两个字符串字面量）。
"""

from __future__ import annotations

import numpy as np

try:
    from ipi.pes._ase import ASEDriver
except ImportError as exc:  # 未装 i-PI：给出官方安装指引
    raise ImportError(
        "TyxonQDriver requires i-PI. Install it with: "
        "pip install -U ipi   (or the dev version: "
        "python -m pip install git+https://github.com/i-pi/i-pi.git)"
    ) from exc

try:
    from .ase_calculator import TyxonQCalculator
except ImportError:
    # 被 `i-pi-driver-py -m custom -P <本文件>` 以独立模块方式加载时
    # 没有父包，相对导入失败，退回绝对导入。
    from tyxonq.applications.chem.interfaces.ase_calculator import TyxonQCalculator

# i-PI driver 注册约定（字符串字面量，供 load_pes 的 AST 扫描）
__DRIVER_NAME__ = "tyxonq"
__DRIVER_CLASS__ = "TyxonQDriver"


def _normalize(kwargs: dict) -> dict:
    """把 ``-o`` 参数串解析结果适配成 ``TyxonQCalculator`` 的 kwargs。

    只处理约定差异：``active_space`` 在命令行里写成 ``4 4`` / ``4x4``
    （逗号是上游参数串的分隔符，不能出现在值里），这里转回 ``(4, 4)`` 元组；
    ``qm_indices`` / ``atom_charges`` 写成空格分隔的数串，这里转成数组；
    ``mm_lattice`` 是空格分隔的 9 个数（行主序 3×3），``rcut_ewald`` /
    ``rcut_hcore`` 是单浮点数；顺带剥掉用户可能加的引号。
    程序化直接传元组/数组时原样放行。
    """
    as_ = kwargs.get("active_space")
    if isinstance(as_, str):
        as_ = as_.strip().strip("'\"()")
        parts = as_.split("x") if "x" in as_ else as_.split()
        kwargs["active_space"] = tuple(int(p) for p in parts)
    for key, cast in (("qm_indices", int), ("atom_charges", float)):
        val = kwargs.get(key)
        if isinstance(val, str):
            kwargs[key] = [cast(tok) for tok in val.strip().strip("'\"[]").split()]
    lat = kwargs.get("mm_lattice")
    if isinstance(lat, str):
        vals = [float(tok) for tok in lat.strip().strip("'\"[]").split()]
        if len(vals) != 9:
            raise ValueError(f"mm_lattice needs 9 numbers (3x3), got {len(vals)}.")
        kwargs["mm_lattice"] = np.array(vals).reshape(3, 3)
    for key in ("rcut_ewald", "rcut_hcore"):
        val = kwargs.get(key)
        if isinstance(val, str):
            kwargs[key] = float(val)
    return kwargs


class TyxonQDriver(ASEDriver):
    """把 TyxonQ 势能面暴露为 i-PI forcefield 的最小适配器。

    构造参数 = ``template``（ASE 可读结构文件，来自 ``-o`` 的第一个 token）
    + 透传给 ``TyxonQCalculator`` 的 kwargs（``basis`` / ``active_space`` /
    ``method`` / ``solver_kwargs`` / ``mm_charges`` / ``qm_indices`` /
    ``atom_charges`` ...）。

    不声明 stress：``has_stress=False``，i-PI 会按零 virial 处理，
    与 ``TyxonQCalculator.implemented_properties`` 一致（分子体系无周期应力）。
    """

    def __init__(self, template, **kwargs):
        # ASEDriver/Dummy_driver 层的开关与 qc_scanner 的 kwargs 分开收
        ase_flags = {k: kwargs.pop(k) for k in
                     ("has_energy", "has_forces", "has_stress", "verbose")
                     if k in kwargs}
        ase_flags.setdefault("has_stress", False)

        self.tq_kwargs = _normalize(kwargs)
        # 注意：Dummy_driver.__init__ 末尾会调用 check_parameters()，
        # 彼时上游把 self.ase_calculator 置为 None；super() 返回后再装真计算器。
        super().__init__(template, **ase_flags)
        self.ase_calculator = TyxonQCalculator(**self.tq_kwargs)

    def compute_structure(self, cell, pos):
        """与上游相同，但为无 stress 的计算器补零应力后再交上游 ``post_process``。

        i-PI 3.3.0 上游 ``post_process`` 在 ``has_stress=False`` 时用
        ``np.zeros(9)``（shape ``(9,)``）充数，随即触发它自己的 ``(3,3)``
        shape 校验直接抛 ValueError——对无应力计算器是上游 bug。
        TyxonQCalculator 不声明 stress（分子体系无周期应力，保持 ASE 侧诚实），
        故这里只算 energy/forces，手动补一个零 ``(3,3)`` stress 再交给上游。
        """
        cell_a, pos_a = self.convert_units(cell, pos)
        structure = self.template_ase.copy()
        structure.positions[:] = pos_a
        structure.cell[:] = cell_a
        structure.calc = self.ase_calculator

        caps = list(self.capabilities)
        properties = dict(structure.get_properties(caps))
        if "stress" not in caps:
            properties["stress"] = np.zeros((3, 3))
            caps.append("stress")

        saved = self.capabilities
        self.capabilities = caps
        try:
            return self.post_process(properties, structure)
        finally:
            self.capabilities = saved
