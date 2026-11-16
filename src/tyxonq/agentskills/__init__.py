"""TyxonQ Agent Skills：面向 AI agent 的权威知识层与只读体检入口。

**定位**：本包不是执行入口。执行一律回到统一链式 ``Circuit.compile().device().run()``。
本包只做两件事：
1. 承载随 pip 包发布的权威 skill（``SKILLS_DIR/<name>/SKILL.md``）——人读 markdown，
   机器读 :mod:`tyxonq.agentskills.checks.knowledge` 解析出的知识块，一处写、两处用、零双写；
2. 提供 :mod:`tyxonq.agentskills.checks` 的六个**只读、不编译、不执行**体检函数，
   让 agent 复制即得真数字（门数 / 深度 / 三态基覆盖 / SWAP 下界 / 参数校验 / 知识检索 / 环境自检）。

**硬规则**（本包所有代码与 skill 文档都必须遵守）：

1. 不新建执行入口；执行一律回到统一链式 ``.run()``（HANDOFF 架构硬约束）。
2. :mod:`checks` 全部函数**严格只读**：入口与出口断言 ``circuit.ops`` 与传入 ``device_rule``
   不变；``compile()`` 会写入参对象的 ``_compiled_source``/``_compile_engine``/``_compile_opts``
   （见 ``compiler/api.py:234/244-245/290/306/322/338``），故本包一律不调 ``compile()``。
3. 不给 ``Circuit.compile()`` / ``Circuit.run()`` 加任何新关键字（``core/ir/circuit.py:806-813``
   的链式 no-op 短路逐参数判定「全默认」）。
4. 不把任何检查注册为 ``CompileStep``（``tests_core_module/test_compiler_pipeline.py:66-88``
   对步骤名序列做全等断言）。
5. 模块级不 import ``torch``/``pyscf``/``qiskit`` 及任何新第三方依赖；本包只 import stdlib，
   所有 ``tyxonq.*`` 原语在函数体内 import，保证 ``import tyxonq`` 的 0.54 s 开销零影响。
6. skill 与 ``.qoder/skills/`` shim **不得双写知识**：shim 只含 frontmatter + 一句
   「权威内容见 ``src/tyxonq/agentskills/<name>/SKILL.md``，先读它」。
7. 能力缺口必须显式写成 ``kind: gap`` 条目（含 ``reason`` 与 ``workaround``），让 agent
   立刻停止错误推理，而不是让它试错。

**公开 API**：
- :data:`SKILLS_DIR`：本文件所在目录（``Path``），skill 子目录的根。
- :func:`list_skills`：动态扫描 ``SKILLS_DIR`` 下含 ``SKILL.md`` 的子目录名，排序返回。
  **严禁硬编码 skill 名单**——并行创建的目录会立刻让硬编码过期。
- :func:`read`：读某 skill 的 ``SKILL.md`` 全文。
- :func:`path`：返回某 skill 目录的 ``Path``。
"""

from __future__ import annotations

from pathlib import Path
from typing import List

# 本文件所在目录 = skill 子目录的根。
SKILLS_DIR: Path = Path(__file__).resolve().parent


def list_skills() -> List[str]:
    """动态扫描 ``SKILLS_DIR`` 下含 ``SKILL.md`` 的子目录名，排序返回。

    严禁硬编码 skill 名单：并行任务正在创建这些目录，硬编码会立刻过期。
    目录不存在或不可读时返回空列表（不抛异常）。
    """
    if not SKILLS_DIR.is_dir():
        return []
    names: List[str] = []
    for entry in SKILLS_DIR.iterdir():
        if entry.is_dir() and (entry / "SKILL.md").is_file():
            names.append(entry.name)
    return sorted(names)


def read(name: str) -> str:
    """读 skill ``name`` 的 ``SKILL.md`` 全文（UTF-8）。

    Raises:
        FileNotFoundError: skill 目录或 ``SKILL.md`` 不存在。
    """
    return path(name).joinpath("SKILL.md").read_text(encoding="utf-8")


def path(name: str) -> Path:
    """返回 skill ``name`` 的目录 ``Path``（``SKILLS_DIR / name``）。

    不校验目录是否存在，调用方自行处理 ``FileNotFoundError``。
    """
    return SKILLS_DIR / str(name)


__all__ = ["SKILLS_DIR", "list_skills", "read", "path"]
