"""受限 YAML 子集解析器：解析内嵌在 SKILL.md 里的 ``tyxonq-knowledge`` 块。

零第三方依赖（只用 stdlib ``re`` / ``pathlib``）。识别 markdown 中 info string 为
``tyxonq-knowledge`` 的 fenced code block（```tyxonq-knowledge ... ```），把块内文本
解析成 ``dict``。

**支持的语法（仅限）**：
- ``key: value``：标量，去除首尾空白；
- ``key: [a, b, c]``：内联列表，元素为字符串（去空白、去引号）；
- ``key: |``：块标量，后续缩进行按原样保留、去掉公共缩进。

**明确不支持**（docstring 强制声明，护栏测试据此拒绝复杂块）：
- 嵌套映射（``key:`` 后再缩进一层 ``subkey: value``）；
- 锚点 / 别名（``&anchor`` / ``*alias``）；
- 多文档（``---`` 分隔）；
- 引号转义规则（``\\n`` / ``\\"`` 等不解析，原样保留）；
- 注释以外的复杂结构（``# comment`` 行被跳过，但行内 ``#`` 不当注释）。

**固定字段集**：``id / kind / tags / applies_when_ops / applies_when_module / statement /
why / source_ref / verified_by / confidence / reason / workaround``。

- ``kind ∈ {"fact","bound","gap","convention"}``（缺省 ``"fact"``）
- ``confidence ∈ {"verified","stated"}``（缺省 ``"stated"``）
- ``id`` 与 ``statement`` 为必填，缺失即报错。

**对外 API**：
- :func:`parse_blocks`：从 markdown 文本解析全部知识块。
- :func:`parse_file`：从文件路径解析全部知识块。

每个返回条目额外附带 ``_skill``（所属 skill 名）与 ``_line``（块在原文件中的起始行号），
便于报错定位与后续护栏测试。解析失败时抛 ``ValueError``，消息中必须含**块 id（若已知）
与文件内行号**，严禁静默跳过。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# fenced code block 正则：```tyxonq-knowledge ... ```
_FENCE_RE = re.compile(
    r"^```tyxonq-knowledge\s*\n(?P<body>.*?)^```\s*$",
    re.MULTILINE | re.DOTALL,
)

# 固定字段集（缺省值见 _DEFAULTS）。
_ALLOWED_FIELDS = frozenset(
    {
        "id", "kind", "tags", "applies_when_ops", "applies_when_module",
        "statement", "why", "source_ref", "verified_by", "confidence",
        "reason", "workaround",
    }
)

_DEFAULTS: Dict[str, Any] = {
    "kind": "fact",
    "confidence": "stated",
    "tags": [],
    "applies_when_ops": [],
    "applies_when_module": "",
    "why": "",
    "source_ref": "",
    "verified_by": "",
    "reason": "",
    "workaround": "",
}

_VALID_KINDS = frozenset({"fact", "bound", "gap", "convention"})
_VALID_CONFIDENCE = frozenset({"verified", "stated"})

# 内联列表正则：[a, b, c]
_INLINE_LIST_RE = re.compile(r"^\[(?P<inner>.*)\]$")


def _strip_quotes(s: str) -> str:
    """去除首尾配对引号（单/双），不解析转义。"""
    s = s.strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    return s


def _parse_inline_list(raw: str) -> List[str]:
    """解析 ``[a, b, c]`` 内联列表，元素为字符串（去空白、去引号）。"""
    m = _INLINE_LIST_RE.match(raw.strip())
    if not m:
        raise ValueError(f"内联列表格式错误，期望 [a, b, c]，收到: {raw!r}")
    inner = m.group("inner").strip()
    if not inner:
        return []
    return [_strip_quotes(part) for part in inner.split(",")]


def _parse_block_body(body: str, start_line: int, skill: str) -> Dict[str, Any]:
    """解析单个知识块的 body 文本，返回 dict。

    Args:
        body: 块内文本（不含 fence 行）。
        start_line: 块在原文件中的起始行号（1-based，用于报错定位）。
        skill: 所属 skill 名（写入 ``_skill``）。

    Raises:
        ValueError: 字段缺失、格式错误、kind/confidence 非法。
    """
    lines = body.splitlines()
    result: Dict[str, Any] = {}
    block_id: Optional[str] = None

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 跳过空行与注释行。
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        # 解析 key: value 或 key: |
        if ":" not in stripped:
            raise ValueError(
                f"块起始行 {start_line}：无法解析的行（缺 ':'）: {line!r}"
                + (f"（块 id={block_id}）" if block_id else "")
            )

        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()

        if key not in _ALLOWED_FIELDS:
            raise ValueError(
                f"块起始行 {start_line}：未知字段 {key!r}"
                + (f"（块 id={block_id}）" if block_id else "")
            )

        if key == "id":
            block_id = value

        # 块标量：key: |
        if value == "|":
            block_lines: List[str] = []
            i += 1
            # 收集后续缩进行（至少 2 空格缩进），直到遇到非缩进行或块结束。
            min_indent: Optional[int] = None
            while i < len(lines):
                raw = lines[i]
                if not raw.strip():
                    block_lines.append("")
                    i += 1
                    continue
                indent = len(raw) - len(raw.lstrip())
                if indent < 2:
                    break
                if min_indent is None or indent < min_indent:
                    min_indent = indent
                block_lines.append(raw)
                i += 1
            # 去掉公共缩进。
            if min_indent is not None:
                block_lines = [
                    ln[min_indent:] if ln.strip() else ""
                    for ln in block_lines
                ]
            result[key] = "\n".join(block_lines).strip()
            continue

        # 内联列表：key: [a, b, c]
        if value.startswith("["):
            try:
                result[key] = _parse_inline_list(value)
            except ValueError as exc:
                raise ValueError(
                    f"块起始行 {start_line}：字段 {key!r} 解析失败: {exc}"
                    + (f"（块 id={block_id}）" if block_id else "")
                ) from exc
            i += 1
            continue

        # 标量：key: value
        result[key] = _strip_quotes(value)
        i += 1

    # 必填字段校验。
    if "id" not in result or not result["id"]:
        raise ValueError(f"块起始行 {start_line}：缺少必填字段 'id'")
    if "statement" not in result or not result["statement"]:
        raise ValueError(
            f"块起始行 {start_line}：缺少必填字段 'statement'（块 id={result['id']}）"
        )

    # kind / confidence 合法性校验。
    kind = result.get("kind", _DEFAULTS["kind"])
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"块起始行 {start_line}：kind={kind!r} 非法，"
            f"期望 {_VALID_KINDS}（块 id={result['id']}）"
        )
    confidence = result.get("confidence", _DEFAULTS["confidence"])
    if confidence not in _VALID_CONFIDENCE:
        raise ValueError(
            f"块起始行 {start_line}：confidence={confidence!r} 非法，"
            f"期望 {_VALID_CONFIDENCE}（块 id={result['id']}）"
        )

    # 填充缺省值。
    for field, default in _DEFAULTS.items():
        if field not in result:
            result[field] = list(default) if isinstance(default, list) else default

    # 附带元信息。
    result["_skill"] = skill
    result["_line"] = start_line
    return result


def parse_blocks(text: str, *, skill: str = "") -> List[Dict[str, Any]]:
    """从 markdown 文本解析全部 ``tyxonq-knowledge`` 块。

    Args:
        text: markdown 全文。
        skill: 所属 skill 名（写入每条结果的 ``_skill``）。

    Returns:
        知识块 dict 列表，按块在文本中的出现顺序排列。

    Raises:
        ValueError: 任一块解析失败（消息含块 id 与行号）。
    """
    results: List[Dict[str, Any]] = []
    for m in _FENCE_RE.finditer(text):
        # 计算块起始行号（1-based）。
        start_line = text[: m.start()].count("\n") + 1
        body = m.group("body")
        results.append(_parse_block_body(body, start_line, skill))
    return results


def parse_file(path: Any) -> List[Dict[str, Any]]:
    """从文件路径解析全部 ``tyxonq-knowledge`` 块。

    Args:
        path: 文件路径（``str`` 或 ``Path``）。skill 名取父目录名。

    Returns:
        知识块 dict 列表。

    Raises:
        ValueError: 任一块解析失败。
        FileNotFoundError: 文件不存在。
    """
    p = Path(path)
    skill = p.parent.name
    text = p.read_text(encoding="utf-8")
    return parse_blocks(text, skill=skill)


__all__ = ["parse_blocks", "parse_file"]
