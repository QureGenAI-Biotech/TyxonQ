"""国盾平台驱动：提交、查询、取消和设备发现。"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from types import MethodType
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from ..config import get_token as _hw_get_token


logger = logging.getLogger(__name__)
_CQLIB_VERSION = "1.3.11"
_DEVICE_CODE = re.compile(r"^(?:gd|guodun)_[A-Za-z0-9_-]+$")
_QUBIT_TOKEN = re.compile(r"\bQ\d+\b")
_COUPLER_TOKEN = re.compile(r"\bG\d+\b")
_PULSE_INSTRUCTIONS = {"PXY", "PZ", "PZ0", "G", "I", "B"}
_TOPOLOGY_DEVICES = {"gd_sim1", "gd_qc1"}
_TOKEN_HELP = (
    "Guodun token required. 请通过显式 token=...、"
    "tq.set_token(..., provider='guodun') 或 TYXONQ_GUODUN_TOKEN 提供密钥。"
)


def _load_platform_class():
    """延迟加载并锁定本期验证过的 cqlib 版本。"""
    try:
        installed_version = version("cqlib")
        from cqlib import GuoDunPlatform
    except (ImportError, PackageNotFoundError) as exc:
        raise RuntimeError(
            "Guodun 驱动需要可选依赖 cqlib==1.3.11；"
            "请执行 `pip install 'tyxonq[guodun]'`。"
        ) from exc
    if installed_version != _CQLIB_VERSION:
        raise RuntimeError(
            "Guodun 驱动当前仅兼容 cqlib==1.3.11，"
            f"检测到版本 {installed_version!r}。"
        )
    return GuoDunPlatform


def _load_get_pulse_class():
    """延迟加载与 cqlib 1.3.11 配套的 GetPulse。"""
    _load_platform_class()
    from cqlib.utils.get_pulse import GetPulse

    return GetPulse


def _resolve_token(token: Optional[str]) -> str:
    """固定顺序：显式参数 -> set_token -> 专用环境变量。"""
    if token:
        return token
    configured = _hw_get_token(provider="guodun", env_fallback=False)
    if configured:
        return configured
    environment = os.getenv("TYXONQ_GUODUN_TOKEN")
    if environment:
        return environment
    raise RuntimeError(_TOKEN_HELP)


def _disable_automatic_request_retry(platform: Any) -> None:
    """移除 cqlib 1.3.11 `_send_request` 上的隐式 HTTP 重试装饰器。"""
    original_request = getattr(platform._send_request, "__wrapped__", None)
    if original_request is None:
        raise RuntimeError(
            "当前 cqlib 结构与 1.3.11 兼容基线不一致，"
            "无法安全关闭自动重试，停止连接和提交。"
        )
    platform._send_request = MethodType(original_request, platform)


def _create_platform(token: str, device: str):
    """先构造未登录对象并关闭重试，再执行一次显式登录。"""
    platform_class = _load_platform_class()
    platform = platform_class(
        login_key=token,
        machine_name=device,
        auto_login=False,
    )
    _disable_automatic_request_retry(platform)
    platform.login()
    return platform


def _normalize_device(device: str) -> str:
    name = str(device).split("::")[-1]
    if not name:
        raise ValueError("Guodun device 不能为空")
    return name


def _normalize_sources(
    source: Optional[Union[str, Sequence[str]]]
) -> List[str]:
    sources = list(source) if isinstance(source, (list, tuple)) else [source]
    if not sources or sources == [None]:
        raise ValueError("Guodun 驱动需要 QCIS source")
    for item in sources:
        if not isinstance(item, str):
            raise TypeError("Guodun source 必须是 QCIS 字符串或字符串列表")
        if not item.strip():
            raise ValueError("Guodun QCIS source 不能为空")
        if not any(line.strip().startswith("M ") for line in item.splitlines()):
            raise ValueError("Guodun QCIS 必须包含显式 M 测量指令")
    return sources  # type: ignore[return-value]


def _normalize_shots(
    shots: Union[int, Sequence[int]], source_count: int
) -> int:
    if isinstance(shots, (list, tuple)):
        shots_list = [int(item) for item in shots]
        if len(shots_list) != source_count:
            raise ValueError("shots 列表长度必须与批量线路数一致")
        if len(set(shots_list)) != 1:
            raise ValueError("Guodun 同一批次中的所有线路必须使用相同 shots")
        num_shots = shots_list[0]
    else:
        num_shots = int(shots)
    if num_shots <= 0:
        raise ValueError("shots 必须是正整数")
    return num_shots


def _contains_pulse_instructions(qcis_sources: Sequence[str]) -> bool:
    """判断批次中是否含需要当前标定校验的 pulse-QCIS。"""
    return any(
        line.split()[0] in _PULSE_INSTRUCTIONS
        for source in qcis_sources
        for line in source.splitlines()
        if line.strip()
    )


def _parse_name_set(value: Any, field_name: str) -> set[str]:
    """解析配置中的逗号字符串；结构异常时失败关闭。"""
    if value in (None, ""):
        return set()
    if not isinstance(value, str):
        raise RuntimeError(f"国盾配置字段 {field_name!r} 结构异常")
    return {item.strip() for item in value.split(",") if item.strip()}


def _validate_topology(qcis_sources: Sequence[str], config: Any) -> None:
    """按当前配置校验禁用比特、禁用耦合器和每一条 CZ 边。"""
    if not isinstance(config, dict):
        raise RuntimeError("国盾配置缺失或不是字典，停止提交")
    if "disabledQubits" not in config or "disabledCouplers" not in config:
        raise RuntimeError("国盾配置缺少 disabledQubits/disabledCouplers，停止提交")
    overview = config.get("overview")
    if not isinstance(overview, dict):
        raise RuntimeError("国盾配置缺少 overview，停止提交")
    coupler_map = overview.get("coupler_map")
    if not isinstance(coupler_map, dict):
        raise RuntimeError("国盾配置缺少有效 coupler_map，停止提交")

    disabled_qubits = _parse_name_set(config["disabledQubits"], "disabledQubits")
    disabled_couplers = _parse_name_set(
        config["disabledCouplers"], "disabledCouplers"
    )
    active_edges: set[frozenset[str]] = set()
    active_couplers: dict[str, frozenset[str]] = {}
    for coupler, qubits in coupler_map.items():
        if coupler in disabled_couplers:
            continue
        if not isinstance(qubits, (list, tuple)) or len(qubits) != 2:
            raise RuntimeError(f"耦合器 {coupler!r} 的配置结构异常")
        q0, q1 = str(qubits[0]), str(qubits[1])
        if q0 in disabled_qubits or q1 in disabled_qubits:
            continue
        edge = frozenset((q0, q1))
        active_edges.add(edge)
        active_couplers[str(coupler)] = edge

    for qcis in qcis_sources:
        used_qubits = set(_QUBIT_TOKEN.findall(qcis))
        blocked_qubits = used_qubits & disabled_qubits
        if blocked_qubits:
            raise RuntimeError(
                f"线路使用了当前禁用物理比特：{sorted(blocked_qubits)}"
            )
        used_couplers = set(_COUPLER_TOKEN.findall(qcis))
        blocked_couplers = used_couplers - set(active_couplers)
        if blocked_couplers:
            raise RuntimeError(
                f"线路使用了当前禁用或未知耦合器：{sorted(blocked_couplers)}"
            )
        for line in qcis.splitlines():
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "CZ":
                if len(tokens) != 3 or not all(
                    _QUBIT_TOKEN.fullmatch(q) for q in tokens[1:]
                ):
                    raise RuntimeError(f"无法解析 CZ 指令：{line!r}")
                edge = frozenset((tokens[1], tokens[2]))
                if edge not in active_edges:
                    raise RuntimeError(
                        f"线路使用了当前不可用的 CZ 边：{tokens[1]}-{tokens[2]}"
                    )
            elif tokens[0] == "B":
                if (
                    len(tokens) != 4
                    or not all(_QUBIT_TOKEN.fullmatch(q) for q in tokens[1:3])
                    or not _COUPLER_TOKEN.fullmatch(tokens[3])
                ):
                    raise RuntimeError(f"无法解析 B 指令：{line!r}")
                edge = frozenset((tokens[1], tokens[2]))
                if active_couplers.get(tokens[3]) != edge:
                    raise RuntimeError(
                        f"B 指令中的 {tokens[3]} 不连接 {tokens[1]}-{tokens[2]}"
                    )


@dataclass
class GuodunTask:
    """保存已有 query ID；后续查询和取消都只操作这个 ID。"""

    id: Any
    device: str
    status: str = "submitted"
    platform: Any = field(default=None, repr=False)
    async_result: bool = True


@dataclass
class GuodunPulseContext:
    """保存同一次只读标定下载得到的平台和 GetPulse 对象。"""

    device: str
    platform: Any = field(repr=False)
    get_pulse: Any = field(repr=False)


@dataclass
class GuodunWaveformTask:
    """保存 waveform query ID；查询不会创建新的波形任务。"""

    id: Any
    device: str
    platform: Any = field(repr=False)
    status: str = "pending"


def open_pulse_context(
    device: str,
    token: Optional[str] = None,
) -> GuodunPulseContext:
    """登录真机并下载一次当前标定；本函数不会提交实验或波形任务。"""
    machine = _normalize_device(device)
    if machine != "gd_qc1":
        raise ValueError("GetPulse 当前只允许用于 gd_qc1 真机")
    platform = _create_platform(_resolve_token(token), machine)
    get_pulse = _load_get_pulse_class()(platform=platform)
    _validate_topology([], get_pulse.config)
    return GuodunPulseContext(
        device=machine,
        platform=platform,
        get_pulse=get_pulse,
    )


def create_waveform(
    context: GuodunPulseContext,
    source: str,
    *,
    circuit_name: Optional[str] = None,
) -> GuodunWaveformTask:
    """创建恰好一个 waveform 任务；不在本函数内查询或重建。"""
    from tyxonq.compiler.compile_engine.guodun.pulse import validate_pulse_qcis

    validate_pulse_qcis(source, get_pulse=context.get_pulse)
    _validate_topology([source], context.get_pulse.config)
    query_id = context.platform.create_waveform_data(
        circuit=source,
        circuit_name=circuit_name,
        is_verify=True,
    )
    if not isinstance(query_id, (str, int)):
        raise RuntimeError("平台没有返回有效的 waveform query ID；停止且不重建")
    return GuodunWaveformTask(
        id=query_id,
        device=context.device,
        platform=context.platform,
    )


def get_waveform(task: GuodunWaveformTask) -> Dict[str, Any]:
    """只查询已有 waveform ID；空 URL 表示尚未提供可下载结果。"""
    url = task.platform.query_waveform_data(task.id)
    status = "completed" if url else "pending"
    task.status = status
    return {
        "query_id": task.id,
        "device": task.device,
        "status": status,
        "url": url,
    }


def run(
    device: str,
    token: Optional[str] = None,
    *,
    source: Optional[Union[str, Sequence[str]]] = None,
    shots: Union[int, Sequence[int]] = 1024,
    **opts: Any,
) -> List[GuodunTask]:
    """一次 `submit_job` 提交一个批次，不做自动重提。"""
    machine = _normalize_device(device)
    resolved_token = _resolve_token(token)
    sources = _normalize_sources(source)
    num_shots = _normalize_shots(shots, len(sources))

    platform = _create_platform(resolved_token, machine)
    if _contains_pulse_instructions(sources):
        if machine != "gd_qc1":
            raise ValueError("pulse-QCIS 当前只允许提交到 gd_qc1")
        from tyxonq.compiler.compile_engine.guodun.pulse import validate_pulse_qcis

        # GetPulse 初始化会下载提交时刻的当前标定；所有脉冲逐点通过后才提交。
        get_pulse = _load_get_pulse_class()(platform=platform)
        for pulse_source in sources:
            validate_pulse_qcis(
                pulse_source,
                get_pulse=get_pulse,
                require_measurement=True,
            )
        _validate_topology(sources, get_pulse.config)
    elif machine in _TOPOLOGY_DEVICES:
        config = platform.download_config()
        _validate_topology(sources, config)

    query_ids = platform.submit_job(
        circuit=sources,
        exp_name=str(opts.pop("exp_name", opts.pop("task_name", "TyxonQJob"))),
        num_shots=num_shots,
        is_verify=True,
    )
    if not isinstance(query_ids, (list, tuple)) or len(query_ids) != len(sources):
        raise RuntimeError(
            "Guodun submit_job 返回的 query ID 数量与线路数不一致；"
            "为避免重提，已停止。"
        )
    return [
        GuodunTask(id=query_id, device=machine, platform=platform)
        for query_id in query_ids
    ]


def _map_status(run_status: Any) -> str:
    if run_status == 2:
        return "completed"
    if run_status == 3:
        return "failed"
    return "unknown" if run_status is None else str(run_status)


def _parse_result_status(result_status: Any) -> tuple[Dict[str, int], Any, str]:
    """保持平台 bit 顺序，统计 resultStatus[1:]。"""
    if result_status in (None, []):
        return {}, None, ""
    if not isinstance(result_status, list) or not isinstance(result_status[0], list):
        return {}, None, "resultStatus 结构异常"
    measured_qubits = result_status[0]
    width = len(measured_qubits)
    counts: Counter[str] = Counter()
    for shot_index, row in enumerate(result_status[1:], start=1):
        if not isinstance(row, list) or len(row) != width:
            return {}, measured_qubits, f"第 {shot_index} 个 shot 宽度异常"
        bits: list[str] = []
        for bit in row:
            if isinstance(bit, str) and bit in ("0", "1"):
                bits.append(bit)
            elif isinstance(bit, int) and not isinstance(bit, bool) and bit in (0, 1):
                bits.append(str(bit))
            else:
                return {}, measured_qubits, f"第 {shot_index} 个 shot 包含非二进制值"
        counts["".join(bits)] += 1
    return dict(counts), measured_qubits, ""


def _is_transient_query_error(exc: Exception) -> bool:
    """识别 cqlib 在任务排队或校准期间可能抛出的查询异常。"""
    try:
        from cqlib.exceptions import CqlibRequestError
    except (ImportError, AttributeError):  # pragma: no cover - 兼容可选依赖缺失
        CqlibRequestError = ()  # type: ignore[assignment]
    return isinstance(exc, CqlibRequestError) or (
        exc.__class__.__name__ == "CqlibRequestError"
    )


def get_task_details(task: GuodunTask, token: Optional[str] = None) -> Dict[str, Any]:
    """仅轮询 task 中已有 ID，不创建任何新任务。"""
    if task.platform is None:
        raise RuntimeError("GuodunTask 缺少平台对象，不能查询")
    try:
        responses = task.platform.query_experiment(
            query_id=[task.id],
            max_wait_time=1,
            sleep_time=1,
        )
    except Exception as exc:
        # 平台校准或任务仍在排队时，cqlib 可能暂时查询失败。
        # 返回非终态，让统一轮询稍后继续查询同一个 ID；绝不重新提交。
        if not _is_transient_query_error(exc):
            raise
        return {
            "result": {},
            "result_meta": {
                "query_id": task.id,
                "device": task.device,
                "measurement_order": None,
                "probability": None,
                "query_error": f"{type(exc).__name__}: {exc}",
                "raw": None,
            },
            "uni_status": "unknown",
            "error": "",
        }
    if not responses:
        return {
            "result": {},
            "result_meta": {
                "query_id": task.id,
                "device": task.device,
                "measurement_order": None,
                "probability": None,
                "raw": responses,
            },
            "uni_status": "unknown",
            "error": "",
        }
    raw = responses[0] if isinstance(responses, list) else responses
    if not isinstance(raw, dict):
        return {
            "result": {},
            "result_meta": {"query_id": task.id, "device": task.device, "raw": raw},
            "uni_status": "error",
            "error": f"Guodun 返回了无法解析的结果：{raw!r}",
        }

    counts, measurement_order, parse_error = _parse_result_status(
        raw.get("resultStatus")
    )
    status = _map_status(raw.get("runStatus"))
    if parse_error:
        status = "error"
    return {
        "result": counts,
        "result_meta": {
            "query_id": task.id,
            "device": task.device,
            "shots": sum(counts.values()) if counts else 0,
            "measurement_order": measurement_order,
            "probability": raw.get("probability"),
            "raw": raw,
        },
        "uni_status": status,
        "error": parse_error or raw.get("msg") or raw.get("error") or "",
    }


def remove_task(task: GuodunTask, token: Optional[str] = None) -> Any:
    """取消 task 中已有 query ID。"""
    if task.platform is None:
        raise RuntimeError("GuodunTask 缺少平台对象，不能取消")
    return task.platform.stop_running_experiments(query_id=task.id)


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_strings(key)
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)


def list_devices(token: Optional[str] = None, **kws: Any) -> List[str]:
    """从 SDK 公共查询结果动态提取机器代码；无密钥时返回空列表。"""
    try:
        resolved_token = _resolve_token(token)
    except RuntimeError as exc:
        logger.warning("Guodun list_devices: %s", exc)
        return []

    try:
        platform = _create_platform(
            resolved_token, str(kws.get("device", "gd_test"))
        )
        rows = platform.query_quantum_computer_list()
    except Exception as exc:
        logger.warning("Guodun list_devices 查询失败: %s", exc)
        return []
    codes = {value for value in _iter_strings(rows) if _DEVICE_CODE.fullmatch(value)}
    return [f"guodun::{code}" for code in sorted(codes)]


__all__ = [
    "GuodunTask",
    "GuodunPulseContext",
    "GuodunWaveformTask",
    "run",
    "get_task_details",
    "remove_task",
    "list_devices",
    "open_pulse_context",
    "create_waveform",
    "get_waveform",
]
