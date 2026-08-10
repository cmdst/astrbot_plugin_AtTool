"""astrbot_plugin_AtTool 纯逻辑工具函数。

本模块刻意不导入任何 astrbot 框架模块，保持"纯正逻辑 + 零框架耦合"，
使下列变更点的核心判定逻辑可被单元测试直接覆盖（CHG-08）：

- CHG-01 权限缓存 key 构建（按 group_id:user_id 维度，修复会话级越权）
- CHG-02 会话黑白名单精确匹配（修复子串误命中）
- CHG-03 @全体频率限制判定、审计记录构建与按日日志路径
- CHG-05 会话级多结果列表格式化
- CHG-06 成员角色中文标注
- CHG-07 缓存过期清理与大小上限淘汰
- CHG-08 [at:xxx] 标签文本解析

缓存统一约定：缓存 value 的最后一个元素为创建时间戳（float），
过期判定为 `now - value[-1] >= ttl`，这样三种缓存（权限/成员/待选）
可共用同一套过期与容量维护逻辑。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

__all__ = [
    "AT_TAG_PATTERN",
    "ROLE_LABELS",
    "format_role_label",
    "match_session_entry",
    "check_session_lists",
    "build_permission_cache_key",
    "is_at_all_in_cooldown",
    "split_text_by_at_tags",
    "format_single_member_result",
    "format_member_choice_list",
    "build_audit_record",
    "audit_log_path",
    "drop_expired",
    "evict_oldest_to_limit",
]

# 角色原始值 -> 中文展示标签。未知角色保留原值（兜底显示，不做猜测）。
ROLE_LABELS = {
    "owner": "群主",
    "admin": "管理员",
    "member": "成员",
}

# @ 标签正则：匹配 [at:数字] 或 [at:all]。作为标签解析的单一事实来源。
AT_TAG_PATTERN = re.compile(r"\[at:(\d+|all)\]")


def format_role_label(role: object) -> str:
    """将群成员角色原始值转换为中文展示标签（CHG-06）。

    Args:
        role: 群成员信息中的 role 字段，常见 'owner'/'admin'/'member'。

    Returns:
        '群主'/'管理员'/'成员'；未知非空角色原样返回，空值兜底为 '成员'。
    """
    key = str(role if role is not None else "member").strip().lower()
    return ROLE_LABELS.get(key, key or "成员")


def match_session_entry(entry: object, unified_msg_origin: str, group_id: str) -> bool:
    """判断单个黑白名单条目是否命中当前会话（CHG-02 精确匹配）。

    修复前的子串匹配 `entry in umo` 会导致群号「123」误命中任意含「123」的
    UMO；此处仅在 entry 等于完整 UMO 或等于 group_id 时才命中。

    Args:
        entry: 单个名单条目（去空白前先做类型归一）。
        unified_msg_origin: 当前会话完整 UMO。
        group_id: 当前群号，私聊场景为空串。

    Returns:
        True 表示命中该会话。
    """
    norm = (str(entry) if entry is not None else "").strip()
    if not norm:
        return False
    umo = unified_msg_origin or ""
    gid = (group_id or "").strip()
    return norm == umo or (bool(gid) and norm == gid)


def check_session_lists(
    whitelist: Optional[List[str]],
    blacklist: Optional[List[str]],
    unified_msg_origin: str,
    group_id: str,
) -> Tuple[bool, str]:
    """综合判定会话是否允许使用艾特功能（CHG-02）。

    黑名单优先级高于白名单：任一黑名单条目命中即拒绝。
    白名单非空时，未命中任何条目即拒绝；白名单为空表示允许全部会话。

    Args:
        whitelist: 白名单条目列表（可为 None / 空列表）。
        blacklist: 黑名单条目列表（可为 None / 空列表）。
        unified_msg_origin: 当前会话完整 UMO。
        group_id: 当前群号（私聊为空串）。

    Returns:
        (allowed, deny_reason)：放行时 deny_reason 为空串。
    """
    umo = unified_msg_origin or ""
    gid = (group_id or "").strip()

    if blacklist:
        for raw in blacklist:
            if match_session_entry(raw, umo, gid):
                return False, "此会话已被列入艾特功能黑名单"

    if whitelist:
        for raw in whitelist:
            if match_session_entry(raw, umo, gid):
                return True, ""
        return False, "此会话未在白名单中，艾特功能已禁用"

    return True, ""


def build_permission_cache_key(group_id: object, user_id: object) -> str:
    """构建 @全体 权限缓存 Key（CHG-01 修复越权）。

    修复前以 unified_msg_origin（会话维度）为 key，导致同群不同用户共享
    缓存、管理员触发后普通成员被误判有权限。现改为 `group_id:user_id`
    维度，使同群不同用户的权限判定相互独立。

    Args:
        group_id: 群号。
        user_id: 操作者 ID。

    Returns:
        形如 `"123456:987654"` 的 key；群号或用户任一为空则返回空串
        （调用方应据此跳过缓存写入）。
    """
    gid = (str(group_id) if group_id is not None else "").strip()
    uid = (str(user_id) if user_id is not None else "").strip()
    if not gid or not uid:
        return ""
    return f"{gid}:{uid}"


def is_at_all_in_cooldown(
    last_trigger_ts: Optional[float], now: float, cooldown: float
) -> Tuple[bool, float]:
    """判定当前时刻是否处于 @全体 冷却期内（CHG-03）。

    Args:
        last_trigger_ts: 上一次放行 @全体 的时间戳；None 表示从未触发。
        now: 当前时间戳。
        cooldown: 冷却秒数；<=0 表示不限制。

    Returns:
        (in_cooldown, remain_seconds)：在冷却时 remain 为剩余秒数。
    """
    if cooldown <= 0 or last_trigger_ts is None:
        return False, 0.0
    elapsed = now - last_trigger_ts
    if elapsed < cooldown:
        return True, cooldown - elapsed
    return False, 0.0


def split_text_by_at_tags(
    text: Optional[str], pattern: Optional[re.Pattern] = None
) -> List[Tuple[str, str]]:
    """把文本按 [at:xxx] 标签拆分为有序段（CHG-08 标签解析纯逻辑）。

    Args:
        text: 待解析的纯文本。
        pattern: 可选自定义标签正则，默认使用 AT_TAG_PATTERN。

    Returns:
        段列表，每段为 (kind, value)：
        - ('text', 普通文本片段) —— 普通片段恒为非空；
        - ('at', 目标ID) —— 目标ID 为纯数字字符串或 'all'。
        空文本返回空列表。
    """
    if pattern is None:
        pattern = AT_TAG_PATTERN
    src = text or ""
    if not src:
        return []
    segments: List[Tuple[str, str]] = []
    last_idx = 0
    for match in pattern.finditer(src):
        start, end = match.span()
        if start > last_idx:
            segments.append(("text", src[last_idx:start]))
        segments.append(("at", match.group(1)))
        last_idx = end
    tail = src[last_idx:]
    if tail:
        segments.append(("text", tail))
    return segments


def format_single_member_result(
    display_name: str, user_id: str, role: object, prefix: str = "已找到："
) -> str:
    """格式化单个成员的搜索/选定结果文本（含角色标注，CHG-06）。

    Args:
        display_name: 展示名（群名片优先，回退昵称）。
        user_id: 成员 QQ 号。
        role: 角色原始值。
        prefix: 结果前缀，搜索用「已找到：」、序号选定用「已选定：」。

    Returns:
        包含角色标注与艾特标签使用提示的文本。
    """
    role_label = format_role_label(role)
    return (
        f"{prefix}{display_name}（{user_id}，{role_label}）\n"
        f"请直接在回复中使用 [at:{user_id}] 标签来艾特他。"
    )


def format_member_choice_list(
    name: str, matches: List[Tuple[str, str, object]]
) -> str:
    """格式化多候选成员列表（带序号与角色，CHG-05/06）。

    Args:
        name: 用户原始搜索词。
        matches: 候选成员列表，每项为 (user_id, display_name, role)。

    Returns:
        引导 LLM 让用户选择序号后调用 select_member_by_index 的多行文本。
    """
    count = len(matches)
    lines = [
        f"找到多个名称包含「{name}」的成员（共 {count} 个），"
        f"请把以下列表展示给用户让其选择序号（1~{count}），"
        f"然后使用用户选择的序号调用 select_member_by_index 工具："
    ]
    for index, (uid, dname, role) in enumerate(matches, start=1):
        lines.append(f"{index}. {dname} (ID: {uid}) - {format_role_label(role)}")
    return "\n".join(lines)


def build_audit_record(
    time_iso: str,
    group_id: str,
    operator_id: str,
    op_type: str,
    target_id: str,
    allowed: bool,
    reason: str = "",
) -> dict:
    """构建一条审计日志记录（CHG-03）。

    Args:
        time_iso: ISO8601 时间字符串（调用方负责格式化）。
        group_id: 群号。
        operator_id: 操作者 QQ 号。
        op_type: 操作类型，'at_member' / 'at_all'。
        target_id: 艾特目标 ID，成员为 QQ 号、@全体为 'all'。
        allowed: 是否放行。
        reason: 原因说明，放行可为 'ok'，拦截为具体原因。

    Returns:
        结构化审计字段字典，供 JSONL 落盘。
    """
    return {
        "time": time_iso,
        "group_id": group_id,
        "operator_id": operator_id,
        "op_type": op_type,
        "target_id": target_id,
        "allowed": bool(allowed),
        "reason": reason or "",
    }


def audit_log_path(data_dir: object, date_str: str) -> Path:
    """计算审计日志文件路径（CHG-03 按日切割）。

    Args:
        data_dir: 插件数据目录（StarTools.get_data_dir 返回值）。
        date_str: 日期字符串，形如 'YYYYMMDD'。

    Returns:
        形如 `<data_dir>/audit/at_audit_YYYYMMDD.jsonl` 的路径。
    """
    return Path(data_dir) / "audit" / f"at_audit_{date_str}.jsonl"


def drop_expired(cache: dict, ttl: float, now: float) -> int:
    """惰性清理缓存中已过期项（CHG-07）。

    约定缓存 value 的最后一个元素为创建时间戳（float），过期判定为
    `now - value[-1] >= ttl`。

    Args:
        cache: 缓存字典（原地修改）。
        ttl: 过期阈值秒数；<=0 时不处理。
        now: 当前时间戳。

    Returns:
        本次清理掉的条目数。
    """
    if ttl <= 0:
        return 0
    expired_keys = [key for key, value in cache.items() if (now - value[-1]) >= ttl]
    for key in expired_keys:
        del cache[key]
    return len(expired_keys)


def evict_oldest_to_limit(cache: dict, max_size: int) -> int:
    """缓存超容时按插入顺序淘汰最早写入的项（CHG-07）。

    依赖 dict 保序特性（Python 3.7+）， longevity 准入式 LRU。

    Args:
        cache: 缓存字典（原地修改）。
        max_size: 最大容量；<=0 时不处理。

    Returns:
        本次淘汰的条目数。
    """
    if max_size <= 0:
        return 0
    evicted = 0
    while len(cache) > max_size:
        oldest_key = next(iter(cache), None)
        if oldest_key is None:
            break
        del cache[oldest_key]
        evicted += 1
    return evicted