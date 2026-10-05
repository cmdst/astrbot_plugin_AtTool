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
- P0-5 标签语法单一事实来源（形态判定 / 载荷分类 / 拆分共用同一份正则）

缓存统一约定：缓存 value 的最后一个元素为创建时间戳（float），
过期判定为 `now - value[-1] >= ttl`，这样三种缓存（权限/成员/待选）
可共用同一套过期与容量维护逻辑。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

__all__ = [
    "AT_TAG_HEAD_PATTERN",
    "AT_TAG_LOOSE_PATTERN",
    "AT_TAG_PATTERN",
    "ROLE_LABELS",
    "format_role_label",
    "match_session_entry",
    "check_session_lists",
    "build_permission_cache_key",
    "has_at_tag",
    "is_at_all_in_cooldown",
    "parse_at_tag_payload",
    "split_text_by_at_tags",
    "format_single_member_result",
    "format_member_choice_list",
    "build_audit_record",
    "audit_log_path",
    "drop_expired",
    "evict_oldest_to_limit",
    "expand_alias_queries",
]

# 角色原始值 -> 中文展示标签。未知角色保留原值（兜底显示，不做猜测）。
ROLE_LABELS = {
    "owner": "群主",
    "admin": "管理员",
    "member": "成员",
}

# --------------------------------------------------------------------------- #
# @ 标签语法（P0-1 / P0-5 单一事实来源）
#
# 宿主序列化（[At:{qq}] / [At:全体成员]）与历史上下文（[At: 名字]）会诱导模型
# 写出多种标签形态：大小写（[At:123]）、冒号两侧空白/零宽（[at: 123]、
# [at\u200b:123]）、全角冒号（[at：123]）、载荷内零宽（[at:12\u200b3]）、
# "[" 与 at 之间夹零宽（[\u200bat:123]，r2 评审 F4）、非数字载荷（[at:柴郡]）。
# 下列正则由同一份片段拼接而成，渲染、降级、判定三处共用，避免多套正则互相
# 漂移（旧实现只认 `[at:数字]` 小写半角，导致上述形态原样穿链发到群里）。
#
# 注意：`[at 123]`（无冒号）、`[at]`、`[avatar:1]` 不是标签，按普通文本处理
# （不进入"裸标签"不变量判定范围）。
# --------------------------------------------------------------------------- #
_AT_TAG_HEAD_SRC = r"\[\u200b*(?i:at)[\s\u200b]*[:：]"
_AT_TAG_PAYLOAD_SRC = r"([^\]]*)"

# 载荷长度约束（r2 评审 F5）：QQ 号最长 12 位 ASCII 数字。更长的"纯数字"
# 只可能是模型幻觉/粘贴物，渲染成 At 反而是"合法但不存在"的艾特目标。
_AT_TAG_MAX_DIGITS: int = 12

# 标签起始语法（含未闭合）：存在性判定的唯一入口，替代旧的裸 "[at:" 子串判定
AT_TAG_HEAD_PATTERN = re.compile(_AT_TAG_HEAD_SRC)

# 完整标签外壳（任意载荷）：宽松判定与非数字载荷降级用
AT_TAG_LOOSE_PATTERN = re.compile(_AT_TAG_HEAD_SRC + _AT_TAG_PAYLOAD_SRC + r"\]")

# 严格解析：载荷为 ASCII 数字或 all（大小写不敏感），容忍冒号两侧空白与零宽
AT_TAG_PATTERN = re.compile(
    _AT_TAG_HEAD_SRC + r"[\s\u200b]*([0-9\u200b]+|(?i:all))[\s\u200b]*\]"
)

# 扫描用（split_text_by_at_tags 内部）：完整标签优先，失败则匹配未闭合语法
_AT_TAG_SCAN_PATTERN = re.compile(
    _AT_TAG_HEAD_SRC + r"(?P<body>[^\]]*)\]" + r"|(?P<head>" + _AT_TAG_HEAD_SRC + r")"
)


def has_at_tag(text: Optional[str]) -> bool:
    """判定文本是否含 @ 标签语法（含未闭合形态，P0-5）。

    Args:
        text: 待判定的纯文本；None/空串视为无标签。

    Returns:
        True 表示存在标签起始语法（`[at:` / `[At：` 等，闭合与否均可）。
    """
    if not text:
        return False
    return AT_TAG_HEAD_PATTERN.search(text) is not None


def parse_at_tag_payload(raw: object) -> Tuple[str, str]:
    """解析标签外壳内载荷的最终形态（P0-1/P0-5 唯一分类实现）。

    渲染（At 组件）与降级（纯文本 `@载荷`）共用本函数，避免两套判断漂移。

    Args:
        raw: 半角/全角冒号与闭合 `]` 之间的原始载荷文本（可含零宽字符）。

    Returns:
        (kind, payload)：
        - ("at", "123") / ("at", "all")：可解析载荷（1..12 位 ASCII 数字或 all）；
        - ("degrade", "柴郡")：不可解析的非空载荷（含全角数字、超长数字、
          名字、占位符）；
        - ("drop", "")：空载荷（`[at:]` / `[at: ]`）。
        载荷内零宽字符与嵌套标签语法会被剔除，保证返回值不再含标签语法。
    """
    payload = (str(raw) if raw is not None else "").replace("\u200b", "").strip()
    if AT_TAG_HEAD_PATTERN.search(payload):
        # 载荷里嵌套了标签语法（如 "[at:[at:123]"）：剔除语法只保留正文
        payload = (
            AT_TAG_HEAD_PATTERN.sub("", payload).replace("\u200b", "").strip()
        )
    if not payload:
        return "drop", ""
    if payload.isascii() and payload.isdigit() and len(payload) <= _AT_TAG_MAX_DIGITS:
        return "at", payload
    if payload.lower() == "all":
        return "at", "all"
    return "degrade", payload


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
    """把文本按 [at:xxx] 标签拆分为有序段（CHG-08 + P0-1 形态容忍）。

    支持的形态见 parse_at_tag_payload：大小写、冒号两侧空白/零宽、全角
    冒号、载荷内零宽、非数字载荷降级、空载荷删除、未闭合语法删除。

    Args:
        text: 待解析的纯文本。
        pattern: 可选自定义正则（须含一个载荷捕获组）；传入时按旧式
            "整体作为 ('at', 载荷)" 拆分，不做未闭合语法处理。

    Returns:
        段列表，每段为 (kind, value)：
        - ('text', 普通文本片段) —— 普通片段恒为非空；
        - ('at', 目标ID) —— ASCII 数字或 'all'，应渲染为 At 组件；
        - ('degrade', 载荷) —— 不可解析载荷，应降级为纯文本 `@载荷`；
        - ('drop', '') —— 空载荷，应删除标签语法；
        - ('unclosed', 起始语法原文) —— 未闭合标签，应删除该语法片段
          （流式 chunk 场景可选择原样保留，见 main.py keep_unclosed）。
        空文本返回空列表。
    """
    src = text or ""
    if not src:
        return []
    segments: List[Tuple[str, str]] = []
    last_idx = 0
    if pattern is not None:
        for match in pattern.finditer(src):
            start, end = match.span()
            if start > last_idx:
                segments.append(("text", src[last_idx:start]))
            segments.append(("at", match.group(1)))
            last_idx = end
    else:
        for match in _AT_TAG_SCAN_PATTERN.finditer(src):
            start, end = match.span()
            if start > last_idx:
                segments.append(("text", src[last_idx:start]))
            body = match.group("body")
            if body is None:
                # 起始语法命中但无闭合 "]": 未闭合标签，回报语法原文
                segments.append(("unclosed", match.group("head")))
            else:
                segments.append(parse_at_tag_payload(body))
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


def expand_alias_queries(query: object, aliases: object) -> List[str]:
    """按别名映射展开用户搜索词，返回去重保序的候选词列表。

    供 search_and_mention 在原始查询无命中时，用配置的别名（如「才俊」
    →「柴郡」）展开真实群名片再次搜索。别名配置兼容两种形态：
    字典 {"才俊": "柴郡"} / {"才俊": ["柴郡", "柴郡〔Bot〕"]}，
    以及 AstrBot 配置列表常用的 "别名=真名" 字符串。

    Args:
        query: 用户原始搜索词。
        aliases: 别名映射；None/空视为无别名。

    Returns:
        去重保序的候选词列表；首项恒为原始查询（非空时），
        其后为命中别名的目标真名/群名片。
    """
    original = (str(query) if query is not None else "").strip()
    candidates: List[str] = [original] if original else []
    if not aliases:
        return candidates

    items: List[Tuple[str, object]] = []
    if isinstance(aliases, dict):
        for key, value in aliases.items():
            alias_key = str(key).strip()
            if alias_key:
                items.append((alias_key, value))
    elif isinstance(aliases, (list, tuple)):
        for raw in aliases:
            if not isinstance(raw, str) or "=" not in raw:
                continue
            alias_key, _, target = raw.partition("=")
            if alias_key.strip() and target.strip():
                items.append((alias_key.strip(), target.strip()))

    for alias_key, target in items:
        if original != alias_key:
            continue
        targets = target if isinstance(target, (list, tuple)) else [target]
        for t in targets:
            cleaned = str(t).strip()
            if cleaned and cleaned not in candidates:
                candidates.append(cleaned)
    return candidates


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