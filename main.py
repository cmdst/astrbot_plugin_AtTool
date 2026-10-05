import asyncio
import json
import re
import time
from datetime import datetime
from typing import Any, List, Optional, Tuple

from astrbot.api.star import Context, Star, StarTools
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api import AstrBotConfig, logger
from astrbot.api.provider import ProviderRequest
from astrbot.api.message_components import Plain, At, BaseMessageComponent
from astrbot.core.agent.message import TextPart  # noqa: F401 常规用法，见 P2-6 说明

from .utils import (
    AT_TAG_HEAD_PATTERN,
    AT_TAG_LOOSE_PATTERN,
    AT_TAG_PATTERN,
    audit_log_path,
    build_audit_record,
    build_permission_cache_key,
    check_session_lists,
    drop_expired,
    evict_oldest_to_limit,
    expand_alias_queries,
    format_member_choice_list,
    format_single_member_result,
    has_at_tag,
    is_at_all_in_cooldown,
    USER_QQ_PATTERN,
    parse_at_tag_payload,
    split_text_by_at_tags,
)

# --------------------------------------------------------------------------- #
# 常量：缓存/冷却/容量上限（CHG-03/04/07）
# --------------------------------------------------------------------------- #
_PERMISSION_CACHE_TTL: float = 300.0  # 权限缓存 TTL（秒），与原 v2.4.0 一致
_PERMISSION_CACHE_MAX: int = 500  # 权限缓存上限，超过淘汰最早项（CHG-07）
_MEMBER_CACHE_MAX: int = 50  # 群成员列表缓存群数上限（CHG-04/07）
_PENDING_CHOICE_TTL: float = 120.0  # 会话级多结果临时存储 TTL（秒，CHG-05）
_MAX_PENDING_CHOICES: int = 200  # 会话级待选缓存上限，超过淘汰最早项（CHG-07，P2-3）
_FALLBACK_AT_TTL: float = 120.0  # 兜底补插缓存 TTL（秒，修复 2）：search_and_mention
# 唯一命中后，若 LLM 最终回复缺 [at:ID] 标签，process_at_tags 在 TTL 内自动补插
_MAX_MULTI_MATCHES: int = 30  # 命中成员过多时拒绝列出，提示细化关键词
_MEMBER_LIST_MAX_RETRIES: int = 1  # 群成员列表拉取失败后的重试次数（P2-7）
# T10：会话级"工具确认过的艾特 ID"记忆的 TTL 与容量（与兜底/待选缓存同款治理）
_KNOWN_AT_ID_TTL: float = 120.0  # 秒；与 _FALLBACK_AT_TTL/_PENDING_CHOICE_TTL 一致
_MAX_TRUSTED_SLOTS: int = 500  # (会话, 发起者) 槽位上限，超限淘汰最早写入（CHG-07）
_MEMBER_LIST_RETRY_DELAY: float = 0.5  # 重试间隔（秒，P2-7）

# 审计操作类型枚举（CHG-03）
_AUDIT_OP_AT_MEMBER = "at_member"
_AUDIT_OP_AT_ALL = "at_all"

# P0-1 跨组件拼合窗口：标签前缀与闭合 "]" 之间最多累积的字符数，
# 超出即认为不是被切碎的标签（避免长正文被误吞）。
_SPLIT_AT_WINDOW: int = 64

# P0-1 跨组件拼合次数上限（每链）：防止病态输入下的反复重建。
_SPLIT_AT_MAX_ROUNDS: int = 8

# 艾特意图关键词（P0-4 可观测：区分"工具未被调用"与"模型未按提示词调用"）
_AT_INTENT_PATTERN = re.compile(r"艾特|叫一下|喊一下|呼叫|@|\bat\b", re.IGNORECASE)


def _strip_clean_prefix(text: str, n: int) -> str:
    """从 text 中剔除前 n 个非零宽字符，其余字符（含零宽）按原序保留。

    供 _recover_cross_component_at_tag 跨组件拼合剥离标签片段使用：
    标签内容按"清洗零宽后"的字符计数消耗，但剥离时保留 At 后补零宽
    前缀（finalize 语义），避免误删平台艾特提示保护字符。
    """
    kept: List[str] = []
    seen = 0
    for ch in text:
        if ch == "\u200b":
            kept.append(ch)  # 零宽始终保留
        elif seen >= n:
            kept.append(ch)
        else:
            seen += 1
    return "".join(kept)

# 插件名（与 metadata.yaml name 保持一致，用于定位插件数据目录）
_PLUGIN_NAME = "astrbot_plugin_AtTool"


def _to_int(value: object, default: int) -> int:
    """把配置值安全转换为 int，None/空串/非法值回退默认值（P1-1 防御）。

    接受 int/float/str：int 原样返回，float 按 int() 截断（含 6.5 -> 6），
    str 去空白后解析；None、空串、非数字字符串、bool、非有限浮点
    （nan/inf）均回退 default，保证插件初始化永不因配置异常崩溃。
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, str) and not value.strip():
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _to_bool(value: object) -> bool:
    """把配置值安全转换为 bool（P2-1 防御）。

    字符串按内容解析：strip().lower() 后为 "false"/"0"/"" 视为 False，
    其余字符串视为 True；非字符串直接 bool(v)。修复 bool("false")==True
    导致配置以字符串形式出现时语义反转的问题。
    """
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "")
    return bool(value)


def _is_aiocqhttp_event(event: object) -> bool:
    """鸭子类型判定事件是否来自 aiocqhttp 平台（P2-6）。

    不再 isinstance 依赖 astrbot.core 内部类 AiocqhttpMessageEvent（内部
    实现路径无稳定 API 承诺，跨版本重构即可能 import 失败）；改为检查
    event 是否暴露 bot.api（aiocqhttp 事件的特征），任何异常降级为 False
    （视为平台不支持）。
    """
    try:
        return hasattr(event, "bot") and hasattr(event.bot, "api")
    except Exception:
        return False


class LLMAtToolPlugin(Star):
    """让 LLM 拥有真实艾特群成员能力的插件。"""

    def __init__(self, context: Context, config: Optional[AstrBotConfig] = None):
        super().__init__(context)
        self.config = config if config is not None else {}

        # ---- 既有配置项 ----
        self.permission_verification = _to_bool(
            self.config.get("permission_verification", True)
        )
        self.allow_direct_qq_at = _to_bool(
            self.config.get("allow_direct_qq_at", True)
        )
        self.enable_fuzzy_search = _to_bool(
            self.config.get("enable_fuzzy_search", True)
        )
        self.session_whitelist: list = self.config.get("session_whitelist", []) or []
        self.session_blacklist: list = self.config.get("session_blacklist", []) or []
        self.llm_prompt_base = self._normalize_editor_text(
            self.config.get("llm_prompt", "")
        )
        # ---- 新增配置项（CHG-03 / CHG-04，均为向下兼容默认值）----
        self.at_all_cooldown = _to_int(
            self.config.get("at_all_cooldown", 60), 60
        )
        self.enable_audit_log = _to_bool(self.config.get("enable_audit_log", True))
        self.member_list_cache_ttl = _to_int(
            self.config.get("member_list_cache_ttl", 180), 180
        )

        # ---- 别名映射（用户确认方案）：搜索未命中时按别名展开再搜 ----
        # 兼容 {"才俊": "柴郡"} 字典与 ["才俊=柴郡"] 列表两种配置形态。
        raw_aliases = self.config.get("member_aliases", []) or []
        self.member_aliases: dict[str, str] = {}
        if isinstance(raw_aliases, dict):
            for alias, target in raw_aliases.items():
                alias_s = str(alias).strip()
                target_s = str(target).strip()
                if alias_s and target_s:
                    self.member_aliases[alias_s] = target_s
        elif isinstance(raw_aliases, list):
            for item in raw_aliases:
                if isinstance(item, str) and "=" in item:
                    alias_s, _, target_s = item.partition("=")
                    if alias_s.strip() and target_s.strip():
                        self.member_aliases[alias_s.strip()] = target_s.strip()

        # ---- 缓存结构：统一 (payload, created_ts) 维度 ----
        # 权限缓存：value = (allowed, deny_reason, created_ts)
        self._permission_cache: dict[str, Tuple[bool, str, float]] = {}
        # 群成员缓存：value = (member_list, created_ts)
        self._member_cache: dict[str, Tuple[list, float]] = {}
        # @全体 路由级冷却记录：key=group_id, value=最近一次放行时间戳
        self._at_all_last_trigger: dict[str, float] = {}
        # 会话级多结果待选：key=unified_msg_origin, value=(matches, created_ts)
        self._pending_choices: dict[str, Tuple[list, float]] = {}
        # 兜底补插缓存（修复 2 + T14/A3）：key=(unified_msg_origin, sender_id),
        # value=(user_id, created_ts)——同群不同用户各持一条，互不覆盖；
        # 他人回复既读不到也清不掉本发起者的缓存（r2/F1 + t12 M1）
        self._fallback_at: dict[tuple[str, str], tuple[str, float]] = {}
        # 工具缺席告警去重（P0-4）：key=unified_msg_origin，每会话只告警一次
        self._tool_absent_warned: set = set()
        # "艾特意图未被满足"告警去重（r2 评审 F8）：key=unified_msg_origin，
        # 每会话只记一次；与 _tool_absent_warned 同形，规模受会话数约束
        self._intent_unmet_warned: set = set()
        # T10 可信 ID 记忆（T14/A3 起按发起者分槽）：key=(umo, sender_id),
        # value=({工具确认过的 ID...}, created_ts)。同群不同用户各持一条，
        # 乙的一次工具命中不会覆盖甲的条目；容量/过期沿用既有缓存治理函数
        self._known_at_ids: dict[tuple[str, str], tuple[set, float]] = {}

        # 审计目录（获取失败时审计降级跳过，不阻断消息流）
        self._audit_dir: Optional[object] = None
        self._ensure_audit_dir()

    # ------------------------------------------------------------------ #
    # 配置/初始化辅助
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_editor_text(text: object) -> str:
        if not isinstance(text, str):
            return ""
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        if "\n" not in normalized and (
            "\\r\\n" in normalized or "\\n" in normalized or "\\r" in normalized
        ):
            normalized = (
                normalized.replace("\\r\\n", "\n")
                .replace("\\r", "\n")
                .replace("\\n", "\n")
            )
        return normalized

    def _ensure_audit_dir(self) -> None:
        """获取插件专属数据目录用于审计日志落盘（CHG-03）。"""
        try:
            self._audit_dir = StarTools.get_data_dir(_PLUGIN_NAME)
        except Exception as exc:
            logger.warning(f"获取插件数据目录失败，审计日志将不可用: {exc}")
            self._audit_dir = None

    # ------------------------------------------------------------------ #
    # 会话准入（CHG-02 精确匹配）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sender_key(event: AstrMessageEvent) -> Tuple[str, str]:
        """构造"会话 + 发起者"维度的缓存 key（T14/A3）。

        `_known_at_ids`（工具确认过的 ID）与 `_fallback_at`（兜底补插）共用
        该 key：同群不同用户各持一条，既不会互相覆盖、也无法读到或清掉对方的
        条目（越权艾特与"合法艾特被降级"两个方向都被堵住）。

        Args:
            event: 当前消息事件。

        Returns:
            (unified_msg_origin, sender_id)；发起者缺失时 sender 为空串。
        """
        return (event.unified_msg_origin, str(event.get_sender_id() or ""))

    def _is_session_allowed(self, event: AstrMessageEvent) -> Tuple[bool, str]:
        return check_session_lists(
            whitelist=self.session_whitelist,
            blacklist=self.session_blacklist,
            unified_msg_origin=event.unified_msg_origin,
            group_id=event.get_group_id() or "",
        )

    # ------------------------------------------------------------------ #
    # Bot 超级管理员判定（不变）
    # ------------------------------------------------------------------ #
    async def _is_bot_super_admin(self, event: AstrMessageEvent) -> bool:
        sender_id = event.get_sender_id()
        if not sender_id:
            return False
        try:
            cfg = self.context.get_config()
            admin_ids = cfg.get("admins_id", ["astrbot"])
            return str(sender_id) in [str(a) for a in admin_ids]
        except Exception as exc:
            logger.warning(f"检查 Bot 超级管理员权限失败: {exc}")
            return False

    @staticmethod
    def _build_deny_reason(sender_id: object, detail: str) -> str:
        if not sender_id:
            return "无法识别身份，拒绝执行"
        return f"{sender_id} {detail}"

    # ------------------------------------------------------------------ #
    # @全体权限校验（不变）+ 带缓存的结果获取
    # ------------------------------------------------------------------ #
    async def _check_at_all_permission(
        self, event: AstrMessageEvent
    ) -> Tuple[bool, str]:
        group_id = event.get_group_id()
        if not group_id:
            return False, "非群聊场景"
        if await self._is_bot_super_admin(event):
            return True, ""
        if not _is_aiocqhttp_event(event):
            return False, "当前平台不支持@全体权限校验"
        sender_id = event.get_sender_id()
        if not sender_id:
            return False, self._build_deny_reason(sender_id, "的身份，拒绝执行")
        try:
            group_member_info = await event.bot.api.call_action(
                "get_group_member_info",
                group_id=group_id,
                user_id=sender_id,
            )
        except Exception as exc:
            logger.warning(
                f"查询@全体权限失败: group_id={group_id}, "
                f"user_id={sender_id}, error={exc}"
            )
            return False, "查询群成员权限失败"
        role = str(group_member_info.get("role", "member")).lower()
        if role in {"owner", "admin"}:
            return True, ""
        return False, self._build_deny_reason(
            sender_id, "不是群主、管理员或 Bot 超级管理员"
        )

    async def _get_at_all_permission_result(
        self, event: AstrMessageEvent
    ) -> Tuple[bool, str]:
        """带缓存的 @全体 权限结果获取。

        CHG-01 修复：缓存 key 由 unified_msg_origin（会话维度）改为
        group_id:user_id（群+用户维度），同群不同用户权限互相独立。
        CHG-07：写入时惰性清理过期项并维持容量上限。
        """
        group_id = event.get_group_id() or ""
        sender_id = event.get_sender_id() or ""
        cache_key = build_permission_cache_key(str(group_id), str(sender_id))
        # 无法定位群或用户时不可缓存，直接现场判定
        if not cache_key:
            return await self._check_at_all_permission(event)

        now = time.time()
        cached = self._permission_cache.get(cache_key)
        if cached and (now - cached[2]) < _PERMISSION_CACHE_TTL:
            return cached[0], cached[1]

        result = await self._check_at_all_permission(event)
        self._permission_cache[cache_key] = (result[0], result[1], now)
        # 写入即清理：过期项 + 容量上限（CHG-07）
        drop_expired(self._permission_cache, _PERMISSION_CACHE_TTL, now)
        evict_oldest_to_limit(self._permission_cache, _PERMISSION_CACHE_MAX)
        return result

    # ------------------------------------------------------------------ #
    # @全体频率限制（CHG-03）
    # ------------------------------------------------------------------ #
    def _check_at_all_cooldown(self, event: AstrMessageEvent) -> Tuple[bool, float]:
        """冷却判定。返回 (allow, remain_seconds)：allow=False 表示在冷却中。"""
        if self.at_all_cooldown <= 0:
            return True, 0.0
        group_id = event.get_group_id() or ""
        if not group_id:
            return True, 0.0
        last_ts = self._at_all_last_trigger.get(group_id)
        in_cd, remain = is_at_all_in_cooldown(
            last_ts, time.time(), self.at_all_cooldown
        )
        return (not in_cd), remain

    def _record_at_all_trigger(self, event: AstrMessageEvent) -> None:
        """记录一次放行的 @全体 触发时间，作为下一次冷却起点。"""
        group_id = event.get_group_id() or ""
        if group_id:
            self._at_all_last_trigger[group_id] = time.time()

    # ------------------------------------------------------------------ #
    # 审计日志（CHG-03）
    # ------------------------------------------------------------------ #
    def _write_audit_log(
        self,
        event: AstrMessageEvent,
        op_type: str,
        target_id: str,
        allowed: bool,
        reason: str = "",
    ) -> None:
        """落盘一条艾特操作审计记录（JSONL，按日切割）。

        写盘失败只记 warning，绝不影响消息收发主链路（审计非强一致）。
        """
        if not self.enable_audit_log:
            return
        if self._audit_dir is None:
            self._ensure_audit_dir()
            if self._audit_dir is None:
                logger.warning("审计日志目录不可用，跳过审计记录写入。")
                return
        try:
            now = datetime.now()
            record = build_audit_record(
                time_iso=now.strftime("%Y-%m-%dT%H:%M:%S"),
                group_id=str(event.get_group_id() or ""),
                operator_id=str(event.get_sender_id() or ""),
                op_type=op_type,
                target_id=str(target_id),
                allowed=allowed,
                reason=reason,
            )
            log_path = audit_log_path(self._audit_dir, now.strftime("%Y%m%d"))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning(f"写入审计日志失败: {exc}")

    # ------------------------------------------------------------------ #
    # 群成员列表缓存（CHG-04）
    # ------------------------------------------------------------------ #
    async def _get_group_members_cached(
        self, event: AstrMessageEvent, group_id: str
    ) -> Optional[list]:
        """命中缓存则复用，否则拉取并按 TTL 缓存；TTL<=0 表示不缓存。

        P2-7：网络瞬时抖动时对 get_group_member_list 重试
        _MEMBER_LIST_MAX_RETRIES 次（间隔 _MEMBER_LIST_RETRY_DELAY 秒）；
        空列表视为合法结果不重试；重试耗尽仍失败返回 None（上层降级为
        「无法获取群成员列表」，不抛出异常）。
        """
        now = time.time()
        if self.member_list_cache_ttl > 0:
            cached = self._member_cache.get(group_id)
            if cached and (now - cached[1]) < self.member_list_cache_ttl:
                return cached[0]
        raw_members = None
        for attempt in range(_MEMBER_LIST_MAX_RETRIES + 1):
            try:
                raw_members = await event.bot.api.call_action(
                    "get_group_member_list", group_id=group_id
                )
                break
            except Exception as exc:
                if attempt < _MEMBER_LIST_MAX_RETRIES:
                    logger.warning(
                        f"获取群成员列表失败 group_id={group_id}（第 {attempt + 1} 次），"
                        f"{_MEMBER_LIST_RETRY_DELAY}s 后重试: {exc}"
                    )
                    await asyncio.sleep(_MEMBER_LIST_RETRY_DELAY)
                else:
                    logger.warning(f"获取群成员列表失败 group_id={group_id}: {exc}")
                    return None
        if not raw_members:
            return None
        if self.member_list_cache_ttl > 0:
            self._member_cache[group_id] = (raw_members, now)
            drop_expired(self._member_cache, self.member_list_cache_ttl, now)
            evict_oldest_to_limit(self._member_cache, _MEMBER_CACHE_MAX)
        return raw_members

    # ------------------------------------------------------------------ #
    # LLM 动态提示词构建（CHG-05 补充 select 工具说明）
    # ------------------------------------------------------------------ #
    def _build_dynamic_instructions(self) -> str:
        parts = []
        if self.allow_direct_qq_at:
            parts.append(
                "### QQ号处理\n"
                "用户可能会直接提供QQ号，此时你可以直接使用 [at:QQ号] 标签，"
                "不需要再调用 search_and_mention 工具搜索。"
                "无需额外确认即可直接输出 [at:ID] 标签。"
            )
        else:
            parts.append(
                "### QQ号处理\n"
                "即使用户直接提供了QQ号，你也必须调用 search_and_mention 工具"
                "来确认该成员存在后才可以使用 [at:ID] 标签。"
            )
        # 新增：多成员选择说明（CHG-05）
        parts.append(
            "### 多个成员的艾特选择\n"
            "当 `search_and_mention` 工具返回多个候选成员（带编号列表）时，"
            "请先把该列表展示给用户让其选择序号；"
            "获取用户选择的序号后，调用 `select_member_by_index` 工具（传入该序号）"
            "获取最终可使用的艾特标签。\n"
            "若 `select_member_by_index` 返回错误（序号无效或记录已过期约120秒），"
            "请重新调用 `search_and_mention` 搜索后再请用户选择。"
        )
        # T20：连"当前说话人"也必须走工具（线上 22:20 实测：模型跳过工具直接写
        # [at:<发送者 QQ>]，降级后把说话人显示成一串裸号码）
        parts.append(
            "### 艾特当前说话人也必须先调用工具\n"
            "除「QQ号处理」小节允许的情形（用户在本条消息里直接给出了他要你艾特的"
            " QQ 号、且该能力已开启）外，即使你要艾特当前正在和你说话的人（也就是"
            "这条消息的发送者），也必须先调用 search_and_mention 搜索取得他的 ID，"
            "再使用 [at:ID] 标签；不要直接写出你从历史消息或上下文里看到的号码。"
        )
        return "\n\n".join(parts)

    @filter.on_llm_request()
    async def inject_at_instruction(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """在 LLM 请求前注入格式指令和动态配置。

        P0-4：先按 `req.func_tool` 判定艾特工具是否真的进了本轮请求（人设
        "工具"白名单 / 插件集过滤都可能剔除它）。缺席时注入降级提示并留一次
        warning，绝不要求模型调用一个不存在的工具（否则模型只能退化成
        "输出裸标签"或"纯文本假装艾特"）。
        """
        # P4：请求阶段即安装 send 兜底渲染（实例级、幂等），保证任何第三方
        # 插件在结果阶段"清空 chain 后直接 event.send()"时，[at:ID] 仍在
        # 真正发往平台前被渲染为 At 组件。
        self._wrap_event_send(event)

        allowed, deny_reason = self._is_session_allowed(event)
        if not allowed:
            req.system_prompt = (req.system_prompt or "") + (
                f"\n\n【注意】{deny_reason}，本会话不允许使用艾特功能。"
                "\n请不要输出任何 [at:ID] 或 [at:all] 标签。"
            )
            return

        # 工具可达性判定：ToolSet.names() 是宿主装配结果的唯一可读视图
        func_tool = getattr(req, "func_tool", None)
        try:
            tool_names = list(func_tool.names()) if func_tool is not None else []
        except Exception:
            tool_names = []
        umo = event.unified_msg_origin
        if "search_and_mention" not in tool_names:
            req.system_prompt = (req.system_prompt or "") + (
                "\n\n【注意】当前会话未启用艾特工具，你无法搜索群成员，"
                "也不要输出任何 [at:ID] 或 [at:all] 标签。"
                "如用户要求艾特某人，请用自然语言说明当前会话未启用艾特工具，"
                "不要伪造艾特。"
            )
            if umo not in self._tool_absent_warned:
                self._tool_absent_warned.add(umo)
                logger.warning(
                    "[AtTool] 本轮请求工具集不含 search_and_mention（会话 "
                    f"{umo}），已注入降级提示；当前可用工具: {tool_names[:20]}"
                )
            return

        static_part = self.llm_prompt_base
        dynamic_part = self._build_dynamic_instructions()
        full_instruction = static_part + "\n\n" + dynamic_part
        req.system_prompt = (req.system_prompt or "") + full_instruction

        # 可观测性：群聊中出现艾特意图关键词时记录注入，并打标记供
        # process_at_tags 判定"有意图但工具从未被调用"（反馈 1 取证）
        try:
            msg_text = getattr(event, "message_str", "") or ""
            if event.get_group_id() and _AT_INTENT_PATTERN.search(msg_text):
                setattr(event, "_attool_at_intent", True)
                logger.info(
                    f"[AtTool] 检测到艾特意图，已注入艾特指令（会话 {umo}）"
                )
        except Exception:
            pass

        if self.permission_verification:
            allowed, deny_message = await self._get_at_all_permission_result(event)
            if allowed:
                permission_text = (
                    "\n【@全体权限】当前操作者具备@全体权限。"
                    "\n如用户明确要求且场景确有必要，你可以输出 [at:all]。"
                    "\n但请注意 [at:all] 有冷却限制，频繁@全体会被自动拦截降级。"
                )
            else:
                permission_text = (
                    f"\n【@全体权限】当前操作者不具备@全体权限。原因：{deny_message}"
                    "\n禁止输出 [at:all]。"
                    "\n如果用户要求@全体，请直接用自然语言说明无法执行。"
                )
            req.extra_user_content_parts.append(
                TextPart(text=permission_text).mark_as_temp()
            )

    # ------------------------------------------------------------------ #
    # 工具：搜索群成员（CHG-04 走缓存 / CHG-06 带角色 / CHG-05 多结果待选）
    # ------------------------------------------------------------------ #
    @filter.llm_tool(name="search_and_mention")
    async def search_and_mention(
        self, event: AstrMessageEvent, name: str
    ) -> str:
        """当用户要求艾特、@、呼叫、叫一下或喊一下某个群成员时，必须先调用本工具搜索群成员；本工具返回可直接使用的 [at:ID] 艾特标签。

        使用时机与约束：
        - 用户表达"艾特/@/呼叫/叫一下/喊一下某人"等意图时，必须先调用本工具，
          不要用 "@名字" 之类的纯文本假装艾特；
        - 拿到返回值后在最终回复中原样输出其中的 [at:ID] 标签，本插件会把它
          渲染成真实艾特，不要改写标签、不要加空格；
        - 返回多个候选时，必须先把列表展示给用户让其选择序号，再调用
          select_member_by_index 选定，不要自行猜测；
        - 未找到成员时如实说明，不要凭昵称猜测 ID 或伪造标签。

        Args:
            name(string): 要搜索的群成员昵称或群名片，例如"张三"、"群主"、"管理员"。
        """
        setattr(event, "_attool_tool_called", True)
        umo = event.unified_msg_origin
        key = self._sender_key(event)
        allowed, deny_reason = self._is_session_allowed(event)
        if not allowed:
            # 修复 2：最新一次搜索未解析出唯一成员，清除兜底缓存避免误艾特
            self._fallback_at.pop(key, None)
            return f"【拒绝】{deny_reason}"

        group_id = event.get_group_id()
        if not group_id:
            self._fallback_at.pop(key, None)
            return "【错误】当前不在群聊环境中。"
        if not _is_aiocqhttp_event(event):
            self._fallback_at.pop(key, None)
            return "【错误】当前平台暂不支持此功能。"

        try:
            name_str = name or ""
            if not name_str.strip():
                # P2-2：空搜索词时 "" in nickname 恒为 True，会误命中全部成员
                self._fallback_at.pop(key, None)
                return "【错误】搜索词为空，请提供要搜索的成员昵称或群名片。"

            raw_members = await self._get_group_members_cached(event, group_id)
            if not raw_members:
                self._fallback_at.pop(key, None)
                return "【错误】无法获取群成员列表。"

            def _collect(term: str) -> Tuple[list, list]:
                """按 term 收集精确/模糊命中（闭包复用原始匹配逻辑）。"""
                exact: list = []
                fuzzy: list = []
                term = (term or "").strip()
                if not term:
                    return exact, fuzzy
                for m in raw_members:
                    user_id = str(m.get("user_id", ""))
                    if not user_id:
                        continue
                    nickname = m.get("nickname", "")
                    card = m.get("card", "")
                    role = m.get("role", "member")
                    display_name = card or nickname
                    if term == nickname or term == card or term == display_name:
                        exact.append((user_id, display_name, role))
                    elif self.enable_fuzzy_search and (
                        term in nickname or term in card
                    ):
                        fuzzy.append((user_id, display_name, role))
                return exact, fuzzy

            exact_matches, fuzzy_matches = _collect(name_str)
            if not exact_matches and not fuzzy_matches:
                # 别名兜底（如「才俊」→「柴郡」）：原始词未命中时，
                # 按 member_aliases 配置展开真名/群名片再搜一次
                for alt in expand_alias_queries(name_str, self.member_aliases):
                    if alt == name_str:
                        continue
                    alt_exact, alt_fuzzy = _collect(alt)
                    if alt_exact or alt_fuzzy:
                        exact_matches, fuzzy_matches = alt_exact, alt_fuzzy
                        break

            matches = exact_matches if exact_matches else fuzzy_matches

            if not matches:
                mode = "包含" if self.enable_fuzzy_search else "完全匹配"
                self._fallback_at.pop(key, None)
                logger.info(
                    f"[AtTool] search_and_mention 未命中: 关键词=「{name_str}」"
                    f"（{mode}匹配，会话 {umo}）"
                )
                return f"【未找到】群聊中没有找到名称{mode}「{name_str}」的成员。"

            if len(matches) > _MAX_MULTI_MATCHES:
                self._fallback_at.pop(key, None)
                return (
                    f"【提示】名称包含「{name_str}」的成员过多（共 {len(matches)} 个），"
                    "请让用户提供更精确的姓名或群名片后再搜索。"
                )

            if len(matches) == 1:
                user_id, display_name, role = matches[0]
                # 修复 2：唯一命中写入兜底缓存；LLM 最终回复缺标签时由
                # process_at_tags 自动补插 [at:ID]
                # r2（评审 F1）：缓存绑定本次发起者 sender_id，宿主按事件
                # 并发，同群他人回复不得被补插本成员
                now = time.time()
                self._fallback_at[key] = (user_id, now)
                drop_expired(self._fallback_at, _FALLBACK_AT_TTL, now)
                evict_oldest_to_limit(self._fallback_at, _MAX_TRUSTED_SLOTS)
                # T10：登记"工具确认过的 ID"，供最终渲染做来源判定
                cached_ids = self._known_at_ids.get(key)
                known = set(cached_ids[0]) if cached_ids else set()
                known.add(user_id)
                self._known_at_ids[key] = (known, now)
                drop_expired(self._known_at_ids, _KNOWN_AT_ID_TTL, now)
                evict_oldest_to_limit(self._known_at_ids, _MAX_TRUSTED_SLOTS)
                logger.info(
                    f"[AtTool] search_and_mention 命中: {display_name} ({user_id}，"
                    f"{role}，会话 {umo})"
                )
                return format_single_member_result(display_name, user_id, role)

            # 多结果：需用户选择序号后经 select_member_by_index 选定，
            # 不写入兜底缓存（多成员匹配不兜底，修复 2）
            self._fallback_at.pop(key, None)
            self._pending_choices[umo] = (matches, time.time())
            drop_expired(self._pending_choices, _PENDING_CHOICE_TTL, time.time())
            # P2-3：待选缓存同样受容量上限约束（对称 CHG-07 治理）
            evict_oldest_to_limit(self._pending_choices, _MAX_PENDING_CHOICES)
            return format_member_choice_list(name_str, matches)

        except Exception as exc:
            logger.error(f"search_and_mention 异常: {exc}")
            self._fallback_at.pop(key, None)
            return f"【错误】搜索时发生异常: {str(exc)}"

    # ------------------------------------------------------------------ #
    # 工具：按序号选定成员（CHG-05 新增）
    # ------------------------------------------------------------------ #
    @filter.llm_tool(name="select_member_by_index")
    async def select_member_by_index(
        self, event: AstrMessageEvent, index: int
    ) -> str:
        """在 search_and_mention 工具返回多个候选成员后，按用户选择的序号选定最终要艾特的成员，并返回可直接使用的 [at:ID] 艾特标签；必须在用户给出序号后再调用本工具，不要凭猜测调用。

        Args:
            index(int): 用户选择的成员序号，从 1 开始，对应上次搜索返回列表中的编号。
        """
        setattr(event, "_attool_tool_called", True)
        allowed, deny_reason = self._is_session_allowed(event)
        if not allowed:
            return f"【拒绝】{deny_reason}"

        try:
            idx = int(index)
        except (TypeError, ValueError):
            return "【错误】序号必须是正整数（从1开始）。"
        if idx <= 0:
            return "【错误】序号必须是正整数（从1开始）。"

        umo = event.unified_msg_origin
        now = time.time()
        cached = self._pending_choices.get(umo)
        if not cached or (now - cached[1]) >= _PENDING_CHOICE_TTL:
            self._pending_choices.pop(umo, None)
            return (
                "【提示】没有可用的成员选择记录，或记录已过期（约120秒）。"
                "请重新调用 search_and_mention 工具搜索。"
            )
        matches = cached[0]
        if idx > len(matches):
            return (
                f"【错误】序号超出范围，有效范围为 1~{len(matches)}。"
                "请提示用户在范围内重新选择。"
            )
        user_id, display_name, role = matches[idx - 1]
        # T10：用户选定（工具确认）的 ID 同样计入可信来源（按发起者分槽）
        key = self._sender_key(event)
        cached_ids = self._known_at_ids.get(key)
        known = set(cached_ids[0]) if cached_ids else set()
        known.add(user_id)
        self._known_at_ids[key] = (known, now)
        drop_expired(self._known_at_ids, _KNOWN_AT_ID_TTL, now)
        evict_oldest_to_limit(self._known_at_ids, _MAX_TRUSTED_SLOTS)
        return format_single_member_result(
            display_name, user_id, role, prefix="已选定："
        )

    # ------------------------------------------------------------------ #
    # 标签渲染主链路（CHG-03 频率限制 + 审计；解析走 utils 纯逻辑）
    # ------------------------------------------------------------------ #
    def _apply_fallback_at_tag(
        self, event: AstrMessageEvent, chain: List[BaseMessageComponent]
    ) -> bool:
        """兜底补插 [at:ID]（修复 2）。

        当本次会话内 search_and_mention 曾唯一命中成员、且待渲染链既无
        [at:ID] 标签也无 At 组件时，在链末尾补插 Plain("[at:ID]")，交由
        后续渲染逻辑正常转为 At 组件——即使 LLM 因角色卡"纯文本"等约束
        省略标签，艾特也不会丢失。

        P4 起 chain 由调用方传入（process_at_tags 主钩子传入 result.chain；
        send 兜底渲染传入待发送消息链），使"清链直发"类第三方插件场景下
        兜底同样生效。

        r2（评审 F1）：缓存绑定发起者（sender_id）——宿主按事件并发处理
        （同群多人同时说话），只按 UMO 作 key 会让 B 的无标签回复被补插
        A 刚找到的成员。发起者不一致时直接消费掉缓存并放弃补插。

        返回 True 表示已补插（调用方应继续渲染）；False 表示无需/不可补插。
        缓存为一次性消费：无论是否补插，读取后即删除，防止跨轮次误用。
        """
        umo = event.unified_msg_origin
        key = self._sender_key(event)
        now = time.time()
        cached = self._fallback_at.get(key)
        if not cached or (now - cached[-1]) >= _FALLBACK_AT_TTL:
            self._fallback_at.pop(key, None)
            return False
        user_id = cached[0]
        # 链中已有其他来源插入的 At 组件时不再兜底（避免重复艾特）
        if any(isinstance(comp, At) for comp in chain):
            self._fallback_at.pop(key, None)
            return False
        # 会话准入：不允许艾特的会话不兜底
        allowed, _ = self._is_session_allowed(event)
        if not allowed:
            self._fallback_at.pop(key, None)
            return False
        self._fallback_at.pop(key, None)
        chain.append(Plain(f"[at:{user_id}]"))
        logger.warning(
            f"兜底补插 [at:{user_id}]：LLM 回复未携带艾特标签（会话 {umo}）"
        )
        return True

    @filter.on_decorating_result(priority=-1000)
    async def process_at_tags(self, event: AstrMessageEvent) -> None:
        """渲染 [at:ID] → At 组件（P4 延迟渲染 + P0-2 唯一补插入口）。

        钩子顺序（实测）：本钩子 priority=-1000，早于分段插件的 -1e17 执行
        （star_handler.py 按 -priority 降序执行）。因此本钩子看到的是**尚未
        被分段拆分的整条链**：标签在这里渲染、兜底补插也在链尾完成；随后
        分段插件按 at_strategy=「跟随下段」把它归入最后一段投递。旧注释
        "本钩子执行时链已空"与实测执行序矛盾，已修正。

        P0-2：本钩子是"兜底补插 + 缓存消费"的唯一入口（enable_fallback=
        True）。send / send_streaming 路径只渲染链内已有标签，绝不补插、
        绝不消费缓存——否则逐段发送时会把 @ 补到第一段、或把后续段里的
        真实标签当成"重复"剥离（探针场景 2/3/4 实测）。

        第三方插件若在本钩子之前清空 chain 后直接 event.send()，本钩子看到
        空链（走缓存清理分支）；该段消息由 _wrap_event_send 的"链内已有标签
        渲染"兜底处理。
        """
        result = event.get_result()
        if not result or not result.chain:
            # 链空时解除兜底缓存：本钩子无法履行补插义务，而 send 路径刻意
            # 不消费缓存（P0-2）；若此处不清理，缓存会残留满 TTL(120s)，
            # 新一轮无标签回复会误补插上一轮成员（跨轮次误艾特）。
            self._fallback_at.pop(self._sender_key(event), None)
            return
        rendered, new_chain = await self._render_at_tags(
            event, result.chain, enable_fallback=True
        )
        if rendered:
            result.chain = new_chain

        # P0-4 可观测（反馈 1）：有艾特意图、工具在本轮可用却从未被调用，
        # 且最终回复没有任何真实艾特 → 记一条 warning，供 F2 判定取证。
        # r2（评审 F8）：按会话去重（_AT_INTENT_PATTERN 含 '@' 与 \bat\b，
        # 触发面宽于"真要艾特"，不去重会在活跃群形成常态噪音）。
        final_chain = new_chain if rendered else result.chain
        if (
            getattr(event, "_attool_at_intent", False)
            and not getattr(event, "_attool_tool_called", False)
            and not any(isinstance(comp, At) for comp in final_chain)
        ):
            umo = event.unified_msg_origin
            if umo not in self._intent_unmet_warned:
                self._intent_unmet_warned.add(umo)
                logger.warning(
                    "[AtTool] 艾特意图未被满足：本轮未调用 search_and_mention，"
                    f"最终回复也无艾特标签（会话 {umo}）"
                )

    # ------------------------------------------------------------------ #
    # P4 公共渲染：主钩子与 send 兜底共用的 [at:ID] → At 渲染流水线
    # ------------------------------------------------------------------ #
    async def _render_at_tags(
        self,
        event: AstrMessageEvent,
        chain: List[BaseMessageComponent],
        enable_fallback: bool = False,
        keep_unclosed: bool = False,
    ) -> Tuple[bool, List[BaseMessageComponent]]:
        """把消息链中的 [at:ID] / [at:all] 渲染为 At 组件（唯一渲染流水线）。

        process_at_tags 钩子与 send / send_streaming 兜底渲染共用本方法，
        保证各路径的形态归一 / 会话准入 / @全体权限+冷却 / 审计 / 零宽清理
        语义一致；标签语法解析统一委托 utils（P0-5 单一事实来源）。

        T10 成员 ID 可信来源约束：只有「本会话内工具确认过的 ID」
        （`_known_at_ids`，绑定发起者）与「用户消息里出现的 QQ 号」
        （`allow_direct_qq_at=true` 时）才渲染为 At，其余按 degrade 语义降级为
        纯文本 `@载荷` 并聚合一条 warning —— 模型凭空的 `[at:<QQ>]`
        （线上实测 [at:2060958352]/[at:3882563785]）不再误伤无关成员。

        Args:
            event: 当前消息事件（会话准入、@全体权限、审计依赖）。
            chain: 待渲染的消息链组件列表（调用方保证非空）。
            enable_fallback: 是否允许"兜底补插 + 缓存消费"（P0-2）。仅主钩子
                process_at_tags 传 True；send / send_streaming 保持默认 False
                ——否则分段插件逐段发送时会出现"@ 被补到第一段""后续段真实
                标签被当重复剥离"（探针场景 2/3/4）。
            keep_unclosed: 未闭合标签语法是否原样保留。流式路径传 True：
                跨 chunk 拼合属非目标（spec P2），残缺 chunk 原样透传，
                不删除用户可见正文。

        Returns:
            (rendered, new_chain)：rendered=True 表示链已被重建，调用方应
            用 new_chain 替换原链；False 表示链中无标签且未触发兜底补插，
            原链无需改动。
        """
        umo = event.unified_msg_origin

        # ⓪ 成员 ID 可信来源判定（T10 / T14-B1 fail-closed）：只认两类 ID——
        #    (a) 本会话内工具确认过的（_known_at_ids，按发起者分槽）；
        #    (b) 用户消息里明确出现、且**是本群成员**的 QQ 号
        #        （allow_direct_qq_at=true 时）。
        #    取不到用户上下文（无 message_str 也无入站链）时可信集退化为 (a)，
        #    编造 ID 一律降级 —— 绝不因"读不到用户消息"而放行（T12 B1）。
        #    该可信集必须早于 ①b 跨组件拼合构造，供拼合路径共用同一份判定
        #    （T12 B2）。
        trusted_ids: set = set()
        now = time.time()
        cached_ids = self._known_at_ids.get(self._sender_key(event))
        if cached_ids and (now - cached_ids[-1]) < _KNOWN_AT_ID_TTL:
            trusted_ids |= cached_ids[0]
        msg_text = getattr(event, "message_str", "") or ""
        try:
            incoming = event.get_messages()
        except Exception:
            incoming = None
        direct_ids: set = set()
        if self.allow_direct_qq_at:
            if msg_text:
                direct_ids |= set(USER_QQ_PATTERN.findall(msg_text))
            for in_comp in incoming or []:
                if isinstance(in_comp, Plain):
                    direct_ids |= set(USER_QQ_PATTERN.findall(in_comp.text))
        group_id = event.get_group_id() or ""
        sender_id = str(event.get_sender_id() or "")
        # T20：链里只要有成员标签，降级时就需要昵称/群名片，因此与 T24 的直连号
        # 过滤共用同一次成员名单拉取（_get_group_members_cached 有 180s 缓存，
        # 工具路径通常已预热；无标签且无直连号时保持 0 次调用）。
        need_members = bool(direct_ids) or any(
            isinstance(comp, Plain) and has_at_tag(comp.text) for comp in chain
        )
        raw_members = (
            await self._get_group_members_cached(event, group_id)
            if need_members and group_id
            else None
        )
        member_ids: set = set()
        member_names: dict[str, str] = {}
        for member in raw_members or []:
            uid = str(member.get("user_id", ""))
            if not uid:
                continue
            member_ids.add(uid)
            # 群名片优先，其次昵称（T20 降级文案用）。T26/F1'：名字本身若命中
            # 标签起始语法（成员把群名片设成 `[at:<QQ>]` 等），降级文案会把标签
            # 原文带进群（非末段经 splitter 的 context.send_message 直发，不经本
            # 插件二次渲染）⇒ 丢弃该名字、回退为号码。复用 has_at_tag 以覆盖
            # 大小写/空白/全角/零宽变体，不另写正则。
            name = str(member.get("card") or member.get("nickname") or "").strip()
            if name and not has_at_tag(name):
                member_names[uid] = name
        if direct_ids:
            # T16：直连号必须过本群成员名单。平台（NapCat/OneBot）对无法解析的
            # QQ 会以 retcode=1200 "Get Uid Error" 拒绝**整条消息**（线上实测
            # 22:02 respond.stage:322，正文全丢），因此只有名单内的号码才可信；
            # 不在名单内的号码不进 trusted_ids ⇒ 走既有 degrade（@数字 + 聚合告警）。
            if raw_members is None:
                # T24/F1'（fail-closed）：名单不可用（非群聊/拉取失败/平台返回为空）
                # 时**不得**放行直连号——否则非法 At 会被平台以 retcode=1200
                # "Get Uid Error" 拒绝，整条回复连正文一起丢（线上 22:02 实测）。
                # 直连号一律按未知来源处理：不进可信集 ⇒ 走既有 degrade（@数字）。
                logger.warning(
                    "无法获取群成员名单（非群聊/拉取失败/平台返回为空），本轮直连 "
                    "QQ 号已按未知来源降级（不渲染 At）："
                    f"{sorted(direct_ids)}（会话 {umo}）；若平台报 retcode=1200 或长期"
                    "拉不到名单，请检查适配器 get_group_member_list 权限/大群截断。"
                )
            else:
                trusted_ids |= direct_ids & member_ids
        if sender_id and group_id and sender_id in member_ids:
            # T20 窄例外（T26/F2' 收紧）：模型想艾特"当前正在跟它说话的人"是常见
            # 合法意图，而该 ID 由事件本身提供（event.get_sender_id()），不依赖
            # 模型自述、无法被伪造，因此即使未经工具确认也允许渲染真 At。
            # 但**必须同时满足**：群聊 + 名单可用 + 发送者确实在名单内 —— 原注释
            # "该 ID 必然是本群成员"是错的前提：名单可能拉取失败/为空/被截断，
            # 或发言后立刻退群导致平台解析不了该 At（同样会以 1200 拒收整条消息，
            # 见 T24/T26 实测）。名单不可用时沿用 T24 的 fail-closed 降级。
            trusted_ids.add(sender_id)
        # 未经工具确认 / 未出现在用户消息里 / 不是本群成员的 ID（聚合告警用）
        untrusted_ids: list[str] = []

        # ① 合并相邻 Plain：修复标签被第三方插件拆分到相邻组件的情况
        #    （如 "[at:12" 与 "345]" 分处两个 Plain，合并后可完整匹配）。
        #    注：合并必须生成新 Plain 对象（非原地修改首个 Plain 的 text）：
        #    merged_input 与调用方 chain 共享组件引用，若原地修改，提前
        #    return False 的路径（无标签且无兜底，rendered=False 调用方保留
        #    原链）会把"首 Plain 已含合并文本 + 尾 Plain 仍在"的重复内容
        #    泄漏回原链，导致消息文本重复渲染。
        merged_input: List[BaseMessageComponent] = []
        for comp in chain:
            if (
                isinstance(comp, Plain)
                and merged_input
                and isinstance(merged_input[-1], Plain)
            ):
                merged_input[-1] = Plain(merged_input[-1].text + comp.text)
            else:
                merged_input.append(comp)

        # ①b 跨组件拼合（P0-1 + T14-B2）："[at:12" 与 "345]" 之间夹着 At 等
        #     非 Plain 组件时，步骤①的相邻合并救不了；先拼回完整标签再统一
        #     解析，否则前缀会被当成"未闭合语法"删除、艾特丢失（spec §7.2
        #     第 13 行）。拼出的成员 ID 必须过同一份 trusted_ids：不可信时
        #     按 degrade 语义降级为 @载荷（不渲染、也不静默删除）
        reassembled = self._merge_split_at_tags(
            merged_input, umo, trusted_ids, untrusted_ids, member_names
        )
        if reassembled is not None:
            merged_input = reassembled

        # ② 扫描标签（P0-5 统一判定入口，替代旧的裸 "[at:" 子串判定）
        has_tag = False
        has_at_all_tag = False
        for comp in merged_input:
            if not isinstance(comp, Plain) or not has_at_tag(comp.text):
                continue
            has_tag = True
            if has_at_all_tag:
                continue
            for match in AT_TAG_LOOSE_PATTERN.finditer(comp.text):
                kind, value = parse_at_tag_payload(match.group(1))
                if kind == "at" and value == "all":
                    has_at_all_tag = True
                    break
        if not has_tag:
            if enable_fallback and self._apply_fallback_at_tag(event, merged_input):
                has_tag = True
            elif reassembled is None:
                # 无标签（send 路径不补插 / 主钩子无可补插）：原链原样返回，
                # 无标签消息零开销透传（P0-2）
                return False, chain
        elif enable_fallback:
            # 回复已含标签：本次兜底义务解除（仅主钩子消费清理缓存）
            self._fallback_at.pop(self._sender_key(event), None)

        if not has_tag:
            # 仅发生跨组件拼合（P0-1）：标签已在 ①b 渲染/降级，仍需返回重建链
            self._warn_untrusted_at_ids(untrusted_ids, umo)
            return True, self._finalize_chain(merged_input)

        # ③ 会话准入：不允许则统一按 P0-1 语义降级（成员标签剥离、@全体转
        #    纯文本、非数字载荷转 @载荷），全程不改写调用方组件（P1-3）
        allowed, _ = self._is_session_allowed(event)
        if not allowed:
            degraded: List[BaseMessageComponent] = []
            for comp in merged_input:
                if not isinstance(comp, Plain):
                    degraded.append(comp)
                    continue
                kept: list[str] = []
                for kind, value in split_text_by_at_tags(comp.text):
                    if kind == "text":
                        kept.append(value)
                    elif kind == "degrade":
                        logger.warning(
                            f"会话不允许艾特，标签载荷 {value!r} 已降级为纯文本"
                        )
                        kept.append("@" + value)
                    elif kind == "at" and value == "all":
                        kept.append("@全体成员")
                    # 成员标签 / 空载荷 / 未闭合语法：删除标签语法
                degraded.append(Plain("".join(kept)))
            self._warn_untrusted_at_ids(untrusted_ids, umo)
            return True, self._finalize_chain(degraded)

        # ④ @全体 权限 + 频率一次性判定（CHG-01 / CHG-03）
        at_all_allowed = True
        at_all_deny_reason = ""
        if has_at_all_tag:
            if self.permission_verification:
                perm_ok, perm_deny = await self._get_at_all_permission_result(event)
                if not perm_ok:
                    at_all_allowed = False
                    at_all_deny_reason = f"无权限：{perm_deny}"
            if at_all_allowed:
                cooldown_ok, remain = self._check_at_all_cooldown(event)
                if not cooldown_ok:
                    at_all_allowed = False
                    at_all_deny_reason = f"冷却中，剩余约{int(remain)}秒"
                elif self.at_all_cooldown > 0:
                    self._record_at_all_trigger(event)

        # ⑤b 逐组件解析标签并渲染（共用 utils 单一解析实现，B2/B5 一并收敛）；
        #     trusted_ids / untrusted_ids 已在 ⓪ 步骤构造（T14：早于跨组件拼合，
        #     两条路径共用同一份可信集）
        new_chain: List[BaseMessageComponent] = []
        at_member_targets: List[str] = []
        for comp in merged_input:
            if not isinstance(comp, Plain):
                new_chain.append(comp)
                continue
            for kind, value in split_text_by_at_tags(comp.text):
                if kind == "text":
                    if value:
                        new_chain.append(Plain(value))
                elif kind == "degrade":
                    # P0-1：非数字/全角/占位符载荷降级为可读纯文本，绝不原样穿链
                    logger.warning(
                        f"标签载荷无法解析（{value!r}），已降级为纯文本 "
                        f"@{value}（会话 {umo}）"
                    )
                    new_chain.append(Plain("@" + value))
                elif kind == "unclosed":
                    if keep_unclosed:
                        # 流式 chunk：跨 chunk 拼合为非目标，原样保留语法片段
                        new_chain.append(Plain(value))
                        continue
                    logger.warning(
                        "检测到未闭合的 [at: 标签语法，已删除语法片段并保留"
                        f"其余正文（会话 {umo}）"
                    )
                elif kind == "drop":
                    continue
                elif value == "all":
                    if at_all_allowed:
                        new_chain.append(At(qq="all"))
                    else:
                        new_chain.append(Plain("@全体成员"))
                elif value not in trusted_ids:
                    # T10：编造/上游残留的 ID 不得渲染成真实艾特（误伤无关成员）
                    # T20：能从成员名单解析出群名片/昵称时不要露裸号码
                    untrusted_ids.append(value)
                    new_chain.append(Plain("@" + (member_names.get(value) or value)))
                else:
                    new_chain.append(At(qq=value))
                    at_member_targets.append(value)

        self._warn_untrusted_at_ids(untrusted_ids, umo)

        # ⑥ 审计写入：@单人逐条（同一事件同一目标只记一次，P1-4）+
        #    @全体整体一条（CHG-03）
        if self.enable_audit_log:
            audited = getattr(event, "_attool_audited_targets", None)
            if not isinstance(audited, set):
                audited = set()
                try:
                    event._attool_audited_targets = audited
                except Exception:
                    pass
            for target_id in dict.fromkeys(at_member_targets):
                if target_id in audited:
                    continue
                audited.add(target_id)
                self._write_audit_log(
                    event, _AUDIT_OP_AT_MEMBER, target_id, True, "ok"
                )
            if has_at_all_tag and "all" not in audited:
                audited.add("all")
                self._write_audit_log(
                    event,
                    _AUDIT_OP_AT_ALL,
                    "all",
                    at_all_allowed,
                    at_all_deny_reason or "ok",
                )

        # ⑦ 收尾（零宽清理/空剔除/相邻合并/At 后补零宽，全程 copy-on-write）
        new_chain = self._finalize_chain(new_chain)

        # ⑦b 最后一道可观测防线：收尾后仍含标签语法的 Plain（正常路径不应
        #    出现；流式 chunk 的未闭合语法属已知限制，不告警）
        residual = [
            comp.text
            for comp in new_chain
            if isinstance(comp, Plain) and has_at_tag(comp.text)
        ]
        if residual and not keep_unclosed:
            logger.warning(
                "检测到 [at: 标签语法残留（可能被第三方插件在分段/重建时"
                f"切断），已按纯文本保留: {residual!r}，会话 {umo}"
            )
        return True, new_chain

    @staticmethod
    def _warn_untrusted_at_ids(untrusted_ids: list, umo: str) -> None:
        """对"未经工具确认的艾特 ID"聚合告警（每轮渲染至多一条）。

        Args:
            untrusted_ids: 本轮渲染中被判为不可信的成员 ID（可为空）。
            umo: 当前会话标识。
        """
        if not untrusted_ids:
            return
        logger.warning(
            "检测到未经工具确认、或不在本群成员名单的艾特 ID（疑似编造/上游"
            "残留），已降级为纯文本 "
            f"@{'、@'.join(dict.fromkeys(untrusted_ids))}"
            f"（会话 {umo}；本轮用户消息与工具结果均未提供该 ID）"
        )

    def _merge_split_at_tags(
        self,
        chain: list[BaseMessageComponent],
        umo: str,
        trusted_ids: set,
        untrusted_ids: list,
        member_names: dict,
    ) -> list[BaseMessageComponent] | None:
        """P0-1 跨组件拼合：把被非 Plain 组件切碎的标签拼回并渲染为 At。

        链形态如 Plain("…[at:12") + At(999) + Plain("345]…")：两个 Plain 被
        At 隔开，步骤①的相邻 Plain 合并无法修复，逐组件解析只会把前缀当成
        未闭合语法删除。此处以起始语法为锚点、跨过中间组件累积后续 Plain
        文本（零宽透明，窗口 _SPLIT_AT_WINDOW 字符），拼出完整成员标签则
        原位插入 At 组件，中间组件保持原序不动。

        仅拼合成员标签（ASCII 数字载荷）：[at:all] 的切碎形态不做猜测，
        避免绕过步骤④的 @全体 权限/冷却判定。

        Args:
            chain: 已合并相邻 Plain 的消息链。
            umo: 当前会话标识（仅用于告警日志）。
            trusted_ids: 与逐组件路径共用的同一份可信 ID 集合（T14-B2）。
            untrusted_ids: 本轮渲染中不可信 ID 的聚合列表（调用方负责告警）。
            member_names: user_id → 群名片/昵称（T20 降级文案用；可为空）。

        Returns:
            重建后的链；无需拼合时返回 None（调用方保持原链不变）。
        """
        working = list(chain)
        recovered = False
        # 上限 _SPLIT_AT_MAX_ROUNDS 轮：每轮至少消费一个起始语法，避免
        # 病态输入下的无限重建（P1-1 明确禁止无上限的 while True 补丁）
        for _ in range(_SPLIT_AT_MAX_ROUNDS):
            hit: list[BaseMessageComponent] | None = None
            for idx, comp in enumerate(working):
                if not isinstance(comp, Plain) or not has_at_tag(comp.text):
                    continue
                hit = self._try_merge_split_at_tag(
                    working, idx, umo, trusted_ids, untrusted_ids, member_names
                )
                if hit is not None:
                    break
            if hit is None:
                break
            working = hit
            recovered = True
        return working if recovered else None

    def _try_merge_split_at_tag(
        self,
        chain: list[BaseMessageComponent],
        idx: int,
        umo: str,
        trusted_ids: set,
        untrusted_ids: list,
        member_names: dict,
    ) -> list[BaseMessageComponent] | None:
        """尝试用 chain[idx] 起的跨组件文本拼出一个完整成员标签（P0-1）。

        仅处理本组件内不闭合的起始语法（已闭合的完整标签交给逐组件解析），
        且拼出的载荷必须是 ASCII 数字，否则不消费任何文本（交由逐组件解析
        按降级/未闭合规则处理）。

        Args:
            chain: 当前消息链。
            idx: 起始语法所在 Plain 的下标。
            umo: 当前会话标识（仅用于告警日志）。
            trusted_ids: 与逐组件路径共用的同一份可信 ID 集合（T14-B2）。
            untrusted_ids: 不可信 ID 聚合列表（本函数只追加，调用方统一告警）。
            member_names: user_id → 群名片/昵称（T20 降级文案用；可为空）。

        Returns:
            重建后的链；无法拼合时返回 None。
        """
        text = chain[idx].text
        for head in AT_TAG_HEAD_PATTERN.finditer(text):
            partial = text[head.end() :].replace("\u200b", "")
            if "]" in partial:
                continue  # 本组件内已闭合：完整标签，交给逐组件解析
            collected = ""
            consumption: list[tuple[int, int]] = []  # (组件索引, 消耗的非零宽字符数)
            closed = False
            for j in range(idx + 1, len(chain)):
                if len(partial) + len(collected) > _SPLIT_AT_WINDOW:
                    break
                nxt = chain[j]
                if not isinstance(nxt, Plain):
                    continue  # 跨过中间组件（At 等）：内容不参与拼合，原位保留
                seg_clean = nxt.text.replace("\u200b", "")
                close_pos = seg_clean.find("]")
                if close_pos == -1:
                    consumption.append((j, len(seg_clean)))
                    collected += seg_clean
                    continue
                # 消耗到 "]" 为止（含），按"清洗零宽后"的字符数记录，
                # 与 _strip_clean_prefix 的语义一致（零宽透明）
                consumption.append((j, close_pos + 1))
                collected += seg_clean[: close_pos + 1]
                closed = True
                break
            if not closed:
                continue  # 后续没有 Plain 能提供闭合 "]"（或链尾截断）
            match = AT_TAG_PATTERN.fullmatch(head.group(0) + partial + collected)
            if match is None:
                continue
            # 复用唯一分类实现（P0-5）：非成员标签（all）与超长载荷（r2/F5）
            # 一律不拼合，交回逐组件解析按降级/未闭合规则处理
            kind, target_id = parse_at_tag_payload(match.group(1))
            if kind != "at" or target_id == "all":
                continue
            # T14-B2：拼出的成员 ID 必须过与逐组件路径同一份可信集；
            # 不可信 → 原位降级为 @载荷（保留数字文本，不渲染 At）
            trusted = target_id in trusted_ids
            if not trusted:
                untrusted_ids.append(target_id)
            # ——重建链：剥离片段、原位插入 At、中间组件保持原序——
            rebuilt: list[BaseMessageComponent] = []
            head_text = text[: head.start()]
            if head_text:
                rebuilt.append(Plain(head_text))
            rebuilt.append(
                At(qq=target_id)
                if trusted
                else Plain("@" + (member_names.get(target_id) or target_id))
            )
            consumed_map = dict(consumption)
            for j in range(idx + 1, len(chain)):
                if j in consumed_map:
                    remain = _strip_clean_prefix(chain[j].text, consumed_map[j])
                    if remain:
                        rebuilt.append(Plain(remain))
                else:
                    rebuilt.append(chain[j])
            logger.warning(
                "宽松恢复 [at:%s]：检测到跨组件切碎的艾特标签残留（严格正则"
                "未命中，前缀与闭合括号分处不同组件），已原位恢复渲染为 At "
                "组件（会话 %s）",
                target_id,
                umo,
            )
            return rebuilt
        return None

    def _finalize_chain(
        self, new_chain: List[BaseMessageComponent]
    ) -> List[BaseMessageComponent]:
        """渲染收尾：零宽字符清理 / 空 Plain 剔除 / 相邻 Plain 合并 / At 后补零宽。

        逻辑与原 process_at_tags 收尾一致（含 P2-4 两遍处理，保证零宽不重复）。
        P1-3 copy-on-write：入参组件可能被调用方（甚至第三方插件）共享，
        所有文本改写都在新建的 Plain 上完成，绝不原地改写共享组件。

        Args:
            new_chain: 待收尾的消息链。

        Returns:
            收尾后的新链（Plain 全部为新对象；非 Plain 组件按引用复用）。
        """
        cleaned: list[BaseMessageComponent] = []
        for comp in new_chain:
            if not isinstance(comp, Plain):
                cleaned.append(comp)
                continue
            text = comp.text.strip().replace("\u200b", "")
            if text:
                cleaned.append(Plain(text))

        merged: List[BaseMessageComponent] = []
        for comp in cleaned:
            if isinstance(comp, Plain) and merged and isinstance(merged[-1], Plain):
                merged[-1] = Plain(merged[-1].text + comp.text)
            else:
                merged.append(comp)
        new_chain = merged

        # At 后补零宽字符，避免部分平台吞艾特提示
        # P2-4：连续 [at:ID] 时，后一个 At 会把前一个 At 刚插入的零宽 Plain
        # 再补一次前缀，产生冗余 "\u200b\u200b"。改为两遍处理：先收集每个 At
        # 后第一个 Plain（去重，每个 Plain 只补一次前缀）与需要插入零宽的
        # 位置，再统一收尾应用，保证零宽不重复。
        plain_to_prefix: set = set()  # 需要补前缀的 Plain 在 new_chain 中的索引
        insert_after: list = []  # (at_index, Plain) 待插入的零宽 Plain
        for idx, comp in enumerate(new_chain):
            if isinstance(comp, At):
                plain_idx = next(
                    (
                        i
                        for i in range(idx + 1, len(new_chain))
                        if isinstance(new_chain[i], Plain)
                    ),
                    None,
                )
                if plain_idx is not None:
                    plain_to_prefix.add(plain_idx)
                else:
                    insert_after.append((idx, Plain("\u200b")))
        for at_idx, zwsp in reversed(insert_after):
            new_chain.insert(at_idx + 1, zwsp)
        for plain_idx in plain_to_prefix:
            new_chain[plain_idx] = Plain("\u200b" + new_chain[plain_idx].text)

        return new_chain

    # ------------------------------------------------------------------ #
    # P4 send 兜底渲染：免疫"清空 chain 后直接 event.send()"的第三方插件
    # ------------------------------------------------------------------ #
    def _wrap_event_send(self, event: AstrMessageEvent) -> None:
        """给当前事件实例的 send / send_streaming 绑定"发送前最后渲染"兜底。

        (a) 覆盖范围：本包装器只覆盖 `event.send()` 与
        `event.send_streaming()` 两条交付入口——清空 result.chain 后直接
        `event.send()` 的第三方插件、以及 respond.stage 直接调
        `event.send_streaming()` 的流式路径，都从这里过一遍，在消息真正发往
        平台前把链内已有标签渲染成 At 组件。

        (b) 覆盖不到、也不该依赖的路径：分段插件（astrbot_plugin_splitter）
        的**非末段**走 `context.send_message()` 直发平台
        （splitter/main.py:296-322 → context.py 直连 platform.send_by_session），
        **不经过任何 event.send 包装**。因此"分段后艾特不丢"的正确性并不靠
        本包装器，而是靠钩子顺序：AtTool 的 on_decorating_result 优先级
        -1000 高于 splitter 的 -1e17，按 `star_handler.py:22-26` 的
        `sort(key=lambda h: -h.extras_configs["priority"])` 降序执行 ⇒ 标签在
        分段之前就已渲染为 At 组件，splitter 只是把 At 按 at_strategy 归段。
        本包装器是"钩子看不到的链"的补充防线，不是分段场景的依赖。

        实例级绑定：仅影响当前事件对象，随事件生命周期结束自动失效，无
        全局副作用、热重载无需恢复。幂等：重复调用只包装一次。

        ⚠️需确认：AstrBot 无官方"发送前"事件钩子（on_decorating_result
        为唯一发送前扩展点，OnAfterMessageSentEvent 在发送后），故采用
        实例方法包装实现最终阶段渲染；包装仅处理含标签语法的消息，其余
        消息零开销原样透传。

        P0-2 收敛（2026-09-07 的"send 路径兜底补插 + 跨段剥离"已移除）：
        本包装器只做"链内已有标签 → At 渲染"（enable_fallback=False）。
        补插与兜底缓存消费只发生在主钩子 process_at_tags（实测早于
        splitter 的 -1e17 执行、看到完整链）；send 路径插手只会把 @ 补到
        第一段、或把后续段里的真实标签当"重复"剥离（探针场景 2/3/4）。
        """
        if getattr(event, "_attool_send_wrapped", False):
            return
        original_send = getattr(event, "send", None)
        if original_send is None:
            return

        async def _render_message(
            message: Any, keep_unclosed: bool = False
        ) -> Tuple[Any, bool]:
            """渲染含标签语法的消息；返回 (最终消息, 是否发生了替换)。

            P0-2 收敛：只做"链内已有标签 → At 渲染"（enable_fallback=
            False），不补插、不消费兜底缓存、不剥离其它段的标签——补插与
            缓存消费只由主钩子 process_at_tags 承担，否则分段插件逐段发送
            时 @ 会被补到第一段、后续段的真实标签会被当"重复"剥离。

            Args:
                message: 待发送的消息（MessageChain 或 str）。
                keep_unclosed: 流式 chunk 场景传 True，未闭合标签语法原样
                    保留（跨 chunk 拼合为非目标，不误删用户可见正文）。

            Returns:
                (最终消息, 是否发生了替换)：未渲染且无字面 \\n 兜底改写时
                返回原消息 + False，保证无标签消息零开销透传。
            """
            chain = getattr(message, "chain", None)
            is_str = isinstance(message, str)
            if is_str and has_at_tag(message):
                chain = [Plain(message)]
            if not isinstance(chain, list):
                return message, False
            # 字面 \n 兜底（2026-08-11）：LLM 偶发把换行转义序列（\n）
            # 当字面文本输出，分段插件只认真实换行切不动、发送管道不
            # 转义，导致明文 \n 暴露在群里。发送前统一转真实换行
            # （角色卡换行规范已治本，此为防御兜底）。P1-3：改写发生在
            # 新建的 Plain 上，不动调用方共享组件。
            # 快速路径：仅含反斜杠的 Plain 才处理，其余零开销。
            needs_newline_fix = any(
                isinstance(comp, Plain) and "\\n" in comp.text for comp in chain
            )
            if needs_newline_fix:
                chain = [
                    Plain(
                        comp.text.replace("\\r\\n", "\n")
                        .replace("\\r", "\n")
                        .replace("\\n", "\n")
                    )
                    if isinstance(comp, Plain)
                    else comp
                    for comp in chain
                ]
            if not any(
                isinstance(comp, Plain) and has_at_tag(comp.text) for comp in chain
            ):
                if not needs_newline_fix:
                    return message, False
                # 仅做了字面 \n 规范化：链需要替换，但没有标签渲染
                message.chain = chain
                return message, True
            rendered, new_chain = await self._render_at_tags(
                event, chain, keep_unclosed=keep_unclosed
            )
            if not rendered:
                return message, False
            if is_str:
                try:
                    from astrbot.core.message.message_event_result import (
                        MessageChain,
                    )
                except Exception:
                    return message, False
                return MessageChain(chain=new_chain), True
            message.chain = new_chain
            return message, True

        async def send_with_at_render(message: Any) -> None:
            try:
                rendered_message, _ = await _render_message(message)
            except Exception as exc:
                # 兜底渲染失败绝不影响消息发出
                logger.error(f"AtTool 发送前渲染兜底异常，已按原样发送: {exc}")
                rendered_message = message
            return await original_send(rendered_message)

        event.send = send_with_at_render  # 实例属性遮蔽类方法

        original_send_streaming = getattr(event, "send_streaming", None)
        if original_send_streaming is not None:

            async def send_streaming_with_at_render(
                generator: Any, use_fallback: bool = False
            ) -> None:
                """流式发送兜底渲染：逐段渲染 [at:ID] 后转交原 send_streaming。

                流式响应由 respond.stage 直接调 event.send_streaming() 交付
                平台适配器，绕过 event.send 包装器；若 [at:ID] 恰好整体落在
                某一段内，在此处渲染，保证引用回复/分段场景下艾特不丢失。
                跨 chunk 拼合属非目标（spec §7.2 第 18 行）：chunk 内的残缺
                标签语法原样透传（keep_unclosed=True），不误删正文。
                """

                async def render_gen() -> Any:
                    async for item in generator:
                        try:
                            rendered_message, _ = await _render_message(
                                item, keep_unclosed=True
                            )
                        except Exception as exc:
                            # 兜底渲染失败绝不影响消息发出
                            logger.error(
                                f"AtTool 流式发送兜底渲染异常，已按原样发送: {exc}"
                            )
                            rendered_message = item
                        yield rendered_message

                await original_send_streaming(render_gen(), use_fallback)

            event.send_streaming = send_streaming_with_at_render

        event._attool_send_wrapped = True

    # ------------------------------------------------------------------ #
    # 资源清理（CHG-07 清理全部新增缓存）
    # ------------------------------------------------------------------ #
    async def terminate(self) -> None:
        self._permission_cache.clear()
        self._member_cache.clear()
        self._at_all_last_trigger.clear()
        self._pending_choices.clear()
        self._fallback_at.clear()
        self._tool_absent_warned.clear()
        self._intent_unmet_warned.clear()
        self._known_at_ids.clear()
        logger.info("LLMAtToolPlugin 已清理权限/成员/冷却/待选/兜底缓存并卸载。")