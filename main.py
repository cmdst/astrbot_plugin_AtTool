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
    AT_TAG_PATTERN,
    audit_log_path,
    build_audit_record,
    build_permission_cache_key,
    check_session_lists,
    drop_expired,
    evict_oldest_to_limit,
    format_member_choice_list,
    format_single_member_result,
    is_at_all_in_cooldown,
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
_MEMBER_LIST_RETRY_DELAY: float = 0.5  # 重试间隔（秒，P2-7）

# 审计操作类型枚举（CHG-03）
_AUDIT_OP_AT_MEMBER = "at_member"
_AUDIT_OP_AT_ALL = "at_all"

# P1-2 宽松恢复正则：在"切碎/污染后重新完整暴露"的残留 Plain 文本中
# 搜索完整 [at:ID]（严格解析 AT_TAG_PATTERN 未能命中的场景），命中则
# 原位恢复为 At 组件。仅匹配成员标签（不含 [at:all]）。
_LOOSE_AT_PATTERN = re.compile(r"\[at:(\d+)\]")


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

        # ---- 缓存结构：统一 (payload, created_ts) 维度 ----
        # 权限缓存：value = (allowed, deny_reason, created_ts)
        self._permission_cache: dict[str, Tuple[bool, str, float]] = {}
        # 群成员缓存：value = (member_list, created_ts)
        self._member_cache: dict[str, Tuple[list, float]] = {}
        # @全体 路由级冷却记录：key=group_id, value=最近一次放行时间戳
        self._at_all_last_trigger: dict[str, float] = {}
        # 会话级多结果待选：key=unified_msg_origin, value=(matches, created_ts)
        self._pending_choices: dict[str, Tuple[list, float]] = {}
        # 兜底补插缓存（修复 2）：key=unified_msg_origin, value=(user_id, created_ts)
        # 仅 search_and_mention 唯一命中成员时写入；process_at_tags 消费后即删除
        self._fallback_at: dict[str, Tuple[str, float]] = {}

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
        return "\n\n".join(parts)

    @filter.on_llm_request()
    async def inject_at_instruction(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """在 LLM 请求前注入格式指令和动态配置。"""
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

        static_part = self.llm_prompt_base
        dynamic_part = self._build_dynamic_instructions()
        full_instruction = static_part + "\n\n" + dynamic_part
        req.system_prompt = (req.system_prompt or "") + full_instruction

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
        """
        在群聊中根据昵称或群名片搜索群成员，并返回可以直接使用的艾特标签。

        Args:
            name(string): 要搜索的群成员昵称或群名片，例如"张三"、"群主"、"管理员"。
        """
        umo = event.unified_msg_origin
        allowed, deny_reason = self._is_session_allowed(event)
        if not allowed:
            # 修复 2：最新一次搜索未解析出唯一成员，清除兜底缓存避免误艾特
            self._fallback_at.pop(umo, None)
            return f"【拒绝】{deny_reason}"

        group_id = event.get_group_id()
        if not group_id:
            self._fallback_at.pop(umo, None)
            return "【错误】当前不在群聊环境中。"
        if not _is_aiocqhttp_event(event):
            self._fallback_at.pop(umo, None)
            return "【错误】当前平台暂不支持此功能。"

        try:
            name_str = name or ""
            if not name_str.strip():
                # P2-2：空搜索词时 "" in nickname 恒为 True，会误命中全部成员
                self._fallback_at.pop(umo, None)
                return "【错误】搜索词为空，请提供要搜索的成员昵称或群名片。"

            raw_members = await self._get_group_members_cached(event, group_id)
            if not raw_members:
                self._fallback_at.pop(umo, None)
                return "【错误】无法获取群成员列表。"

            exact_matches = []
            fuzzy_matches = []

            for m in raw_members:
                user_id = str(m.get("user_id", ""))
                if not user_id:
                    continue
                nickname = m.get("nickname", "")
                card = m.get("card", "")
                role = m.get("role", "member")
                display_name = card or nickname

                if name_str == nickname or name_str == card or name_str == display_name:
                    exact_matches.append((user_id, display_name, role))
                elif self.enable_fuzzy_search and (
                    name_str in nickname or name_str in card
                ):
                    fuzzy_matches.append((user_id, display_name, role))

            matches = exact_matches if exact_matches else fuzzy_matches

            if not matches:
                mode = "包含" if self.enable_fuzzy_search else "完全匹配"
                self._fallback_at.pop(umo, None)
                return f"【未找到】群聊中没有找到名称{mode}「{name_str}」的成员。"

            if len(matches) > _MAX_MULTI_MATCHES:
                self._fallback_at.pop(umo, None)
                return (
                    f"【提示】名称包含「{name_str}」的成员过多（共 {len(matches)} 个），"
                    "请让用户提供更精确的姓名或群名片后再搜索。"
                )

            if len(matches) == 1:
                user_id, display_name, role = matches[0]
                # 修复 2：唯一命中写入兜底缓存；LLM 最终回复缺标签时由
                # process_at_tags 自动补插 [at:ID]
                self._fallback_at[umo] = (user_id, time.time())
                drop_expired(self._fallback_at, _FALLBACK_AT_TTL, time.time())
                return format_single_member_result(display_name, user_id, role)

            # 多结果：需用户选择序号后经 select_member_by_index 选定，
            # 不写入兜底缓存（多成员匹配不兜底，修复 2）
            self._fallback_at.pop(umo, None)
            self._pending_choices[umo] = (matches, time.time())
            drop_expired(self._pending_choices, _PENDING_CHOICE_TTL, time.time())
            # P2-3：待选缓存同样受容量上限约束（对称 CHG-07 治理）
            evict_oldest_to_limit(self._pending_choices, _MAX_PENDING_CHOICES)
            return format_member_choice_list(name_str, matches)

        except Exception as exc:
            logger.error(f"search_and_mention 异常: {exc}")
            self._fallback_at.pop(umo, None)
            return f"【错误】搜索时发生异常: {str(exc)}"

    # ------------------------------------------------------------------ #
    # 工具：按序号选定成员（CHG-05 新增）
    # ------------------------------------------------------------------ #
    @filter.llm_tool(name="select_member_by_index")
    async def select_member_by_index(
        self, event: AstrMessageEvent, index: int
    ) -> str:
        """
        在 search_and_mention 工具返回多个候选成员后，按用户选择的序号选定最终要艾特的成员，并返回可直接使用的艾特标签。

        Args:
            index(int): 用户选择的成员序号，从 1 开始，对应上次搜索返回列表中的编号。
        """
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

        返回 True 表示已补插（调用方应继续渲染）；False 表示无需/不可补插。
        缓存为一次性消费：无论是否补插，读取后即删除，防止跨轮次误用。
        """
        umo = event.unified_msg_origin
        now = time.time()
        cached = self._fallback_at.get(umo)
        if not cached or (now - cached[1]) >= _FALLBACK_AT_TTL:
            self._fallback_at.pop(umo, None)
            return False
        # 链中已有其他来源插入的 At 组件时不再兜底（避免重复艾特）
        if any(isinstance(comp, At) for comp in chain):
            self._fallback_at.pop(umo, None)
            return False
        # 会话准入：不允许艾特的会话不兜底
        allowed, _ = self._is_session_allowed(event)
        if not allowed:
            self._fallback_at.pop(umo, None)
            return False
        user_id = cached[0]
        self._fallback_at.pop(umo, None)
        chain.append(Plain(f"[at:{user_id}]"))
        logger.warning(
            f"兜底补插 [at:{user_id}]：LLM 回复未携带艾特标签（会话 {umo}）"
        )
        return True

    @filter.on_decorating_result(priority=-1000)
    async def process_at_tags(self, event: AstrMessageEvent) -> None:
        """渲染 [at:ID] → At 组件（P4 延迟渲染：最后执行，免疫第三方插件破坏）。

        P4 兼容改造：priority 由 2 改为 -1000，按 AstrBot 钩子排序
        （star_handler.py: sort(key=lambda h: -h.extras_configs["priority"])）
        在全部 on_decorating_result 钩子之后执行。[at:ID] 以纯文本形式贯穿
        所有中间插件（包括重建 chain 的插件，只要其保留 Plain 文本），由本
        钩子统一最后渲染为 At 组件——任何插件都无法再丢弃。

        对"清空 result.chain 后直接 event.send()"的插件（如分段插件），
        本钩子执行时链已空、标签已作为文本发出，由 on_llm_request 阶段安装
        的 send 兜底渲染（_wrap_event_send → _render_at_tags）在消息真正发往
        平台前完成渲染，保证艾特不丢失。

        P1-1（QA 方案 A）：本钩子是"兜底补插 + 缓存消费"的唯一入口，
        enable_fallback=True——send 兜底路径绝不补插/消费，避免分段插件
        逐段 send 时（段1无标签先触发兜底、段2含标签再渲染）同一成员被
        重复艾特。
        """
        result = event.get_result()
        if not result or not result.chain:
            # 缺陷 2 修复：主钩子链空时同样解除兜底缓存。
            # 分段插件"清链直发"后本钩子执行时链已空，无法履行兜底补插
            # 义务；而 send 路径刻意不消费缓存（P1-1 方案 A，保留给主钩子），
            # 若此处不清理，缓存将残留满 TTL(120s)，新一轮 LLM 回复无标签
            # 时会误补插上一轮成员（跨轮次误艾特）。故链空时一并解除缓存。
            self._fallback_at.pop(event.unified_msg_origin, None)
            return
        rendered, new_chain = await self._render_at_tags(
            event, result.chain, enable_fallback=True
        )
        if rendered:
            result.chain = new_chain

    # ------------------------------------------------------------------ #
    # P4 公共渲染：主钩子与 send 兜底共用的 [at:ID] → At 渲染流水线
    # ------------------------------------------------------------------ #
    async def _render_at_tags(
        self,
        event: AstrMessageEvent,
        chain: List[BaseMessageComponent],
        enable_fallback: bool = False,
    ) -> Tuple[bool, List[BaseMessageComponent]]:
        """把消息链中的 [at:ID] / [at:all] 渲染为 At 组件（唯一渲染流水线）。

        process_at_tags 钩子与 send 兜底渲染共用本方法，保证两条路径的
        会话准入 / @全体权限+冷却 / 审计 / 零宽清理语义完全一致。

        Args:
            event: 当前消息事件（会话准入、@全体权限、审计依赖）。
            chain: 待渲染的消息链组件列表（调用方保证非空）。
            enable_fallback: 是否允许"兜底补插 + 缓存消费"（P1-1，QA 方案
                A）。仅主钩子 process_at_tags（整链最后渲染）传 True；send
                兜底路径（_wrap_event_send）保持默认 False，只做"已有标签
                时的渲染"，绝不补插、绝不消费缓存——否则分段插件逐段 send
                时：段1无标签触发兜底补插（消费缓存）、段2含标签再渲染一次，
                同一成员被重复艾特。

        Returns:
            (rendered, new_chain)：rendered=True 表示链已被重建，调用方应
            用 new_chain 替换原链；False 表示链中无标签且未触发兜底补插，
            原链无需改动。
        """
        umo = event.unified_msg_origin

        # ① 合并相邻 Plain：修复标签被第三方插件拆分到相邻组件的情况
        #    （如 "[at:12" 与 "345]" 分处两个 Plain，合并后可完整匹配）。
        #    注：此处原地修改首个 Plain 的 text；渲染后整链被替换，原对象
        #    不再被 result.chain 引用，无副作用。
        merged_input: List[BaseMessageComponent] = []
        for comp in chain:
            if (
                isinstance(comp, Plain)
                and merged_input
                and isinstance(merged_input[-1], Plain)
            ):
                merged_input[-1].text += comp.text
            else:
                merged_input.append(comp)

        # ② 扫描标签（无标签时走兜底补插，修复 2）
        has_tag = False
        has_at_all_tag = False
        for comp in merged_input:
            if isinstance(comp, Plain) and "[at:" in comp.text:
                has_tag = True
                if "[at:all]" in comp.text:
                    has_at_all_tag = True
        if not has_tag:
            if not enable_fallback:
                # P1-1（QA 方案 A）：send 兜底路径不补插、不消费缓存——
                # 补插与消费必须成对发生在主钩子（整链最后渲染），否则
                # 分段插件逐段 send 时：段1无标签触发兜底补插（消费缓存）、
                # 段2含标签再渲染一次，导致同一成员被重复艾特。
                return False, chain
            if not self._apply_fallback_at_tag(event, merged_input):
                return False, chain
            has_tag = True
        elif enable_fallback:
            # 回复已含标签：本次兜底义务解除，仅主钩子消费清理缓存。
            # send 路径不消费（保留给主钩子），保证"补插"与"消费"只由
            # 主钩子成对执行，杜绝跨路径竞争导致的重复/漏补。
            self._fallback_at.pop(umo, None)

        # ③ 会话准入：不允许则统一降级（剥离 [at:ID]，[at:all] 转纯文本）
        allowed, _ = self._is_session_allowed(event)
        if not allowed:
            degraded: List[BaseMessageComponent] = []
            for comp in merged_input:
                if isinstance(comp, Plain):
                    comp.text = re.sub(r"\[at:\d+\]", "", comp.text)
                    comp.text = re.sub(r"\[at:all\]", "@全体成员", comp.text)
                degraded.append(comp)
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

        # ⑤ 逐组件解析标签并渲染
        new_chain: List[BaseMessageComponent] = []
        at_member_targets: List[str] = []
        for comp in merged_input:
            if isinstance(comp, Plain):
                for kind, value in split_text_by_at_tags(comp.text):
                    if kind == "text":
                        if value:
                            new_chain.append(Plain(value))
                    elif value == "all":
                        if at_all_allowed:
                            new_chain.append(At(qq="all"))
                        else:
                            new_chain.append(Plain("@全体成员"))
                    else:
                        new_chain.append(At(qq=value))
                        at_member_targets.append(value)
            else:
                new_chain.append(comp)

        # ⑥ 审计写入：@单人逐条 + @全体整体一条（CHG-03）
        if self.enable_audit_log:
            for target_id in at_member_targets:
                self._write_audit_log(
                    event, _AUDIT_OP_AT_MEMBER, target_id, True, "ok"
                )
            if has_at_all_tag:
                self._write_audit_log(
                    event,
                    _AUDIT_OP_AT_ALL,
                    "all",
                    at_all_allowed,
                    at_all_deny_reason or "ok",
                )

        # ⑦ 收尾（零宽清理/空剔除/相邻合并/At 后补零宽）
        new_chain = self._finalize_chain(new_chain)

        # ⑦b P1-2 宽松恢复：合并+清洗后仍含 "[at:" 的 Plain，若其中重新
        #     暴露出完整 [at:ID]（如标签被切碎且混入零宽字符，finalize 清洗
        #     零宽后标签恢复完整），则原位恢复为 At 组件（每链至多一次）。
        recovered_chain = self._recover_loose_at_tags(event, new_chain)
        if recovered_chain is not None:
            new_chain = recovered_chain

        # ⑦c 残缺标签告警（宽松恢复后仍未消除的 [at: 残留）
        residual = [
            comp.text
            for comp in new_chain
            if isinstance(comp, Plain) and "[at:" in comp.text
        ]
        if residual:
            logger.warning(
                "检测到未闭合的 [at: 标签残留（可能被第三方插件在分段/重建时"
                f"切断），已按纯文本保留: {residual!r}，会话 {umo}"
            )
        return True, new_chain

    def _recover_loose_at_tags(
        self, event: AstrMessageEvent, chain: List[BaseMessageComponent]
    ) -> Optional[List[BaseMessageComponent]]:
        """P1-2 宽松恢复：把切碎/污染后重新完整暴露的 [at:ID] 恢复为 At。

        背景：P4 延迟渲染（priority=-1000）使 [at:ID] 以纯文本穿过所有
        中间插件；AstrBot 内建 process_buffer（astr_message_event.py 按
        [^。？！~…]+[。？！~…]+ 切文本流）或分段类第三方插件可能把标签切
        成碎片（如 [at:12] 与 [345] 分处组件、标签内混入零宽字符/空白），
        严格正则 AT_TAG_PATTERN 无法命中 → 标签以裸文本发出。

        本方法在渲染收尾处兜底，按两种形态恢复（每链至多恢复一次，防止
        多次恢复导致组件乱序；命中即记 warning 告警保证可观测）：
          a) 单组件完整暴露：对合并+清洗后仍含 "[at:" 的 Plain 组件，用
             宽松正则 _LOOSE_AT_PATTERN 搜索，命中完整 [at:ID] 则在原
             位置前插入 At(qq=ID) 组件并移除标签文本；
          b) 跨组件切碎（缺陷 1 修复）：Plain 含 "[at:" 前缀但缺闭合 "]"、
             后续 Plain 含闭合 "]"（中间可隔 At 等非 Plain 组件）时，跨
             组件拼合出完整 [at:ID]，从各组件剥离对应片段并在原位插入
             At(qq=ID)，中间组件保持原序。
        前后文本均按原序保留，不误删用户文本；零宽字符视为透明（不参与
        标签内容，At 后补零宽前缀保留给原组件）。

        返回恢复后的新链；无命中返回 None（调用方保持原链不变）。
        """
        umo = event.unified_msg_origin
        for idx, comp in enumerate(chain):
            if not isinstance(comp, Plain) or "[at:" not in comp.text:
                continue
            # 形态 a：单组件内完整 [at:ID]（零宽污染经 finalize 清洗后重新暴露）
            match = _LOOSE_AT_PATTERN.search(comp.text)
            if match:
                target_id = match.group(1)
                before = comp.text[: match.start()]
                after = comp.text[match.end() :]
                logger.warning(
                    "宽松恢复 [at:%s]：检测到被切碎/污染的艾特标签残留（严格"
                    "正则未命中），已原位恢复渲染为 At 组件（会话 %s）",
                    target_id,
                    umo,
                )
                restored: List[BaseMessageComponent] = []
                if before:
                    restored.append(Plain(before))
                restored.append(At(qq=target_id))
                if after:
                    restored.append(Plain(after))
                chain[idx : idx + 1] = restored
                return chain
            # 形态 b：跨组件切碎（[at: 前缀与闭合 ] 分处不同 Plain，中间
            # 隔着 At 等组件，步骤①的相邻 Plain 合并无法修复）
            recovered = self._recover_cross_component_at_tag(chain, idx, comp)
            if recovered is not None:
                target_id, rebuilt = recovered
                logger.warning(
                    "宽松恢复 [at:%s]：检测到跨组件切碎的艾特标签残留（严格"
                    "正则未命中，前缀与闭合括号分处不同组件），已原位恢复"
                    "渲染为 At 组件（会话 %s）",
                    target_id,
                    umo,
                )
                return rebuilt
        return None

    def _recover_cross_component_at_tag(
        self,
        chain: List[BaseMessageComponent],
        idx: int,
        comp: Plain,
    ) -> Optional[Tuple[str, List[BaseMessageComponent]]]:
        """跨组件拼合恢复（P1-2 缺陷 1 修复）。

        链形态如 Plain("…[at:12") + At(999) + Plain("345]…")："[at:" 前缀
        与闭合 "]" 分处两个 Plain，中间隔着 At 等非 Plain 组件——步骤①的
        相邻 Plain 合并无法把它们拼到一起（被 At 隔开），单组件宽松正则
        也因缺闭合 "]" 不命中，标签以裸文本残留。

        本方法以 comp.text 中每个 "[at:" 位置为起点，跨过中间组件累积
        后续 Plain 的文本（零宽字符视为透明、不参与标签内容），直到找到
        闭合 "]"；若拼出完整 [at:\\d+] 标签则重建链：
          - 当前 Plain 剥离 "[at:" 起的片段（此前文本保留）；
          - 后续 Plain 按消耗量剥离被标签占用的文本（含 "]"），剩余文本
            及其开头零宽（At 后补零宽语义）保留；
          - 在原位置插入 At(qq=ID)，中间组件保持原序不动。
        成功返回 (target_id, 重建后的链)；所有 "[at:" 起点均无法拼合时返回
        None（调用方保持原链，由 ⑦c 残缺告警兜底）。
        """
        text = comp.text
        for at_start in (m.start() for m in re.finditer(r"\[at:", text)):
            partial = text[at_start + 4 :].replace("\u200b", "")
            collected = ""
            consumption: List[Tuple[int, int]] = []  # (组件索引, 消耗的非零宽字符数)
            for j in range(idx + 1, len(chain)):
                nxt = chain[j]
                if not isinstance(nxt, Plain):
                    # 跨过中间组件（At 等）：其内容不参与标签拼接，原位保留
                    continue
                seg = nxt.text
                seg_clean = seg.replace("\u200b", "")
                close_pos = seg_clean.find("]")
                if close_pos == -1:
                    consumption.append((j, len(seg_clean)))
                    collected += seg_clean
                    continue
                # 找到闭合 "]"：消耗到 "]" 为止（含），按"清洗零宽后"的
                # 字符数记录，与 _strip_clean_prefix 的语义一致（零宽透明）
                consumption.append((j, close_pos + 1))
                collected += seg_clean[: close_pos + 1]
                break
            else:
                continue  # 后续无 Plain 可提供闭合 "]"（或根本没有后续组件）
            match = _LOOSE_AT_PATTERN.fullmatch("[at:" + partial + collected)
            if match is None:
                continue
            target_id = match.group(1)
            # ——重建链：剥离片段、原位插入 At、中间组件保持原序——
            rebuilt: List[BaseMessageComponent] = []
            head = text[:at_start]
            if head:
                rebuilt.append(Plain(head))
            rebuilt.append(At(qq=target_id))
            consumed_map = dict(consumption)
            for j in range(idx + 1, len(chain)):
                if j in consumed_map:
                    remain = _strip_clean_prefix(chain[j].text, consumed_map[j])
                    if remain:
                        rebuilt.append(Plain(remain))
                else:
                    rebuilt.append(chain[j])
            return target_id, rebuilt
        return None

    def _finalize_chain(
        self, new_chain: List[BaseMessageComponent]
    ) -> List[BaseMessageComponent]:
        """渲染收尾：零宽字符清理 / 空 Plain 剔除 / 相邻 Plain 合并 / At 后补零宽。

        逻辑与原 process_at_tags 收尾一致（含 P2-4 两遍处理，保证零宽不重复）。
        """
        for comp in new_chain:
            if isinstance(comp, Plain):
                comp.text = comp.text.strip().replace("\u200b", "")

        new_chain = [
            comp
            for comp in new_chain
            if not (isinstance(comp, Plain) and not comp.text)
        ]

        merged: List[BaseMessageComponent] = []
        for comp in new_chain:
            if isinstance(comp, Plain) and merged and isinstance(merged[-1], Plain):
                merged[-1].text += comp.text
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
            new_chain[plain_idx].text = "\u200b" + new_chain[plain_idx].text

        return new_chain

    # ------------------------------------------------------------------ #
    # P4 send 兜底渲染：免疫"清空 chain 后直接 event.send()"的第三方插件
    # ------------------------------------------------------------------ #
    def _wrap_event_send(self, event: AstrMessageEvent) -> None:
        """给当前事件实例的 send 绑定"发送前最后渲染"兜底（实例级、幂等）。

        第三方插件（如分段插件）可能在 on_decorating_result 钩子内
        result.chain.clear() 后直接 event.send() 纯文本段——此时 [at:ID]
        仍是文本，延迟渲染钩子（priority=-1000）执行时链已空、标签已发出。
        本包装器在消息真正发往平台前完成渲染，使艾特免疫"清链直发"类插件。

        实例级绑定：仅影响当前事件对象，随事件生命周期结束自动失效，无
        全局副作用、热重载无需恢复。幂等：重复调用只包装一次。

        ⚠️需确认：AstrBot 无官方"发送前"事件钩子（on_decorating_result
        为唯一发送前扩展点，OnAfterMessageSentEvent 在发送后），故采用
        实例方法包装实现最终阶段渲染；包装仅检查含 "[at:" 的链，其余消息
        零开销原样透传。
        """
        if getattr(event, "_attool_send_wrapped", False):
            return
        original_send = getattr(event, "send", None)
        if original_send is None:
            return

        async def send_with_at_render(message: Any) -> None:
            try:
                chain = getattr(message, "chain", None)
                if isinstance(chain, list):
                    rendered, new_chain = await self._render_at_tags(event, chain)
                    if rendered:
                        message.chain = new_chain
            except Exception as exc:
                # 兜底渲染失败绝不影响消息发出
                logger.error(f"AtTool 发送前渲染兜底异常，已按原样发送: {exc}")
            return await original_send(message)

        event.send = send_with_at_render  # 实例属性遮蔽类方法
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
        logger.info("LLMAtToolPlugin 已清理权限/成员/冷却/待选/兜底缓存并卸载。")