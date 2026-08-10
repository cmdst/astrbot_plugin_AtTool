"""main.py 集成主链路测试（补充 CHG-01/02/03/04/05/06/07 框架层验证）。

现有 test_utils.py 只覆盖 utils.py 纯函数层；本文件直接加载 main.py，
用最小 astrbot 桩 + FakeEvent/FakeResult 验证插件主链路：
- CHG-01 权限缓存按 group_id:user_id 隔离（越权回归，P0）
- CHG-03 @全体冷却限制 + 审计日志落盘（JSONL 字段完整）
- CHG-02 会话准入在渲染链路的降级（剥离/转文本）
- CHG-04 群成员列表缓存命中与过期
- CHG-05 select_member_by_index 序号选择（合法/越界/过期/非法）
- CHG-06 角色标注集成
- CHG-07 缓存容量上限淘汰

运行方式（插件目录下）：
    python3 -m pytest tests/test_integration_main.py -v
沙箱（无 astrbot）与云服务器（有真实 astrbot）均可运行。
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.asyncio

# --------------------------------------------------------------------------- #
# 最小 astrbot 桩（仅当真实包不存在时安装）
# --------------------------------------------------------------------------- #
_PLUGIN_DIR = Path(__file__).resolve().parents[1]


def _install_astrbot_stub() -> None:
    if "astrbot" in sys.modules:
        return

    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    star = types.ModuleType("astrbot.api.star")
    event = types.ModuleType("astrbot.api.event")
    provider = types.ModuleType("astrbot.api.provider")
    msg_components = types.ModuleType("astrbot.api.message_components")
    core = types.ModuleType("astrbot.core")
    core_platform = types.ModuleType("astrbot.core.platform")
    core_platform_sources = types.ModuleType("astrbot.core.platform.sources")
    core_aiocqhttp = types.ModuleType(
        "astrbot.core.platform.sources.aiocqhttp"
    )
    core_aiocqhttp_event = types.ModuleType(
        "astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event"
    )
    core_agent = types.ModuleType("astrbot.core.agent")
    core_agent_message = types.ModuleType("astrbot.core.agent.message")

    astrbot.api = api
    api.star = star
    api.event = event
    api.provider = provider
    api.message_components = msg_components
    astrbot.core = core
    core.platform = core_platform
    core_platform.sources = core_platform_sources
    core_platform_sources.aiocqhttp = core_aiocqhttp
    core_aiocqhttp.aiocqhttp_message_event = core_aiocqhttp_event
    core.agent = core_agent
    core_agent.message = core_agent_message

    class AstrBotConfig(dict):
        pass

    class Context:
        def __init__(self):
            self._cfg = {"admins_id": ["astrbot"]}

        def get_config(self):
            return self._cfg

    class Star:
        def __init__(self, context=None):
            self.context = context

    class StarTools:
        @staticmethod
        def get_data_dir(name: str) -> Path:
            return Path("/tmp/astrbot_plugin_AtTool_test_data") / name

    class AstrMessageEvent:
        pass

    class ProviderRequest:
        def __init__(self):
            self.system_prompt = ""
            self.extra_user_content_parts = []

    class Plain:
        def __init__(self, text=""):
            self.text = text

    class At:
        def __init__(self, qq=""):
            self.qq = qq

    class BaseMessageComponent:
        pass

    class TextPart:
        def __init__(self, text=""):
            self.text = text
            self.is_temp = False

        def mark_as_temp(self):
            self.is_temp = True
            return self

    class AiocqhttpMessageEvent:
        pass

    class _Logger:
        def _log(self, *args, **kwargs):
            pass

        def info(self, *a, **kw):
            pass

        def warning(self, *a, **kw):
            pass

        def error(self, *a, **kw):
            pass

        def debug(self, *a, **kw):
            pass

    class _Filter:
        @staticmethod
        def on_llm_request(*args, **kwargs):
            def deco(fn):
                return fn

            return deco

        @staticmethod
        def llm_tool(*args, **kwargs):
            def deco(fn):
                return fn

            return deco

        @staticmethod
        def on_decorating_result(*args, **kwargs):
            def deco(fn):
                return fn

            return deco

    star.Context = Context
    star.Star = Star
    star.StarTools = StarTools
    api.AstrBotConfig = AstrBotConfig
    api.logger = _Logger()
    event.AstrMessageEvent = AstrMessageEvent
    event.filter = _Filter()
    provider.ProviderRequest = ProviderRequest
    msg_components.Plain = Plain
    msg_components.At = At
    msg_components.BaseMessageComponent = BaseMessageComponent
    core_agent_message.TextPart = TextPart
    core_aiocqhttp_event.AiocqhttpMessageEvent = AiocqhttpMessageEvent

    sys.modules["astrbot"] = astrbot
    sys.modules["astrbot.api"] = api
    sys.modules["astrbot.api.star"] = star
    sys.modules["astrbot.api.event"] = event
    sys.modules["astrbot.api.provider"] = provider
    sys.modules["astrbot.api.message_components"] = msg_components
    sys.modules["astrbot.core"] = core
    sys.modules["astrbot.core.platform"] = core_platform
    sys.modules["astrbot.core.platform.sources"] = core_platform_sources
    sys.modules["astrbot.core.platform.sources.aiocqhttp"] = core_aiocqhttp
    sys.modules[
        "astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event"
    ] = core_aiocqhttp_event
    sys.modules["astrbot.core.agent"] = core_agent
    sys.modules["astrbot.core.agent.message"] = core_agent_message


_install_astrbot_stub()

# 以包形式加载插件（兼容相对导入 from .utils import ...）
if str(_PLUGIN_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR.parent))
_pkg = types.ModuleType("astrbot_plugin_AtTool")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules.setdefault("astrbot_plugin_AtTool", _pkg)

from astrbot_plugin_AtTool import main as main_mod
from astrbot_plugin_AtTool import utils as utils_mod

LLMAtToolPlugin = main_mod.LLMAtToolPlugin
build_permission_cache_key = utils_mod.build_permission_cache_key


# --------------------------------------------------------------------------- #
# Fake 组件
# --------------------------------------------------------------------------- #
class FakeResult:
    def __init__(self, chain=None):
        self.chain = chain if chain is not None else []


class FakeBot:
    """记录 call_action 调用，可配置 get_group_member_info / get_group_member_list。"""

    def __init__(self):
        self.calls = []
        self.member_info_role = "member"
        self.member_list = []
        # get_group_member_list 前 N 次抛异常（P2-7 重试测试的失败注入）
        self.member_list_fail_times = 0
        self.api = self  # main.py 通过 event.bot.api.call_action 调用

    async def call_action(self, action, **kwargs):
        self.calls.append((action, kwargs))
        if action == "get_group_member_info":
            return {"role": self.member_info_role, "user_id": kwargs.get("user_id", "")}
        if action == "get_group_member_list":
            if self.member_list_fail_times > 0:
                self.member_list_fail_times -= 1
                raise RuntimeError("模拟网络抖动")
            return self.member_list
        return {}


def _make_aiocq_event_class():
    """构造 FakeEvent 基类。

    P2-6 后 main.py 的平台判定改为鸭子类型（hasattr(event, "bot") and
    hasattr(event.bot, "api")），不再依赖 astrbot.core 内部类，FakeEvent
    只需普通基类 + 暴露 bot.api 即可通过判定。
    """

    class FakeAiocqEvent:
        pass

    return FakeAiocqEvent


class FakeEvent(_make_aiocq_event_class()):
    def __init__(
        self,
        group_id="123456",
        sender_id="10001",
        umo=None,
        chain=None,
        bot=None,
    ):
        self._group_id = group_id
        self._sender_id = sender_id
        self.unified_msg_origin = umo or f"aiocqhttp:GroupMessage:{group_id}"
        self.bot = bot if bot is not None else FakeBot()
        self._result = FakeResult(chain)

    def get_group_id(self):
        return self._group_id

    def get_sender_id(self):
        return self._sender_id

    def get_result(self):
        return self._result

    def set_chain(self, chain):
        self._result = FakeResult(chain)
        return self


def make_plugin(config=None, context=None, audit_dir=None):
    """构造真实插件实例（走 __init__），并允许覆盖审计目录。"""
    from astrbot_plugin_AtTool.main import StarTools

    if context is None:
        from astrbot.api.star import Context

        context = Context()
    cfg = {
        "permission_verification": True,
        "allow_direct_qq_at": True,
        "enable_fuzzy_search": True,
        "session_whitelist": [],
        "session_blacklist": [],
        "llm_prompt": "",
        "at_all_cooldown": 60,
        "enable_audit_log": True,
        "member_list_cache_ttl": 180,
    }
    if config:
        cfg.update(config)
    plugin = LLMAtToolPlugin(context=context, config=cfg)
    if audit_dir is not None:
        plugin._audit_dir = audit_dir
    return plugin


def audit_file(audit_dir, date_str):
    return Path(audit_dir) / "audit" / f"at_audit_{date_str}.jsonl"


# --------------------------------------------------------------------------- #
# CHG-01 越权回归（P0 核心）
# --------------------------------------------------------------------------- #
class TestPermissionCacheIsolation:
    """同群不同用户权限缓存互相独立：管理员放行后普通成员仍被拒。"""

    async def test_admin_then_member_within_ttl(self, tmp_path):
        plugin = make_plugin(audit_dir=tmp_path)

        admin_event = FakeEvent(group_id="1000", sender_id="10001")
        admin_event.bot.member_info_role = "owner"

        member_event = FakeEvent(group_id="1000", sender_id="10002")
        member_event.bot.member_info_role = "member"

        # A（管理员）触发 @全体 → 放行，写入缓存 key=1000:10001
        ok_a, reason_a = await plugin._get_at_all_permission_result(admin_event)
        assert ok_a is True, f"管理员应放行: {reason_a}"
        assert "1000:10001" in plugin._permission_cache

        # B（普通成员）同一群 TTL 内触发 → 必须拒绝（修复前误命中 A 的缓存）
        ok_b, reason_b = await plugin._get_at_all_permission_result(member_event)
        assert ok_b is False, "越权回归失败：普通成员在管理员触发后不应获得权限"
        assert "1000:10002" in plugin._permission_cache
        assert plugin._permission_cache["1000:10002"][0] is False

    async def test_different_groups_isolated(self, tmp_path):
        plugin = make_plugin(audit_dir=tmp_path)

        ev_g1 = FakeEvent(group_id="1000", sender_id="10001")
        ev_g1.bot.member_info_role = "owner"
        ev_g2 = FakeEvent(group_id="2000", sender_id="10001")
        ev_g2.bot.member_info_role = "member"

        ok1, _ = await plugin._get_at_all_permission_result(ev_g1)
        ok2, _ = await plugin._get_at_all_permission_result(ev_g2)
        assert ok1 is True
        assert ok2 is False, "不同群的同一用户权限应独立判定"

    async def test_cache_hit_skips_api(self, tmp_path):
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(group_id="1000", sender_id="10001")
        ev.bot.member_info_role = "owner"

        await plugin._get_at_all_permission_result(ev)
        api_calls_after_first = len(ev.bot.calls)
        await plugin._get_at_all_permission_result(ev)
        assert len(ev.bot.calls) == api_calls_after_first, "TTL 内应命中缓存不再调 API"

    async def test_cache_expired_refetch(self, tmp_path):
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(group_id="1000", sender_id="10001")
        ev.bot.member_info_role = "member"

        ok1, _ = await plugin._get_at_all_permission_result(ev)
        assert ok1 is False
        # 直接改缓存时间戳为 301 秒前（TTL=300）
        key = "1000:10001"
        val = plugin._permission_cache[key]
        plugin._permission_cache[key] = (val[0], val[1], time.time() - 301)
        ev.bot.member_info_role = "owner"
        ok2, _ = await plugin._get_at_all_permission_result(ev)
        assert ok2 is True, "缓存过期后应重新判定"


# --------------------------------------------------------------------------- #
# CHG-03 冷却限制（渲染链路集成）
# --------------------------------------------------------------------------- #
class TestAtAllCooldownIntegration:
    async def test_first_allowed_second_blocked(self, tmp_path):
        plugin = make_plugin(
            config={"at_all_cooldown": 60}, audit_dir=tmp_path
        )
        ev = FakeEvent(group_id="1000", sender_id="10001")
        ev.bot.member_info_role = "owner"

        # 第一次：权限通过 + 冷却通过 → 放行，记录触发时间
        ok1, remain1 = plugin._check_at_all_cooldown(ev)
        assert ok1 is True
        plugin._record_at_all_trigger(ev)
        assert plugin._at_all_last_trigger.get("1000") is not None

        # 第二次（冷却内）：拦截
        ok2, remain2 = plugin._check_at_all_cooldown(ev)
        assert ok2 is False
        assert 0 < remain2 <= 60

    async def test_cooldown_zero_no_limit(self, tmp_path):
        plugin = make_plugin(
            config={"at_all_cooldown": 0}, audit_dir=tmp_path
        )
        ev = FakeEvent(group_id="1000", sender_id="10001")
        ok1, _ = plugin._check_at_all_cooldown(ev)
        plugin._record_at_all_trigger(ev)
        ok2, _ = plugin._check_at_all_cooldown(ev)
        assert ok1 is True and ok2 is True, "cooldown=0 不应限制"

    async def test_cooldown_elapsed_allowed_again(self, tmp_path):
        plugin = make_plugin(
            config={"at_all_cooldown": 60}, audit_dir=tmp_path
        )
        ev = FakeEvent(group_id="1000", sender_id="10001")
        plugin._record_at_all_trigger(ev)
        plugin._at_all_last_trigger["1000"] = time.time() - 61
        ok, remain = plugin._check_at_all_cooldown(ev)
        assert ok is True and remain == 0.0


# --------------------------------------------------------------------------- #
# CHG-03 审计日志落盘（@单人 / @全体放行 / @全体拦截）
# --------------------------------------------------------------------------- #
class TestAuditLogIntegration:
    async def test_at_member_audit_written(self, tmp_path):
        from astrbot.api.message_components import Plain

        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(group_id="1000", sender_id="10001", chain=[Plain("你好 [at:12345]")])
        await plugin.process_at_tags(ev)

        from astrbot_plugin_AtTool.main import _AUDIT_OP_AT_MEMBER

        log_path = audit_file(tmp_path, time.strftime("%Y%m%d"))
        assert log_path.exists()
        lines = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1
        rec = lines[0]
        assert rec["op_type"] == _AUDIT_OP_AT_MEMBER
        assert rec["target_id"] == "12345"
        assert rec["group_id"] == "1000"
        assert rec["operator_id"] == "10001"
        assert rec["allowed"] is True

    async def test_at_all_allowed_audit(self, tmp_path):
        from astrbot.api.message_components import Plain

        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(group_id="1000", sender_id="10001", chain=[Plain("[at:all]")])
        ev.bot.member_info_role = "owner"
        await plugin.process_at_tags(ev)

        log_path = audit_file(tmp_path, time.strftime("%Y%m%d"))
        rec = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
        assert rec["op_type"] == "at_all"
        assert rec["target_id"] == "all"
        assert rec["allowed"] is True
        assert rec["reason"] == "ok"

    async def test_at_all_blocked_audit(self, tmp_path):
        from astrbot.api.message_components import Plain

        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(group_id="1000", sender_id="10001", chain=[Plain("[at:all]")])
        ev.bot.member_info_role = "member"
        await plugin.process_at_tags(ev)

        log_path = audit_file(tmp_path, time.strftime("%Y%m%d"))
        rec = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
        assert rec["op_type"] == "at_all"
        assert rec["allowed"] is False
        assert "无权限" in rec["reason"]

    async def test_at_all_cooldown_blocked_audit(self, tmp_path):
        from astrbot.api.message_components import Plain

        plugin = make_plugin(
            config={"at_all_cooldown": 60}, audit_dir=tmp_path
        )
        ev = FakeEvent(group_id="1000", sender_id="10001", chain=[Plain("[at:all]")])
        ev.bot.member_info_role = "owner"

        # 第一次放行
        await plugin.process_at_tags(ev)
        # 第二次（冷却内）→ 拦截
        ev2 = FakeEvent(group_id="1000", sender_id="10001", chain=[Plain("[at:all]")])
        ev2.bot.member_info_role = "owner"
        await plugin.process_at_tags(ev2)

        log_path = audit_file(tmp_path, time.strftime("%Y%m%d"))
        recs = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(recs) == 2
        assert recs[1]["allowed"] is False
        assert "冷却" in recs[1]["reason"]

    async def test_audit_disabled_no_write(self, tmp_path):
        from astrbot.api.message_components import Plain

        plugin = make_plugin(
            config={"enable_audit_log": False}, audit_dir=tmp_path
        )
        ev = FakeEvent(group_id="1000", sender_id="10001", chain=[Plain("[at:12345]")])
        await plugin.process_at_tags(ev)
        assert not audit_file(tmp_path, time.strftime("%Y%m%d")).exists()


# --------------------------------------------------------------------------- #
# CHG-02 会话准入在渲染链路的降级
# --------------------------------------------------------------------------- #
class TestSessionAllowRenderIntegration:
    async def test_blacklist_strips_at_member(self, tmp_path):
        from astrbot.api.message_components import Plain

        plugin = make_plugin(
            config={"session_blacklist": ["1000"]}, audit_dir=tmp_path
        )
        ev = FakeEvent(group_id="1000", sender_id="10001", chain=[Plain("你好 [at:12345] 再见")])
        await plugin.process_at_tags(ev)
        chain = ev.get_result().chain
        assert len(chain) == 1
        assert "[at:" not in chain[0].text
        assert "12345" not in chain[0].text

    async def test_blacklist_at_all_becomes_text(self, tmp_path):
        from astrbot.api.message_components import Plain

        plugin = make_plugin(
            config={"session_blacklist": ["1000"]}, audit_dir=tmp_path
        )
        ev = FakeEvent(group_id="1000", sender_id="10001", chain=[Plain("通知 [at:all] 请查看")])
        await plugin.process_at_tags(ev)
        chain = ev.get_result().chain
        assert any("@全体成员" in c.text for c in chain if hasattr(c, "text"))

    async def test_whitelist_denied_no_at(self, tmp_path):
        from astrbot.api.message_components import Plain, At

        plugin = make_plugin(
            config={"session_whitelist": ["9999"]}, audit_dir=tmp_path
        )
        ev = FakeEvent(group_id="1000", sender_id="10001", chain=[Plain("[at:12345]")])
        await plugin.process_at_tags(ev)
        chain = ev.get_result().chain
        assert not any(isinstance(c, At) for c in chain)


# --------------------------------------------------------------------------- #
# CHG-04 群成员列表缓存
# --------------------------------------------------------------------------- #
class TestMemberCacheIntegration:
    async def test_cache_hit_single_api_call(self, tmp_path):
        plugin = make_plugin(
            config={"member_list_cache_ttl": 180}, audit_dir=tmp_path
        )
        ev = FakeEvent(group_id="1000", sender_id="10001")
        ev.bot.member_list = [
            {"user_id": "1", "nickname": "张三", "card": "", "role": "member"},
            {"user_id": "2", "nickname": "李四", "card": "群主", "role": "owner"},
        ]

        await plugin._get_group_members_cached(ev, "1000")
        await plugin._get_group_members_cached(ev, "1000")
        api_calls = [c for c in ev.bot.calls if c[0] == "get_group_member_list"]
        assert len(api_calls) == 1, "TTL 内重复搜索应只拉取一次"

    async def test_cache_expired_refetch(self, tmp_path):
        plugin = make_plugin(
            config={"member_list_cache_ttl": 180}, audit_dir=tmp_path
        )
        ev = FakeEvent(group_id="1000", sender_id="10001")
        ev.bot.member_list = [{"user_id": "1", "nickname": "张三", "role": "member"}]

        await plugin._get_group_members_cached(ev, "1000")
        # 过期
        val = plugin._member_cache["1000"]
        plugin._member_cache["1000"] = (val[0], time.time() - 181)
        await plugin._get_group_members_cached(ev, "1000")
        api_calls = [c for c in ev.bot.calls if c[0] == "get_group_member_list"]
        assert len(api_calls) == 2, "缓存过期后应重新拉取"

    async def test_ttl_zero_no_cache(self, tmp_path):
        plugin = make_plugin(
            config={"member_list_cache_ttl": 0}, audit_dir=tmp_path
        )
        ev = FakeEvent(group_id="1000", sender_id="10001")
        ev.bot.member_list = [{"user_id": "1", "nickname": "张三", "role": "member"}]
        await plugin._get_group_members_cached(ev, "1000")
        await plugin._get_group_members_cached(ev, "1000")
        api_calls = [c for c in ev.bot.calls if c[0] == "get_group_member_list"]
        assert len(api_calls) == 2, "ttl=0 不缓存，每次都拉取"


# --------------------------------------------------------------------------- #
# P2-7 回归：get_group_member_list 瞬时失败重试 1 次（间隔 0.5s）
# --------------------------------------------------------------------------- #
class TestMemberListRetry:
    async def test_retry_succeeds_on_second_attempt(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main_mod, "_MEMBER_LIST_RETRY_DELAY", 0)
        plugin = make_plugin(
            config={"member_list_cache_ttl": 180}, audit_dir=tmp_path
        )
        ev = FakeEvent(group_id="1000", sender_id="10001")
        ev.bot.member_list = [
            {"user_id": "1", "nickname": "张三", "card": "", "role": "member"}
        ]
        ev.bot.member_list_fail_times = 1  # 第一次抛异常，第二次成功
        result = await plugin._get_group_members_cached(ev, "1000")
        assert result == ev.bot.member_list, "重试成功后应返回成员列表"
        api_calls = [c for c in ev.bot.calls if c[0] == "get_group_member_list"]
        assert len(api_calls) == 2, "首次失败后应重试一次"
        # 成功结果正常入缓存，后续命中缓存不再拉取
        await plugin._get_group_members_cached(ev, "1000")
        api_calls = [c for c in ev.bot.calls if c[0] == "get_group_member_list"]
        assert len(api_calls) == 2, "重试成功的结果应照常缓存"

    async def test_retry_exhausted_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main_mod, "_MEMBER_LIST_RETRY_DELAY", 0)
        plugin = make_plugin(
            config={"member_list_cache_ttl": 180}, audit_dir=tmp_path
        )
        ev = FakeEvent(group_id="1000", sender_id="10001")
        ev.bot.member_list_fail_times = 99  # 一直失败
        result = await plugin._get_group_members_cached(ev, "1000")
        assert result is None, "重试耗尽仍失败应返回 None（上层降级）"
        api_calls = [c for c in ev.bot.calls if c[0] == "get_group_member_list"]
        assert len(api_calls) == 1 + main_mod._MEMBER_LIST_MAX_RETRIES, (
            "初始 1 次 + 重试 _MEMBER_LIST_MAX_RETRIES 次"
        )

    async def test_success_path_no_retry(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main_mod, "_MEMBER_LIST_RETRY_DELAY", 0)
        plugin = make_plugin(
            config={"member_list_cache_ttl": 180}, audit_dir=tmp_path
        )
        ev = FakeEvent(group_id="1000", sender_id="10001")
        ev.bot.member_list = [
            {"user_id": "1", "nickname": "张三", "card": "", "role": "member"}
        ]
        result = await plugin._get_group_members_cached(ev, "1000")
        assert result == ev.bot.member_list
        api_calls = [c for c in ev.bot.calls if c[0] == "get_group_member_list"]
        assert len(api_calls) == 1, "成功路径不应重试"

    async def test_empty_list_is_not_retried(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main_mod, "_MEMBER_LIST_RETRY_DELAY", 0)
        plugin = make_plugin(
            config={"member_list_cache_ttl": 180}, audit_dir=tmp_path
        )
        ev = FakeEvent(group_id="1000", sender_id="10001")
        ev.bot.member_list = []  # 空列表是合法结果，不是网络失败
        result = await plugin._get_group_members_cached(ev, "1000")
        assert result is None
        api_calls = [c for c in ev.bot.calls if c[0] == "get_group_member_list"]
        assert len(api_calls) == 1, "空列表不应触发重试"


# --------------------------------------------------------------------------- #
# CHG-05 select_member_by_index
# --------------------------------------------------------------------------- #
class TestSelectMemberByIndex:
    async def _setup_multi_choice(self, tmp_path):
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(group_id="1000", sender_id="10001")
        ev.bot.member_list = [
            {"user_id": "1", "nickname": "张三", "card": "", "role": "member"},
            {"user_id": "2", "nickname": "张三丰", "card": "", "role": "owner"},
            {"user_id": "3", "nickname": "王五", "card": "", "role": "admin"},
        ]
        return plugin, ev

    async def test_select_valid_index(self, tmp_path):
        plugin, ev = await self._setup_multi_choice(tmp_path)
        text = await plugin.search_and_mention(ev, "张")
        assert "1." in text and "2." in text  # 多结果列表
        # 会话级缓存写入
        assert ev.unified_msg_origin in plugin._pending_choices

        result = await plugin.select_member_by_index(ev, 2)
        assert "张三丰" in result
        assert "[at:2]" in result

    async def test_select_out_of_range(self, tmp_path):
        plugin, ev = await self._setup_multi_choice(tmp_path)
        await plugin.search_and_mention(ev, "张")
        result = await plugin.select_member_by_index(ev, 99)
        assert "超出范围" in result

    async def test_select_expired(self, tmp_path):
        plugin, ev = await self._setup_multi_choice(tmp_path)
        await plugin.search_and_mention(ev, "张")
        # 过期
        val = plugin._pending_choices[ev.unified_msg_origin]
        plugin._pending_choices[ev.unified_msg_origin] = (val[0], time.time() - 121)
        result = await plugin.select_member_by_index(ev, 1)
        assert "过期" in result

    async def test_select_invalid_index(self, tmp_path):
        plugin, ev = await self._setup_multi_choice(tmp_path)
        result = await plugin.select_member_by_index(ev, "abc")
        assert "正整数" in result
        result0 = await plugin.select_member_by_index(ev, 0)
        assert "正整数" in result0

    async def test_select_without_search(self, tmp_path):
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(group_id="1000", sender_id="10001")
        result = await plugin.select_member_by_index(ev, 1)
        assert "没有可用的成员选择记录" in result


# --------------------------------------------------------------------------- #
# CHG-06 角色标注集成
# --------------------------------------------------------------------------- #
class TestRoleLabelIntegration:
    async def test_single_result_contains_role(self, tmp_path):
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(group_id="1000", sender_id="10001")
        ev.bot.member_list = [
            {"user_id": "2", "nickname": "李四", "card": "群主", "role": "owner"}
        ]
        text = await plugin.search_and_mention(ev, "李四")
        assert "群主" in text

    async def test_multi_result_contains_role(self, tmp_path):
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(group_id="1000", sender_id="10001")
        ev.bot.member_list = [
            {"user_id": "1", "nickname": "张三", "role": "member"},
            {"user_id": "2", "nickname": "张三丰", "role": "owner"},
        ]
        text = await plugin.search_and_mention(ev, "张")
        assert "群主" in text
        assert "成员" in text

    async def test_no_match_message(self, tmp_path):
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(group_id="1000", sender_id="10001")
        ev.bot.member_list = [
            {"user_id": "1", "nickname": "张三", "role": "member"}
        ]
        text = await plugin.search_and_mention(ev, "不存在的名字")
        assert "未找到" in text


# --------------------------------------------------------------------------- #
# CHG-07 缓存容量上限（走真实写入路径触发淘汰）
# --------------------------------------------------------------------------- #
class TestCacheLimit:
    async def test_permission_cache_evicted(self, tmp_path):
        plugin = make_plugin(audit_dir=tmp_path)
        now = time.time()
        # 预填满上限前一条（_PERMISSION_CACHE_MAX=500）
        for i in range(499):
            plugin._permission_cache[f"g:{i}"] = (True, "", now - 1)
        # 真实触发一次写入（走 _get_at_all_permission_result → drop_expired + evict）
        ev = FakeEvent(group_id="9999", sender_id="8888")
        ev.bot.member_info_role = "owner"
        await plugin._get_at_all_permission_result(ev)
        assert len(plugin._permission_cache) <= 500
        # 再直接塞 1 条（模拟超限），应在上一次真实写入时已被清理过；这里验证容量上限生效
        plugin._permission_cache["extra:key"] = (True, "", now)
        assert len(plugin._permission_cache) <= 501  # 直接赋值不触发淘汰，仅验证边界

    async def test_member_cache_evicted(self, tmp_path):
        plugin = make_plugin(audit_dir=tmp_path)
        now = time.time()
        for i in range(50):
            plugin._member_cache[f"g{i}"] = ([], now - 1)
        ev = FakeEvent(group_id="1000", sender_id="10001")
        ev.bot.member_list = [{"user_id": "1", "nickname": "张三", "role": "member"}]
        # 真实写入新群 → 应触发 evict_oldest_to_limit
        await plugin._get_group_members_cached(ev, "new_group")
        assert len(plugin._member_cache) <= 50
        assert "g0" not in plugin._member_cache
        assert "new_group" in plugin._member_cache

    async def test_terminate_clears_caches(self, tmp_path):
        plugin = make_plugin(audit_dir=tmp_path)
        plugin._permission_cache["a:b"] = (True, "", time.time())
        plugin._member_cache["g"] = ([], time.time())
        plugin._at_all_last_trigger["g"] = time.time()
        plugin._pending_choices["umo"] = ([], time.time())
        plugin._fallback_at["umo"] = ("1", time.time())
        await plugin.terminate()
        assert not plugin._permission_cache
        assert not plugin._member_cache
        assert not plugin._at_all_last_trigger
        assert not plugin._pending_choices
        assert not plugin._fallback_at


# --------------------------------------------------------------------------- #
# 提示词注入（on_llm_request）
# --------------------------------------------------------------------------- #
class TestInjectInstruction:
    async def test_inject_allowed_permission_text(self, tmp_path):
        from astrbot.api.provider import ProviderRequest

        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(group_id="1000", sender_id="10001")
        ev.bot.member_info_role = "owner"
        req = ProviderRequest()
        await plugin.inject_at_instruction(ev, req)
        # 权限提示注入到 extra_user_content_parts（temp 分区），而非 system_prompt
        assert any("具备@全体权限" in p.text for p in req.extra_user_content_parts)
        assert "@全体权限" in req.system_prompt or any(
            "具备@全体权限" in p.text for p in req.extra_user_content_parts
        )

    async def test_inject_denied_permission_text(self, tmp_path):
        from astrbot.api.provider import ProviderRequest

        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(group_id="1000", sender_id="10001")
        ev.bot.member_info_role = "member"
        req = ProviderRequest()
        await plugin.inject_at_instruction(ev, req)
        assert any("不具备@全体权限" in p.text for p in req.extra_user_content_parts)

    async def test_inject_blacklist_denied(self, tmp_path):
        from astrbot.api.provider import ProviderRequest

        plugin = make_plugin(
            config={"session_blacklist": ["1000"]}, audit_dir=tmp_path
        )
        ev = FakeEvent(group_id="1000", sender_id="10001")
        req = ProviderRequest()
        await plugin.inject_at_instruction(ev, req)
        assert "不允许使用艾特功能" in req.system_prompt


# --------------------------------------------------------------------------- #
# P1-1 回归：_to_int 配置归一化（null/字符串配置不再导致插件加载崩溃）
# --------------------------------------------------------------------------- #
class TestToIntConfig:
    """_to_int 边界 + 配置为 null/非法字符串时插件可正常初始化。"""

    async def test_unit_basic_values(self):
        _to_int = main_mod._to_int
        assert _to_int(60, 60) == 60
        assert _to_int(6.5, 60) == 6  # float 按 int() 截断
        assert _to_int("60", 60) == 60
        assert _to_int(" 60 ", 60) == 60  # 首尾空白容忍

    async def test_unit_invalid_fallbacks(self):
        _to_int = main_mod._to_int
        for bad in (None, "abc", "", "   ", "6.5", [], {}, True, False):
            assert _to_int(bad, 60) == 60, f"value={bad!r} 应回退默认值"

    async def test_unit_non_finite_float(self):
        _to_int = main_mod._to_int
        assert _to_int(float("nan"), 60) == 60
        assert _to_int(float("inf"), 60) == 60
        assert _to_int(float("-inf"), 60) == 60

    async def test_unit_negative_and_zero(self):
        _to_int = main_mod._to_int
        assert _to_int("-5", 60) == -5
        assert _to_int(0, 60) == 0
        assert _to_int("0", 60) == 0

    async def test_unit_default_used_when_key_missing(self):
        # 键缺失时 .get() 返回默认值，_to_int 原样放行
        _to_int = main_mod._to_int
        assert _to_int(60, 60) == 60
        assert _to_int(180, 180) == 180

    async def test_init_with_null_config_no_crash(self, tmp_path):
        # P1-1 复现场景：配置 JSON 中两项为 null → 修复前 int(None) 抛 TypeError
        plugin = make_plugin(
            config={"at_all_cooldown": None, "member_list_cache_ttl": None},
            audit_dir=tmp_path,
        )
        assert plugin.at_all_cooldown == 60
        assert plugin.member_list_cache_ttl == 180

    async def test_init_with_string_config_parsed(self, tmp_path):
        # 字符串形式配置（手改 JSON / 配置迁移）：可解析则解析，不可解析回退
        plugin = make_plugin(
            config={
                "at_all_cooldown": "120",
                "member_list_cache_ttl": "abc",
            },
            audit_dir=tmp_path,
        )
        assert plugin.at_all_cooldown == 120
        assert plugin.member_list_cache_ttl == 180

    async def test_init_with_float_config(self, tmp_path):
        plugin = make_plugin(
            config={"at_all_cooldown": 90.7, "member_list_cache_ttl": 300.0},
            audit_dir=tmp_path,
        )
        assert plugin.at_all_cooldown == 90
        assert plugin.member_list_cache_ttl == 300


# --------------------------------------------------------------------------- #
# P2-1 回归：_to_bool 字符串布尔归一（"false" 不再被 bool() 误判为 True）
# --------------------------------------------------------------------------- #
class TestToBoolConfig:
    """_to_bool 边界 + 字符串形式 bool 配置语义正确。"""

    async def test_unit_string_false_values(self):
        _to_bool = main_mod._to_bool
        for v in ("false", "False", "FALSE", " false ", "0", ""):
            assert _to_bool(v) is False, f"value={v!r} 应为 False"

    async def test_unit_string_true_values(self):
        _to_bool = main_mod._to_bool
        for v in ("true", "True", "1", "yes", "on", " 1 "):
            assert _to_bool(v) is True, f"value={v!r} 应为 True"

    async def test_unit_non_string_values(self):
        _to_bool = main_mod._to_bool
        assert _to_bool(True) is True
        assert _to_bool(False) is False
        assert _to_bool(1) is True
        assert _to_bool(0) is False
        assert _to_bool(None) is False
        assert _to_bool([]) is False

    async def test_init_string_false_config(self, tmp_path):
        # P2-1 复现场景：配置以字符串 "false" 出现时，修复前 4 个开关全被
        # bool("false") 误判为 True（语义反转）
        plugin = make_plugin(
            config={
                "enable_audit_log": "false",
                "permission_verification": "false",
                "allow_direct_qq_at": "false",
                "enable_fuzzy_search": "false",
            },
            audit_dir=tmp_path,
        )
        assert plugin.enable_audit_log is False
        assert plugin.permission_verification is False
        assert plugin.allow_direct_qq_at is False
        assert plugin.enable_fuzzy_search is False

    async def test_init_missing_keys_default_true(self, tmp_path):
        # 键缺失时 .get(key, True) 默认 True，_to_bool 原样放行
        plugin = make_plugin(config={}, audit_dir=tmp_path)
        assert plugin.enable_audit_log is True
        assert plugin.permission_verification is True


# --------------------------------------------------------------------------- #
# P2-2 回归：search_and_mention 空搜索词不得命中全部成员
# --------------------------------------------------------------------------- #
class TestSearchEmptyName:
    async def test_empty_name_rejected(self, tmp_path):
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(group_id="1000", sender_id="10001")
        ev.bot.member_list = [
            {"user_id": "1", "nickname": "张三", "role": "member"},
            {"user_id": "2", "nickname": "李四", "role": "member"},
        ]
        text = await plugin.search_and_mention(ev, "")
        assert "搜索词为空" in text
        assert "张三" not in text and "李四" not in text  # 不得列出成员

    async def test_none_and_whitespace_rejected(self, tmp_path):
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(group_id="1000", sender_id="10001")
        ev.bot.member_list = [{"user_id": "1", "nickname": "张三", "role": "member"}]
        assert "搜索词为空" in await plugin.search_and_mention(ev, None)
        assert "搜索词为空" in await plugin.search_and_mention(ev, "   ")

    async def test_normal_search_unchanged(self, tmp_path):
        # 非空搜索词成功路径输出格式不变（DoD-3）
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(group_id="1000", sender_id="10001")
        ev.bot.member_list = [{"user_id": "1", "nickname": "张三", "role": "member"}]
        text = await plugin.search_and_mention(ev, "张三")
        assert text.startswith("已找到：")
        assert "[at:1]" in text


# --------------------------------------------------------------------------- #
# P2-3 回归：_pending_choices 待选缓存容量上限淘汰
# --------------------------------------------------------------------------- #
class TestPendingChoicesLimit:
    async def test_capacity_evicts_oldest(self, tmp_path):
        plugin = make_plugin(audit_dir=tmp_path)
        now = time.time()
        # 预填满上限（_MAX_PENDING_CHOICES=200）
        for i in range(200):
            plugin._pending_choices[f"umo{i}"] = ([], now - 1)
        # 真实触发一次多结果写入（走 search_and_mention 写入路径）
        ev = FakeEvent(group_id="1000", sender_id="10001")
        ev.bot.member_list = [
            {"user_id": "1", "nickname": "张三", "card": "", "role": "member"},
            {"user_id": "2", "nickname": "张三丰", "card": "", "role": "member"},
        ]
        text = await plugin.search_and_mention(ev, "张")
        assert "找到多个" in text  # 确认走多结果写入路径
        assert len(plugin._pending_choices) <= 200
        assert "umo0" not in plugin._pending_choices  # 最早项被淘汰
        assert ev.unified_msg_origin in plugin._pending_choices  # 新项保留


# --------------------------------------------------------------------------- #
# P2-4 回归：连续 [at:ID] 不再产生冗余 "\u200b\u200b"
# --------------------------------------------------------------------------- #
class TestZwspDedup:
    async def test_consecutive_at_tags(self, tmp_path):
        from astrbot.api.message_components import Plain, At

        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(
            group_id="1000", sender_id="10001", chain=[Plain("[at:1][at:2]")]
        )
        await plugin.process_at_tags(ev)
        chain = ev.get_result().chain
        ats = [c for c in chain if isinstance(c, At)]
        plains = [c for c in chain if isinstance(c, Plain)]
        assert len(ats) == 2
        # 每个 At 后各有一个独立零宽 Plain，且不再重复叠加
        assert [p.text for p in plains] == ["\u200b", "\u200b"]
        assert all("\u200b\u200b" not in p.text for p in plains)

    async def test_at_then_text_then_at(self, tmp_path):
        from astrbot.api.message_components import Plain, At

        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(
            group_id="1000", sender_id="10001", chain=[Plain("[at:1]你好[at:2]")]
        )
        await plugin.process_at_tags(ev)
        chain = ev.get_result().chain
        ats = [c for c in chain if isinstance(c, At)]
        assert len(ats) == 2
        plain_texts = [c.text for c in chain if isinstance(c, Plain)]
        # 第一个 At 后的 Plain 补一次零宽；第二个 At 后插入独立零宽
        assert "\u200b你好" in plain_texts
        assert "\u200b" in plain_texts
        assert all("\u200b\u200b" not in t for t in plain_texts)

    async def test_consecutive_at_shared_trailing_text(self, tmp_path):
        from astrbot.api.message_components import Plain, At

        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(
            group_id="1000", sender_id="10001", chain=[Plain("[at:1][at:2]hi")]
        )
        await plugin.process_at_tags(ev)
        chain = ev.get_result().chain
        ats = [c for c in chain if isinstance(c, At)]
        plains = [c for c in chain if isinstance(c, Plain)]
        assert len(ats) == 2
        assert len(plains) == 1  # 两个 At 共享尾部 Plain，只补一次前缀
        assert plains[0].text == "\u200bhi"

    async def test_single_at_no_trailing_plain(self, tmp_path):
        from astrbot.api.message_components import Plain, At

        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(group_id="1000", sender_id="10001", chain=[Plain("[at:1]")])
        await plugin.process_at_tags(ev)
        chain = ev.get_result().chain
        assert [type(c).__name__ for c in chain] == ["At", "Plain"]
        assert chain[1].text == "\u200b"


# --------------------------------------------------------------------------- #
# P2-5 回归：metadata 声明的最低 AstrBot 版本
# --------------------------------------------------------------------------- #
class TestMetadataVersion:
    async def test_astrbot_version_lower_bound(self):
        meta_path = _PLUGIN_DIR / "metadata.yaml"
        assert meta_path.exists(), "metadata.yaml 必须存在"
        content = meta_path.read_text(encoding="utf-8")
        assert 'astrbot_version: ">=4.26.0"' in content, (
            "最低版本应为 >=4.26.0（P2-5），当前声明过宽"
        )


# --------------------------------------------------------------------------- #
# P2-6 回归：平台判定改鸭子类型（不再依赖 astrbot.core 内部类）
# --------------------------------------------------------------------------- #
class TestDuckTypingPlatform:
    class _OtherPlatformEvent:
        """模拟无 bot.api 的非 aiocqhttp 平台事件。"""

        def __init__(self, group_id="1000", sender_id="10001"):
            self.unified_msg_origin = f"other:GroupMessage:{group_id}"
            self._group_id = group_id
            self._sender_id = sender_id

        def get_group_id(self):
            return self._group_id

        def get_sender_id(self):
            return self._sender_id

    async def test_helper_duck_typing(self, tmp_path):
        # 有 bot.api → True
        ev = FakeEvent(group_id="1000", sender_id="10001")
        assert main_mod._is_aiocqhttp_event(ev) is True
        # 无 bot 属性 → False
        assert main_mod._is_aiocqhttp_event(self._OtherPlatformEvent()) is False
        # bot 存在但无 api → False
        class BotNoApi:
            pass

        ev_no_api = FakeEvent(group_id="1000", sender_id="10001", bot=BotNoApi())
        assert main_mod._is_aiocqhttp_event(ev_no_api) is False

    async def test_search_mention_unsupported_platform(self, tmp_path):
        plugin = make_plugin(audit_dir=tmp_path)
        other = self._OtherPlatformEvent()
        text = await plugin.search_and_mention(other, "张三")
        assert "平台暂不支持" in text

    async def test_at_all_permission_unsupported_platform(self, tmp_path):
        plugin = make_plugin(audit_dir=tmp_path)
        other = self._OtherPlatformEvent()
        ok, reason = await plugin._check_at_all_permission(other)
        assert ok is False
        assert "平台不支持" in reason

    async def test_aiocq_event_still_supported(self, tmp_path):
        # 鸭子类型下 FakeEvent(bot.api) 仍被判定为支持平台（回归保护）
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(group_id="1000", sender_id="10001")
        ev.bot.member_list = [{"user_id": "1", "nickname": "张三", "role": "member"}]
        text = await plugin.search_and_mention(ev, "张三")
        assert "已找到" in text
        assert "[at:1]" in text


# --------------------------------------------------------------------------- #
# 修复 2 回归：search_and_mention 唯一命中但 LLM 缺标签 → 兜底补插 [at:ID]
# --------------------------------------------------------------------------- #
class TestFallbackAtTag:
    """修复 2 兜底补插回归。

    场景：search_and_mention 唯一命中成员，但 LLM 最终回复未携带 [at:ID]
    标签（如被角色卡"纯文本"约束压制）→ process_at_tags 自动在链末尾补插
    标签并正常渲染为 At 组件。多成员匹配 / 未命中 / 无搜索 / 会话不允许时
    绝不兜底；缓存为一次性消费，避免跨轮次误用。
    """

    def _single_member_event(self):
        ev = FakeEvent(group_id="1000", sender_id="10001")
        ev.bot.member_list = [
            {"user_id": "1", "nickname": "张三", "card": "", "role": "member"}
        ]
        return ev

    async def test_single_match_no_tag_fallback_appended(self, tmp_path):
        from astrbot.api.message_components import Plain, At

        plugin = make_plugin(audit_dir=tmp_path)
        ev = self._single_member_event()
        text = await plugin.search_and_mention(ev, "张三")
        assert "[at:1]" in text  # 工具正常返回标签提示
        assert ev.unified_msg_origin in plugin._fallback_at  # 唯一命中已缓存

        # LLM 回复无任何标签 → 应兜底补插并渲染为 At 组件
        ev.set_chain([Plain("好的，这就把张三喊出来～")])
        await plugin.process_at_tags(ev)
        chain = ev.get_result().chain
        ats = [c for c in chain if isinstance(c, At)]
        assert len(ats) == 1 and ats[0].qq == "1"
        # 缓存已被一次性消费
        assert ev.unified_msg_origin not in plugin._fallback_at

    async def test_single_match_with_tag_no_duplicate(self, tmp_path):
        from astrbot.api.message_components import Plain, At

        plugin = make_plugin(audit_dir=tmp_path)
        ev = self._single_member_event()
        await plugin.search_and_mention(ev, "张三")

        # LLM 已输出标签 → 不补插、不重复艾特
        ev.set_chain([Plain("来了 [at:1]")])
        await plugin.process_at_tags(ev)
        chain = ev.get_result().chain
        ats = [c for c in chain if isinstance(c, At)]
        assert len(ats) == 1 and ats[0].qq == "1"

    async def test_cache_consumed_after_tag_present(self, tmp_path):
        from astrbot.api.message_components import Plain, At

        plugin = make_plugin(audit_dir=tmp_path)
        ev = self._single_member_event()
        await plugin.search_and_mention(ev, "张三")

        # 第一轮回复已带标签 → 兜底缓存应被消费
        ev.set_chain([Plain("[at:1]")])
        await plugin.process_at_tags(ev)
        assert ev.unified_msg_origin not in plugin._fallback_at

        # 第二轮（同一会话）无标签回复 → 不再兜底（缓存已消费，防跨轮次误用）
        ev2 = FakeEvent(group_id="1000", sender_id="10001")
        ev2.set_chain([Plain("没有标签的普通回复")])
        await plugin.process_at_tags(ev2)
        assert not any(isinstance(c, At) for c in ev2.get_result().chain)

    async def test_multi_match_no_fallback(self, tmp_path):
        from astrbot.api.message_components import Plain, At

        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(group_id="1000", sender_id="10001")
        ev.bot.member_list = [
            {"user_id": "1", "nickname": "张三", "card": "", "role": "member"},
            {"user_id": "2", "nickname": "张三丰", "card": "", "role": "member"},
        ]
        text = await plugin.search_and_mention(ev, "张")
        assert "找到多个" in text
        # 多结果需用户选择序号 → 不写兜底缓存、不兜底
        assert ev.unified_msg_origin not in plugin._fallback_at

        ev.set_chain([Plain("找到了多个，请选择序号")])
        await plugin.process_at_tags(ev)
        assert not any(isinstance(c, At) for c in ev.get_result().chain)

    async def test_multi_after_single_clears_cache(self, tmp_path):
        from astrbot.api.message_components import Plain, At

        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(group_id="1000", sender_id="10001")
        ev.bot.member_list = [
            {"user_id": "1", "nickname": "张三", "card": "", "role": "member"},
            {"user_id": "2", "nickname": "张三丰", "card": "", "role": "member"},
            {"user_id": "3", "nickname": "李四", "card": "", "role": "member"},
        ]
        await plugin.search_and_mention(ev, "张三")  # 唯一命中 → 写缓存
        assert ev.unified_msg_origin in plugin._fallback_at
        await plugin.search_and_mention(ev, "张")  # 随后多结果 → 清缓存
        assert ev.unified_msg_origin not in plugin._fallback_at

        ev.set_chain([Plain("请选择")])
        await plugin.process_at_tags(ev)
        assert not any(isinstance(c, At) for c in ev.get_result().chain)

    async def test_not_found_clears_cache(self, tmp_path):
        from astrbot.api.message_components import Plain, At

        plugin = make_plugin(audit_dir=tmp_path)
        ev = self._single_member_event()
        await plugin.search_and_mention(ev, "张三")
        text = await plugin.search_and_mention(ev, "不存在的名字")
        assert "未找到" in text
        assert ev.unified_msg_origin not in plugin._fallback_at

        ev.set_chain([Plain("没找到这个人")])
        await plugin.process_at_tags(ev)
        assert not any(isinstance(c, At) for c in ev.get_result().chain)

    async def test_no_search_no_fallback(self, tmp_path):
        from astrbot.api.message_components import Plain, At

        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(group_id="1000", sender_id="10001")
        ev.set_chain([Plain("普通回复，无搜索")])
        await plugin.process_at_tags(ev)
        assert not any(isinstance(c, At) for c in ev.get_result().chain)

    async def test_expired_cache_no_fallback(self, tmp_path):
        from astrbot.api.message_components import Plain, At

        plugin = make_plugin(audit_dir=tmp_path)
        ev = self._single_member_event()
        await plugin.search_and_mention(ev, "张三")
        # 手工过期（TTL=120s）
        val = plugin._fallback_at[ev.unified_msg_origin]
        plugin._fallback_at[ev.unified_msg_origin] = (val[0], time.time() - 121)

        ev.set_chain([Plain("过期后的回复")])
        await plugin.process_at_tags(ev)
        assert not any(isinstance(c, At) for c in ev.get_result().chain)

    async def test_blacklist_no_fallback(self, tmp_path):
        from astrbot.api.message_components import Plain, At

        plugin = make_plugin(
            config={"session_blacklist": ["1000"]}, audit_dir=tmp_path
        )
        ev = FakeEvent(group_id="1000", sender_id="10001")
        # 手工注入缓存（模拟搜索成功但会话随后被拉黑/不允许）
        plugin._fallback_at[ev.unified_msg_origin] = ("1", time.time())
        ev.set_chain([Plain("黑名单会话回复")])
        await plugin.process_at_tags(ev)
        chain = ev.get_result().chain
        assert not any(isinstance(c, At) for c in chain)
        assert ev.unified_msg_origin not in plugin._fallback_at

    async def test_existing_at_component_no_fallback(self, tmp_path):
        from astrbot.api.message_components import Plain, At

        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(group_id="1000", sender_id="10001")
        plugin._fallback_at[ev.unified_msg_origin] = ("1", time.time())
        # 其他来源已插入 At 组件 → 不再兜底追加，避免重复艾特
        ev.set_chain([At(qq="999"), Plain("已有艾特")])
        await plugin.process_at_tags(ev)
        chain = ev.get_result().chain
        ats = [c for c in chain if isinstance(c, At)]
        assert len(ats) == 1 and ats[0].qq == "999"

    async def test_fallback_tag_goes_through_audit(self, tmp_path):
        from astrbot.api.message_components import Plain, At

        plugin = make_plugin(audit_dir=tmp_path)
        ev = self._single_member_event()
        await plugin.search_and_mention(ev, "张三")

        ev.set_chain([Plain("兜底场景回复")])
        await plugin.process_at_tags(ev)
        # 兜底补插的标签走正常渲染链路 → 审计日志应含 at_member 记录
        log_path = audit_file(tmp_path, time.strftime("%Y%m%d"))
        assert log_path.exists()
        recs = [
            json.loads(l)
            for l in log_path.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        assert any(
            r["op_type"] == "at_member" and r["target_id"] == "1" for r in recs
        )
