"""T3 独立验证 harness：第三方插件互操作（清链直发 / 分段 / 跨组件切碎）。

作者：verifier（独立验证员）。本文件**不依赖**实现者 tests/test_integration_main.py
的测试桩与断言，自带最小 astrbot 桩、FakeEvent/FakeContext/FakeBot 与交付录制器，
可单文件独立运行：

    cd /root/AstrBot/data/plugins/astrbot_plugin_AtTool
    /root/AstrBot/.venv/bin/python -m pytest tests/test_third_party_interplay.py -q   # 修复后：全绿

同一份用例在 HEAD(v2.5.2) 基线 worktree 上复跑即为「修复前」对照：

    git -C /root/AstrBot/data/plugins/astrbot_plugin_AtTool worktree add /tmp/attool-baseline HEAD
    cp tests/test_third_party_interplay.py /tmp/attool-baseline/tests/
    cd /tmp/attool-baseline && ATTOOL_PLUGIN_ROOT=/tmp/attool-baseline \
      /root/AstrBot/.venv/bin/python -m pytest tests/test_third_party_interplay.py -q

第三方分段插件的投递语义按 astrbot_plugin_splitter 真实代码建模（只读核对）：
- main.py:504-582：非末段 `_send_proactive_segment()` 立即直发、末段回填
  `result.chain`（由框架 respond stage 调 `event.send()` 交付）；
- main.py:296-322：`_send_proactive_segment` 走 `context.send_message`，
  **绕过 event.send 包装器**（AtTool 看不见该路径）；
- main.py:470-471：`at_strategy=跟随下段` 时按链中是否含 At 组件决定重排。

不变量（spec §7.1）：
- N1：任一交付链中 Plain.text 不得匹配 ``\\[(?i:at)\\s*[:：]``；
- N2：每个可解析标签恰好产出 1 个 At。
"""

from __future__ import annotations

import os
import re
import sys
import time
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.asyncio

# --------------------------------------------------------------------------- #
# 0. 被测插件根：环境变量可指向 HEAD 基线 worktree（红/绿对照）
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PLUGIN_ROOT = Path(
    os.environ.get("ATTOOL_PLUGIN_ROOT") or _REPO_ROOT
).resolve()
_PKG_NAME = "astrbot_plugin_AtTool"

_NAKED_TAG = re.compile(r"\[(?i:at)\s*[:：]")  # N1 判定函数（spec §7.1）

# 真实用户消息的默认内容：**不含任何数字**，避免把"用户给出的 QQ 号"
# 意外喂给 T10 的可信来源判定（那是专门用例才刻意构造的输入）。
DEFAULT_USER_MESSAGE = "帮我艾特一下张三"


# --------------------------------------------------------------------------- #
# 1. 最小 astrbot 桩：仅当真实包（或实现者的桩）尚未安装时自建
# --------------------------------------------------------------------------- #
_LOGGED_WARNINGS: list = []


def _install_minimal_stub() -> None:
    """自建最小桩，保证本文件在「单文件运行」时也能加载插件主模块。"""
    if "astrbot" in sys.modules:
        return

    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    star = types.ModuleType("astrbot.api.star")
    event = types.ModuleType("astrbot.api.event")
    provider = types.ModuleType("astrbot.api.provider")
    comps = types.ModuleType("astrbot.api.message_components")
    core = types.ModuleType("astrbot.core")
    core_agent = types.ModuleType("astrbot.core.agent")
    core_agent_message = types.ModuleType("astrbot.core.agent.message")
    core_message = types.ModuleType("astrbot.core.message")
    core_message_result = types.ModuleType(
        "astrbot.core.message.message_event_result"
    )

    class _Logger:
        def _record(self, level, *args, **kwargs):
            _LOGGED_WARNINGS.append((level, " ".join(str(a) for a in args)))

        def info(self, *a, **kw):
            self._record("info", *a, **kw)

        def warning(self, *a, **kw):
            self._record("warning", *a, **kw)

        def error(self, *a, **kw):
            self._record("error", *a, **kw)

        def debug(self, *a, **kw):
            self._record("debug", *a, **kw)

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
            return Path("/tmp/attool-verifier-data") / name

    class AstrMessageEvent:
        pass

    class ProviderRequest:
        def __init__(self):
            self.system_prompt = ""
            self.extra_user_content_parts = []
            self.func_tool = None

    class Plain:
        def __init__(self, text=""):
            self.text = text

    class At:
        """与真机语义一致：astrbot At 是 pydantic 模型（qq: int | str），
        纯数字串会被强制转 int；非数字串（如 all）保留字符串。"""

        def __init__(self, qq="", name=""):
            self.qq = int(qq) if isinstance(qq, str) and qq.isdigit() else qq
            self.name = name

        def toDict(self):
            return {"type": "at", "data": {"qq": str(self.qq)}}

    class BaseMessageComponent:
        pass

    class MessageChain:
        def __init__(self, chain=None):
            self.chain = chain if chain is not None else []

    class TextPart:
        def __init__(self, text=""):
            self.text = text

        def mark_as_temp(self):
            self.is_temp = True
            return self

    class _Filter:
        @staticmethod
        def _identity(*args, **kwargs):
            def deco(fn):
                return fn

            return deco

        on_llm_request = _identity
        llm_tool = _identity
        on_decorating_result = _identity

    astrbot.api = api
    astrbot.core = core
    api.star = star
    api.event = event
    api.provider = provider
    api.message_components = comps
    api.AstrBotConfig = dict
    api.logger = _Logger()
    star.Context = Context
    star.Star = Star
    star.StarTools = StarTools
    event.AstrMessageEvent = AstrMessageEvent
    event.filter = _Filter()
    provider.ProviderRequest = ProviderRequest
    comps.Plain = Plain
    comps.At = At
    comps.BaseMessageComponent = BaseMessageComponent
    core.agent = core_agent
    core_agent.message = core_agent_message
    core_agent_message.TextPart = TextPart
    core.message = core_message
    core_message.message_event_result = core_message_result
    core_message_result.MessageChain = MessageChain

    for name, mod in {
        "astrbot": astrbot,
        "astrbot.api": api,
        "astrbot.api.star": star,
        "astrbot.api.event": event,
        "astrbot.api.provider": provider,
        "astrbot.api.message_components": comps,
        "astrbot.core": core,
        "astrbot.core.agent": core_agent,
        "astrbot.core.agent.message": core_agent_message,
        "astrbot.core.message": core_message,
        "astrbot.core.message.message_event_result": core_message_result,
    }.items():
        sys.modules[name] = mod


_install_minimal_stub()

# 以包形式加载被测插件树（相对导入 from .utils import ... 需要 __path__）
if str(_PLUGIN_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT.parent))
if _PKG_NAME not in sys.modules:
    _pkg = types.ModuleType(_PKG_NAME)
    _pkg.__path__ = [str(_PLUGIN_ROOT)]
    sys.modules[_PKG_NAME] = _pkg

import importlib  # noqa: E402

main_mod = importlib.import_module(f"{_PKG_NAME}.main")
utils_mod = importlib.import_module(f"{_PKG_NAME}.utils")

Plain = sys.modules["astrbot.api.message_components"].Plain
At = sys.modules["astrbot.api.message_components"].At


# --------------------------------------------------------------------------- #
# 2. Fake 组件与交付录制器
# --------------------------------------------------------------------------- #
class FakeResult:
    def __init__(self, chain=None):
        self.chain = list(chain) if chain else []


class FakeBot:
    def __init__(self, members=None):
        self.calls: list = []
        self.member_info_role = "owner"
        self.member_list = members if members is not None else []
        self.member_list_fail_times = 0
        self.api = self

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


class DeliveryRecorder:
    """录制每一条真实交付路径（顺序敏感）：事件路径 + context 直发路径。"""

    def __init__(self):
        self.entries: list = []  # (path, MessageChain)

    def record(self, path: str, message):
        chain = getattr(message, "chain", None)
        if chain is None and isinstance(message, str):
            chain = [Plain(message)]
        self.entries.append((path, list(chain or [])))

    # --- 断言辅助 ---
    def chains(self, path: str | None = None) -> list:
        return [c for p, c in self.entries if path is None or p == path]

    def all_components(self) -> list:
        return [comp for chain in self.chains() for comp in chain]

    def at_targets(self) -> list:
        return [str(c.qq) for c in self.all_components() if isinstance(c, At)]

    def naked_tags(self) -> list:
        return [
            comp.text
            for comp in self.all_components()
            if isinstance(comp, Plain) and _NAKED_TAG.search(comp.text)
        ]

    def text_of(self, index: int, path: str | None = None) -> str:
        chain = self.chains(path)[index]
        return "".join(
            c.text for c in chain if isinstance(c, Plain)
        ).replace("\u200b", "")


class FakeEvent:
    """最小事件对象：暴露 bot.api（过 aiocqhttp 鸭子类型判定）与交付路径。

    T11 建模修正：真机 `AstrMessageEvent.message_str` 与 `get_messages()` 恒可用
    （用户那条入站消息），而 T10 的 ID 可信来源判定依赖它们。若事件两者皆空，
    插件会走"无用户上下文则跳过来源判定"的 fail-open 分支，用例就不再表达真实
    会话语义。故这里默认构造一条真实入站消息（内容不含数字，避免意外把数字
    当成用户给出的 QQ）。
    """

    def __init__(self, chain=None, group_id="1035699087", sender_id="10001",
                 umo=None, bot=None, recorder=None,
                 user_message=DEFAULT_USER_MESSAGE, incoming=None):
        self._group_id = group_id
        self._sender_id = sender_id
        self.unified_msg_origin = umo or f"aiocqhttp:GroupMessage:{group_id}"
        self.bot = bot if bot is not None else FakeBot()
        self.message_str = user_message or ""
        self._incoming = (
            list(incoming)
            if incoming is not None
            else ([Plain(self.message_str)] if self.message_str else [])
        )
        self._result = FakeResult(chain)
        self.delivery = recorder if recorder is not None else DeliveryRecorder()

    def get_messages(self):
        """入站消息链（真机同签名；T10 判定用户是否给出过 QQ 号）。"""
        return list(self._incoming)

    def get_group_id(self):
        return self._group_id

    def get_sender_id(self):
        return self._sender_id

    def get_result(self):
        return self._result

    def set_chain(self, chain):
        self._result = FakeResult(chain)
        return self

    # --- 真实交付路径（AtTool 包装 / 框架 respond stage 最终调用）---
    async def send(self, message):
        self.delivery.record("event.send", message)

    async def send_streaming(self, generator, use_fallback=False):
        async for item in generator:
            self.delivery.record("event.send_streaming", item)


class FakeContext:
    """splitter 真实直发路径：context.send_message（AtTool 不可见）。"""

    def __init__(self, recorder=None):
        self._cfg = {"admins_id": ["astrbot"]}
        self.sent: list = []
        self.delivery = recorder if recorder is not None else DeliveryRecorder()

    def get_config(self):
        return self._cfg

    async def send_message(self, umo, message):
        self.sent.append((umo, message))
        self.delivery.record("context.send_message", message)


# --------------------------------------------------------------------------- #
# 3. 驱动辅助：真实插件的公开入口（on_llm_request → on_decorating_result）
# --------------------------------------------------------------------------- #
def make_plugin(recorder=None):
    """构造真实插件实例；用 FakeContext 录制 splitter 直发路径。

    返回的 plugin.context.delivery 与该 recorder 是同一对象，测试用
    ``new_plugin_and_event()`` 可让事件路径与 context 直发路径共用一个
    录制器（否则两条路径分别录到不同对象，断言会看不到直发段）。
    """
    context = FakeContext(recorder)
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
    plugin = main_mod.LLMAtToolPlugin(context=context, config=cfg)
    plugin._audit_dir = Path("/tmp/attool-verifier-audit")
    return plugin


def mark_tool_confirmed(plugin, event, *ids, fallback: bool = False):
    """模拟"工具确认过这些 ID"——与 search_and_mention/select_member_by_index
    的真实写入同形：两个缓存都按 **(umo, sender_id)** 分槽（T14/A3/M1）——
    `_known_at_ids[(umo, sender)] = ({ids}, now)`；`fallback=True` 时同时写
    兜底补插缓存 `_fallback_at[(umo, sender)] = (user_id, now)`。

    T10 之后，"工具确认"是 ID 能被渲染成 At 的两条可信来源之一，因此凡是
    断言"[at:ID] 应渲染"的用例，都必须显式表达这个前提。
    """
    umo = event.unified_msg_origin
    sender = str(event.get_sender_id() or "")
    now = time.time()
    key = (umo, sender)
    cached = plugin._known_at_ids.get(key)
    known = set(cached[0]) if cached else set()
    known |= {str(i) for i in ids}
    plugin._known_at_ids[key] = (known, now)
    if fallback and ids:
        plugin._fallback_at[key] = (str(ids[0]), now)


def fallback_entry(plugin, event):
    """读取该事件发起者名下的兜底补插条目（T14/A3 起按 (umo, sender) 分槽）。"""
    return plugin._fallback_at.get(
        (event.unified_msg_origin, str(event.get_sender_id() or ""))
    )


def _auto_trusted_ids(chain) -> set:
    """从初始链里收集所有标签载荷，作为"工具已确认"的模拟前提。

    仅用于形态/分段类用例（它们考察的是标签**形态与投递路径**，把
    "ID 来源"这一维度固定为可信）；ID 来源本身的用例在
    TestTrustedIdSourceVerifier 里显式构造不可信输入。
    """
    ids = set()
    for comp in chain or []:
        if isinstance(comp, Plain) and comp.text:
            for kind, value in utils_mod.split_text_by_at_tags(comp.text):
                if kind == "at" and value != "all":
                    ids.add(value)
    return ids


def new_plugin_and_event(chain=None, *, trusted_ids="auto", **event_kwargs):
    """构造 (plugin, event)，两者共享同一个交付录制器。

    AtTool 的 send 包装器与第三方 splitter 的 context.send_message 是两条
    独立投递路径，必须录到同一个 DeliveryRecorder 里才能做"全量交付"断言。

    Args:
        chain: 初始结果链。
        trusted_ids: "auto"（默认）= 把初始链里的标签载荷视为"工具已确认"；
            显式传入集合（含空集）则按传入值登记，用于验证"未确认即不渲染"。
        **event_kwargs: 透传给 FakeEvent（user_message / incoming / sender_id 等）。
    """
    recorder = DeliveryRecorder()
    plugin = make_plugin(recorder)
    event = FakeEvent(chain=chain, recorder=recorder, **event_kwargs)
    if trusted_ids == "auto":
        trusted_ids = _auto_trusted_ids(chain)
    if trusted_ids:
        mark_tool_confirmed(plugin, event, *trusted_ids)
    return plugin, event


class FakeToolSet:
    def __init__(self, names):
        self._names = list(names)

    def names(self):
        return list(self._names)


def make_request(tool_names=("search_and_mention", "select_member_by_index")):
    req = sys.modules["astrbot.api.provider"].ProviderRequest()
    req.func_tool = FakeToolSet(tool_names) if tool_names else None
    return req


async def run_llm_request(plugin, event, tool_names=("search_and_mention",)):
    """on_llm_request 阶段：安装 send 包装器 + 注入提示词。"""
    req = make_request(tool_names)
    await plugin.inject_at_instruction(event, req)
    return req


async def run_main_hook(plugin, event):
    """AtTool 的 on_decorating_result(priority=-1000) 主钩子。"""
    await plugin.process_at_tags(event)


async def framework_deliver_final(event):
    """模型回复阶段的框架交付：respond stage 调 event.send(result.chain)。"""
    result = event.get_result()
    if result and result.chain:
        await event.send(sys.modules["astrbot.core.message.message_event_result"].MessageChain(chain=result.chain))


def segment_chain(chain, marker="\n\n", follows_next=True):
    """按分段插件语义把已渲染链切成若干段（model of splitter main.py:470-582）。

    Args:
        chain: 已渲染的消息链。
        marker: 分段边界（真实插件用可配置正则，此处固定段落空行）。
        follows_next: True 时非 Plain 组件（At）采用 at_strategy=跟随下段，
            即移动到下一个文本段；False 时原位保留（用于位置可判定的断言）。

    Returns:
        段链列表。
    """
    segments: list = [[]]

    def push_to_next(comp):
        if segments[-1] and follows_next:
            segments.append([comp])
        else:
            segments[-1].append(comp)

    for comp in chain:
        if not isinstance(comp, Plain):
            push_to_next(comp)
            continue
        parts = comp.text.split(marker)
        for i, part in enumerate(parts):
            if i > 0:
                segments.append([])
            if part:
                segments[-1].append(Plain(part))
    return [seg for seg in segments if seg]


async def splitter_real_path(plugin, event, follows_next=False):
    """分段插件真实投递路径：非末段 context.send_message 直发、末段回填 result.chain。

    对应 astrbot_plugin_splitter/main.py:504-582（`_send_proactive_segment`
    走 context.send_message，末段 `result.chain.clear(); extend(last_seg)`）。
    """
    result = event.get_result()
    segments = segment_chain(list(result.chain), follows_next=follows_next)
    if not segments:
        return []
    context = plugin.context
    for seg in segments[:-1]:
        text = "".join(c.text for c in seg if isinstance(c, Plain))
        if not text.strip(" \t\r\n\u200b") and not any(
            not isinstance(c, Plain) for c in seg
        ):
            continue
        mc = sys.modules["astrbot.core.message.message_event_result"].MessageChain()
        mc.chain = list(seg)
        await context.send_message(event.unified_msg_origin, mc)
    result.chain.clear()
    result.chain.extend(segments[-1])
    return segments


async def third_party_clear_and_send_each(plugin, event):
    """第三方插件（priority 0，早于 AtTool 的 -1000）清链后逐段 event.send。

    这是反馈 3 上报的"09-07 实测"场景：清链直发发生时，AtTool 主钩子尚未
    执行，兜底缓存仍在——修复前 send 路径会消费缓存并补插到第一段。
    """
    result = event.get_result()
    segments = segment_chain(list(result.chain))
    result.chain.clear()
    mc_cls = sys.modules["astrbot.core.message.message_event_result"].MessageChain
    for seg in segments:
        await event.send(mc_cls(chain=list(seg)))
    return segments


def chain_components(*items):
    """按 text/at 速记构造链：[("t", "文本"), ("at", "123")]。"""
    out = []
    for kind, value in items:
        out.append(Plain(value) if kind == "t" else At(qq=value))
    return out


# --------------------------------------------------------------------------- #
# 4. 反馈 3：分段互操作 harness（(i)(ii)(iii)(iv) 四条路径）
# --------------------------------------------------------------------------- #
class TestSplitterInterop:
    """艾特次数 / 位置 / 裸标签 / 权限冷却四项实测断言。"""

    async def test_i_clear_chain_then_send_each_segment(self):
        """(i) 清链后逐段 event.send：艾特必须恰好 1 次、跟着标签所在段。"""
        plugin, event = new_plugin_and_event(chain=[Plain("第一段闲聊\n\n第二段内容\n\n第三段 [at:888]")])
        await run_llm_request(plugin, event)
        # 模拟工具唯一命中 777：修复前 send 路径会用该缓存补插到第一段
        mark_tool_confirmed(plugin, event, "10001", fallback=True)

        await third_party_clear_and_send_each(plugin, event)
        await run_main_hook(plugin, event)  # 主钩子最后执行（链已空）

        assert event.delivery.at_targets() == ["888"], (
            f"艾特应恰为 888 一次，实测 {event.delivery.at_targets()}"
            f"（交付段：{[[c.text for c in ch if isinstance(c, Plain)] for ch in event.delivery.chains()]}）"
        )
        assert event.delivery.naked_tags() == [], (
            f"出现裸标签：{event.delivery.naked_tags()}"
        )
        # 位置：At 落在带标签的第三段（而非被补插到第一段）
        chains = event.delivery.chains()
        assert any(isinstance(c, At) for c in chains[2]), (
            f"At 应跟随标签所在段（第 3 段），实测 "
            f"{[[getattr(c, 'text', getattr(c, 'qq', None)) for c in ch] for ch in chains]}"
        )
        assert not any(isinstance(c, At) for c in chains[0]), (
            "第一段不得出现补插的 At"
        )

    async def test_i_fallback_cache_not_consumed_by_send_path(self):
        """send 路径不得消费兜底缓存（P0-2）：缓存只归主钩子。"""
        plugin, event = new_plugin_and_event(chain=[Plain("无标签的一段\n\n第二段")])
        await run_llm_request(plugin, event)
        mark_tool_confirmed(plugin, event, "10001", fallback=True)

        await third_party_clear_and_send_each(plugin, event)
        assert event.delivery.at_targets() == []
        assert fallback_entry(plugin, event) is not None, (
            "send 路径不应消费兜底缓存（否则主钩子无标签可补）"
        )
        await run_main_hook(plugin, event)
        assert fallback_entry(plugin, event) is None

    async def test_ii_splitter_real_path_context_send_message(self):
        """(ii) splitter 真实路径：非末段 context.send_message、末段 result.chain。"""
        plugin, event = new_plugin_and_event(chain=[Plain("甲段\n\n乙段\n\n丙段 [at:888]")])
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)  # AtTool(-1000) 早于 splitter(-1e17)
        await splitter_real_path(plugin, event)
        await framework_deliver_final(event)  # 末段由框架 event.send 交付

        delivered = event.delivery
        assert delivered.at_targets() == ["888"], (
            f"艾特应恰为 888 一次，实测 {delivered.at_targets()}"
        )
        assert delivered.naked_tags() == [], f"出现裸标签：{delivered.naked_tags()}"
        paths = [p for p, _ in delivered.entries]
        assert "context.send_message" in paths and "event.send" in paths, (
            f"两条投递路径都应被覆盖，实测 {paths}"
        )
        # 位置：At 在末段（走 event.send 的那条），且该段文本仍是丙段
        final_chain = delivered.chains("event.send")[-1]
        assert any(str(c.qq) == "888" for c in final_chain if isinstance(c, At)), (
            "末段应携带 At(888)"
        )
        assert "丙段" in "".join(
            c.text for c in final_chain if isinstance(c, Plain)
        )

    async def test_ii_no_at_leaks_into_non_final_segments(self):
        plugin, event = new_plugin_and_event(chain=[Plain("前段\n\n中段\n\n尾段 [at:888]")])
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)
        await splitter_real_path(plugin, event)

        non_final = event.delivery.chains("context.send_message")
        assert non_final, "应至少有一个非末段走 context.send_message"
        for chain in non_final:
            assert not any(isinstance(c, At) for c in chain), (
                f"非末段不得被塞入 At：{[getattr(c, 'qq', None) for c in chain]}"
            )
            assert not any(
                isinstance(c, Plain) and _NAKED_TAG.search(c.text) for c in chain
            )

    async def test_iii_tag_split_across_adjacent_plains(self):
        """(iii) 原始标签被切到相邻两个 Plain：必须拼回并渲染为 At。"""
        plugin, event = new_plugin_and_event(
            chain=[Plain("你好 [at:12"), Plain("345] 收工")],
            trusted_ids={"12345"},
        )
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)

        chain = event.get_result().chain
        assert [str(c.qq) for c in chain if isinstance(c, At)] == ["12345"], (
            f"相邻 Plain 切碎的标签应拼回：{[getattr(c, 'text', getattr(c, 'qq', None)) for c in chain]}"
        )
        assert not any(
            isinstance(c, Plain) and _NAKED_TAG.search(c.text) for c in chain
        )

    async def test_iii_tag_split_across_plains_with_middle_component(self):
        """(iii) 变体：标签被 At 组件隔开（跨组件切碎）。"""
        plugin, event = new_plugin_and_event(
            chain=[Plain("[at:12"), At(qq="999"), Plain("345]")],
            trusted_ids={"12345", "999"},
        )
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)

        targets = [str(c.qq) for c in event.get_result().chain if isinstance(c, At)]
        assert targets == ["12345", "999"], f"应原位恢复并保序，实测 {targets}"
        assert not any(
            isinstance(c, Plain) and _NAKED_TAG.search(c.text)
            for c in event.get_result().chain
        )

    async def test_iv_tag_in_only_one_of_many_segments(self):
        """(iv) 同事件多段、仅一处含标签：全量交付后艾特恰好一次。"""
        plugin, event = new_plugin_and_event(chain=[Plain("其一\n\n其二\n\n其三 [at:888]\n\n其四")])
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)
        await splitter_real_path(plugin, event)
        await framework_deliver_final(event)

        assert event.delivery.at_targets() == ["888"], (
            f"多段场景艾特应恰好一次，实测 {event.delivery.at_targets()}"
        )
        assert event.delivery.naked_tags() == []

    async def test_iv_all_segments_carry_own_tag(self):
        """(iv) 变体：每段各带自己的标签 → 各自渲染，不互相剥离。"""
        plugin, event = new_plugin_and_event(chain=[Plain("甲 [at:111]\n\n乙 [at:222]")])
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)
        await third_party_clear_and_send_each(plugin, event)

        assert event.delivery.at_targets() == ["111", "222"], (
            f"每段自己的标签都要渲染，实测 {event.delivery.at_targets()}"
        )
        assert event.delivery.naked_tags() == []

    async def test_send_path_renders_tag_when_main_hook_missed_it(self):
        """第三方在 AtTool 之前清链：主钩子看不到标签，send 路径必须补渲染。"""
        plugin, event = new_plugin_and_event(chain=[Plain("先说一句\n\n然后 [at:777]")])
        await run_llm_request(plugin, event)
        # 第三方（priority 0）抢先接管并清链
        await third_party_clear_and_send_each(plugin, event)
        await run_main_hook(plugin, event)  # 链已空

        assert event.delivery.at_targets() == ["777"], (
            f"send 路径应渲染链内已有标签，实测 {event.delivery.at_targets()}"
        )
        assert event.delivery.naked_tags() == []


# --------------------------------------------------------------------------- #
# 5. 裸标签不变量：spec §7.2 形态矩阵（独立参数化，逐条交付级实测）
# --------------------------------------------------------------------------- #
FORM_MATRIX = [
    # (编号, 输入, 期望 At 目标列表, 期望交付文本, 说明)
    (1, "[at:123]", ["123"], "", "基线形态"),
    (2, "[At:123]", ["123"], "", "大写 A（宿主序列化 / 反馈 2 主形态）"),
    (3, "[AT:123]", ["123"], "", "全大写"),
    (4, "[at: 123]", ["123"], "", "冒号后空格"),
    (5, "[at：123]", ["123"], "", "全角冒号"),
    (6, "[At : 123]", ["123"], "", "大小写 + 两侧空格"),
    (7, "[at:all]", ["all"], "", "权限/冷却放行 @全体"),
    (8, "[at:柴郡]", [], "@柴郡", "非数字载荷降级"),
    (9, "[at:ID]", [], "@ID", "占位符照抄降级"),
    (10, "[at:１２３]", [], "@１２３", "全角数字不得产生 At"),
    (11, "[at:] 尾巴", [], "尾巴", "空载荷删除语法"),
    (12, "[at:]", [], "", "空载荷删除语法（独立出现）"),
    (13, "[at: 123", [], "123", "未闭合：删语法留正文"),
    (14, "[at 123] [at] [avatar:1]", [], "[at 123] [at] [avatar:1]", "非标签原样"),
    (15, "[at:\u200b123]", ["123"], "", "标签内零宽"),
]


class TestNakedTagInvariant:
    @pytest.mark.parametrize(
        "idx,src,expect_at,expect_text,note",
        FORM_MATRIX,
        ids=[f"{f[0]}-{f[4]}" for f in FORM_MATRIX],
    )
    async def test_form_matrix_delivery(self, idx, src, expect_at, expect_text, note):
        plugin, event = new_plugin_and_event(chain=[Plain(src)])
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)
        await framework_deliver_final(event)

        assert event.delivery.at_targets() == expect_at, (
            f"形态 {idx}（{note}）At 目标不符：{event.delivery.at_targets()}"
        )
        # 交付文本精确相等；正文被完全删除时框架不投递任何消息（链为空）→ ""
        delivered_chains = event.delivery.chains()
        delivered_text = event.delivery.text_of(-1) if delivered_chains else ""
        assert delivered_text == expect_text, (
            f"形态 {idx}（{note}）交付文本应为 {expect_text!r}，实测 {delivered_text!r}"
        )
        assert event.delivery.naked_tags() == [], (
            f"形态 {idx}（{note}）裸标签穿链：{event.delivery.naked_tags()}"
        )

    async def test_form_16_streaming_single_chunk_renders(self):
        """形态 16（spec 第 17 行）：单 chunk 内完整标签应渲染。"""
        plugin, event = new_plugin_and_event(chain=[], trusted_ids={"123"})
        await run_llm_request(plugin, event)

        mc_cls = sys.modules["astrbot.core.message.message_event_result"].MessageChain

        async def gen():
            yield mc_cls(chain=[Plain("流式一段 [At:123]")])

        await event.send_streaming(gen(), False)
        assert event.delivery.at_targets() == ["123"]
        assert event.delivery.naked_tags() == []

    async def test_form_17_streaming_split_chunk_keeps_text(self):
        """形态 18（spec 第 18 行）：跨 chunk 切碎属已知限制，只验证"不误删正文"。"""
        plugin, event = new_plugin_and_event(chain=[])
        await run_llm_request(plugin, event)

        mc_cls = sys.modules["astrbot.core.message.message_event_result"].MessageChain

        async def gen():
            yield mc_cls(chain=[Plain("前缀 [at:12")])
            yield mc_cls(chain=[Plain("3] 后缀")])

        await event.send_streaming(gen(), False)
        joined = "".join(event.delivery.text_of(i) for i in range(len(event.delivery.chains())))
        assert "前缀" in joined and "3" in joined and "后缀" in joined, (
            f"流式跨 chunk 不得误删用户正文，实测 {joined!r}"
        )
        # 已知限制：跨 chunk 不渲染（spec §6.3 非目标，aiocqhttp 无流式）
        assert event.delivery.at_targets() == []

    async def test_two_polluted_tags_both_recovered(self):
        """P1-1：一条链中两处零宽污染标签都要恢复（旧实现每链只恢复 1 个）。"""
        plugin, event = new_plugin_and_event(chain=[Plain("[at:11\u200b1] 和 [at:22\u200b2]")])
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)
        await framework_deliver_final(event)

        assert event.delivery.at_targets() == ["111", "222"], (
            f"两处污染都应恢复，实测 {event.delivery.at_targets()}"
        )
        assert event.delivery.naked_tags() == []

    async def test_nested_tag_never_leaks(self):
        """对抗输入：嵌套标签语法不得穿链。"""
        plugin, event = new_plugin_and_event(chain=[Plain("[at:[at:123]]")])
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)
        await framework_deliver_final(event)
        assert event.delivery.naked_tags() == [], (
            f"嵌套标签穿链：{event.delivery.naked_tags()}"
        )


# --------------------------------------------------------------------------- #
# 5b. 对抗 fuzz：不变量 N1 的"永不穿链"性质（输入不预设期望，只验不变量）
# --------------------------------------------------------------------------- #
HOSTILE_INPUTS = [
    "[at:123]", "[At:123]", "[AT:123]", "[aT:123]", "[at: 123]", "[at：123]",
    "[At : 123]", "[at\u200b:123]", "[at:\u200b123]", "[at:12\u200b3]",
    "[at:all]", "[At:All]", "[AT:ALL]", "[at:all ]", "[at: all]",
    "[at:柴郡]", "[at:ID]", "[at:１２３]", "[at:]", "[at: ]", "[at:\u200b]",
    "[at 123]", "[at]", "[avatar:1]", "[at:123", "[At：123", "[at:",
    "[at:\n123]", "[at:123\n]", "[at:+123]", "[at:01]", "[at:1.5]",
    "[at:1e5]", "[at:99999999999999999999999999]", "[at:-1]",
    "[Ａt:123]", "［at:123］", "[[at:123]", "[at:123]]", "[at:123][at:456]",
    "[at:[at:123]]", "[at:[at:[at:1]]]", "[at:at:1]", "[at::123]",
    "text[at:1]text", "[at:1] [At:2] [at:all]", "[at:12" + "x" * 100 + "3]",
    "[at:٣]", "[at:１]", "\\[at:123\\]", "[@123]", "[at=123]", "[at: ",
    "[A" * 20 + "t:1]",
]


class TestAdversarialFuzz:
    @pytest.mark.parametrize(
        "src", HOSTILE_INPUTS, ids=[f"h{i}" for i in range(len(HOSTILE_INPUTS))]
    )
    async def test_n1_never_leaks_on_hostile_input(self, src):
        """N1：任意输入经「主钩子 + 框架交付」后不得出现裸标签。"""
        plugin, event = new_plugin_and_event(chain=[Plain(src)])
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)
        await framework_deliver_final(event)

        assert event.delivery.naked_tags() == [], (
            f"输入 {src!r} 裸标签穿链：{event.delivery.naked_tags()}"
        )
        bad_at = [
            a
            for a in event.delivery.at_targets()
            if not (a == "all" or (a.isascii() and a.isdigit()))
        ]
        assert bad_at == [], f"输入 {src!r} 产生了非法 At 载荷：{bad_at}"

    @pytest.mark.parametrize(
        "src", HOSTILE_INPUTS, ids=[f"s{i}" for i in range(len(HOSTILE_INPUTS))]
    )
    async def test_n1_never_leaks_through_send_path(self, src):
        """N1：同一批输入经「第三方清链逐段 send」路径也不得穿链。"""
        plugin, event = new_plugin_and_event(chain=[Plain(src)])
        await run_llm_request(plugin, event)
        await third_party_clear_and_send_each(plugin, event)
        await run_main_hook(plugin, event)

        assert event.delivery.naked_tags() == [], (
            f"输入 {src!r} 经 send 路径裸标签穿链：{event.delivery.naked_tags()}"
        )

    async def test_single_component_long_payload_degrades_not_merged(self):
        """单组件长载荷（标签在本组件内已闭合）走降级路径。

        F2（r2）更正：原用例名 `test_split_window_overflow_does_not_merge`
        会让人以为它覆盖了 `_SPLIT_AT_WINDOW` 边界，实际输入是**单个 Plain**
        且 `]` 在本组件内已闭合 —— `_try_merge_split_at_tag` 在"本组件内已
        闭合"处直接 continue，跨组件窗口分支从未执行。真实语义是不可解析
        载荷降级为纯文本；跨组件窗口覆盖见
        `test_cross_component_over_window_payload_does_not_merge` 与
        `test_split_window_guard_bounds_cross_component_merge`。
        """
        plugin, event = new_plugin_and_event(
            chain=[Plain("[at:12" + "x" * 100 + "3]")]
        )
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)
        await framework_deliver_final(event)

        assert event.delivery.at_targets() == [], "超窗内容不得拼成 At"
        assert event.delivery.naked_tags() == []
        assert "x" * 100 in event.delivery.text_of(-1), "正文必须保留"

    async def test_split_window_within_limit_merges(self):
        """窗口内**真正的跨组件**切碎（前缀 + 中间 At + 闭合括号）应拼回。

        F2（r2）更正：原输入是两个相邻 Plain，实际由步骤①"相邻 Plain 合并"
        修复，走不到窗口分支；此处改为中间夹 At 组件，才真正经过
        `_try_merge_split_at_tag` 的累积路径（载荷 3 位，远小于窗口 64）。
        """
        plugin, event = new_plugin_and_event(
            chain=[Plain("[at:12"), At(qq="999"), Plain("3]")],
            trusted_ids={"123"},  # 拼出的成员 ID：本用例考察窗口机制，来源维度固定为可信
        )
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)

        targets = [
            str(c.qq) for c in event.get_result().chain if isinstance(c, At)
        ]
        assert targets == ["123", "999"], (
            f"窗口内跨组件标签应拼回且中间组件保序，实测 {targets}"
        )

    async def test_cross_component_over_window_payload_does_not_merge(self):
        """F2（r2）：真正的跨组件超窗输入不得拼成 At。

        链 = Plain("[at:1" + "2"*100) + At(999) + Plain("3]")：拼合候选载荷
        103 位，超出 `_SPLIT_AT_WINDOW=64`（也超出载荷长度上限 1..12）→ 只
        保留中间组件的 At(999)，正文按原样保留、无裸标签。
        """
        plugin, event = new_plugin_and_event(
            chain=[Plain("[at:1" + "2" * 100), At(qq="999"), Plain("3]")]
        )
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)
        await framework_deliver_final(event)

        assert event.delivery.at_targets() == ["999"], (
            f"超窗载荷不得拼成 At，实测 {event.delivery.at_targets()}"
        )
        assert event.delivery.naked_tags() == []

    async def test_split_window_guard_bounds_cross_component_merge(
        self, monkeypatch
    ):
        """F2（r2）：`_SPLIT_AT_WINDOW` 守卫在拼合路径上可达（隔离验证）。

        同一输入（Plain("[at:12") + At(999) + Plain("3]")）：
        - 默认窗口 64 → 拼回 At(123)；
        - 窗口压到 1 → 守卫阻断拼合，只剩 At(999)，正文保留、无裸标签。
        证明该分支不是"删除后仍全绿"的死代码（F2 指出的零覆盖）。
        """
        import astrbot_plugin_AtTool.main as main_mod

        def build_chain():
            return [Plain("[at:12"), At(qq="999"), Plain("3]")]

        plugin, event = new_plugin_and_event(
            chain=build_chain(), trusted_ids={"123"}
        )
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)
        assert [
            str(c.qq) for c in event.get_result().chain if isinstance(c, At)
        ] == ["123", "999"]

        monkeypatch.setattr(main_mod, "_SPLIT_AT_WINDOW", 1)
        plugin2, event2 = new_plugin_and_event(
            chain=build_chain(), trusted_ids={"123"}
        )
        await run_llm_request(plugin2, event2)
        await run_main_hook(plugin2, event2)
        await framework_deliver_final(event2)
        assert event2.delivery.at_targets() == ["999"]
        assert event2.delivery.naked_tags() == []

    async def test_render_is_robust_to_non_str_plain(self):
        """反例：Plain.text 为 None 时不得抛异常。"""
        plugin, event = new_plugin_and_event(chain=[])
        event.get_result().chain.append(Plain(None))
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)  # 不抛异常即通过


# --------------------------------------------------------------------------- #
# 6. 权限 / 冷却不得被绕过（含第三方分段路径）
# --------------------------------------------------------------------------- #
class TestPermissionNotBypassed:
    async def test_blacklist_degrades_case_variant_and_never_leaks(self):
        plugin, event = new_plugin_and_event(chain=[Plain("你好 [At:123] 和 [at:all]")])
        plugin.session_blacklist = [event.unified_msg_origin]
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)
        await splitter_real_path(plugin, event)
        await framework_deliver_final(event)

        assert event.delivery.at_targets() == [], "黑名单会话不得产生任何 At"
        assert event.delivery.naked_tags() == [], (
            f"黑名单会话仍泄漏裸标签：{event.delivery.naked_tags()}"
        )
        assert "@全体成员" in event.delivery.text_of(-1)

    async def test_blacklist_blocks_send_path_injection(self):
        """send 路径不得因黑名单外或内而绕过会话准入（含兜底缓存）。"""
        plugin, event = new_plugin_and_event(chain=[Plain("第一段\n\n第二段 [at:123]")])
        plugin.session_blacklist = [event.unified_msg_origin]
        await run_llm_request(plugin, event)
        mark_tool_confirmed(plugin, event, "10001", fallback=True)
        await third_party_clear_and_send_each(plugin, event)

        assert event.delivery.at_targets() == []
        assert event.delivery.naked_tags() == []

    async def test_at_all_cooldown_second_time_degrades(self):
        plugin, event = new_plugin_and_event(chain=[Plain("[at:all]")])
        event.bot.member_info_role = "owner"
        await run_llm_request(plugin, event)

        await run_main_hook(plugin, event)
        first = [
            str(c.qq) for c in event.get_result().chain if isinstance(c, At)
        ]
        event.set_chain([Plain("[at:all]")])
        await run_main_hook(plugin, event)
        second_text = "".join(
            c.text for c in event.get_result().chain if isinstance(c, Plain)
        )

        assert first == ["all"], f"首次应放行 @全体，实测 {first}"
        assert "@全体成员" in second_text, (
            f"冷却内应降级为纯文本 @全体成员，实测 {second_text!r}"
        )
        assert not any(
            isinstance(c, At) for c in event.get_result().chain
        ), "冷却内不得再产生 At"

    async def test_at_all_permission_denied_for_member(self):
        plugin = make_plugin()
        plugin.permission_verification = True
        event = FakeEvent(chain=[Plain("[at:all]")])
        event.bot.member_info_role = "member"
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)
        await framework_deliver_final(event)

        assert event.delivery.at_targets() == [], "普通成员不得 @全体"
        assert "@全体成员" in event.delivery.text_of(-1)


# --------------------------------------------------------------------------- #
# 6b. 流式（send_streaming）与延迟渲染路径
# --------------------------------------------------------------------------- #
class TestStreamingAndDelayedRender:
    async def test_streaming_chunk_without_tag_never_gets_fallback(self):
        """P0-2：流式段无标签时不得补插（基线曾经补插）。"""
        plugin, event = new_plugin_and_event(chain=[])
        await run_llm_request(plugin, event)
        mark_tool_confirmed(plugin, event, "10001", fallback=True)

        mc_cls = sys.modules["astrbot.core.message.message_event_result"].MessageChain

        async def gen():
            yield mc_cls(chain=[Plain("第一块没有标签")])
            yield mc_cls(chain=[Plain("第二块也没有")])

        await event.send_streaming(gen(), False)
        assert event.delivery.at_targets() == [], (
            f"流式路径不得补插兜底艾特，实测 {event.delivery.at_targets()}"
        )
        assert fallback_entry(plugin, event) is not None, (
            "流式路径不得消费兜底缓存"
        )

    async def test_streaming_per_chunk_renders_own_tag(self):
        plugin, event = new_plugin_and_event(chain=[], trusted_ids={"111", "222"})
        await run_llm_request(plugin, event)

        mc_cls = sys.modules["astrbot.core.message.message_event_result"].MessageChain

        async def gen():
            yield mc_cls(chain=[Plain("第一块 [at:111]")])
            yield mc_cls(chain=[Plain("第二块 [At:222]")])

        await event.send_streaming(gen(), False)
        assert event.delivery.at_targets() == ["111", "222"], (
            f"逐块标签应各自渲染，实测 {event.delivery.at_targets()}"
        )
        assert event.delivery.naked_tags() == []

    async def test_literal_backslash_n_normalized_on_send(self):
        """\\n 字面量兜底（分组插件历史问题）：发送前转真实换行。"""
        plugin, event = new_plugin_and_event(chain=[])
        await run_llm_request(plugin, event)

        mc_cls = sys.modules["astrbot.core.message.message_event_result"].MessageChain
        await event.send(mc_cls(chain=[Plain("第一行\\n第二行")]))
        assert "\n" in event.delivery.text_of(-1), (
            f"字面 \\\\n 应转真实换行，实测 {event.delivery.text_of(-1)!r}"
        )

    async def test_send_path_does_not_mutate_caller_component(self):
        """P1-3 copy-on-write：send 渲染不得原地改写调用方 Plain。"""
        plugin, event = new_plugin_and_event(chain=[], trusted_ids={"123"})
        await run_llm_request(plugin, event)

        mc_cls = sys.modules["astrbot.core.message.message_event_result"].MessageChain
        original = Plain("原文 [At:123] 尾部")
        await event.send(mc_cls(chain=[original]))
        assert original.text == "原文 [At:123] 尾部", (
            f"调用方组件被原地改写：{original.text!r}"
        )


# --------------------------------------------------------------------------- #
# 7. 反馈 2/4：边界与反例
# --------------------------------------------------------------------------- #
class TestBoundaries:
    async def test_empty_search_keyword_rejected(self):
        plugin, event = new_plugin_and_event(chain=[])
        event.bot.member_list = [{"user_id": "1", "nickname": "张三", "card": ""}]
        out = await plugin.search_and_mention(event, name="")
        assert out.startswith("【错误】"), f"空搜索词应报错，实测 {out!r}"
        assert plugin._fallback_at == {}, "空搜索词不得写入兜底缓存"

    async def test_whitespace_search_keyword_rejected(self):
        plugin, event = new_plugin_and_event(chain=[])
        out = await plugin.search_and_mention(event, name="   ")
        assert out.startswith("【错误】"), f"空白搜索词应报错，实测 {out!r}"

    async def test_super_long_keyword_does_not_crash(self):
        plugin, event = new_plugin_and_event(chain=[])
        event.bot.member_list = [{"user_id": "1", "nickname": "张三", "card": ""}]
        out = await plugin.search_and_mention(event, name="超" * 5000)
        assert out.startswith("【未找到】"), f"超长关键词应稳定返回未找到，实测 {out[:60]!r}"

    async def test_member_list_retry_then_success(self):
        plugin, event = new_plugin_and_event(chain=[])
        event.bot.member_list = [{"user_id": "42", "nickname": "柴郡", "card": ""}]
        event.bot.member_list_fail_times = 1
        out = await plugin.search_and_mention(event, name="柴郡")
        assert "42" in out and out.startswith("已找到"), (
            f"首次抖动后重试应成功，实测 {out!r}"
        )

    async def test_member_list_always_fails_degrades(self):
        plugin, event = new_plugin_and_event(chain=[])
        event.bot.member_list_fail_times = 99
        plugin._member_cache.clear()
        out = await plugin.search_and_mention(event, name="柴郡")
        assert out.startswith("【错误】"), f"持续失败应降级报错，实测 {out!r}"
        assert plugin._fallback_at == {}

    async def test_member_list_empty_is_not_found(self):
        plugin, event = new_plugin_and_event(chain=[])
        event.bot.member_list = []
        out = await plugin.search_and_mention(event, name="柴郡")
        assert out.startswith("【错误】"), f"空成员列表应降级，实测 {out!r}"

    async def test_member_cache_ttl_boundary(self):
        plugin = make_plugin()
        plugin.member_list_cache_ttl = 180
        event = FakeEvent(chain=[])
        event.bot.member_list = [{"user_id": "42", "nickname": "柴郡", "card": ""}]

        await plugin.search_and_mention(event, name="柴郡")
        calls_after_first = len(event.bot.calls)
        await plugin.search_and_mention(event, name="柴郡")
        assert len(event.bot.calls) == calls_after_first, "TTL 内应命中缓存"

        gid = event.get_group_id()
        members, ts = plugin._member_cache[gid]
        plugin._member_cache[gid] = (members, ts - 181)
        await plugin.search_and_mention(event, name="柴郡")
        assert len(event.bot.calls) > calls_after_first, "TTL 过期后应重新拉取"

    async def test_member_cache_ttl_zero_disables_cache(self):
        plugin = make_plugin()
        plugin.member_list_cache_ttl = 0
        event = FakeEvent(chain=[])
        event.bot.member_list = [{"user_id": "42", "nickname": "柴郡", "card": ""}]
        await plugin.search_and_mention(event, name="柴郡")
        n1 = len(event.bot.calls)
        await plugin.search_and_mention(event, name="柴郡")
        assert len(event.bot.calls) > n1, "ttl=0 时不得缓存"
        assert plugin._member_cache == {}

    async def test_fallback_ttl_expired_no_injection(self):
        plugin, event = new_plugin_and_event(chain=[Plain("本轮没有标签")])
        await run_llm_request(plugin, event)
        plugin._fallback_at[(event.unified_msg_origin, "10001")] = (
            "777",
            time.time() - 121,
        )
        await run_main_hook(plugin, event)
        await framework_deliver_final(event)

        assert event.delivery.at_targets() == [], "兜底缓存过期后不得补插"

    async def test_non_aiocqhttp_platform_degrades(self):
        class NoBotEvent(FakeEvent):
            def __init__(self, **kw):
                super().__init__(**kw)
                self.bot = types.SimpleNamespace()

        plugin = make_plugin()
        event = NoBotEvent(chain=[])
        out = await plugin.search_and_mention(event, name="柴郡")
        assert out.startswith("【错误】"), f"非 aiocqhttp 平台应降级，实测 {out!r}"

    async def test_at_all_cooldown_boundary_exact(self):
        plugin, event = new_plugin_and_event(chain=[Plain("[at:all]")])
        ok, remain = plugin._check_at_all_cooldown(event)
        assert ok is True and remain == 0.0
        plugin._record_at_all_trigger(event)
        plugin._at_all_last_trigger[event.get_group_id()] = time.time() - 60
        ok2, remain2 = plugin._check_at_all_cooldown(event)
        assert ok2 is True and remain2 == 0.0, "冷却恰好到期应放行"


# --------------------------------------------------------------------------- #
# 8. 反馈 1：工具可达性（提示词降级 / 不调用不存在的工具）
# --------------------------------------------------------------------------- #
class TestToolReachability:
    async def test_tool_absent_prompt_degrades_and_warns_once(self, monkeypatch):
        warnings: list = []
        monkeypatch.setattr(
            main_mod.logger, "warning", lambda *a, **kw: warnings.append(a[0] if a else "")
        )
        plugin, event = new_plugin_and_event(chain=[Plain("帮我艾特柴郡")])

        req1 = await run_llm_request(plugin, event, tool_names=())
        prompt1 = req1.system_prompt
        req2 = await run_llm_request(plugin, event, tool_names=())
        assert "必须调用" not in prompt1, "工具缺席时不得要求调用不存在的工具"
        assert "未启用艾特工具" in prompt1
        assert req2.system_prompt == prompt1, "缺席分支提示词应稳定"
        absent_warns = [w for w in warnings if "search_and_mention" in w]
        assert len(absent_warns) == 1, (
            f"每会话只应告警一次，实测 {len(absent_warns)} 次：{absent_warns}"
        )

    async def test_tool_present_prompt_requires_call(self):
        plugin, event = new_plugin_and_event(chain=[Plain("帮我艾特柴郡")])
        req = await run_llm_request(plugin, event)
        assert "search_and_mention" in req.system_prompt
        assert "未启用艾特工具" not in req.system_prompt

    async def test_docstring_declares_trigger_semantics(self):
        """P0-3：工具描述必须含触发语义（模型据此决定是否调用）。"""
        import docstring_parser

        for fn in (main_mod.LLMAtToolPlugin.search_and_mention,
                   main_mod.LLMAtToolPlugin.select_member_by_index):
            desc = docstring_parser.parse(fn.__doc__).short_description or ""
            assert any(k in desc for k in ("艾特", "@", "呼叫", "叫")), (
                f"{fn.__name__} 描述缺触发语义：{desc!r}"
            )
        assert "name" in docstring_parser.parse(
            main_mod.LLMAtToolPlugin.search_and_mention.__doc__
        ).params[0].arg_name


# --------------------------------------------------------------------------- #
# 9. 自检：本 harness 自身有效（防止永真断言）
# --------------------------------------------------------------------------- #
class TestHarnessSelfCheck:
    async def test_delivery_recorder_detects_naked_tag(self):
        rec = DeliveryRecorder()
        rec.record("event.send", types.SimpleNamespace(chain=[Plain("[At:1]")]))
        assert rec.naked_tags() == ["[At:1]"], "录制器必须能抓到裸标签"

    async def test_delivery_recorder_counts_at(self):
        rec = DeliveryRecorder()
        rec.record("event.send", types.SimpleNamespace(chain=[At(qq="1"), At(qq="1")]))
        assert rec.at_targets() == ["1", "1"]

    async def test_plugin_root_has_expected_module(self):
        assert (Path(main_mod.__file__).resolve().parent == _PLUGIN_ROOT), (
            f"被测插件根不符：{main_mod.__file__} vs {_PLUGIN_ROOT}"
        )


# =========================================================================== #
# T11 追加：跨插件「吞标签」机制复现 + T10 ID 可信来源约束的独立验证
# =========================================================================== #

# --------------------------------------------------------------------------- #
# 10. 真实 meme_manager 解析器（只读导入；不修改其任何文件）
# --------------------------------------------------------------------------- #
_MEME_BACKEND = Path("/root/AstrBot/data/plugins/meme_manager/backend")

# 线上 /root/AstrBot/data/config/meme_manager_config.json 实测取值：
#   generation.markup.enable_alternative        = true
#   generation.markup.remove_invalid_alternative= true
#   generation.markup.enable_repeated_detection = true
#   generation.markup.filter_all_tags           = false
#   generation.matching.enable_loose_matching   = true
# 且 mixins/event_handlers.py:_make_meme_parser 恒传 strip_references=True。
_MEME_LIVE_OPTS = dict(
    alternative=True,
    loose=True,
    repeated=True,
    remove_invalid=True,
    strip_references=True,
    filter_all=False,
)
# 活动表情包分类键（线上不包含 at:xxxx，故任何 [at:...] 都是 invalid 标记）
_MEME_CATS = {"开心", "疑惑", "无语", "赞同"}


def _real_meme_parser():
    """只读加载真实 MemeParser（离线、无网络无 IO）。"""
    if not (_MEME_BACKEND / "meme_parser" / "parser.py").exists():
        pytest.skip("meme_manager/meme_parser 不在本机，跳过上游吞标签复现")
    if str(_MEME_BACKEND) not in sys.path:
        sys.path.insert(0, str(_MEME_BACKEND))
    from meme_parser import MemeParser  # noqa: PLC0415

    return MemeParser


def _audit_files(audit_dir) -> list:
    audit_dir = Path(audit_dir) / "audit"
    return sorted(audit_dir.glob("at_audit_*.jsonl")) if audit_dir.exists() else []


class TestUpstreamSwallowMechanism:
    """meme_manager 的 on_llm_response（priority=99999）改写 completion_text，
    在 AtTool 的 on_decorating_result（priority=-1000）之前把标签吞掉。"""

    async def test_production_options_swallow_live_tag_and_keep_spaces(self):
        MemeParser = _real_meme_parser()
        src = "好的，这就帮你艾特 [at:2060958352] "
        out = MemeParser.parse(src, _MEME_CATS, **_MEME_LIVE_OPTS)

        assert "[at:" not in out.text, f"线上组合应吞掉标签，实测 {out.text!r}"
        assert out.text == "好的，这就帮你艾特  ", (
            f"标签被删但两侧空白保留（线上同形），实测 {out.text!r}"
        )
        assert [tok.kind for tok in out.tokens] == ["bracket"]
        assert out.tokens[0].valid is False, "at:xxxx 不是有效表情标记"

    async def test_both_live_ids_swallowed(self):
        MemeParser = _real_meme_parser()
        src = "先 [at:2060958352] 再 [at:3882563785] 完"
        out = MemeParser.parse(src, _MEME_CATS, **_MEME_LIVE_OPTS)
        assert "[at:" not in out.text
        assert out.text == "先  再  完", f"两个标签都消失，空白保留：{out.text!r}"

    async def test_strip_references_alone_does_not_swallow(self):
        """解释「有时吞、有时不吞」：单纯 strip_references 不吞标签。"""
        MemeParser = _real_meme_parser()
        src = "好的 [at:2060958352] "
        out = MemeParser.parse(src, _MEME_CATS, strip_references=True)
        assert out.text == src, f"仅 strip_references 时标签应存活，实测 {out.text!r}"

    async def test_remove_invalid_off_does_not_swallow(self):
        MemeParser = _real_meme_parser()
        src = "好的 [at:2060958352] "
        out = MemeParser.parse(
            src, _MEME_CATS, **{**_MEME_LIVE_OPTS, "remove_invalid": False}
        )
        assert out.text == src, f"关掉 remove_invalid 后标签应存活，实测 {out.text!r}"

    async def test_alternative_off_does_not_swallow(self):
        MemeParser = _real_meme_parser()
        src = "好的 [at:2060958352] "
        out = MemeParser.parse(
            src, _MEME_CATS, **{**_MEME_LIVE_OPTS, "alternative": False}
        )
        assert out.text == src, f"关掉 alternative 后标签应存活，实测 {out.text!r}"

    @pytest.mark.parametrize(
        "form",
        [
            "[at:2060958352]", "[At:2060958352]", "[AT:2060958352]",
            "[at: 2060958352]", "[at：2060958352]", "[at:all]", "[At:All]",
            "[at:柴郡]", "[at:ID]", "[at 123]", "[avatar:1]", "[at:]",
        ],
        ids=lambda f: f"swallow-{abs(len(f))}",
    )
    async def test_every_bracket_form_is_swallowed_upstream(self, form):
        """上游吞的是「任意 [] token」，与是不是合法 AtTool 标签无关。"""
        MemeParser = _real_meme_parser()
        out = MemeParser.parse(f"文本{form}后缀", _MEME_CATS, **_MEME_LIVE_OPTS)
        assert "[at:" not in out.text and "[" not in out.text, (
            f"{form} 应被上游吞掉，实测 {out.text!r}"
        )
        assert out.text == "文本后缀"

    async def test_unclosed_tag_survives_upstream(self):
        """未闭合的 `[at:12`（无 `]`）不匹配 bracket 标记 → 上游保留。"""
        MemeParser = _real_meme_parser()
        out = MemeParser.parse("文本[at:12后缀", _MEME_CATS, **_MEME_LIVE_OPTS)
        assert out.text == "文本[at:12后缀", f"实测 {out.text!r}"


class TestUpstreamSwallowE2E:
    """吞标签之后 AtTool 的两条链路（用真实解析器产出被吞文本）。"""

    async def _chain_after_upstream(self, text: str, swallowed: bool):
        """模拟 LLM 响应阶段：meme_manager 改写 completion_text 之后的最终文本。"""
        MemeParser = _real_meme_parser()
        opts = _MEME_LIVE_OPTS if swallowed else {"strip_references": True}
        return MemeParser.parse(text, _MEME_CATS, **opts).text

    async def test_a_swallowed_tag_is_invisible_to_attool_no_tool_call(self, tmp_path):
        """(a) 标签已被吞且模型没调工具：AtTool 无从渲染 → 群里什么都没有。"""
        LIVE = "2060958352"
        visible = await self._chain_after_upstream(f"好的 [at:{LIVE}] 就这样", True)

        plugin, event = new_plugin_and_event(chain=[Plain(visible)])
        plugin._audit_dir = tmp_path
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)
        await framework_deliver_final(event)

        assert event.delivery.at_targets() == [], "标签已被上游吞掉，不可能有 At"
        assert event.delivery.naked_tags() == [], "上游吞掉后更不会出现裸标签"
        assert _audit_files(tmp_path) == [], "无渲染则无审计（与线上现象一致）"
        assert event.delivery.text_of(-1) == "好的  就这样"

    async def test_a2_swallowed_tag_rescued_by_tool_fallback(self, tmp_path):
        """(a2) 标签被吞但模型调用过工具 → 兜底补插把艾特救回。"""
        LIVE = "2060958352"
        visible = await self._chain_after_upstream(f"好的 [at:{LIVE}] 就这样", True)

        plugin, event = new_plugin_and_event(chain=[Plain(visible)], trusted_ids=set())
        plugin._audit_dir = tmp_path
        await run_llm_request(plugin, event)
        mark_tool_confirmed(plugin, event, LIVE, fallback=True)  # search_and_mention 唯一命中
        await run_main_hook(plugin, event)
        await framework_deliver_final(event)

        assert event.delivery.at_targets() == [LIVE], (
            f"工具命中过的 ID 应由兜底补插救回，实测 {event.delivery.at_targets()}"
        )
        assert event.delivery.naked_tags() == []
        assert _audit_files(tmp_path), "补插渲染应留下审计记录"

    async def test_b_surviving_tag_renders_normally(self, tmp_path):
        """(b) 标签在上游存活（如 remove_invalid=false）→ AtTool 正常渲染。"""
        LIVE = "2060958352"
        visible = await self._chain_after_upstream(f"好的 [at:{LIVE}] 就这样", False)
        assert LIVE in visible

        plugin, event = new_plugin_and_event(chain=[Plain(visible)], trusted_ids={LIVE})
        plugin._audit_dir = tmp_path
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)
        await framework_deliver_final(event)

        assert event.delivery.at_targets() == [LIVE]
        assert event.delivery.naked_tags() == []
        assert _audit_files(tmp_path), "正常渲染应留下审计记录"


# --------------------------------------------------------------------------- #
# 11. T10：ID 可信来源约束（我自己的断言，不跑实现者用例）
# --------------------------------------------------------------------------- #
LIVE_FABRICATED_IDS = ("2060958352", "3882563785")


class TestTrustedIdSourceVerifier:
    async def test_fabricated_live_ids_hard_rejected(self, tmp_path, monkeypatch):
        """线上两个编造 ID：既无工具确认、也不在用户消息里 → 硬拒绝。"""
        warnings: list = []
        monkeypatch.setattr(
            main_mod.logger, "warning",
            lambda *a, **k: warnings.append(a[0] if a else ""),
        )
        plugin, event = new_plugin_and_event(
            chain=[Plain("好的 [at:2060958352] 与 [at:3882563785] 都是他")],
            trusted_ids=set(),
        )
        plugin._audit_dir = tmp_path
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)
        await framework_deliver_final(event)

        assert event.delivery.at_targets() == [], (
            f"编造 ID 绝不能被渲染，实测 {event.delivery.at_targets()}"
        )
        text = event.delivery.text_of(-1)
        assert "@2060958352" in text and "@3882563785" in text, f"应降级为纯文本：{text!r}"
        assert event.delivery.naked_tags() == [], f"不得残留标签语法：{text!r}"
        assert _audit_files(tmp_path) == [], "被拒绝的 ID 不得写审计"
        untrusted_warns = [w for w in warnings if "未经工具确认" in w]
        assert len(untrusted_warns) == 1, (
            f"一轮渲染聚合一条告警，实测 {len(untrusted_warns)} 条：{untrusted_warns}"
        )

    async def test_tool_confirmed_id_renders_end_to_end(self):
        """真调 search_and_mention 命中后，模型输出的 [at:ID] 必须渲染。"""
        plugin, event = new_plugin_and_event(chain=[], trusted_ids=set())
        event.bot.member_list = [
            {"user_id": "2060958352", "nickname": "柴郡", "card": "柴郡"},
        ]
        out = await plugin.search_and_mention(event, name="柴郡")
        assert out.startswith("已找到"), f"工具应命中：{out!r}"

        event.set_chain([Plain("就是他 [at:2060958352] 了")])
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)
        await framework_deliver_final(event)

        assert event.delivery.at_targets() == ["2060958352"], (
            f"工具确认过的 ID 必须渲染，实测 {event.delivery.at_targets()}"
        )
        assert event.delivery.naked_tags() == []

    async def test_user_message_qq_renders(self):
        """用户在消息里给出 QQ 号 → 该 ID 可信（allow_direct_qq_at=true）。"""
        plugin, event = new_plugin_and_event(
            chain=[Plain("好的 [at:2060958352]")],
            trusted_ids=set(),
            user_message="你艾特 2060958352 这个人",
        )
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)
        assert [str(c.qq) for c in event.get_result().chain if isinstance(c, At)] == ["2060958352"]

    async def test_inbound_chain_qq_renders(self):
        """用户 QQ 号出现在入站消息链的 Plain 里（message_str 为空）→ 同样可信。"""
        plugin, event = new_plugin_and_event(
            chain=[Plain("好的 [at:2060958352]")],
            trusted_ids=set(),
            user_message="",
            incoming=[Plain("帮我艾特 2060958352")],
        )
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)
        assert [str(c.qq) for c in event.get_result().chain if isinstance(c, At)] == ["2060958352"]

    async def test_direct_qq_switch_off_requires_tool(self):
        """allow_direct_qq_at=false：用户给了 QQ 也不算可信，必须走工具。"""
        plugin, event = new_plugin_and_event(
            chain=[Plain("好的 [at:2060958352]")],
            trusted_ids=set(),
            user_message="你艾特 2060958352 这个人",
        )
        plugin.allow_direct_qq_at = False
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)

        chain = event.get_result().chain
        assert not any(isinstance(c, At) for c in chain), "开关关闭时不得凭用户消息渲染"
        assert "@2060958352" in "".join(
            c.text for c in chain if isinstance(c, Plain)
        )

    async def test_direct_qq_switch_off_tool_hit_still_renders(self):
        plugin, event = new_plugin_and_event(chain=[], trusted_ids=set())
        plugin.allow_direct_qq_at = False
        event.bot.member_list = [
            {"user_id": "2060958352", "nickname": "柴郡", "card": "柴郡"},
        ]
        await plugin.search_and_mention(event, name="柴郡")

        event.set_chain([Plain("[at:2060958352]")])
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)
        assert [str(c.qq) for c in event.get_result().chain if isinstance(c, At)] == ["2060958352"]

    async def test_fabricated_id_degrades_through_send_path(self):
        """第三方清链逐段 send（绕过主钩子）时同样不得渲染编造 ID。"""
        plugin, event = new_plugin_and_event(
            chain=[Plain("前段\n\n后段 [at:2060958352]")],
            trusted_ids=set(),
        )
        await run_llm_request(plugin, event)
        await third_party_clear_and_send_each(plugin, event)
        await run_main_hook(plugin, event)

        assert event.delivery.at_targets() == [], (
            f"send 路径也必须拒绝编造 ID，实测 {event.delivery.at_targets()}"
        )
        assert event.delivery.naked_tags() == []
        assert "@2060958352" in "".join(
            event.delivery.text_of(i) for i in range(len(event.delivery.chains()))
        )

    async def test_fabricated_id_degrades_through_streaming(self):
        """流式路径同样受 ID 可信来源约束。"""
        plugin, event = new_plugin_and_event(chain=[], trusted_ids=set())
        await run_llm_request(plugin, event)
        mc_cls = sys.modules["astrbot.core.message.message_event_result"].MessageChain

        async def gen():
            yield mc_cls(chain=[Plain("最 [at:3882563785] 后")])

        await event.send_streaming(gen(), False)
        assert event.delivery.at_targets() == [], (
            f"流式路径也必须拒绝编造 ID，实测 {event.delivery.at_targets()}"
        )
        assert event.delivery.naked_tags() == []


# --------------------------------------------------------------------------- #
# 12. 误判边界（实测并如实记录，内含 findings）
# --------------------------------------------------------------------------- #
class TestTrustedIdBoundaries:
    async def test_non_qq_digits_in_user_message_are_trusted(self):
        """【观察】用户消息里任意 5..12 位数字都会被当成可信 QQ（如金额/单号）。"""
        results = {}
        for num in ("19999", "1234567", "10086"):
            plugin, event = new_plugin_and_event(
                chain=[Plain(f"[at:{num}]")],
                trusted_ids=set(),
                user_message=f"我花了 {num} 元",
            )
            await run_llm_request(plugin, event)
            await run_main_hook(plugin, event)
            results[num] = [
                str(c.qq) for c in event.get_result().chain if isinstance(c, At)
            ]
        assert results == {
            "19999": ["19999"],
            "1234567": ["1234567"],
            "10086": ["10086"],
        }, f"实测行为：{results}（非 QQ 数字会被当作可信来源）"

    async def test_four_digit_year_is_not_a_source(self):
        plugin, event = new_plugin_and_event(
            chain=[Plain("[at:2025]")],
            trusted_ids=set(),
            user_message="2025 年的时候",
        )
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)
        assert not any(isinstance(c, At) for c in event.get_result().chain), (
            "4 位数字不匹配 5..12 位规则，不得成为可信来源"
        )

    async def test_previous_turn_qq_not_trusted_next_turn(self):
        """【观察】上一轮给过 QQ、本轮只说"就他吧" → 无工具确认则不渲染（保守）。"""
        plugin, event = new_plugin_and_event(
            chain=[Plain("[at:2060958352]")],
            trusted_ids=set(),
            user_message="就他吧",
        )
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)
        assert not any(isinstance(c, At) for c in event.get_result().chain)
        assert plugin._known_at_ids == {}, "用户消息来源不写会话记忆（只按本轮判定）"

    async def test_tool_confirmation_persists_across_turns_within_ttl(self):
        """工具确认过 → 下一轮（TTL 内、同 sender）只写标签也能渲染。"""
        plugin, event = new_plugin_and_event(chain=[], trusted_ids=set())
        event.bot.member_list = [
            {"user_id": "2060958352", "nickname": "柴郡", "card": "柴郡"},
        ]
        await plugin.search_and_mention(event, name="柴郡")

        # 下一轮：用户只说"就他吧"，模型输出标签
        event2 = FakeEvent(
            chain=[Plain("[at:2060958352]")],
            sender_id=event.get_sender_id(),
            umo=event.unified_msg_origin,
            group_id=event.get_group_id(),
        )
        await run_llm_request(plugin, event2)
        await run_main_hook(plugin, event2)
        assert [str(c.qq) for c in event2.get_result().chain if isinstance(c, At)] == [
            "2060958352"
        ]

    @pytest.mark.xfail(
        reason="T11 finding F2：跨组件拼合路径（_merge_split_at_tags）直接插入 At，"
        "绕过 ID 可信来源判定 —— 待 repair 轮统一收口",
        strict=False,
    )
    async def test_split_fabricated_id_with_middle_component_not_rendered(self):
        """严格语义期望：被切碎且夹着其它组件的编造 ID 也不得渲染。"""
        plugin, event = new_plugin_and_event(
            chain=[Plain("[at:20609"), At(qq="999"), Plain("58352]")],
            trusted_ids=set(),
        )
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)
        targets = [str(c.qq) for c in event.get_result().chain if isinstance(c, At)]
        assert "2060958352" not in targets, f"实测渲染了编造 ID：{targets}"

    async def test_split_fabricated_id_observed_bypass(self, monkeypatch):
        """T14/B2 起对齐到严格期望：切碎的编造 ID 也必须降级、不得拼成 At。

        本用例原本记录 T11 时的"绕过行为实测"（`targets == ["2060958352", "999"]`）；
        跨组件拼合纳入同一份可信集后，期望反转为：编造 ID 不渲染、原位降级为
        `@2060958352`（数字文本保留、不静默删除），同时可信的 `At(999)` 保留。
        """
        warnings: list = []
        plugin, event = new_plugin_and_event(
            chain=[Plain("[at:20609"), At(qq="999"), Plain("58352]")],
            trusted_ids=set(),
        )
        import astrbot_plugin_AtTool.main as main_mod

        monkeypatch.setattr(
            main_mod.logger,
            "warning",
            lambda *a, **k: warnings.append(a[0] if a else ""),
        )
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)
        await framework_deliver_final(event)

        targets = [str(c.qq) for c in event.get_result().chain if isinstance(c, At)]
        assert targets == ["999"], f"切碎的编造 ID 不得拼成 At（T14/B2），实测 {targets}"
        text = "".join(
            c.text for c in event.get_result().chain if isinstance(c, Plain)
        ).replace("\u200b", "")
        assert "@2060958352" in text, f"应原位降级为 @载荷、不得静默删除：{text!r}"
        assert event.delivery.naked_tags() == [], "不得残留标签语法"
        assert any("未经工具确认" in str(w) for w in warnings), (
            f"降级必须留聚合告警，实测 {warnings}"
        )


# --------------------------------------------------------------------------- #
# 13. 状态治理：TTL / 容量 / 发送者绑定 / terminate
# --------------------------------------------------------------------------- #
class TestTrustedIdGovernance:
    async def test_known_ids_expire_after_ttl(self):
        plugin, event = new_plugin_and_event(chain=[Plain("[at:2060958352]")], trusted_ids=set())
        mark_tool_confirmed(plugin, event, "2060958352")
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)
        assert [str(c.qq) for c in event.get_result().chain if isinstance(c, At)] == ["2060958352"]

        umo = event.unified_msg_origin
        key = (umo, str(event.get_sender_id() or ""))
        ids, ts = plugin._known_at_ids[key]
        plugin._known_at_ids[key] = (ids, ts - 121)  # 超过 _KNOWN_AT_ID_TTL
        event.set_chain([Plain("[at:2060958352]")])
        await run_main_hook(plugin, event)
        assert not any(isinstance(c, At) for c in event.get_result().chain), (
            "TTL 过期后不得再凭会话记忆渲染"
        )

    async def test_known_ids_sender_bound_same_umo(self):
        plugin, event = new_plugin_and_event(chain=[], trusted_ids=set())
        mark_tool_confirmed(plugin, event, "2060958352", fallback=False)

        other = FakeEvent(
            chain=[Plain("[at:2060958352]")],
            sender_id="99999",
            umo=event.unified_msg_origin,
            group_id=event.get_group_id(),
        )
        await run_llm_request(plugin, other)
        await run_main_hook(plugin, other)
        assert not any(isinstance(c, At) for c in other.get_result().chain), (
            "同会话换人后不得复用他人的工具确认结果（越权艾特）"
        )
        assert "2060958352" not in str(other.get_result().chain)

    async def test_capacity_evicts_oldest_session(self):
        plugin, event = new_plugin_and_event(chain=[], trusted_ids=set())
        # 501 个会话共用同一个 bot（成员列表一致），逐个真实调用工具登记 ID
        shared_bot = FakeBot(
            members=[{"user_id": "2060958352", "nickname": "柴郡", "card": "柴郡"}]
        )
        umos = []
        for i in range(501):
            ev = FakeEvent(
                chain=[],
                umo=f"aiocqhttp:GroupMessage:100{i}",
                group_id="4242",
                bot=shared_bot,
                user_message="帮我艾特一下柴郡",
            )
            umos.append(ev.unified_msg_origin)
            out = await plugin.search_and_mention(ev, name="柴郡")
            assert out.startswith("已找到"), f"第 {i} 次工具调用应命中：{out!r}"

        assert len(plugin._known_at_ids) == 500, (
            f"超限应淘汰到容量上限 500，实测 {len(plugin._known_at_ids)}"
        )
        slot = (umos[0], "10001")
        assert slot not in plugin._known_at_ids, "最早写入的会话槽位应被淘汰（FIFO）"
        assert (umos[-1], "10001") in plugin._known_at_ids, "最新会话槽位必须保留"

        evicted = FakeEvent(
            chain=[Plain("[at:2060958352]")],
            umo=umos[0],
            group_id="4242",
            sender_id="10001",
        )
        await run_llm_request(plugin, evicted)
        await run_main_hook(plugin, evicted)
        assert not any(isinstance(c, At) for c in evicted.get_result().chain), (
            "被淘汰会话的确认结果不应复活"
        )

    async def test_terminate_clears_known_ids(self):
        plugin, event = new_plugin_and_event(chain=[], trusted_ids=set())
        mark_tool_confirmed(plugin, event, "2060958352")
        assert plugin._known_at_ids
        await plugin.terminate()
        assert plugin._known_at_ids == {}, "terminate 必须清空可信 ID 记忆"
