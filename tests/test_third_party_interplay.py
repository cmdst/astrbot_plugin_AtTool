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
        def __init__(self, qq="", name=""):
            self.qq = qq
            self.name = name

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
    """最小事件对象：暴露 bot.api（过 aiocqhttp 鸭子类型判定）与交付路径。"""

    def __init__(self, chain=None, group_id="1035699087", sender_id="10001",
                 umo=None, bot=None, recorder=None):
        self._group_id = group_id
        self._sender_id = sender_id
        self.unified_msg_origin = umo or f"aiocqhttp:GroupMessage:{group_id}"
        self.bot = bot if bot is not None else FakeBot()
        self.message_str = ""
        self._result = FakeResult(chain)
        self.delivery = recorder if recorder is not None else DeliveryRecorder()

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


def new_plugin_and_event(chain=None, **event_kwargs):
    """构造 (plugin, event)，两者共享同一个交付录制器。

    AtTool 的 send 包装器与第三方 splitter 的 context.send_message 是两条
    独立投递路径，必须录到同一个 DeliveryRecorder 里才能做"全量交付"断言。
    """
    recorder = DeliveryRecorder()
    plugin = make_plugin(recorder)
    event = FakeEvent(chain=chain, recorder=recorder, **event_kwargs)
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
        plugin._fallback_at[event.unified_msg_origin] = ("10001", "777", time.time())

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
        plugin._fallback_at[event.unified_msg_origin] = ("10001", "777", time.time())

        await third_party_clear_and_send_each(plugin, event)
        assert event.delivery.at_targets() == []
        assert plugin._fallback_at.get(event.unified_msg_origin) is not None, (
            "send 路径不应消费兜底缓存（否则主钩子无标签可补）"
        )
        await run_main_hook(plugin, event)
        assert plugin._fallback_at.get(event.unified_msg_origin) is None

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
        plugin, event = new_plugin_and_event(chain=[Plain("你好 [at:12"), Plain("345] 收工")])
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
        plugin, event = new_plugin_and_event(chain=[Plain("[at:12"), At(qq="999"), Plain("345]")])
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
        plugin, event = new_plugin_and_event(chain=[])
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
            chain=[Plain("[at:12"), At(qq="999"), Plain("3]")]
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

        plugin, event = new_plugin_and_event(chain=build_chain())
        await run_llm_request(plugin, event)
        await run_main_hook(plugin, event)
        assert [
            str(c.qq) for c in event.get_result().chain if isinstance(c, At)
        ] == ["123", "999"]

        monkeypatch.setattr(main_mod, "_SPLIT_AT_WINDOW", 1)
        plugin2, event2 = new_plugin_and_event(chain=build_chain())
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
        plugin._fallback_at[event.unified_msg_origin] = ("10001", "777", time.time())
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
        plugin._fallback_at[event.unified_msg_origin] = ("10001", "777", time.time())

        mc_cls = sys.modules["astrbot.core.message.message_event_result"].MessageChain

        async def gen():
            yield mc_cls(chain=[Plain("第一块没有标签")])
            yield mc_cls(chain=[Plain("第二块也没有")])

        await event.send_streaming(gen(), False)
        assert event.delivery.at_targets() == [], (
            f"流式路径不得补插兜底艾特，实测 {event.delivery.at_targets()}"
        )
        assert plugin._fallback_at.get(event.unified_msg_origin) is not None, (
            "流式路径不得消费兜底缓存"
        )

    async def test_streaming_per_chunk_renders_own_tag(self):
        plugin, event = new_plugin_and_event(chain=[])
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
        plugin, event = new_plugin_and_event(chain=[])
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
        plugin._fallback_at[event.unified_msg_origin] = ("10001", "777", time.time() - 121)
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
