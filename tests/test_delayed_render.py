"""P4 延迟渲染 + send 兜底渲染回归测试。

背景（T-SRE-20260809-01 放大缺陷）：AtTool 的 process_at_tags 钩子
priority=2（最先执行）把 [at:ID] 渲染为 At 组件后，任何后执行（priority<2）
的插件若重建/清空消息链（result.chain.clear()），At 组件即被丢弃。

P4 改造：
1. process_at_tags priority 2 → -1000（AstrBot 钩子按 -priority 降序执行，
   使其在全部 on_decorating_result 钩子之后执行，延迟到最后渲染）。
2. 新增 send 兜底渲染（_wrap_event_send）：针对"清空 chain 后直接
   event.send()"的插件（如分段插件），在消息真正发往平台前完成渲染。

本文件覆盖：
- 验收标准 1：模拟"重建 chain 只保留 Plain 文本"的恶意/第三方插件场景，
  断言最终消息链仍含 At 组件、QQ 号正确；
- 分段插件模式（clear chain + 直接 send）下艾特仍生效；
- send 兜底渲染的不变量：会话准入降级 / @全体冷却 / 审计 / 兜底补插 /
  零宽清理，与主钩子语义一致；
- 残缺标签容错与告警、跨组件残缺标签合并修复、包装幂等与异常回退。

运行方式（插件目录下）：
    python3 -m pytest tests/test_delayed_render.py -v
"""

from __future__ import annotations

import inspect
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

pytestmark = pytest.mark.asyncio

# 复用 test_integration_main.py 的 astrbot 桩与 Fake 基础设施
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import test_integration_main as ti  # noqa: E402

# 桩已在 import test_integration_main 时安装，可直接导入组件类
from astrbot.api.message_components import Plain, At  # noqa: E402

FakeEvent = ti.FakeEvent
FakeBot = ti.FakeBot
make_plugin = ti.make_plugin
audit_file = ti.audit_file


class MessageChainStub:
    """最小 MessageChain 桩：暴露 .chain 列表（真实框架 MessageChain 同构）。"""

    def __init__(self, chain):
        self.chain = chain


class SendableEvent(FakeEvent):
    """带 send / send_streaming 记录能力的 FakeEvent。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sent = []

    async def send(self, chain):
        self.sent.append(chain)

    async def send_streaming(self, generator, use_fallback=False):
        self.sent.extend([item async for item in generator])


# --------------------------------------------------------------------------- #
# 验收标准 1：延迟渲染主链路 —— 恶意插件重建 chain（只保留 Plain）后艾特仍生效
# --------------------------------------------------------------------------- #
class TestDelayedRenderImmunity:
    async def test_malicious_rebuild_chain_keeps_at(self, tmp_path):
        """模拟恶意钩子（priority=1）：clear chain 后只放 Plain 文本。

        AtTool 钩子（priority=-1000）在所有钩子之后执行，[at:ID] 以纯文本
        形式穿过恶意插件，最终仍被渲染为 At 组件。
        """
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[])
        # 恶意插件：清空链，重建时只保留 Plain 文本（含 [at:ID]）
        ev.set_chain([Plain("好的[at:10001]这就喊他！")])
        # AtTool 钩子最后执行（模拟 priority=-1000 排序）
        await plugin.process_at_tags(ev)

        chain = ev.get_result().chain
        ats = [c for c in chain if isinstance(c, At)]
        assert len(ats) == 1 and ats[0].qq == "10001", "艾特必须保留且 QQ 号正确"
        texts = "".join(c.text for c in chain if isinstance(c, Plain))
        assert "好的" in texts and "这就喊他" in texts

    async def test_hook_order_simulation_malicious_plugin_first(self, tmp_path):
        """按 AstrBot 钩子排序语义（-priority 降序）模拟完整调用序列。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[])
        ev.set_chain([Plain("长回复[at:10002]结尾")])

        async def malicious_hook(event):
            result = event.get_result()
            plain_text = "".join(
                c.text for c in result.chain if isinstance(c, Plain)
            )
            result.chain.clear()
            result.chain.append(Plain(plain_text))  # 重建链，只保留 Plain

        # 排序语义：priority=1（恶意）先执行，priority=-1000（AtTool）后执行
        await malicious_hook(ev)
        await plugin.process_at_tags(ev)

        ats = [c for c in ev.get_result().chain if isinstance(c, At)]
        assert len(ats) == 1 and ats[0].qq == "10002"

    async def test_adjacent_plain_merge_repairs_split_tag(self, tmp_path):
        """链内跨组件残缺标签（[at:12 + 345] 分处相邻 Plain）合并后仍可渲染。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain("开头[at:12"), Plain("345]结尾")])
        await plugin.process_at_tags(ev)

        ats = [c for c in ev.get_result().chain if isinstance(c, At)]
        assert len(ats) == 1 and ats[0].qq == "12345"

    async def test_no_tag_adjacent_plain_merge_must_not_duplicate_text(self, tmp_path):
        """回归：无标签链的相邻 Plain 合并不得原地突变调用方组件。

        修复前：步骤①以 `merged_input[-1].text += comp.text` 原地合并，
        merged_input 与 result.chain 共享组件引用；rendered=False 提前返回
        （链内无 [at: 标签且无兜底缓存）时原链首 Plain 已含合并文本而尾
        Plain 仍在，消息文本重复渲染（如 "你好" + "世界" → "你好世界世界"）。
        """
        plugin = make_plugin(audit_dir=tmp_path)
        first, second = Plain("你好"), Plain("世界")
        ev = FakeEvent(chain=[first, second])
        await plugin.process_at_tags(ev)

        chain = ev.get_result().chain
        texts = [c.text for c in chain if isinstance(c, Plain)]
        assert texts == ["你好", "世界"], (
            f"无标签链不得被合并突变，实际渲染文本: {texts}"
        )

    async def test_process_at_tags_priority_is_minus_1000(self):
        """防回归：priority 必须为 -1000（延迟到最后渲染）。"""
        src = inspect.getsource(ti.LLMAtToolPlugin.process_at_tags)
        assert "priority=-1000" in src, (
            "process_at_tags 的 priority 必须是 -1000，否则无法免疫"
            "后执行（priority<2）插件对 At 组件的破坏"
        )


# --------------------------------------------------------------------------- #
# send 兜底渲染：分段插件模式（clear chain + 直接 event.send()）艾特仍生效
# --------------------------------------------------------------------------- #
class TestSendFallbackRender:
    async def test_clear_chain_direct_send_keeps_at(self, tmp_path):
        """分段插件模式：clear chain 后直接 event.send() 纯文本段。

        AtTool 的 send 兜底渲染在消息真正发往平台前把 [at:ID] 渲染为 At。
        """
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._wrap_event_send(ev)

        seg = MessageChainStub([Plain("第一段[at:10001]喊人")])
        await ev.send(seg)

        assert len(ev.sent) == 1, "消息必须被发出"
        assert ev.sent[0] is seg, "必须发送原消息链对象（渲染后替换其 chain）"
        ats = [c for c in seg.chain if isinstance(c, At)]
        assert len(ats) == 1 and ats[0].qq == "10001"

    async def test_multi_segment_partial_tags(self, tmp_path):
        """多段 send：仅含标签的段被渲染，无标签段原样透传。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._wrap_event_send(ev)

        seg1 = MessageChainStub([Plain("第一段[at:10001]喊人")])
        seg2 = MessageChainStub([Plain("第二段无标签")])
        await ev.send(seg1)
        await ev.send(seg2)

        assert len(ev.sent) == 2
        assert any(isinstance(c, At) and c.qq == "10001" for c in seg1.chain)
        assert not any(isinstance(c, At) for c in seg2.chain)
        assert seg2.chain[0].text == "第二段无标签"

    async def test_send_without_tag_passthrough(self):
        """无标签消息：零干预原样透传（同一对象、未重建）。"""
        plugin = make_plugin()
        ev = SendableEvent(chain=[])
        plugin._wrap_event_send(ev)

        seg = MessageChainStub([Plain("普通文本")])
        await ev.send(seg)

        assert ev.sent == [seg]
        assert ev.sent[0] is seg
        assert seg.chain[0].text == "普通文本"

    async def test_send_wrapper_non_list_chain_passthrough(self):
        """非 list 形态的 chain：防御性跳过，不崩溃。"""
        plugin = make_plugin()
        ev = SendableEvent(chain=[])
        plugin._wrap_event_send(ev)

        seg = MessageChainStub("not-a-list")
        await ev.send(seg)

        assert ev.sent == [seg]

    async def test_send_wrap_idempotent(self):
        """包装幂等：重复调用只包装一次、只发送一次。"""
        plugin = make_plugin()
        ev = SendableEvent(chain=[])
        plugin._wrap_event_send(ev)
        plugin._wrap_event_send(ev)

        seg = MessageChainStub([Plain("文本[at:10001]")])
        await ev.send(seg)

        assert len(ev.sent) == 1
        assert any(isinstance(c, At) and c.qq == "10001" for c in seg.chain)

    async def test_send_wrapper_exception_falls_back(self, monkeypatch):
        """渲染异常：回退原样发送，绝不影响消息发出。"""
        plugin = make_plugin()

        async def boom(event, chain):
            raise RuntimeError("boom")

        monkeypatch.setattr(plugin, "_render_at_tags", boom)

        ev = SendableEvent(chain=[])
        plugin._wrap_event_send(ev)
        seg = MessageChainStub([Plain("文本[at:10001]")])
        await ev.send(seg)

        assert len(ev.sent) == 1
        assert seg.chain[0].text == "文本[at:10001]", "异常时按原样发送"

    async def test_inject_installs_send_wrapper(self, tmp_path):
        """on_llm_request（inject_at_instruction）安装 send 兜底包装。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        from astrbot.api.provider import ProviderRequest

        req = ProviderRequest()
        await plugin.inject_at_instruction(ev, req)

        assert getattr(ev, "_attool_send_wrapped", False) is True
        seg = MessageChainStub([Plain("文本[at:10001]")])
        await ev.send(seg)
        assert any(isinstance(c, At) and c.qq == "10001" for c in seg.chain)


# --------------------------------------------------------------------------- #
# send 兜底渲染路径的不变量：审计 / 兜底补插 / 会话准入 / @全体冷却
# --------------------------------------------------------------------------- #
class TestSendFallbackInvariants:
    async def test_send_render_writes_audit(self, tmp_path):
        """send 兜底渲染同样写入 at_member 审计（与主钩子语义一致）。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._wrap_event_send(ev)

        seg = MessageChainStub([Plain("文本[at:10001]尾巴")])
        await ev.send(seg)

        date_str = datetime.now().strftime("%Y%m%d")
        records = audit_file(tmp_path, date_str).read_text(encoding="utf-8").splitlines()
        assert any("at_member" in r and "10001" in r for r in records)

    async def test_send_fallback_injection_when_tag_missing(self, tmp_path):
        """send 路径兜底补插：分段直发 + 模型漏写标签时艾特不丢（09-07 修复）。

        旧语义（P1-1 方案 A）：send 路径绝不补插，补插只发生在主钩子；
        但分段插件清空 result.chain 直发各段时主钩子链空无法补插，模型又
        漏写标签 → 艾特全丢。新语义：send 路径在兜底缓存存在、本事件尚未
        送出艾特且本段无标签时补插一次并消费缓存。
        """
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._fallback_at[ev.unified_msg_origin] = ("10001", time.time())
        plugin._wrap_event_send(ev)

        seg = MessageChainStub([Plain("回复没带标签")])
        await ev.send(seg)

        # send 路径：补插一次并消费缓存
        ats = [c for c in seg.chain if isinstance(c, At)]
        assert len(ats) == 1 and ats[0].qq == "10001", "send 路径必须兜底补插"
        assert ev.unified_msg_origin not in plugin._fallback_at, "缓存一次性消费"
        # 补插的标签走正常渲染链路 → 审计落盘
        date_str = datetime.now().strftime("%Y%m%d")
        records = (
            audit_file(tmp_path, date_str).read_text(encoding="utf-8").splitlines()
        )
        assert any("at_member" in r and "10001" in r for r in records)

    async def test_send_render_respects_blacklist(self, tmp_path):
        """会话准入降级在 send 兜底路径同样生效（[at:ID] 被剥离）。"""
        plugin = make_plugin(
            config={"session_blacklist": ["123456"]}, audit_dir=tmp_path
        )
        ev = SendableEvent(chain=[])
        plugin._wrap_event_send(ev)

        seg = MessageChainStub([Plain("文本[at:10001]尾巴")])
        await ev.send(seg)

        assert not any(isinstance(c, At) for c in seg.chain)
        assert not any(
            "[at:" in c.text for c in seg.chain if isinstance(c, Plain)
        )

    async def test_send_at_all_cooldown_enforced(self, tmp_path):
        """@全体 冷却限制在 send 兜底路径生效（第二次 [at:all] 转纯文本）。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        ev.bot.member_info_role = "owner"  # 群主：具备 @全体 权限
        plugin._wrap_event_send(ev)

        seg1 = MessageChainStub([Plain("第一次[at:all]全体")])
        await ev.send(seg1)
        assert any(isinstance(c, At) and c.qq == "all" for c in seg1.chain)

        seg2 = MessageChainStub([Plain("第二次[at:all]全体")])
        await ev.send(seg2)
        assert not any(isinstance(c, At) for c in seg2.chain)
        assert any(
            isinstance(c, Plain) and "全体成员" in c.text for c in seg2.chain
        )


# --------------------------------------------------------------------------- #
# send_streaming 兜底渲染（respond.stage 流式路径绕过 event.send）
# --------------------------------------------------------------------------- #
class TestSendStreamingWrapper:
    """send_streaming 逐段渲染 [at:ID]（引用回复/分段场景的流式兜底）。"""

    async def test_streaming_chunk_with_tag_rendered(self, tmp_path):
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._wrap_event_send(ev)

        async def gen():
            yield MessageChainStub([Plain("文本[at:10001]尾巴")])

        await ev.send_streaming(gen())

        sent = ev.sent
        assert len(sent) == 1
        ats = [c for c in sent[0].chain if isinstance(c, At)]
        assert len(ats) == 1 and ats[0].qq == "10001"
        # 与 send 兜底一致：流式渲染同样落审计
        date_str = datetime.now().strftime("%Y%m%d")
        records = (
            audit_file(tmp_path, date_str).read_text(encoding="utf-8").splitlines()
        )
        assert any("at_member" in r and "10001" in r for r in records)

    async def test_streaming_chunk_without_tag_passthrough(self, tmp_path):
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._wrap_event_send(ev)

        async def gen():
            yield MessageChainStub([Plain("普通流式文本")])

        await ev.send_streaming(gen())

        sent = ev.sent
        assert len(sent) == 1
        assert sent[0].chain[0].text == "普通流式文本"

    async def test_send_str_message_with_tag_rendered(self, tmp_path):
        """部分发送路径直接传字符串；含 [at:] 时应渲染为 At 后发出。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._wrap_event_send(ev)

        await ev.send("[at:10001]字符串消息")

        sent = ev.sent
        assert len(sent) == 1
        assert hasattr(sent[0], "chain"), "字符串消息渲染后应转为 MessageChain"
        ats = [c for c in sent[0].chain if isinstance(c, At)]
        assert len(ats) == 1 and ats[0].qq == "10001"

    async def test_send_str_message_without_tag_passthrough(self, tmp_path):
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._wrap_event_send(ev)

        await ev.send("普通字符串")

        assert ev.sent == ["普通字符串"]


    async def test_streaming_fallback_injection(self, tmp_path):
        """流式段无标签 + 兜底缓存存在 → send_streaming 路径补插一次。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._fallback_at[ev.unified_msg_origin] = ("10001", time.time())
        plugin._wrap_event_send(ev)

        async def gen():
            yield MessageChainStub([Plain("流式段无标签")])

        await ev.send_streaming(gen())

        sent = ev.sent
        assert len(sent) == 1
        ats = [c for c in sent[0].chain if isinstance(c, At)]
        assert len(ats) == 1 and ats[0].qq == "10001", "流式 send 路径必须兜底补插"
        assert ev.unified_msg_origin not in plugin._fallback_at, "缓存一次性消费"


# --------------------------------------------------------------------------- #
# 兜底补插语义回归：send 路径补插 + 事件级防重复（分段清链丢艾特修复）
# --------------------------------------------------------------------------- #
class TestP1_1FallbackSemantics:
    """09-07 修复后语义：send/send_streaming 路径可兜底补插一次并消费缓存，
    事件级标记 _attool_send_at_done 保证同一事件至多一个艾特。"""

    async def test_segmented_send_no_duplicate_at(self, tmp_path):
        """分段插件逐段 send：段1无标签 → 补插一次；段2含标签 → 被剥离。

        旧语义下段1无标签不补插、段2含标签渲染一次，看似无重复，但模型
        漏写标签时艾特全丢。新语义：段1补插（消费缓存）后，段2的 [at:ID]
        剥离为纯文本，全局仍恰好一次 At 且不会丢。
        """
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._fallback_at[ev.unified_msg_origin] = ("10001", time.time())
        plugin._wrap_event_send(ev)

        seg1 = MessageChainStub([Plain("第一段无标签")])
        seg2 = MessageChainStub([Plain("第二段[at:10001]喊人")])
        await ev.send(seg1)
        await ev.send(seg2)

        # 段1：无标签 → send 路径兜底补插一次（消费缓存）
        ats1 = [c for c in seg1.chain if isinstance(c, At)]
        assert len(ats1) == 1 and ats1[0].qq == "10001", "段1兜底补插一次"
        # 段2：事件已送出艾特 → 标签剥离为纯文本，不得重复艾特
        assert not any(isinstance(c, At) for c in seg2.chain)
        assert not any(
            "[at:" in c.text for c in seg2.chain if isinstance(c, Plain)
        ), "段2标签应被剥离"
        # 全局合计仅一次 At（重复艾特回归点）
        total = sum(1 for c in seg1.chain if isinstance(c, At)) + sum(
            1 for c in seg2.chain if isinstance(c, At)
        )
        assert total == 1, "同一成员不得被重复艾特"
        assert ev.unified_msg_origin not in plugin._fallback_at, "缓存一次性消费"

    async def test_main_hook_no_double_inject_after_send_fallback(self, tmp_path):
        """send 兜底已补插消费后，主钩子不再重复补插。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._fallback_at[ev.unified_msg_origin] = ("10001", time.time())
        plugin._wrap_event_send(ev)

        seg1 = MessageChainStub([Plain("第一段无标签")])
        await ev.send(seg1)
        assert ev.unified_msg_origin not in plugin._fallback_at, "缓存已被 send 路径消费"

        # 主钩子执行：缓存已消费 → 无标签链不再补插
        ev.set_chain([Plain("第一段无标签")])
        await plugin.process_at_tags(ev)
        chain = ev.get_result().chain
        assert not any(isinstance(c, At) for c in chain), "不得重复补插"
        assert ev.unified_msg_origin not in plugin._fallback_at

    async def test_main_hook_consumes_cache_when_has_tag(self, tmp_path):
        """回复已含标签：主钩子消费清理 send 路径保留的缓存（防跨轮次误用）。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain("回复带标签[at:10002]了")])
        plugin._fallback_at[ev.unified_msg_origin] = ("10001", time.time())
        await plugin.process_at_tags(ev)

        ats = [c for c in ev.get_result().chain if isinstance(c, At)]
        assert len(ats) == 1 and ats[0].qq == "10002"
        assert ev.unified_msg_origin not in plugin._fallback_at, (
            "含标签时主钩子必须清理缓存，否则下一轮无标签回复会误补插上一轮成员"
        )

    async def test_main_hook_clears_cache_when_chain_empty(self, tmp_path):
        """链空（分段插件清链直发后主钩子执行）：必须解除残留兜底缓存。

        分段插件清空 result.chain 后直接 event.send()，补插义务已由 send
        路径兜底履行并消费缓存（09-07 修复）；主钩子执行时链已空，此处
        再清理一次防残留（如 send 路径未消费的极端场景）。若链空时不清理，
        缓存将残留满 TTL(120s)，新一轮 LLM 回复无标签时会误补插上一轮成员
        （跨轮次误艾特）。
        """
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[])
        plugin._fallback_at[ev.unified_msg_origin] = ("10001", time.time())
        await plugin.process_at_tags(ev)

        assert ev.unified_msg_origin not in plugin._fallback_at, (
            "主钩子链空时无法履行兜底义务，必须解除缓存，"
            "否则新一轮无标签回复会误补插上一轮成员"
        )
        # 链空时不得产生任何补插（result.chain 保持空）
        assert ev.get_result().chain == []

    async def test_main_hook_clears_cache_when_result_none(self, tmp_path):
        """result 为 None（极端异常场景）：同样解除兜底缓存，不崩溃。"""
        plugin = make_plugin(audit_dir=tmp_path)

        class NoResultEvent:
            unified_msg_origin = "aiocqhttp:GroupMessage:123456"

            def get_result(self):
                return None

        ev = NoResultEvent()
        plugin._fallback_at[ev.unified_msg_origin] = ("10001", time.time())
        await plugin.process_at_tags(ev)

        assert ev.unified_msg_origin not in plugin._fallback_at, (
            "result 为空时同样必须解除缓存，防止跨轮次误用"
        )


# --------------------------------------------------------------------------- #
# P1-2 回归：宽松恢复被切碎/污染的 [at:ID]（_recover_loose_at_tags）
# --------------------------------------------------------------------------- #
class TestP1_2LooseRecovery:
    async def test_zwsp_polluted_tag_recovered_in_main_hook(
        self, monkeypatch, tmp_path
    ):
        """切碎形态①：Plain 含零宽字符污染标签内部（"[at:12\\u200b345]"）。

        finalize 清洗零宽后完整标签重新暴露，宽松恢复原位替换为 At 组件，
        且产生告警日志。
        """
        import astrbot_plugin_AtTool.main as main_mod

        warnings = []
        monkeypatch.setattr(
            main_mod.logger, "warning", lambda *a, **k: warnings.append(a)
        )

        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain("开头[at:12\u200b345]结尾")])
        await plugin.process_at_tags(ev)

        chain = ev.get_result().chain
        ats = [c for c in chain if isinstance(c, At)]
        assert len(ats) == 1 and ats[0].qq == "12345", "宽松恢复必须还原完整 QQ"
        # 无 "[at:" 裸文本残留
        assert not any("[at:" in c.text for c in chain if isinstance(c, Plain))
        # 上下文文本按原序保留
        texts = "".join(c.text for c in chain if isinstance(c, Plain))
        assert "开头" in texts and "结尾" in texts
        assert any("宽松恢复" in str(w) for w in warnings), "必须产生宽松恢复告警"

    async def test_zwsp_polluted_tag_recovered_in_send_path(
        self, monkeypatch, tmp_path
    ):
        """切碎形态①（send 兜底路径）：零宽污染标签同样被宽松恢复。"""
        import astrbot_plugin_AtTool.main as main_mod

        warnings = []
        monkeypatch.setattr(
            main_mod.logger, "warning", lambda *a, **k: warnings.append(a)
        )

        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._wrap_event_send(ev)

        seg = MessageChainStub([Plain("分段[at:12\u200b345]喊人")])
        await ev.send(seg)

        ats = [c for c in seg.chain if isinstance(c, At)]
        assert len(ats) == 1 and ats[0].qq == "12345"
        assert not any("[at:" in c.text for c in seg.chain if isinstance(c, Plain))
        assert any("宽松恢复" in str(w) for w in warnings), "必须产生宽松恢复告警"

    async def test_cross_component_split_tag_recovered(self, monkeypatch, tmp_path):
        """切碎形态②：非相邻 Plain 跨组件切碎（中间夹 At 组件）。

        链 = Plain("[at:12") + At(999) + Plain("345]")：两个 Plain 被 At
        隔开，步骤①的相邻 Plain 合并无法修复。需求语义：宽松恢复应把
        重新完整暴露的 [at:12345] 原位恢复为 At(qq=12345)，无 "[at:" 裸
        文本残留，并产生告警日志。
        """
        import astrbot_plugin_AtTool.main as main_mod

        warnings = []
        monkeypatch.setattr(
            main_mod.logger, "warning", lambda *a, **k: warnings.append(a)
        )

        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain("开头[at:12"), At(qq="999"), Plain("345]结尾")])
        await plugin.process_at_tags(ev)

        chain = ev.get_result().chain
        ats = [c for c in chain if isinstance(c, At)]
        assert any(a.qq == "12345" for a in ats), "切碎的 [at:12345] 必须恢复为 At"
        assert not any("[at:" in c.text for c in chain if isinstance(c, Plain)), (
            "恢复后不得残留 [at: 裸文本"
        )
        assert any("宽松恢复" in str(w) for w in warnings), "必须产生宽松恢复告警"


# --------------------------------------------------------------------------- #
# 残缺标签容错与告警（任务第 6 点）
# --------------------------------------------------------------------------- #
class TestBrokenTagTolerance:
    async def test_broken_tag_warns_and_keeps_text(self, monkeypatch, tmp_path):
        """未闭合标签（[at:12345 无 ]）：告警 + 按纯文本保留，不崩溃。"""
        import astrbot_plugin_AtTool.main as main_mod

        warnings = []
        monkeypatch.setattr(
            main_mod.logger, "warning", lambda *a, **k: warnings.append(a)
        )

        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain("文本[at:12345")])
        await plugin.process_at_tags(ev)

        chain = ev.get_result().chain
        assert not any(isinstance(c, At) for c in chain)
        assert any("文本[at:12345" in c.text for c in chain if isinstance(c, Plain))
        assert any("[at:" in str(w) for w in warnings), "必须产生告警日志"

    async def test_broken_tag_send_path_warns(self, monkeypatch, tmp_path):
        """send 兜底路径同样告警未闭合标签。"""
        import astrbot_plugin_AtTool.main as main_mod

        warnings = []
        monkeypatch.setattr(
            main_mod.logger, "warning", lambda *a, **k: warnings.append(a)
        )

        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._wrap_event_send(ev)

        seg = MessageChainStub([Plain("文本[at:12345尾巴")])
        await ev.send(seg)

        assert len(ev.sent) == 1, "残缺标签不影响消息发出"
        assert not any(isinstance(c, At) for c in seg.chain)
        assert any("[at:" in str(w) for w in warnings), "必须产生告警日志"
