"""P4 延迟渲染 + send 兜底渲染回归测试。

背景（T-SRE-20260809-01 放大缺陷）：AtTool 的 process_at_tags 钩子
priority=2（最先执行）把 [at:ID] 渲染为 At 组件后，任何后执行（priority<2）
的插件若重建/清空消息链（result.chain.clear()），At 组件即被丢弃。

P4 改造：
1. process_at_tags priority 2 → -1000（AstrBot 钩子按 -priority 降序执行，
   使其在全部 on_decorating_result 钩子之后执行，延迟到最后渲染）。
2. 新增 send 兜底渲染（_wrap_event_send）：针对"清空 chain 后直接
   event.send()"的插件（如分段插件），在消息真正发往平台前完成渲染。

P0-2 send 路径收敛（分段互操作收口）：
send / send_streaming 路径只做"链内已有标签 → 渲染 At"，**不补插、不消费
兜底缓存、不剥离其它段的标签**——补插与缓存消费只发生在主钩子
process_at_tags（实测早于 splitter 的 -1e17 执行），否则分段插件逐段 send
时会出现 "@ 被补到第一段""后续段真实标签被当重复剥离""换人艾特"。

本文件覆盖：
- 验收标准 1：模拟"重建 chain 只保留 Plain 文本"的恶意/第三方插件场景，
  断言最终消息链仍含 At 组件、QQ 号正确；
- 分段插件模式（clear chain + 直接 send）下艾特仍生效；
- send 兜底渲染的不变量：会话准入降级 / @全体冷却 / 审计，与主钩子一致；
- send 路径收敛（不补插/不消费/不剥离）与分段互操作回归；
- 标签形态容错（大小写/空白/全角/非数字载荷/未闭合）、copy-on-write、
  审计去重、包装幂等与异常回退。

运行方式（插件目录下）：
    python3 -m pytest tests/test_delayed_render.py -v
"""

from __future__ import annotations

import inspect
import json
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
        assert len(ats) == 1 and str(ats[0].qq) == "10001", "艾特必须保留且 QQ 号正确"
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
        assert len(ats) == 1 and str(ats[0].qq) == "10002"

    async def test_adjacent_plain_merge_repairs_split_tag(self, tmp_path):
        """链内跨组件残缺标签（[at:12 + 345] 分处相邻 Plain）合并后仍可渲染。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain("开头[at:12"), Plain("345]结尾")])
        await plugin.process_at_tags(ev)

        ats = [c for c in ev.get_result().chain if isinstance(c, At)]
        assert len(ats) == 1 and str(ats[0].qq) == "12345"

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
        assert len(ats) == 1 and str(ats[0].qq) == "10001"

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
        assert any(isinstance(c, At) and str(c.qq) == "10001" for c in seg1.chain)
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
        assert any(isinstance(c, At) and str(c.qq) == "10001" for c in seg.chain)

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
        assert any(isinstance(c, At) and str(c.qq) == "10001" for c in seg.chain)


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
        """send 路径不补插：分段直发 + 模型漏写标签时由主钩子兜底（P0-2）。

        P0-2 收敛语义：send/send_streaming 只渲染"链内已有标签"，补插与
        缓存消费只发生在主钩子 process_at_tags。send 路径补插会把 @ 补到
        第一段（分段插件逐段直发时），因此必须禁止。
        """
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._fallback_at[ev.unified_msg_origin] = ("10001", "10001", time.time())
        plugin._wrap_event_send(ev)

        seg = MessageChainStub([Plain("回复没带标签")])
        await ev.send(seg)

        assert not any(isinstance(c, At) for c in seg.chain), "send 路径不得补插"
        assert seg.chain[0].text == "回复没带标签"
        assert ev.unified_msg_origin in plugin._fallback_at, "send 路径不得消费缓存"

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
        assert any(isinstance(c, At) and str(c.qq) == "all" for c in seg1.chain)

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
        assert len(ats) == 1 and str(ats[0].qq) == "10001"
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
        assert len(ats) == 1 and str(ats[0].qq) == "10001"

    async def test_send_str_message_without_tag_passthrough(self, tmp_path):
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._wrap_event_send(ev)

        await ev.send("普通字符串")

        assert ev.sent == ["普通字符串"]


    async def test_streaming_fallback_injection(self, tmp_path):
        """流式段无标签 + 兜底缓存存在 → 流式路径同样不补插（P0-2）。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._fallback_at[ev.unified_msg_origin] = ("10001", "10001", time.time())
        plugin._wrap_event_send(ev)

        async def gen():
            yield MessageChainStub([Plain("流式段无标签")])

        await ev.send_streaming(gen())

        sent = ev.sent
        assert len(sent) == 1
        assert not any(isinstance(c, At) for c in sent[0].chain), (
            "流式 send 路径不得补插"
        )
        assert sent[0].chain[0].text == "流式段无标签"
        assert ev.unified_msg_origin in plugin._fallback_at, "不得消费兜底缓存"


# --------------------------------------------------------------------------- #
# 兜底补插语义回归：补插 + 缓存消费只发生在主钩子（P0-2 收敛）
# --------------------------------------------------------------------------- #
class TestP1_1FallbackSemantics:
    """P0-2 收敛语义：send/send_streaming 路径只渲染"链内已有标签"，
    补插与兜底缓存消费只由主钩子 process_at_tags 承担（其实测早于
    splitter 的 -1e17 执行），杜绝跨段错位与换人艾特。"""

    async def test_segmented_send_no_duplicate_at(self, tmp_path):
        """分段插件逐段 send：段1无标签不补插；段2的真实标签正常渲染。

        旧语义（09-07）下段1补插并消费缓存、段2标签被剥离，导致 @ 被补到
        第一段、后续段的真实标签被当"重复"删除（探针场景 3/4）。收敛后由
        主钩子统一补插（补插在链尾 → splitter「跟随下段」落到最后一段），
        send 路径只负责渲染本段已有标签。
        """
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._fallback_at[ev.unified_msg_origin] = ("10001", "10001", time.time())
        plugin._wrap_event_send(ev)

        seg1 = MessageChainStub([Plain("第一段无标签")])
        seg2 = MessageChainStub([Plain("第二段[at:10001]喊人")])
        await ev.send(seg1)
        await ev.send(seg2)

        # 段1：无标签 → send 路径不补插（补插只由主钩子承担）
        assert not any(isinstance(c, At) for c in seg1.chain), "段1不得被补插"
        # 段2：链内标签必须渲染，不得被当"重复"剥离
        ats2 = [c for c in seg2.chain if isinstance(c, At)]
        assert len(ats2) == 1 and str(ats2[0].qq) == "10001", "段2标签必须渲染"
        assert not any(
            "[at:" in c.text for c in seg2.chain if isinstance(c, Plain)
        ), "段2标签不得被剥离为裸文本"
        # 缓存保留，交由主钩子消费
        assert ev.unified_msg_origin in plugin._fallback_at, "缓存不得被 send 路径消费"

    async def test_main_hook_consumes_cache_when_has_tag(self, tmp_path):
        """回复已含标签：主钩子消费清理 send 路径保留的缓存（防跨轮次误用）。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain("回复带标签[at:10002]了")])
        plugin._fallback_at[ev.unified_msg_origin] = ("10001", "10001", time.time())
        await plugin.process_at_tags(ev)

        ats = [c for c in ev.get_result().chain if isinstance(c, At)]
        assert len(ats) == 1 and str(ats[0].qq) == "10002"
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
        plugin._fallback_at[ev.unified_msg_origin] = ("10001", "10001", time.time())
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
        plugin._fallback_at[ev.unified_msg_origin] = ("10001", "10001", time.time())
        await plugin.process_at_tags(ev)

        assert ev.unified_msg_origin not in plugin._fallback_at, (
            "result 为空时同样必须解除缓存，防止跨轮次误用"
        )


# --------------------------------------------------------------------------- #
# P1-1/P1-2 回归：被切碎/污染的 [at:ID] 全部恢复（标签内零宽、跨组件切碎）
# --------------------------------------------------------------------------- #
class TestP1_2LooseRecovery:
    async def test_zwsp_polluted_tag_recovered_in_main_hook(self, tmp_path):
        """切碎形态①：Plain 含零宽字符污染标签内部（"[at:12\\u200b345]"）。

        P0-1 起标签载荷解析统一剔除零宽字符 → 直接渲染为 At，且无裸文本残留。
        """
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain("开头[at:12\u200b345]结尾")])
        await plugin.process_at_tags(ev)

        chain = ev.get_result().chain
        ats = [c for c in chain if isinstance(c, At)]
        assert len(ats) == 1 and str(ats[0].qq) == "12345", "零宽污染标签必须还原完整 QQ"
        # 无 "[at:" 裸文本残留
        assert not any("[at:" in c.text for c in chain if isinstance(c, Plain))
        # 上下文文本按原序保留
        texts = "".join(c.text for c in chain if isinstance(c, Plain))
        assert "开头" in texts and "结尾" in texts

    async def test_zwsp_polluted_tag_recovered_in_send_path(self, tmp_path):
        """切碎形态①（send 兜底路径）：零宽污染标签同样被恢复渲染。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._wrap_event_send(ev)

        seg = MessageChainStub([Plain("分段[at:12\u200b345]喊人")])
        await ev.send(seg)

        ats = [c for c in seg.chain if isinstance(c, At)]
        assert len(ats) == 1 and str(ats[0].qq) == "12345"
        assert not any("[at:" in c.text for c in seg.chain if isinstance(c, Plain))

    async def test_two_polluted_tags_both_recovered(self, tmp_path):
        """P1-1：同一链内两处零宽污染标签必须全部恢复（不复现"只救一个"）。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain("[at:11\u200b1] 和 [at:22\u200b2]")])
        await plugin.process_at_tags(ev)

        chain = ev.get_result().chain
        assert [str(c.qq) for c in chain if isinstance(c, At)] == ["111", "222"]
        assert not any("[at:" in c.text for c in chain if isinstance(c, Plain))

    async def test_many_broken_tags_terminates(self, tmp_path):
        """P1-1：大量残缺/污染标签在同一链内不会死循环，且全部得到处理。"""
        plugin = make_plugin(audit_dir=tmp_path)
        raw = "".join(f"[at:{i}\u200b00]尾巴" for i in range(1, 9))
        ev = FakeEvent(chain=[Plain(raw)])
        await plugin.process_at_tags(ev)

        chain = ev.get_result().chain
        qqs = [str(c.qq) for c in chain if isinstance(c, At)]
        assert qqs == [f"{i}00" for i in range(1, 9)]
        assert not any("[" in c.text for c in chain if isinstance(c, Plain))

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
        assert any(str(a.qq) == "12345" for a in ats), "切碎的 [at:12345] 必须恢复为 At"
        assert not any("[at:" in c.text for c in chain if isinstance(c, Plain)), (
            "恢复后不得残留 [at: 裸文本"
        )
        assert any("宽松恢复" in str(w) for w in warnings), "必须产生宽松恢复告警"


# --------------------------------------------------------------------------- #
# 残缺标签容错与告警（任务第 6 点）
# --------------------------------------------------------------------------- #
class TestBrokenTagTolerance:
    async def test_broken_tag_warns_and_keeps_text(self, monkeypatch, tmp_path):
        """未闭合标签（[at:12345 无 ]）：删除起始语法、保留正文，并告警。

        spec §7.2 第 15 行冻结语义：仅删除标签语法片段（[at:/[At：），
        保留其余正文，绝不把裸标签语法留在交付链里（不变量 N1）。
        """
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
        assert "".join(c.text for c in chain if isinstance(c, Plain)) == "文本12345"
        assert not any("[at:" in c.text for c in chain if isinstance(c, Plain))
        assert any("未闭合" in str(w) for w in warnings), "必须产生告警日志"

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


# --------------------------------------------------------------------------- #
# P0-2 send 路径收敛：不补插 / 不消费缓存 / 不剥离其它段标签 / 仍渲染链内标签
# --------------------------------------------------------------------------- #
class TestSendPathContraction:
    """send 与 send_streaming 只做"链内已有标签 → At 渲染"（P0-2）。"""

    async def test_send_path_never_injects_fallback(self, tmp_path):
        """兜底缓存存在但本段无标签：send 路径不得补插（补插只由主钩子承担）。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._fallback_at[ev.unified_msg_origin] = ("10001", "10001", time.time())
        plugin._wrap_event_send(ev)

        seg = MessageChainStub([Plain("第一段文字")])
        await ev.send(seg)

        assert not any(isinstance(c, At) for c in seg.chain)
        assert seg.chain[0].text == "第一段文字"

    async def test_send_path_never_strips_other_segment_tag(self, tmp_path):
        """后续段的真实标签必须原样渲染，不得被当"重复"剥离（换人艾特回归）。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._wrap_event_send(ev)

        seg1 = MessageChainStub([Plain("第一段[at:10001]喊人")])
        seg2 = MessageChainStub([Plain("第二段[at:10002]再喊一个")])
        await ev.send(seg1)
        await ev.send(seg2)

        assert [str(c.qq) for c in seg1.chain if isinstance(c, At)] == ["10001"]
        assert [str(c.qq) for c in seg2.chain if isinstance(c, At)] == ["10002"], (
            "第二段真实标签不得被剥离"
        )
        assert not any(
            "[at:" in c.text for c in seg1.chain + seg2.chain if isinstance(c, Plain)
        )

    async def test_send_path_does_not_consume_cache(self, tmp_path):
        """send 路径不得消费兜底缓存（缓存必须留给主钩子）。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._fallback_at[ev.unified_msg_origin] = ("10001", "10001", time.time())
        plugin._wrap_event_send(ev)

        await ev.send(MessageChainStub([Plain("无标签段")]))
        assert ev.unified_msg_origin in plugin._fallback_at

        # 主钩子随后仍能履行补插义务
        ev.set_chain([Plain("整链最后渲染")])
        await plugin.process_at_tags(ev)
        assert [str(c.qq) for c in ev.get_result().chain if isinstance(c, At)] == ["10001"]

    async def test_send_path_still_renders_existing_tag(self, tmp_path):
        """链内已有标签：send 路径必须渲染（不得被剥离）。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._wrap_event_send(ev)

        seg = MessageChainStub([Plain("文本[At:10001]尾巴")])
        await ev.send(seg)

        assert [str(c.qq) for c in seg.chain if isinstance(c, At)] == ["10001"]
        assert not any("[at:" in c.text for c in seg.chain if isinstance(c, Plain))


# --------------------------------------------------------------------------- #
# P0-1 标签形态不变量：主钩子与 send 路径同语义（单一解析实现）
# --------------------------------------------------------------------------- #
class TestTagFormInvariant:
    async def test_case_and_space_and_fullwidth_forms_render(self, tmp_path):
        """大小写/空格/全角冒号/零宽形态全部渲染为 At。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(
            chain=[Plain("[At:1] [AT:2] [at: 3] [at：4] [at : 5] [at\u200b:6]")]
        )
        await plugin.process_at_tags(ev)

        chain = ev.get_result().chain
        assert [str(c.qq) for c in chain if isinstance(c, At)] == [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
        ]
        assert not any("[at:" in c.text.lower() for c in chain if isinstance(c, Plain))

    async def test_non_numeric_payload_degrades_to_plain_at(
        self, monkeypatch, tmp_path
    ):
        """非数字载荷降级为纯文本 @载荷，并留 warning（绝不原样穿链）。"""
        import astrbot_plugin_AtTool.main as main_mod

        warnings = []
        monkeypatch.setattr(
            main_mod.logger, "warning", lambda *a, **k: warnings.append(a)
        )

        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain("[at:柴郡]和[at:ID]")])
        await plugin.process_at_tags(ev)

        chain = ev.get_result().chain
        assert not any(isinstance(c, At) for c in chain)
        assert "".join(c.text for c in chain if isinstance(c, Plain)) == "@柴郡和@ID"
        assert any("无法解析" in str(w) for w in warnings)

    async def test_fullwidth_digits_never_become_at(self, tmp_path):
        """全角数字不得产生 At 组件（Python \\d 陷阱回归）。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain("[at:１２３]")])
        await plugin.process_at_tags(ev)

        chain = ev.get_result().chain
        assert not any(isinstance(c, At) for c in chain)
        assert "".join(c.text for c in chain if isinstance(c, Plain)) == "@１２３"

    async def test_unclosed_tag_fragment_removed_keeps_text(self, tmp_path):
        """未闭合标签：删除起始语法、保留正文（spec §7.2 第 15 行）。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain("前面[at:123")])
        await plugin.process_at_tags(ev)

        chain = ev.get_result().chain
        assert not any(isinstance(c, At) for c in chain)
        assert "".join(c.text for c in chain if isinstance(c, Plain)) == "前面123"
        assert not any("[at:" in c.text for c in chain if isinstance(c, Plain))

    async def test_main_hook_and_send_path_agree(self, tmp_path):
        """同一形态在主钩子与 send 路径给出相同结果（P0-5 一致性）。"""
        text = "A[At:1]B[at：2]C[at:柴郡]D[at:]E[At：3"

        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain(text)])
        await plugin.process_at_tags(ev)

        plugin2 = make_plugin(audit_dir=tmp_path)
        ev2 = SendableEvent(chain=[])
        plugin2._wrap_event_send(ev2)
        await ev2.send(MessageChainStub([Plain(text)]))

        def summarize(chain):
            return (
                [str(c.qq) for c in chain if isinstance(c, At)],
                [c.text.replace("\u200b", "") for c in chain if isinstance(c, Plain)],
            )

        assert summarize(ev.get_result().chain) == summarize(ev2.sent[0].chain)


# --------------------------------------------------------------------------- #
# 分段插件互操作 harness（spec §4.5）：前 n-1 段直发 + 最后一段经 event.send
# --------------------------------------------------------------------------- #
def _simulate_splitter(chain, split_char="\n"):
    """模拟 astrbot_plugin_splitter 的分段结果（at_strategy=跟随下段）。

    Args:
        chain: 已由 AtTool 主钩子渲染过的消息链。
        split_char: 分段字符（splitter 按换行/句末切分）。

    Returns:
        段链列表：At 组件跟随下一段，正文按 split_char 切分。
    """
    segments = [[]]
    pending = []
    for comp in chain:
        if isinstance(comp, Plain) and split_char in comp.text:
            for i, piece in enumerate(comp.text.split(split_char)):
                if i:
                    segments.append([])
                if pending:
                    segments[-1].extend(pending)
                    pending = []
                if piece:
                    segments[-1].append(Plain(piece))
        elif isinstance(comp, At):
            pending.append(comp)  # 跟随下段
        else:
            if pending:
                segments[-1].extend(pending)
                pending = []
            segments[-1].append(comp)
    if pending:
        segments[-1].extend(pending)
    return [seg for seg in segments if seg]


async def _deliver_segments(plugin, ev, chain):
    """按 splitter 行为投递：前 n-1 段直发（绕过包装），最后一段经 event.send。"""
    segments = _simulate_splitter(chain)
    delivered = list(segments[:-1])  # context.send_message 直连平台，包装看不到
    last = segments[-1]
    ev.set_chain(last)
    plugin._wrap_event_send(ev)
    await ev.send(MessageChainStub(last))
    delivered.append(last)
    return delivered


class TestSplitterInterop:
    async def test_splitter_segments_exactly_one_at(self, tmp_path):
        """§4.5 五条断言：分段后恰好一次真实艾特、无裸标签、不换人。"""
        # 场景 1+5：模型写 [at:123] → 恰好一次 At(123)，全程无裸标签
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[Plain("第一段无标签\n第二段[at:123]喊人\n第三段")])
        await plugin.process_at_tags(ev)
        delivered = await _deliver_segments(plugin, ev, ev.get_result().chain)
        assert sum(
            1 for seg in delivered for c in seg if isinstance(c, At) and str(c.qq) == "123"
        ) == 1
        assert sum(1 for seg in delivered for c in seg if isinstance(c, At)) == 1
        for seg in delivered:
            for comp in seg:
                assert not (
                    isinstance(comp, Plain) and "[at:" in comp.text.lower()
                ), f"裸标签泄漏: {comp.text!r}"

        # 场景 3：模型写大写 [At:123] → 同样恰好一次 At(123) 且无 "[At:" 明文
        plugin2 = make_plugin(audit_dir=tmp_path)
        ev2 = SendableEvent(chain=[Plain("甲段\n乙段[At:123]喊人\n丙段")])
        await plugin2.process_at_tags(ev2)
        delivered2 = await _deliver_segments(plugin2, ev2, ev2.get_result().chain)
        assert [str(c.qq) for seg in delivered2 for c in seg if isinstance(c, At)] == ["123"]
        for seg in delivered2:
            for comp in seg:
                assert not (isinstance(comp, Plain) and "[at:" in comp.text.lower())

        # 场景 2：模型漏写标签 + 唯一命中缓存 → 恰好一次 At，位置在最后一段
        plugin3 = make_plugin(audit_dir=tmp_path)
        ev3 = SendableEvent(chain=[Plain("一段没有标签\n二段也没有\n三段")])
        plugin3._fallback_at[ev3.unified_msg_origin] = ("10001", "10001", time.time())
        await plugin3.process_at_tags(ev3)
        delivered3 = await _deliver_segments(plugin3, ev3, ev3.get_result().chain)
        assert sum(1 for seg in delivered3 for c in seg if isinstance(c, At)) == 1
        assert not any(
            isinstance(c, At) for seg in delivered3[:-1] for c in seg
        ), "补插位置必须在最后一段（splitter「跟随下段」）"
        assert [str(c.qq) for c in delivered3[-1] if isinstance(c, At)] == ["10001"]

        # 场景 4：模型写 [at:999] 而缓存是 123（不同人）→ 只出现 At(999)
        plugin4 = make_plugin(audit_dir=tmp_path)
        ev4 = SendableEvent(chain=[Plain("一段[at:999]喊人\n二段收尾")])
        plugin4._fallback_at[ev4.unified_msg_origin] = ("10001", "123", time.time())
        await plugin4.process_at_tags(ev4)
        delivered4 = await _deliver_segments(plugin4, ev4, ev4.get_result().chain)
        assert [str(c.qq) for seg in delivered4 for c in seg if isinstance(c, At)] == ["999"]


# --------------------------------------------------------------------------- #
# P1-3 输入组件不可变（copy-on-write）
# --------------------------------------------------------------------------- #
class TestChainMutationIsolation:
    async def test_input_components_not_mutated(self, tmp_path):
        """渲染/收尾不得原地改写调用方共享的 Plain 组件。"""
        plugin = make_plugin(audit_dir=tmp_path)
        originals = [Plain("前置 \u200b[at:10001]"), Plain(" 后置\u200b")]
        ev = FakeEvent(chain=list(originals))
        await plugin.process_at_tags(ev)

        assert originals[0].text == "前置 \u200b[at:10001]"
        assert originals[1].text == " 后置\u200b"
        assert [str(c.qq) for c in ev.get_result().chain if isinstance(c, At)] == ["10001"]


# --------------------------------------------------------------------------- #
# P1-4 审计去重：同一事件同一目标只记一条
# --------------------------------------------------------------------------- #
class TestAuditDedup:
    async def test_same_target_audited_once_per_event(self, tmp_path):
        """同一事件内同目标重复渲染只审计一次（第三方插件重建链场景）。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain("喊人[at:10001]")])
        await plugin.process_at_tags(ev)
        assert [str(c.qq) for c in ev.get_result().chain if isinstance(c, At)] == ["10001"]

        # 第三方插件用原文本重建链 → 主钩子（或 send）再渲染一次同一目标
        ev.set_chain([Plain("喊人[at:10001]")])
        await plugin.process_at_tags(ev)

        records = [
            json.loads(line)
            for line in audit_file(tmp_path, datetime.now().strftime("%Y%m%d"))
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        assert len([r for r in records if r["target_id"] == "10001"]) == 1

    async def test_at_all_audited_once_per_event(self, tmp_path):
        """P1-4：@全体 同一事件内重复渲染同样只审计一次。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain("[at:all]集合")])
        ev.bot.member_info_role = "owner"
        await plugin.process_at_tags(ev)

        ev.set_chain([Plain("[at:all]集合")])
        await plugin.process_at_tags(ev)

        records = [
            json.loads(line)
            for line in audit_file(tmp_path, datetime.now().strftime("%Y%m%d"))
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        assert len([r for r in records if r["op_type"] == "at_all"]) == 1
