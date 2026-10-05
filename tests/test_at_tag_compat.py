"""标签形态兼容 e2e 回归（P0-1 / P0-5 + 不变量 N1）。

背景（反馈 2）：宿主序列化与历史上下文会诱导模型写出大写/全角/带空格的
标签形态，而旧正则只认 `[at:数字]` 小写半角形态 → 标签原样穿链发到群里。

本文件按 spec §7.2「e2e 标签形态清单」逐条覆盖端到端行为：
- 可解析载荷（大小写 / 空白 / 全角冒号 / 零宽污染）必须渲染为 At 组件；
- 不可解析载荷（名字、占位符、全角数字）降级为 `@载荷` 纯文本并告警；
- 空载荷删除标签语法；未闭合标签删除起始语法、保留正文；
- 不变量 N1：交付链的任何 Plain 不得再包含标签起始语法；
- 非标签形态（`[at 123]` / `[at]` / `[avatar:1]`）保持原样。

运行方式（插件目录下）：
    python3 -m pytest tests/test_at_tag_compat.py -v
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import pytest

pytestmark = pytest.mark.asyncio

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import test_integration_main as ti  # noqa: E402
from astrbot_plugin_AtTool import main as main_mod  # noqa: E402
from test_delayed_render import MessageChainStub, SendableEvent  # noqa: E402

from astrbot.api.message_components import At, Plain  # noqa: E402

FakeEvent = ti.FakeEvent
make_plugin = ti.make_plugin
audit_file = ti.audit_file

# 不变量 N1（spec §7.1 + r2 评审 F4）：任何 Plain 的 text 不得匹配该正则。
# 比 spec 原文更严：额外覆盖"[" 与 at 之间夹零宽（[\u200bat:123]）的形态。
_NAKED_TAG = re.compile(r"\[\u200b*(?i:at)[\s\u200b]*[:：]")


def _plain_texts(chain) -> list:
    """收集链中全部 Plain 文本。"""
    return [c.text for c in chain if isinstance(c, Plain)]


def _ats(chain) -> list:
    """收集链中全部 At 载荷（按顺序，str 形态：真机 qq 为 int | str）。"""
    return [str(c.qq) for c in chain if isinstance(c, At)]


def assert_n1(chain) -> None:
    """断言不变量 N1：交付链中不存在标签起始语法残留。"""
    for text in _plain_texts(chain):
        assert not _NAKED_TAG.search(text), f"裸标签泄漏: {text!r}"


# spec §7.2 形态清单： (输入, 期望 At 载荷, 期望保留/降级文本片段)
FORM_CASES = [
    ("[at:123]", ["123"], []),
    ("[At:123]", ["123"], []),
    ("[AT:123]", ["123"], []),
    ("[aT:123]", ["123"], []),
    ("[at: 123]", ["123"], []),
    ("[at：123]", ["123"], []),
    ("[at : 123]", ["123"], []),
    ("[at\u200b:123]", ["123"], []),
    ("[at:\u200b123]", ["123"], []),
    ("[\u200bat:123]", ["123"], []),          # F4：'[' 与 at 之间夹零宽
    ("[\u200bAt\u200b:\u200b123\u200b]", ["123"], []),  # F4：零宽多点混排
    ("[at:123456789012]", ["123456789012"], []),   # F5：12 位（合法上界）
    ("[at:1234567890123]", [], ["@1234567890123"]),  # F5：13 位 → 降级
    ("[at:" + "1" * 103 + "]", [], ["@" + "1" * 103]),  # F5：超长载荷 → 降级
    ("[at:柴郡]", [], ["@柴郡"]),
    ("[At: 张三]", [], ["@张三"]),
    ("[at:ID]", [], ["@ID"]),
    ("[at:１２３]", [], ["@１２３"]),
    ("[at:]", [], []),
    ("[at: ]", [], []),
    ("[at 123]", [], ["[at 123]"]),
    ("[at]", [], ["[at]"]),
    ("[avatar:1]", [], ["[avatar:1]"]),
]


class TestTagFormMatrixE2E:
    """形态矩阵端到端：主钩子渲染 + 降级 + 不变量 N1。"""

    @pytest.mark.parametrize("text,expected_ats,expected_frags", FORM_CASES)
    async def test_main_hook_form_matrix(
        self, text, expected_ats, expected_frags, tmp_path
    ):
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain(f"前缀{text}后缀")])
        await plugin.process_at_tags(ev)

        chain = ev.get_result().chain
        assert _ats(chain) == expected_ats
        assert_n1(chain)
        joined = "".join(_plain_texts(chain))
        assert "前缀" in joined and "后缀" in joined
        for frag in expected_frags:
            assert frag in joined, joined

        # F3（r2）：真机 At.toDict() 恒为 str(qq)，且 pydantic 会把可整数化的
        # 载荷归一为 int（"0123"→123、"１２３"→123）——序列化形态必须与
        # 期望载荷逐字一致，否则就是"对象层通过、序列化层变形"
        serialized = [
            c.toDict()["data"]["qq"] for c in chain if isinstance(c, At)
        ]
        assert serialized == expected_ats
        for comp in chain:
            if isinstance(comp, At):
                assert comp.toDict() == {
                    "type": "at",
                    "data": {"qq": str(comp.qq)},
                }
            elif isinstance(comp, Plain):
                assert comp.toDict() == {
                    "type": "text",
                    "data": {"text": comp.text},
                }

    async def test_payload_never_deforms_at_serialization_layer(self, tmp_path):
        """F3（r2）：真机 toDict 会把可整数化载荷归一（`'１２３'`→`'123'`、
        `'0123'`→`'123'`）——这类载荷必须被降级，绝不能进入 At 组件。

        维护级套件此前只有对象层断言（`At.qq`），真机 pydantic 的
        `qq: int | str` 会在序列化前把全角/超长数字归一，属"对象层通过、
        序列化层变形"；本用例按 toDict() 形态把关。
        """
        plugin = make_plugin(audit_dir=tmp_path)
        for raw in (
            "[at:１２３]",  # 全角数字 → int 123
            "[at:1２3]",  # 全角/半角混排
            "[at:1234567890123]",  # 13 位超长（F5）
            "[at:" + "1" * 103 + "]",  # 极端超长（F5）
        ):
            ev = FakeEvent(chain=[Plain(raw)])
            await plugin.process_at_tags(ev)
            chain = ev.get_result().chain
            assert not any(isinstance(c, At) for c in chain), raw
            assert_n1(chain)

    async def test_valid_payload_serializes_verbatim(self, tmp_path):
        """F3：合法载荷（1..12 位 ASCII 数字）序列化后与输入逐字一致。"""
        plugin = make_plugin(audit_dir=tmp_path)
        for payload in ("1", "10001", "123456789012"):
            ev = FakeEvent(chain=[Plain(f"[at:{payload}]")])
            await plugin.process_at_tags(ev)
            ats = [c for c in ev.get_result().chain if isinstance(c, At)]
            assert len(ats) == 1
            assert ats[0].toDict() == {"type": "at", "data": {"qq": payload}}

    async def test_all_tag_permission_and_cooldown_paths(self, tmp_path):
        """[at:all] 全形态走既有 @全体 权限路径（放行 At('all')，拒绝转文本）。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain("[At:All] 集合")])
        ev.bot.member_info_role = "owner"  # 群主：具备 @全体 权限
        await plugin.process_at_tags(ev)
        assert _ats(ev.get_result().chain) == ["all"]
        assert_n1(ev.get_result().chain)

        plugin2 = make_plugin(audit_dir=tmp_path)
        ev2 = FakeEvent(chain=[Plain("[AT：ALL] 集合")])
        ev2.bot.member_info_role = "member"  # 普通成员：无权限
        await plugin2.process_at_tags(ev2)
        chain2 = ev2.get_result().chain
        assert _ats(chain2) == []
        assert "全体成员" in "".join(_plain_texts(chain2))
        assert_n1(chain2)

    async def test_downgrade_payload_written_to_audit_as_plain(self, tmp_path):
        """降级为纯文本的载荷不得写审计（没有真实艾特发生）。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain("看[at:柴郡]")])
        await plugin.process_at_tags(ev)

        log_path = audit_file(tmp_path, datetime.now().strftime("%Y%m%d"))
        assert not log_path.exists(), "降级为纯文本不应产生审计记录"

    async def test_unclosed_tag_syntax_removed_with_warning(
        self, monkeypatch, tmp_path
    ):
        """未闭合标签：删除标签起始语法、保留正文，并留 warning。"""
        warnings = []
        monkeypatch.setattr(
            main_mod.logger, "warning", lambda *a, **k: warnings.append(a)
        )

        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain("正文[At：123")])
        await plugin.process_at_tags(ev)

        chain = ev.get_result().chain
        assert _ats(chain) == []
        assert_n1(chain)
        assert "".join(_plain_texts(chain)) == "正文123"
        assert any("未闭合" in str(w) for w in warnings), warnings

    async def test_downgrade_leaves_warning(self, monkeypatch, tmp_path):
        """不可解析载荷：降级 + warning（可观测，不静默吞掉）。"""
        warnings = []
        monkeypatch.setattr(
            main_mod.logger, "warning", lambda *a, **k: warnings.append(a)
        )

        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain("[at:１２３]")])
        await plugin.process_at_tags(ev)

        assert any("无法解析" in str(w) for w in warnings), warnings

    async def test_cross_component_split_tag_reassembled(self, tmp_path):
        """跨组件切碎（相邻 Plain / 中间夹 At）必须拼回完整标签。"""
        plugin = make_plugin(audit_dir=tmp_path)

        ev = FakeEvent(chain=[Plain("开头[at:12"), Plain("3]结尾")])
        await plugin.process_at_tags(ev)
        assert _ats(ev.get_result().chain) == ["123"]
        assert_n1(ev.get_result().chain)

        ev2 = FakeEvent(chain=[Plain("开头[at:12"), At(qq="888"), Plain("3]结尾")])
        await plugin.process_at_tags(ev2)
        chain2 = ev2.get_result().chain
        assert _ats(chain2) == ["123", "888"]
        assert_n1(chain2)

    async def test_multiple_polluted_tags_all_rendered(self, tmp_path):
        """P1-1：同一链内多处零宽污染标签必须全部恢复（不再只救一个）。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain("[at:11\u200b1] 和 [at:22\u200b2]")])
        await plugin.process_at_tags(ev)

        chain = ev.get_result().chain
        assert _ats(chain) == ["111", "222"]
        assert_n1(chain)

    async def test_nested_tag_payload_never_leaks(self, tmp_path):
        """对抗输入：载荷本身含标签语法时，渲染/降级都不得残留标签语法。"""
        plugin = make_plugin(audit_dir=tmp_path)
        for raw in ("[at:[at:123]", "[at:柴郡[at:123]"):
            ev = FakeEvent(chain=[Plain(raw)])
            await plugin.process_at_tags(ev)
            assert_n1(ev.get_result().chain)

    async def test_send_path_form_matrix(self, tmp_path):
        """send 兜底路径与主钩子形态语义一致（同一份解析实现）。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._wrap_event_send(ev)

        seg = MessageChainStub(
            [Plain("A[At:10001]B[at：10002]C[at:柴郡]D[at:１２３]E")]
        )
        await ev.send(seg)

        chain = seg.chain
        assert _ats(chain) == ["10001", "10002"]
        assert_n1(chain)
        joined = "".join(_plain_texts(chain))
        for frag in ("A", "B", "C", "@柴郡", "D", "@１２３", "E"):
            assert frag in joined, joined

    async def test_send_path_downgrade_not_audited(self, tmp_path):
        """send 路径降级同样不写审计。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._wrap_event_send(ev)

        await ev.send(MessageChainStub([Plain("[at:柴郡]")]))
        log_path = audit_file(tmp_path, datetime.now().strftime("%Y%m%d"))
        assert not log_path.exists()

    async def test_render_does_not_mutate_input_components(self, tmp_path):
        """P1-3：渲染不改写调用方传入的 Plain 对象（copy-on-write）。"""
        plugin = make_plugin(audit_dir=tmp_path)
        origin = Plain("前置 [At:10001] 后置")
        chain = [origin]
        ev = FakeEvent(chain=chain)
        await plugin.process_at_tags(ev)

        assert origin.text == "前置 [At:10001] 后置", "输入 Plain 不得被原地改写"
        assert _ats(ev.get_result().chain) == ["10001"]

    async def test_render_does_not_mutate_input_on_denied_session(self, tmp_path):
        """P1-3 + P1-2：会话禁艾特降级同样不改写输入组件。"""
        plugin = make_plugin(
            config={"session_blacklist": ["123456"]}, audit_dir=tmp_path
        )
        origin = Plain("前置 [At:10001] 后置")
        ev = FakeEvent(chain=[origin])
        await plugin.process_at_tags(ev)

        assert origin.text == "前置 [At:10001] 后置"
        chain = ev.get_result().chain
        # 标签被删除，其两侧空白原样保留（与旧行为一致，只是不再残留语法）
        assert "".join(_plain_texts(chain)) == "前置  后置"
        assert_n1(chain)


class TestStreamingFormRows:
    """流式路径形态（spec §7.2 第 17/18 行）。"""

    async def test_streaming_single_chunk_tag_rendered(self, tmp_path):
        """第 17 行：单 chunk 内完整的标签正常渲染。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._wrap_event_send(ev)

        async def gen():
            yield MessageChainStub([Plain("[At:10001]来啦")])

        await ev.send_streaming(gen())
        assert _ats(ev.sent[0].chain) == ["10001"]
        assert_n1(ev.sent[0].chain)

    async def test_streaming_cross_chunk_tag_preserved_untouched(self, tmp_path):
        """第 18 行（非目标/已知限制）：跨 chunk 拼合的标签不渲染也不误删。

        spec §7.1 明确不变量 N1 只覆盖非流式交付路径；流式 chunk 内残缺的
        标签语法按原文透传，避免把用户可见正文当作残留删除。
        """
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._wrap_event_send(ev)

        async def gen():
            yield MessageChainStub([Plain("[at:12")])
            yield MessageChainStub([Plain("3]")])

        await ev.send_streaming(gen())
        assert _ats(ev.sent[0].chain) == [] and _ats(ev.sent[1].chain) == []
        assert "".join(_plain_texts(ev.sent[0].chain)) == "[at:12"
        assert "".join(_plain_texts(ev.sent[1].chain)) == "3]"
