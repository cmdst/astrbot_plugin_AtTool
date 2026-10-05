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
import time
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
        await ti.run_main_hook(plugin, ev)

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
            await ti.run_main_hook(plugin, ev)
            chain = ev.get_result().chain
            assert not any(isinstance(c, At) for c in chain), raw
            assert_n1(chain)

    async def test_valid_payload_serializes_verbatim(self, tmp_path):
        """F3：合法载荷（1..12 位 ASCII 数字）序列化后与输入逐字一致。"""
        plugin = make_plugin(audit_dir=tmp_path)
        for payload in ("1", "10001", "123456789012"):
            ev = FakeEvent(chain=[Plain(f"[at:{payload}]")])
            await ti.run_main_hook(plugin, ev)
            ats = [c for c in ev.get_result().chain if isinstance(c, At)]
            assert len(ats) == 1
            assert ats[0].toDict() == {"type": "at", "data": {"qq": payload}}

    async def test_all_tag_permission_and_cooldown_paths(self, tmp_path):
        """[at:all] 全形态走既有 @全体 权限路径（放行 At('all')，拒绝转文本）。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain("[At:All] 集合")])
        ev.bot.member_info_role = "owner"  # 群主：具备 @全体 权限
        await ti.run_main_hook(plugin, ev)
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
        await ti.run_main_hook(plugin, ev)

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
        await ti.run_main_hook(plugin, ev)

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
        await ti.run_main_hook(plugin, ev)

        assert any("无法解析" in str(w) for w in warnings), warnings

    async def test_cross_component_split_tag_reassembled(self, tmp_path):
        """跨组件切碎（相邻 Plain / 中间夹 At）必须拼回完整标签。"""
        plugin = make_plugin(audit_dir=tmp_path)

        ev = FakeEvent(chain=[Plain("开头[at:12"), Plain("3]结尾")])
        ti.trust_ids(plugin, ev, "123")
        await ti.run_main_hook(plugin, ev)
        assert _ats(ev.get_result().chain) == ["123"]
        assert_n1(ev.get_result().chain)

        ev2 = FakeEvent(chain=[Plain("开头[at:12"), At(qq="888"), Plain("3]结尾")])
        ti.trust_ids(plugin, ev2, "123")
        await ti.run_main_hook(plugin, ev2)
        chain2 = ev2.get_result().chain
        assert _ats(chain2) == ["123", "888"]
        assert_n1(chain2)

    async def test_multiple_polluted_tags_all_rendered(self, tmp_path):
        """P1-1：同一链内多处零宽污染标签必须全部恢复（不再只救一个）。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain("[at:11\u200b1] 和 [at:22\u200b2]")])
        await ti.run_main_hook(plugin, ev)

        chain = ev.get_result().chain
        assert _ats(chain) == ["111", "222"]
        assert_n1(chain)

    async def test_nested_tag_payload_never_leaks(self, tmp_path):
        """对抗输入：载荷本身含标签语法时，渲染/降级都不得残留标签语法。"""
        plugin = make_plugin(audit_dir=tmp_path)
        for raw in ("[at:[at:123]", "[at:柴郡[at:123]"):
            ev = FakeEvent(chain=[Plain(raw)])
            await ti.run_main_hook(plugin, ev)
            assert_n1(ev.get_result().chain)

    async def test_send_path_form_matrix(self, tmp_path):
        """send 兜底路径与主钩子形态语义一致（同一份解析实现）。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._wrap_event_send(ev)

        seg = MessageChainStub(
            [Plain("A[At:10001]B[at：10002]C[at:柴郡]D[at:１２３]E")]
        )
        ti.trust_ids(plugin, ev, "10001", "10002")
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
        await ti.run_main_hook(plugin, ev)

        assert origin.text == "前置 [At:10001] 后置", "输入 Plain 不得被原地改写"
        assert _ats(ev.get_result().chain) == ["10001"]

    async def test_render_does_not_mutate_input_on_denied_session(self, tmp_path):
        """P1-3 + P1-2：会话禁艾特降级同样不改写输入组件。"""
        plugin = make_plugin(
            config={"session_blacklist": ["123456"]}, audit_dir=tmp_path
        )
        origin = Plain("前置 [At:10001] 后置")
        ev = FakeEvent(chain=[origin])
        await ti.run_main_hook(plugin, ev)

        assert origin.text == "前置 [At:10001] 后置"
        chain = ev.get_result().chain
        # 标签被删除，其两侧空白原样保留（与旧行为一致，只是不再残留语法）
        assert "".join(_plain_texts(chain)) == "前置  后置"
        assert_n1(chain)


class TestTrustedIdSource:
    """T10：只有「工具确认过的 ID」与「用户消息里出现的 QQ 号」才能渲染为 At。

    线上实测（19:39，群 744868236）：模型未调用任何工具却自行输出
    ``[at:2060958352]``（发送者自己的 QQ）与 ``[at:3882563785]``（他人），
    插件照单渲染 → 误伤无关成员。本组用例把「ID 可信来源」钉死。
    """

    # 线上实测的两个 ID：前者=发送者自己（误艾特），后者=群成员但来源不明
    FABRICATED = ("2060958352", "3882563785")

    async def test_fabricated_ids_never_render(self, monkeypatch, tmp_path):
        """编造 ID（无工具调用、用户消息里也没有该号）→ 降级为 @载荷，不渲染。"""
        warnings = []
        monkeypatch.setattr(
            main_mod.logger, "warning", lambda *a, **k: warnings.append(a)
        )

        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(
            chain=[
                Plain(
                    f"好的 [at:{self.FABRICATED[0]}] 还有 [at:{self.FABRICATED[1]}] 来了"
                )
            ]
        )
        ev.message_str = "帮我看看群里谁在"  # 用户消息里没有任何 QQ 号
        await plugin.process_at_tags(ev)

        chain = ev.get_result().chain
        assert not any(isinstance(c, At) for c in chain), "编造 ID 不得渲染成真实艾特"
        joined = "".join(_plain_texts(chain))
        for uid in self.FABRICATED:
            assert f"@{uid}" in joined, joined
        assert any("未经工具确认" in str(w) for w in warnings), warnings
        assert_n1(chain)
        # 没有真实艾特 → 不得写审计
        assert not audit_file(tmp_path, datetime.now().strftime("%Y%m%d")).exists()

    async def test_tool_confirmed_id_renders(self, tmp_path):
        """工具命中过的 ID（search_and_mention 唯一命中）→ 正常渲染。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[])
        ev.message_str = "帮我艾特一下陨落星辰"
        ev.bot.member_list = [
            {
                "user_id": self.FABRICATED[1],
                "nickname": "陨落星辰",
                "card": "",
                "role": "member",
            }
        ]
        text = await plugin.search_and_mention(ev, "陨落星辰")
        assert self.FABRICATED[1] in text

        ev.set_chain([Plain(f"来了 [at:{self.FABRICATED[1]}]")])
        await plugin.process_at_tags(ev)
        assert _ats(ev.get_result().chain) == [self.FABRICATED[1]]
        assert_n1(ev.get_result().chain)

    async def test_user_message_qq_renders_when_direct_allowed(self, tmp_path):
        """用户消息里明确给出的 QQ 号（allow_direct_qq_at=true）→ 正常渲染。"""
        # (a) 来源是 event.message_str（T16 起该号还需是本群成员）
        plugin = make_plugin(config={"allow_direct_qq_at": True}, audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain(f"来了 [at:{self.FABRICATED[1]}]")])
        ev.message_str = f"帮我艾特 {self.FABRICATED[1]} 谢谢"
        ev.bot.member_list = [
            {"user_id": self.FABRICATED[1], "nickname": "陨落星辰", "card": "", "role": "member"}
        ]
        await plugin.process_at_tags(ev)
        assert _ats(ev.get_result().chain) == [self.FABRICATED[1]]

        # (b) 来源是入站消息链里的 Plain 文本
        plugin2 = make_plugin(
            config={"allow_direct_qq_at": True}, audit_dir=tmp_path
        )
        ev2 = FakeEvent(
            chain=[Plain(f"[At:{self.FABRICATED[0]}] 已通知")],
            incoming=[Plain(f"艾特下 {self.FABRICATED[0]}")],
        )
        ev2.message_str = ""  # 入站链存在即视为有用户上下文
        ev2.bot.member_list = [
            {"user_id": self.FABRICATED[0], "nickname": "张三", "card": "", "role": "member"}
        ]
        await plugin2.process_at_tags(ev2)
        assert _ats(ev2.get_result().chain) == [self.FABRICATED[0]]

    async def test_direct_qq_not_trusted_when_switch_off(self, tmp_path):
        """allow_direct_qq_at=false：用户给的号也不可信，必须走工具路径。"""
        plugin = make_plugin(config={"allow_direct_qq_at": False}, audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain(f"来了 [at:{self.FABRICATED[1]}]")])
        ev.message_str = f"帮我艾特 {self.FABRICATED[1]} 谢谢"
        await plugin.process_at_tags(ev)

        chain = ev.get_result().chain
        assert not any(isinstance(c, At) for c in chain), "关闭直连后仍不得凭用户消息渲染"
        assert f"@{self.FABRICATED[1]}" in "".join(_plain_texts(chain))

        # 走工具确认后同样可渲染
        ev.bot.member_list = [
            {
                "user_id": self.FABRICATED[1],
                "nickname": "陨落星辰",
                "card": "",
                "role": "member",
            }
        ]
        await plugin.search_and_mention(ev, "陨落星辰")
        ev.set_chain([Plain(f"来了 [at:{self.FABRICATED[1]}]")])
        await plugin.process_at_tags(ev)
        assert _ats(ev.get_result().chain) == [self.FABRICATED[1]]

    async def test_selected_member_id_renders(self, tmp_path):
        """select_member_by_index 选定过的 ID → 计入可信来源，可渲染。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[])
        ev.message_str = "帮我艾特第二个"
        ev.bot.member_list = [
            {"user_id": "111", "nickname": "张三", "card": "", "role": "member"},
            {"user_id": self.FABRICATED[1], "nickname": "张三丰", "card": "", "role": "member"},
        ]
        listed = await plugin.search_and_mention(ev, "张")
        assert "找到多个" in listed
        selected = await plugin.select_member_by_index(ev, 2)
        assert self.FABRICATED[1] in selected

        ev.set_chain([Plain(f"就是他了 [at:{self.FABRICATED[1]}]")])
        await plugin.process_at_tags(ev)
        assert _ats(ev.get_result().chain) == [self.FABRICATED[1]]

    async def test_short_number_in_message_is_not_a_qq_source(self, tmp_path):
        """用户消息里的 4 位数字不是 QQ 来源（需 5..12 位），不得据此渲染。"""
        plugin = make_plugin(config={"allow_direct_qq_at": True}, audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain("[at:1234]")])
        ev.message_str = "订单号 1234 帮我艾特一下"
        await plugin.process_at_tags(ev)

        chain = ev.get_result().chain
        assert not any(isinstance(c, At) for c in chain)
        assert "@1234" in "".join(_plain_texts(chain))

    async def test_other_sender_write_does_not_downgrade_mine(self, tmp_path):
        """T14/A3-①：乙的工具命中不得覆盖甲的可信 ID，甲仍可渲染为 At。"""
        plugin = make_plugin(audit_dir=tmp_path)
        members = [
            {"user_id": self.FABRICATED[0], "nickname": "张三", "card": "", "role": "member"},
            {"user_id": self.FABRICATED[1], "nickname": "李四", "card": "", "role": "member"},
        ]
        ev_a = FakeEvent(group_id="1000", sender_id="10001", chain=[])
        ev_a.message_str = "帮我艾特张三"
        ev_a.bot.member_list = members
        await plugin.search_and_mention(ev_a, "张三")

        ev_b = FakeEvent(group_id="1000", sender_id="20002", chain=[])
        ev_b.message_str = "帮我艾特李四"
        ev_b.bot.member_list = members
        await plugin.search_and_mention(ev_b, "李四")

        ev_a.set_chain([Plain(f"[at:{self.FABRICATED[0]}]")])
        await plugin.process_at_tags(ev_a)
        assert _ats(ev_a.get_result().chain) == [self.FABRICATED[0]], (
            "乙写入后甲的工具确认结果必须仍有效（T14/A3：各持一条）"
        )

    async def test_other_sender_write_keeps_my_fallback_injection(self, tmp_path):
        """T14/A3-②：乙写入后，甲的兜底补插仍能渲染出 At（不丢艾特）。"""
        plugin = make_plugin(audit_dir=tmp_path)
        members = [
            {"user_id": self.FABRICATED[0], "nickname": "张三", "card": "", "role": "member"},
            {"user_id": self.FABRICATED[1], "nickname": "李四", "card": "", "role": "member"},
        ]
        ev_a = FakeEvent(group_id="1000", sender_id="10001", chain=[])
        ev_a.message_str = "帮我艾特张三"
        ev_a.bot.member_list = members
        await plugin.search_and_mention(ev_a, "张三")  # 唯一命中 → 写甲的兜底缓存

        ev_b = FakeEvent(group_id="1000", sender_id="20002", chain=[])
        ev_b.message_str = "帮我艾特李四"
        ev_b.bot.member_list = members
        await plugin.search_and_mention(ev_b, "李四")  # 乙的写入不得冲掉甲的

        ev_a.set_chain([Plain("好的，这就喊他")])  # 模型漏写标签 → 走兜底补插
        await plugin.process_at_tags(ev_a)
        assert _ats(ev_a.get_result().chain) == [self.FABRICATED[0]], (
            "乙写入后甲的兜底补插不得丢失（T14/A3）"
        )

    async def test_known_ids_expire_and_are_capped(self, tmp_path):
        """会话级已知 ID 记忆：过期即失效；会话数受容量上限约束（不无界增长）。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain(f"[at:{self.FABRICATED[1]}]")])
        ev.message_str = "在吗"
        # 已过期（> TTL）→ 不信任
        plugin._known_at_ids[(ev.unified_msg_origin, "10001")] = (
            {self.FABRICATED[1]},
            time.time() - 121,
        )
        await plugin.process_at_tags(ev)
        assert not any(isinstance(c, At) for c in ev.get_result().chain)
        assert f"@{self.FABRICATED[1]}" in "".join(_plain_texts(ev.get_result().chain))

        # 容量上限：逐会话写入超过上限后，最早的会话被淘汰
        plugin2 = make_plugin(audit_dir=tmp_path)
        cap = main_mod._MAX_TRUSTED_SLOTS
        plugin2.bot_sender = None
        for i in range(cap + 3):
            ev_i = FakeEvent(group_id=f"9{i}", sender_id="10001", chain=[])
            ev_i.message_str = "帮我艾特张三"
            ev_i.bot.member_list = [
                {"user_id": "1", "nickname": "张三", "card": "", "role": "member"}
            ]
            await plugin2.search_and_mention(ev_i, "张三")
        assert len(plugin2._known_at_ids) <= cap
        assert ("aiocqhttp:GroupMessage:90", "10001") not in plugin2._known_at_ids

    async def test_known_ids_are_sender_bound_same_umo(self, tmp_path):
        """同群不同用户不得串用（沿用 F1 的 sender 绑定语义）。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev_a = FakeEvent(group_id="1000", sender_id="10001", chain=[])
        ev_a.message_str = "帮我艾特张三"
        ev_a.bot.member_list = [
            {
                "user_id": self.FABRICATED[1],
                "nickname": "张三",
                "card": "",
                "role": "member",
            }
        ]
        await plugin.search_and_mention(ev_a, "张三")

        ev_a.set_chain([Plain(f"来了 [at:{self.FABRICATED[1]}]")])
        await plugin.process_at_tags(ev_a)
        assert _ats(ev_a.get_result().chain) == [self.FABRICATED[1]]

        ev_b = FakeEvent(
            group_id="1000",
            sender_id="20002",
            chain=[Plain(f"来了 [at:{self.FABRICATED[1]}]")],
        )
        ev_b.message_str = "我也说一句"
        await plugin.process_at_tags(ev_b)
        assert not any(isinstance(c, At) for c in ev_b.get_result().chain), (
            "同一 UMO 下另一用户不得借用他人工具确认过的 ID"
        )

    async def test_no_user_context_still_degrades_fabricated_id(
        self, monkeypatch, tmp_path
    ):
        """fail-closed（T14/B1）：取不到用户上下文时可信集退化为工具确认集，

        编造 ID 仍必须降级为 `@载荷` 并留聚合 warning —— 绝不因"读不到用户
        消息"而放行。真实宿主 `message_str`/`get_messages()` 恒可用，本用例
        覆盖的是"宿主/测试桩未暴露用户消息"这一最坏情形。
        """
        warnings: list = []
        monkeypatch.setattr(
            main_mod.logger, "warning", lambda *a, **k: warnings.append(a)
        )
        plugin = make_plugin(audit_dir=tmp_path)
        ev = FakeEvent(chain=[Plain(f"[at:{self.FABRICATED[1]}]")])
        assert getattr(ev, "message_str", "") == ""
        assert ev.get_messages() == []
        await plugin.process_at_tags(ev)

        chain = ev.get_result().chain
        assert not any(isinstance(c, At) for c in chain), (
            "无用户上下文时编造 ID 也必须降级（fail-closed）"
        )
        assert f"@{self.FABRICATED[1]}" in "".join(_plain_texts(chain))
        assert any("未经工具确认" in str(w) for w in warnings), warnings


class TestDirectQqGroupMembership:
    """T16：直连 QQ 号必须过**本群成员名单**才能渲染成 At。

    线上实测（22:02，群 744868236）：用户给了自编号 1645896432（"是1645896432了"），
    模型直接输出 ``[at:1645896432]``；该号在当轮用户消息里 ⇒ v2.6.1 的"用户提供的
    QQ 可信"规则让它渲染成 At，但平台以 ``retcode=1200 Get Uid Error`` **拒绝整条
    消息**（``respond.stage:322``），群里正文全丢。故直连号必须先在本群成员名单里。
    """

    LIVE_BOGUS = "1645896432"  # 线上实测：用户自编、非本群成员的号码
    MEMBER = "3882563785"  # 真实群成员（测试名单里）

    @staticmethod
    def _member_list():
        return [
            {
                "user_id": TestDirectQqGroupMembership.MEMBER,
                "nickname": "陨落星辰",
                "card": "",
                "role": "member",
            }
        ]

    async def test_non_member_direct_id_degrades_and_message_survives(
        self, monkeypatch, tmp_path
    ):
        """① 非本群成员的直连号 → 降级为 `@数字`、消息可交付、无 at_member 审计。"""
        warnings: list = []
        monkeypatch.setattr(
            main_mod.logger, "warning", lambda *a, **k: warnings.append(a)
        )
        plugin = make_plugin(config={"allow_direct_qq_at": True}, audit_dir=tmp_path)
        ev = FakeEvent(
            group_id="1000",
            sender_id="10001",
            chain=[Plain(f"[at:{self.LIVE_BOGUS}] 抓到一只隐身的小可爱啦~")],
        )
        ev.message_str = f"是{self.LIVE_BOGUS}了"
        ev.bot.member_list = self._member_list()  # 名单里没有这个号
        await plugin.process_at_tags(ev)

        chain = ev.get_result().chain
        assert not any(isinstance(c, At) for c in chain), "非本群号不得渲染成 At"
        assert f"@{self.LIVE_BOGUS}" in "".join(_plain_texts(chain))
        # 复用既有成员缓存：工具调用已拉过一次，渲染判定不得再打一次 API
        assert [
            call for call in ev.bot.calls if call[0] == "get_group_member_list"
        ].__len__() == 1, "直连号成员校验必须复用 _get_group_members_cached 的缓存"
        assert chain, "降级后仍必须是一条可交付的消息"
        assert_n1(chain)
        assert not audit_file(tmp_path, datetime.now().strftime("%Y%m%d")).exists(), (
            "被拒绝的直连号不得写 at_member 审计"
        )
        assert any("未经工具确认" in str(w) for w in warnings), warnings

    async def test_member_direct_id_renders(self, tmp_path):
        """② 当轮用户消息里给出的**真实群成员**号 → 照常渲染 At。"""
        plugin = make_plugin(config={"allow_direct_qq_at": True}, audit_dir=tmp_path)
        ev = FakeEvent(
            group_id="1000",
            sender_id="10001",
            chain=[Plain(f"[at:{self.MEMBER}] 来")],
        )
        ev.message_str = f"帮我艾特 {self.MEMBER} 谢谢"
        ev.bot.member_list = self._member_list()
        await plugin.process_at_tags(ev)

        assert _ats(ev.get_result().chain) == [self.MEMBER]
        assert_n1(ev.get_result().chain)

    async def test_member_list_unavailable_keeps_rendering_with_warning(
        self, monkeypatch, tmp_path
    ):
        """③【T24 起翻转】名单不可用（非群聊/拉取为空/异常 ⇒ None）→ **降级**。

        旧语义（T16）是"查不到名单就保持原行为渲染 At"，但线上实测表明该分支同
        样会被平台以 retcode=1200 Get Uid Error 拒绝、整条回复连正文一起丢，因此
        T24 改为 fail-closed：直连号按未知来源降级为 `@数字` + 区分文案 warning。
        """
        warnings: list = []
        monkeypatch.setattr(
            main_mod.logger, "warning", lambda *a, **k: warnings.append(a)
        )
        plugin = make_plugin(config={"allow_direct_qq_at": True}, audit_dir=tmp_path)
        ev = FakeEvent(
            group_id="1000",
            sender_id="10001",
            chain=[Plain(f"[at:{self.LIVE_BOGUS}] 来")],
        )
        ev.message_str = f"是{self.LIVE_BOGUS}了"
        ev.bot.member_list = []  # 空/异常/私聊都归到 None 分支
        await plugin.process_at_tags(ev)

        chain = ev.get_result().chain
        assert not any(isinstance(c, At) for c in chain), (
            "名单不可用时直连号不得渲染 At（否则整条消息可能被平台拒收）"
        )
        assert f"@{self.LIVE_BOGUS}" in "".join(_plain_texts(chain)), "降级为 @数字"
        assert chain, "降级后仍必须是一条可交付的消息"
        assert_n1(chain)
        assert any("无法获取群成员名单" in str(w) for w in warnings), warnings
        assert any("已按未知来源降级" in str(w) for w in warnings), warnings

    async def test_tool_confirmed_path_unaffected_by_membership_check(
        self, monkeypatch, tmp_path
    ):
        """④ 工具确认路径不受影响：命中 ID 本就来自成员名单，仍直接渲染。"""
        warnings: list = []
        monkeypatch.setattr(
            main_mod.logger, "warning", lambda *a, **k: warnings.append(a)
        )
        plugin = make_plugin(config={"allow_direct_qq_at": True}, audit_dir=tmp_path)
        ev = FakeEvent(group_id="1000", sender_id="10001", chain=[])
        ev.message_str = f"是{self.LIVE_BOGUS}了"  # 用户只给了编造号
        ev.bot.member_list = self._member_list()
        text = await plugin.search_and_mention(ev, "陨落星辰")  # 工具确认 MEMBER
        assert self.MEMBER in text

        ev.set_chain([Plain(f"[at:{self.MEMBER}] 和 [at:{self.LIVE_BOGUS}]")])
        await plugin.process_at_tags(ev)
        chain = ev.get_result().chain
        assert _ats(chain) == [self.MEMBER], (
            "工具确认路径不受名单校验影响；非本群的直连号仍降级"
        )
        assert f"@{self.LIVE_BOGUS}" in "".join(_plain_texts(chain))


class TestStreamingFormRows:
    """流式路径形态（spec §7.2 第 17/18 行）。"""

    async def test_streaming_single_chunk_tag_rendered(self, tmp_path):
        """第 17 行：单 chunk 内完整的标签正常渲染。"""
        plugin = make_plugin(audit_dir=tmp_path)
        ev = SendableEvent(chain=[])
        plugin._wrap_event_send(ev)

        async def gen():
            yield MessageChainStub([Plain("[At:10001]来啦")])

        ti.trust_ids(plugin, ev, "10001")
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
