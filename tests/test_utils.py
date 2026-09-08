"""utils.py 纯逻辑单元测试（CHG-08）。

覆盖变更点：
- CHG-01 build_permission_cache_key（群+用户维度，修复越权）
- CHG-02 match_session_entry / check_session_lists（精确匹配）
- CHG-03 is_at_all_in_cooldown / build_audit_record / audit_log_path
- CHG-05 format_member_choice_list（多结果序号）
- CHG-06 format_role_label / format_single_member_result（角色标注）
- CHG-07 drop_expired / evict_oldest_to_limit（缓存上限清理）
- CHG-08 split_text_by_at_tags（标签解析）
"""

from __future__ import annotations

from pathlib import Path

import pytest

from utils import (
    AT_TAG_PATTERN,
    audit_log_path,
    build_audit_record,
    build_permission_cache_key,
    check_session_lists,
    drop_expired,
    evict_oldest_to_limit,
    expand_alias_queries,
    format_member_choice_list,
    format_role_label,
    format_single_member_result,
    is_at_all_in_cooldown,
    match_session_entry,
    split_text_by_at_tags,
)

# 测试用会话标识
UMO = "aiocqhttp:GroupMessage:123456"


# --------------------------------------------------------------------------- #
# CHG-06 角色标注
# --------------------------------------------------------------------------- #
class TestFormatRoleLabel:
    """format_role_label：原始角色 -> 中文标签。"""

    @pytest.mark.parametrize(
        "role,expected",
        [
            ("owner", "群主"),
            ("admin", "管理员"),
            ("member", "成员"),
            ("OWNER", "群主"),  # 大小写归一
            ("Admin", "管理员"),
            ("", "成员"),  # 空兜底
            (None, "成员"),  # None 兜底
        ],
    )
    def test_known_roles(self, role, expected):
        assert format_role_label(role) == expected

    def test_unknown_role_kept_as_is(self):
        # 未知非空角色原样保留，不强行猜测
        assert format_role_label("vip") == "vip"


# --------------------------------------------------------------------------- #
# CHG-02 会话黑白名单精确匹配
# --------------------------------------------------------------------------- #
class TestMatchSessionEntry:
    """match_session_entry：修复子串误命中，仅精确等于命中。"""

    @pytest.mark.parametrize(
        "entry,umo,group_id,expected",
        [
            (UMO, UMO, "123456", True),  # 等于完整 UMO
            ("123456", UMO, "123456", True),  # 等于 group_id
            ("123", UMO, "123456", False),  # 关键回归：子串不误命中
            ("1234567", UMO, "123456", False),  # 相似群号不误命中
            ("", UMO, "123456", False),  # 空条目
            ("   ", UMO, "123456", False),  # 纯空白条目
            ("123456", UMO, "", False),  # 私聊场景 group_id 为空，群号条目不应命中
            (" 123456 ", UMO, "123456", True),  # 带空白被 strip 后命中
        ],
    )
    def test_match(self, entry, umo, group_id, expected):
        assert match_session_entry(entry, umo, group_id) is expected


class TestCheckSessionLists:
    """check_session_lists：黑白名单组合判定（黑名单优先）。"""

    def test_empty_lists_all_allowed(self):
        assert check_session_lists(None, None, UMO, "123456") == (True, "")
        assert check_session_lists([], [], UMO, "123456") == (True, "")

    def test_blacklist_hit_denied(self):
        allowed, reason = check_session_lists(None, ["123456"], UMO, "123456")
        assert allowed is False
        assert "黑名单" in reason

    def test_blacklist_similar_not_misfire(self):
        # 条目 "123" 不应误命中含 123 的 UMO（精确匹配回归）
        assert check_session_lists(None, ["123"], UMO, "123456") == (True, "")

    def test_whitelist_hit_allowed(self):
        assert check_session_lists(["123456"], None, UMO, "123456") == (True, "")

    def test_whitelist_miss_denied(self):
        allowed, reason = check_session_lists(["999999"], None, UMO, "123456")
        assert allowed is False
        assert "白名单" in reason

    def test_whitelist_similar_not_misfire(self):
        # "123" 不命中，应被白名单拒绝而非放行
        assert check_session_lists(["123"], None, UMO, "123456")[0] is False

    def test_blacklist_priority_over_whitelist(self):
        # 同时在黑白名单中：黑名单优先
        allowed, reason = check_session_lists(
            whitelist=["123456"], blacklist=["123456"], unified_msg_origin=UMO,
            group_id="123456",
        )
        assert allowed is False
        assert "黑名单" in reason


# --------------------------------------------------------------------------- #
# CHG-01 权限缓存 Key（群+用户维度）
# --------------------------------------------------------------------------- #
class TestBuildPermissionCacheKey:
    """build_permission_cache_key：修复同群不同用户共享缓存越权。"""

    def test_normal_key(self):
        assert build_permission_cache_key("123456", "10001") == "123456:10001"

    def test_different_users_isolated(self):
        # 同群不同用户得到不同 key —— 正是 v2.4.0 越权 bug 的根因
        assert build_permission_cache_key("123456", "10001") != \
               build_permission_cache_key("123456", "10002")
        assert build_permission_cache_key("123456", "10001") != \
               build_permission_cache_key("789012", "10001")

    @pytest.mark.parametrize("group_id,user_id", [("", "10001"), ("123456", ""), ("", "")])
    def test_missing_parts_returns_empty(self, group_id, user_id):
        assert build_permission_cache_key(group_id, user_id) == ""

    def test_whitespace_stripped(self):
        assert build_permission_cache_key("  123456  ", " 10001 ") == "123456:10001"


# --------------------------------------------------------------------------- #
# CHG-03 @全体冷却
# --------------------------------------------------------------------------- #
class TestIsAtAllInCooldown:
    """is_at_all_in_cooldown：冷却判定与剩余时间。"""

    def test_zero_cooldown_not_limited(self):
        assert is_at_all_in_cooldown(100.0, 200.0, 0)[0] is False

    def test_negative_cooldown_not_limited(self):
        assert is_at_all_in_cooldown(100.0, 200.0, -5)[0] is False

    def test_never_triggered_not_in_cooldown(self):
        assert is_at_all_in_cooldown(None, 200.0, 60) == (False, 0.0)

    def test_within_cooldown(self):
        # 距上次触发 10s < 60s，在冷却中，剩余 50s
        in_cd, remain = is_at_all_in_cooldown(190.0, 200.0, 60)
        assert in_cd is True
        assert abs(remain - 50.0) < 1e-6

    def test_boundary_elapsed_equals_cooldown_not_in_cooldown(self):
        # elapsed == cooldown，左闭右开，恰好在冷却外
        assert is_at_all_in_cooldown(140.0, 200.0, 60)[0] is False

    def test_already_expired_not_in_cooldown(self):
        assert is_at_all_in_cooldown(100.0, 200.0, 60)[0] is False


# --------------------------------------------------------------------------- #
# CHG-08 标签解析
# --------------------------------------------------------------------------- #
class TestSplitTextByAtTags:
    """split_text_by_at_tags：[at:xxx] 标签拆分纯逻辑。"""

    def test_plain_text_only(self):
        assert split_text_by_at_tags("hello world") == [("text", "hello world")]

    def test_single_numeric_tag(self):
        assert split_text_by_at_tags("[at:123]") == [("at", "123")]

    def test_all_tag(self):
        assert split_text_by_at_tags("[at:all]") == [("at", "all")]

    def test_mixed_segments(self):
        segments = split_text_by_at_tags("a[at:1]b[at:2]c")
        assert segments == [
            ("text", "a"),
            ("at", "1"),
            ("text", "b"),
            ("at", "2"),
            ("text", "c"),
        ]

    def test_leading_and_trailing_text(self):
        assert split_text_by_at_tags("hi [at:9]") == [("text", "hi "), ("at", "9")]

    def test_empty_and_none(self):
        assert split_text_by_at_tags("") == []
        assert split_text_by_at_tags(None) == []

    def test_invalid_tag_treated_as_text(self):
        # 非法标签（非数字/非 all）不被识别，整段作为普通文本
        assert split_text_by_at_tags("[at:abc]") == [("text", "[at:abc]")]

    def test_pattern_consistent_with_module_constant(self):
        # 模块正则与 main.py 约定一致
        assert AT_TAG_PATTERN.search("[at:all]")[1] == "all"
        assert AT_TAG_PATTERN.search("[at:0000]") is not None


# --------------------------------------------------------------------------- #
# CHG-05 / CHG-06 成员结果格式化
# --------------------------------------------------------------------------- #
class TestFormatSingleMemberResult:
    """format_single_member_result：含角色标注与艾特标签提示。"""

    def test_default_prefix_and_role(self):
        result = format_single_member_result("张三", "10001", "member")
        assert result.startswith("已找到：")
        assert "张三" in result and "10001" in result and "成员" in result
        assert "[at:10001]" in result

    def test_custom_prefix(self):
        result = format_single_member_result("李四", "10002", "owner", prefix="已选定：")
        assert result.startswith("已选定：")
        assert "群主" in result


class TestFormatMemberChoiceList:
    """format_member_choice_list：多候选序号列表 + 角色标注。"""

    def test_multi_choice_layout(self):
        matches = [("10001", "张三", "owner"), ("10002", "李四", "member")]
        text = format_member_choice_list("张", matches)
        assert "共 2 个" in text
        assert "1. 张三 (ID: 10001) - 群主" in text
        assert "2. 李四 (ID: 10002) - 成员" in text
        assert "select_member_by_index" in text

    def test_empty_matches(self):
        text = format_member_choice_list("x", [])
        assert "共 0 个" in text


# --------------------------------------------------------------------------- #
# 别名展开（用户确认方案：才俊 -> 柴郡）
# --------------------------------------------------------------------------- #
class TestExpandAliasQueries:
    """expand_alias_queries：别名展开去重保序。"""

    def test_no_alias_returns_original(self):
        assert expand_alias_queries("才俊", None) == ["才俊"]
        assert expand_alias_queries("才俊", {}) == ["才俊"]

    def test_dict_alias_matched(self):
        out = expand_alias_queries("才俊", {"才俊": "柴郡", "柴俊": "柴郡"})
        assert out == ["才俊", "柴郡"]

    def test_dict_alias_list_target(self):
        out = expand_alias_queries("才俊", {"才俊": ["柴郡", "柴郡〔Bot〕"]})
        assert out == ["才俊", "柴郡", "柴郡〔Bot〕"]

    def test_other_alias_not_expanded(self):
        assert expand_alias_queries("才俊", {"阿俊": "阿郡"}) == ["才俊"]

    def test_eq_string_list_form(self):
        out = expand_alias_queries("才俊", ["才俊=柴郡", "阿俊=阿郡"])
        assert out == ["才俊", "柴郡"]

    def test_dedupe_preserves_order(self):
        out = expand_alias_queries("才俊", {"才俊": ["柴郡", "柴郡"]})
        assert out == ["才俊", "柴郡"]

    def test_empty_query(self):
        assert expand_alias_queries("", {"": "x"}) == []
        assert expand_alias_queries(None, {}) == []


# --------------------------------------------------------------------------- #
# CHG-03 审计记录与日志路径
# --------------------------------------------------------------------------- #
class TestBuildAuditRecord:
    """build_audit_record：审计字段完整性与类型归一。"""

    def test_full_fields(self):
        rec = build_audit_record(
            "2026-08-08T12:00:00", "123456", "999", "at_all", "all", True, "ok"
        )
        assert rec == {
            "time": "2026-08-08T12:00:00",
            "group_id": "123456",
            "operator_id": "999",
            "op_type": "at_all",
            "target_id": "all",
            "allowed": True,
            "reason": "ok",
        }

    def test_allowed_bool_coercion(self):
        rec = build_audit_record("t", "g", "o", "at_member", "100", 0)
        assert rec["allowed"] is False

    def test_reason_default_empty(self):
        rec = build_audit_record("t", "g", "o", "at_all", "all", False, "")
        assert rec["reason"] == ""


class TestAuditLogPath:
    """audit_log_path：按日切割的 JSONL 路径构造。"""

    def test_path_with_string_dir(self):
        p = audit_log_path("/data/plugin_data/AtTool", "20260808")
        assert p == Path("/data/plugin_data/AtTool/audit/at_audit_20260808.jsonl")

    def test_path_with_pathlib_dir(self):
        p = audit_log_path(Path("/data/plugin_data/AtTool"), "20260101")
        assert p.name == "at_audit_20260101.jsonl"
        assert p.parent.name == "audit"


# --------------------------------------------------------------------------- #
# CHG-07 缓存过期清理与容量上限
# --------------------------------------------------------------------------- #
class TestDropExpired:
    """drop_expired：惰性清理过期项（约定 value[-1] 为创建时间戳）。"""

    def test_remove_expired_keep_valid(self):
        cache = {"a": ("p1", 100.0), "b": ("p2", 200.0)}
        removed = drop_expired(cache, ttl=50, now=155.0)
        # a: 155-100=55>=50 过期；b: 155-200<0 不过期
        assert removed == 1
        assert "a" not in cache and "b" in cache

    def test_all_fresh_no_removal(self):
        cache = {"a": ("p1", 150.0), "b": ("p2", 140.0)}
        assert drop_expired(cache, ttl=50, now=155.0) == 0
        assert len(cache) == 2

    def test_zero_ttl_no_op(self):
        cache = {"a": ("p1", 0.0)}
        assert drop_expired(cache, ttl=0, now=1000.0) == 0
        assert "a" in cache


class TestEvictOldestToLimit:
    """evict_oldest_to_limit：超容淘汰最早写入项。"""

    def test_evict_oldest_to_limit_k(self):
        cache = {"a": (1, 0.0), "b": (2, 0.0), "c": (3, 0.0)}
        evicted = evict_oldest_to_limit(cache, 2)
        assert evicted == 1
        assert list(cache.keys()) == ["b", "c"]

    def test_within_limit_no_eviction(self):
        cache = {"a": (1, 0.0), "b": (2, 0.0)}
        assert evict_oldest_to_limit(cache, 2) == 0
        assert len(cache) == 2

    def test_zero_or_negative_limit_no_op(self):
        cache = {"a": (1, 0.0), "b": (2, 0.0)}
        evict_oldest_to_limit(cache, 0)
        assert len(cache) == 2

    def test_permission_cache_shape_compatible(self):
        # 验证权限缓存 value=(allowed, deny, ts) 也兼容 drop/evict（ts 在末位）
        cache = {"k": (True, "", 100.0)}
        drop_expired(cache, ttl=50, now=200.0)  # 200-100=100>=50 过期
        assert cache == {}