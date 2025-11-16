"""
单元测试：测试 EnhancedNetwork.parse_response 方法
"""

import unittest

from agent import EnhancedNetwork


class TestParseResponse(unittest.TestCase):
    """测试 parse_response 方法"""

    def test_parse_multiple_tool_calls_with_separator(self):
        """测试解析包含多个 tool calls（用 --- 分隔）的字符串"""
        test_string = """next_thought:"I need to examine the _combinator_query method and the order_by method to understand how they interact. Also, let me look at the SQL compiler code to see how the ORDER BY clause is generated for union queries."

tool_name: get_file_content

tool_args: {"file_path": "./django/db/models/query.py", "search_start_line": 920, "search_end_line": 940}

---

tool_name: get_file_content

tool_args: {"file_path": "./django/db/models/query.py", "search_start_line": 1060, "search_end_line": 1080}

---

tool_name: search_in_all_files_content

tool_args: {"search_term": "ORDER BY position", "case_sensitive": false}"""

        result = EnhancedNetwork.parse_response(test_string)
        thought, tool_name, tool_args, error_msg = result
        self.assertEqual(
            thought,
            '"I need to examine the _combinator_query method and the order_by method to understand how they interact. Also, let me look at the SQL compiler code to see how the ORDER BY clause is generated for union queries."',
        )
        self.assertEqual(tool_name, ["get_file_content", "get_file_content", "search_in_all_files_content"])
        self.assertEqual(
            tool_args,
            [
                {"file_path": "./django/db/models/query.py", "search_start_line": 920, "search_end_line": 940},
                {"file_path": "./django/db/models/query.py", "search_start_line": 1060, "search_end_line": 1080},
                {"search_term": "ORDER BY position", "case_sensitive": False},
            ],
        )
        self.assertIsNone(error_msg, "error_msg 应该为 None")

    def test_parse_single_tool_call(self):
        """测试解析单个 tool call 的字符串"""
        test_string = """next_thought:"Now I need to look at the get_combinator_sql method which handles the SQL generation for union queries. This is where the issue likely occurs - the ORDER BY clause is being applied to the combined query but the column positions don't match up correctly."
tool_name: search_in_specified_file_v2
tool_args: {"file_path": "./django/db/models/sql/compiler.py", "search_term": "def get_combinator_sql"}"""

        # 调用 parse_response
        result = EnhancedNetwork.parse_response(test_string)
        thought, tool_name, tool_args, error_msg = result
        self.assertIsNotNone(thought, "thought 不应该为 None")
        self.assertEqual(
            thought,
            '"Now I need to look at the get_combinator_sql method which handles the SQL generation for union queries. This is where the issue likely occurs - the ORDER BY clause is being applied to the combined query but the column positions don\'t match up correctly."',
        )
        self.assertEqual(tool_name, "search_in_specified_file_v2")
        self.assertIsInstance(tool_args, dict)
        self.assertEqual(tool_args["file_path"], "./django/db/models/sql/compiler.py")
        self.assertEqual(tool_args["search_term"], "def get_combinator_sql")

    def test_parse_single_tool_call_with_mixed_fields(self):
        """测试解析包含 mixed fields 的字符串"""
        test_string = """next_thought:"I need to look at the get_combinator_sql method which handles the SQL generation for union queries. This is where the issue likely occurs - the ORDER BY clause is being applied to the combined query but the column positions don't match up correctly."
next_tool_name: search_in_specified_file_v2
next_tool_args: {"file_path": "./django/db/models/sql/compiler.py", "search_term": "def get_combinator_sql"}"""

        # 调用 parse_response
        result = EnhancedNetwork.parse_response(test_string)
        thought, tool_name, tool_args, error_msg = result
        self.assertIsNotNone(thought, "thought 不应该为 None")
        self.assertEqual(
            thought,
            '"I need to look at the get_combinator_sql method which handles the SQL generation for union queries. This is where the issue likely occurs - the ORDER BY clause is being applied to the combined query but the column positions don\'t match up correctly."',
        )
        self.assertEqual(tool_name, "search_in_specified_file_v2")
        self.assertIsInstance(tool_args, dict)
        self.assertEqual(tool_args["file_path"], "./django/db/models/sql/compiler.py")
        self.assertEqual(tool_args["search_term"], "def get_combinator_sql")
        self.assertIsNone(error_msg, "不应该有错误信息")
