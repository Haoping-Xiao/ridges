#!/usr/bin/env python3
"""
提取 agent 日志中的 next_thought/next_tool_name/next_tool_args traces 用于诊断。

提取的 traces 包含以下字段:
    - thought: agent 的思考过程 (next_thought)
    - tool: 使用的工具名称 (next_tool_name)
    - input: 工具输入参数 (next_tool_args)

用法:
    python utils/extract_agent_traces.py <agent_logs.txt> [output.json]

示例:
    python utils/extract_agent_traces.py test_agent_results/.../agent_logs.txt
    python utils/extract_agent_traces.py test_agent_results/.../agent_logs.txt traces.json
"""

import json
import os
import sys


def extract_traces(log_file_path):
    """从 agent 日志文件中提取 traces

    提取每一步的 next_thought, next_tool_name, next_tool_args
    """
    with open(log_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    traces = []
    current_trace = {}
    collecting_thought = False
    collecting_tool_args = False
    thought_lines = []
    tool_args_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]
        line_stripped = line.strip()

        # Match next_thought (可能在日志行中，也可能跨多行)
        if "next_thought:" in line:
            # 如果当前 trace 已经完整，先保存
            if current_trace and "tool" in current_trace and "input" in current_trace:
                traces.append(current_trace)
                current_trace = {}

            # 提取 thought 文本
            thought_text = line.split("next_thought:")[1].strip()
            # 移除开头的引号（如果有）
            if thought_text.startswith('"'):
                thought_text = thought_text[1:]
            elif thought_text.startswith("'"):
                thought_text = thought_text[1:]

            # 检查是否在同一行结束（以引号结尾）
            if thought_text.endswith('"') or thought_text.endswith("'"):
                # 单行 thought
                thought_text = thought_text.rstrip("\"'")
                current_trace["thought"] = thought_text
                collecting_thought = False
            else:
                # 多行 thought，开始收集
                collecting_thought = True
                thought_lines = [thought_text]

            collecting_tool_args = False
            tool_args_lines = []
            i += 1
            continue

        # 如果正在收集 thought
        elif collecting_thought:
            # 检查是否遇到 next_tool_name（thought 结束）
            if line_stripped.startswith("next_tool_name:"):
                # 合并 thought 行并移除末尾引号
                thought_text = "\n".join(thought_lines).rstrip("\"'")
                current_trace["thought"] = thought_text
                collecting_thought = False
                thought_lines = []
                # 继续处理这一行的 next_tool_name
                # 不 continue，让下面的逻辑处理
            else:
                # 继续收集 thought 行
                thought_lines.append(line_stripped)
                i += 1
                continue

        # Match next_tool_name
        if line_stripped.startswith("next_tool_name:"):
            tool_name = line_stripped.split("next_tool_name:")[1].strip()
            current_trace["tool"] = tool_name
            collecting_thought = False
            collecting_tool_args = False
            i += 1
            continue

        # Match next_tool_args (可能跨多行)
        elif line_stripped.startswith("next_tool_args:"):
            args_str = line_stripped.split("next_tool_args:")[1].strip()
            tool_args_lines = [args_str]

            # 检查括号是否匹配，判断是否需要继续收集
            open_brackets = args_str.count("[") + args_str.count("{") + args_str.count("(")
            close_brackets = args_str.count("]") + args_str.count("}") + args_str.count(")")

            if open_brackets > close_brackets:
                # 需要继续收集
                collecting_tool_args = True
            else:
                # 单行完成
                current_trace["input"] = args_str
                collecting_tool_args = False
                # 如果 trace 完整，可以保存
                if "thought" in current_trace and "tool" in current_trace:
                    traces.append(current_trace)
                    current_trace = {}

            i += 1
            continue

        # 如果正在收集 tool_args
        elif collecting_tool_args:
            tool_args_lines.append(line_stripped)
            # 检查括号是否匹配
            all_args = " ".join(tool_args_lines)
            open_brackets = all_args.count("[") + all_args.count("{") + all_args.count("(")
            close_brackets = all_args.count("]") + all_args.count("}") + all_args.count(")")

            if open_brackets == close_brackets:
                # 括号匹配，收集完成
                current_trace["input"] = " ".join(tool_args_lines)
                collecting_tool_args = False
                tool_args_lines = []
                # 如果 trace 完整，可以保存
                if "thought" in current_trace and "tool" in current_trace:
                    traces.append(current_trace)
                    current_trace = {}

            i += 1
            continue

        i += 1

    # 处理最后未完成的收集
    if collecting_thought and thought_lines:
        thought_text = "\n".join(thought_lines).rstrip("\"'")
        current_trace["thought"] = thought_text

    if collecting_tool_args and tool_args_lines:
        current_trace["input"] = " ".join(tool_args_lines)

    # 添加最后一个 trace（如果完整）
    if current_trace and "thought" in current_trace and "tool" in current_trace and "input" in current_trace:
        traces.append(current_trace)

    return traces


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    log_file_path = sys.argv[1]

    if not os.path.exists(log_file_path):
        print(f"错误: 文件不存在: {log_file_path}", file=sys.stderr)
        sys.exit(1)

    # 提取 traces
    traces = extract_traces(log_file_path)

    # 确定输出文件
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        # 默认输出到同目录下的 traces.json
        base_name = os.path.splitext(os.path.basename(log_file_path))[0]
        output_dir = os.path.dirname(log_file_path)
        output_file = os.path.join(output_dir, f"{base_name}_traces.json")

    # 保存 traces
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2, ensure_ascii=False)

    print(f"成功提取 {len(traces)} 条 traces")
    print(f"输出文件: {output_file}")


if __name__ == "__main__":
    main()
