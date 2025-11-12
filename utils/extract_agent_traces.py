#!/usr/bin/env python3
"""
提取 agent 日志中的 thought/action/input/output traces 用于诊断。

提取的 traces 包含以下字段:
    - thought: agent 的思考过程
    - action: 执行的操作名称
    - tool: 使用的工具名称
    - input: 工具输入参数
    - output: 工具输出结果

用法:
    python utils/extract_agent_traces.py <agent_logs.txt> [output.json]
    
示例:
    python utils/extract_agent_traces.py test_agent_results/.../agent_logs.txt
    python utils/extract_agent_traces.py test_agent_results/.../agent_logs.txt traces.json
"""
import json
import sys
import os


def extract_traces(log_file_path):
    """从 agent 日志文件中提取 traces"""
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    traces = []
    current_trace = {}
    collecting_observation = False
    observation_lines = []

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Match tool execution
        if 'About to execute operation:' in line:
            if current_trace and 'tool' in current_trace:
                traces.append(current_trace)
            op_name = line.split('About to execute operation:')[1].strip()
            current_trace = {'action': op_name}
            collecting_observation = False
            observation_lines = []
        
        # Match thought (appears before tool name)
        elif 'next_thought:' in line:
            thought_text = line.split('next_thought:')[1].strip()
            # Remove quotes if present
            if thought_text.startswith('"') and thought_text.endswith('"'):
                thought_text = thought_text[1:-1]
            elif thought_text.startswith("'") and thought_text.endswith("'"):
                thought_text = thought_text[1:-1]
            current_trace['thought'] = thought_text
            collecting_observation = False
            observation_lines = []
        
        # Match tool name
        elif line_stripped.startswith('next_tool_name:'):
            tool_name = line_stripped.split('next_tool_name:')[1].strip()
            if 'action' not in current_trace:
                current_trace['action'] = tool_name
            current_trace['tool'] = tool_name
            collecting_observation = False
            observation_lines = []
        
        # Match tool args
        elif line_stripped.startswith('next_tool_args:'):
            args_str = line_stripped.split('next_tool_args:')[1].strip()
            current_trace['input'] = args_str
            collecting_observation = False
            observation_lines = []
        
        # Match observation start
        elif 'next_observation:' in line:
            obs_start = line.split('next_observation:')[1].strip()
            collecting_observation = True
            observation_lines = [obs_start] if obs_start else []
        
        # Collect observation continuation lines (until next marker)
        elif collecting_observation:
            # Stop collecting if we hit certain markers
            if (line_stripped.startswith('[CRITICAL]') or 
                line_stripped.startswith('[REQUEST]') or
                line_stripped.startswith('next_tool_name:') or
                line_stripped.startswith('next_tool_args:') or
                line_stripped.startswith('Execution step') or
                'About to execute operation:' in line or
                'next_observation:' in line or
                'next_thought:' in line):
                collecting_observation = False
                if observation_lines:
                    current_trace['output'] = ' '.join(observation_lines)
                if 'tool' in current_trace:
                    traces.append(current_trace)
                    current_trace = {}
                observation_lines = []
            elif line_stripped and not line_stripped.startswith('2025-'):
                # Collect non-empty, non-timestamp lines
                observation_lines.append(line_stripped)

    # Add last trace if exists
    if current_trace and 'tool' in current_trace:
        if observation_lines:
            current_trace['output'] = ' '.join(observation_lines)
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
        output_file = os.path.join(output_dir, f'{base_name}_traces.json')
    
    # 保存 traces
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(traces, f, indent=2, ensure_ascii=False)
    
    print(f"成功提取 {len(traces)} 条 traces")
    print(f"输出文件: {output_file}")


if __name__ == '__main__':
    main()

