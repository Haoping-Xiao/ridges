from __future__ import annotations

import ast
import inspect
import json
import logging
import os
import random
import re
import subprocess
import sys
import textwrap
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for h in list(logger.handlers):
    logger.removeHandler(h)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
run_id = None
STOP_INSTRUCTION = textwrap.dedent("""
# 🎨 
DO NOT generate `observation:` in your response. It will be provided by user for you.
Generate only SINGLE triplet of `next_thought`, `next_tool_name`, `next_tool_args` in your response.
""")
DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "2200"))
PROBLEM_TYPE_CREATE, PROBLEM_TYPE_FIX = "CREATE", "FIX"
GLM_MODEL_NAME, KIMI_MODEL_NAME, DEEPSEEK_MODEL_NAME, QWEN_MODEL_NAME = (
    "zai-org/GLM-4.6-FP8",
    "moonshotai/Kimi-K2-Instruct",
    "deepseek-ai/DeepSeek-V3-0324",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
)
AGENT_MODELS = [GLM_MODEL_NAME, KIMI_MODEL_NAME, DEEPSEEK_MODEL_NAME, QWEN_MODEL_NAME]
(
    MAX_FIX_TASK_STEPS,
    BASIC_APPROACH_MAX_RETRIES,
    MAX_INFERENCE_RETRIES,
    RETRY_BASE_DELAY,
    TEMPERATURE_INCREMENT,
    MAX_TEMPERATURE,
    COT_OBSERVATIONS_TO_KEEP,
) = 400, 20, 10, 2.0, 0.1, 0.7, 30
DO_NOT_REPEAT_TOOL_CALLS = textwrap.dedent("""
You're not allowed to repeat the same tool call with the same arguments.
Your previous response: 
{previous_response}

Try to use something different!
""")
FORMAT_PROMPT_V0 = textwrap.dedent("""
**📝 Response Format Requirements**

1. **Strict Triplet Format**:
   - `next_thought`: Detailed reasoning (include:
     - Problem understanding
     - Code analysis
     - Solution justification
     - Validation plan)
   - `next_tool_name`: Must be an exact tool name from the tool list, OR a JSON array of tool names for parallel execution
   - `next_tool_args`: Valid JSON with:
     - Proper escaping
     - No trailing commas
     - Tool-specific parameters
     - OR a JSON array of argument objects when using parallel tool calls

2. **Parallel Tool Calls (Recommended for Exploration)**:
   - You can call multiple tools in parallel to speed up exploration
   - Format for parallel calls:
     next_thought: "I'll search for multiple patterns in parallel to understand the codebase"
     next_tool_name: ["search_in_all_files_content", "search_in_all_files_content", "get_file_content"]
     next_tool_args: [
       {"search_term": "def process_data", "case_sensitive": false},
       {"search_term": "class DataProcessor", "case_sensitive": false},
       {"file_path": "main.py"}
     ]
   - All parallel tool calls will execute simultaneously
   - Results will be combined and returned together
   - Use parallel calls when:
     * Searching for multiple related patterns
     * Reading multiple files simultaneously
     * Exploring different aspects of the codebase

3. **Error Handling Format**:
   - For errors: 
     next_thought: "Error: [detailed explanation]"
     next_tool_name: ""
     next_tool_args: {}

4. **Example Valid Format (Single Tool)**:
   next_thought: "I'll fix the JSON parsing issue by adding proper error handling and validation"
   next_tool_name: "apply_code_edit"
   next_tool_args: {
     "file_path": "network.py",
     "search": "return json.loads(response)",
     "replace": "try:\n    return json.loads(response)\nexcept JSONDecodeError:\n    logger.error(f'Invalid JSON: {{response}}')\n    raise"
   }

5. **Example Valid Format (Parallel Tools)**:
   next_thought: "I'll search for all occurrences of the bug pattern and read related files in parallel"
   next_tool_name: ["search_in_all_files_content", "get_file_content", "get_file_content"]
   next_tool_args: [
     {"search_term": "def buggy_function"},
     {"file_path": "utils.py"},
     {"file_path": "helpers.py"}
   ]

6. **Invalid Format Examples** (Avoid These):
   - Missing any of the three required fields
   - JSON syntax errors in next_tool_args
   - Extra text outside the triplet format
   - Using incorrect tool names
   - Not quoting special characters properly
   - Mismatch between number of tool names and argument objects in parallel calls
""")

FIX_TASK_SYSTEM_PROMPT = textwrap.dedent("""
# Hey there! You're a Coding Assistant 🚀. I have uploaded all files of a python repository. Your current working directory is at the root of that repo. You will be provided with a problem statement and you need to make the necessary changes to fix the issue.

## Follow these steps to fix the issue:
1. As a first step, find the relevant files in the repo to work on.
2. Localise the code causing the issue.
3. Edit the sourcecode of the repo to resolve the issue.
4. Think about edgecases and make sure the fix handles them as well.
5. Code must always be backward compatible unless explicitly mentioned otherwise in the problem statement.
6. Thoroughly check the entire code base to ensure the changes made are exhaustive and does not break any other functionality.
7. Thoroughly check the entire code base to ensure the changes user requested are only limited to the ones you have identified.
8. Never edit/update the existing test files directly when validating a hypothesis. Instead, when you need a new or focused test to reproduce or protect the fix, use the dedicated test generation tool.
9. Do not create any new files or directories unless absolutely necessary for the fix. Generated tests are allowed but are excluded from the final patch automatically.
10. Always check all the test cases which will be impacted with your change and ensure they don't fail.
11. You need to propose at least 2 meaningfully different and accurate solutions to the problem to the user for approval.
12. You need to look at both expected output mentioned in the problem statement AND the output in the most relevant test case. This is very important.
13. If you find that the error while running the run_code or run_repo_tests tool due to missing dependencies, do not try to solve it as you don't have any internet access.
14. You can add debug prints and then run_repo_tests. For debug prints: use `print("DEBUG: <message>")` or `print(f"DEBUG: <message> {{<variable>}}")`

## Multi-file awareness (critical):
- Tests and patch contexts may span multiple files. Do not stop after the first similar match or applied fix.
- Keep searching the repository after each match and apply consistent changes to every relevant file before finishing.
- Prefer using `search_in_all_files_content` to enumerate matches across the codebase and `search_in_specified_file_v2` to drill into each file; iterate until no applicable occurrences remain.
- Re-run tests only after covering all discovered occurrences to avoid partial fixes.

You have access to the following tools:-
{tools_docs}

{format_prompt}""")
FIX_TASK_INSTANCE_PROMPT_TEMPLATE = textwrap.dedent("""
# Now let's start. Here is the problem statement:
{problem_statement}""")
PYTEST_COMMAND_TEMPLATE = textwrap.dedent("""\
python -c "import sys, pytest, collections, collections.abc, urllib3.exceptions, _pytest.pytester, numpy;
collections.Mapping = collections.abc.Mapping;
collections.MutableMapping = collections.abc.MutableMapping;
collections.MutableSet = collections.abc.MutableSet;
collections.Sequence = collections.abc.Sequence;
collections.Callable = collections.abc.Callable;
collections.Iterable = collections.abc.Iterable;
collections.Iterator = collections.abc.Iterator;
urllib3.exceptions.SNIMissingWarning = urllib3.exceptions.DependencyWarning;
pytest.RemovedInPytest4Warning = DeprecationWarning;
_pytest.pytester.Testdir = _pytest.pytester.Pytester;
numpy.PINF = numpy.inf;
sys.exit(pytest.main([{file_paths}, '-vv', '-s', '--tb=long', '--showlocals', '-W', 'ignore']))"\
""")


class EnhancedCOT:
    class Action:
        def __init__(
            self,
            next_thought: str,
            next_tool_name: list[str],
            next_tool_args: list[dict],
            observation: list[tuple[int, str]],
            is_error: bool = False,
            raw_response: str = None,
            total_attempts: int = 0,
            inference_error_counter: dict = None,
            request_data: list = None,
            is_parallel: bool = False,
        ):
            self.next_thought = next_thought
            self.next_tool_name = next_tool_name
            self.next_tool_args = next_tool_args
            self.observation = observation
            self.is_error = is_error
            self.raw_response = raw_response
            self.total_attempts = total_attempts
            self.inference_error_counter = inference_error_counter
            self.request_data = request_data
            self.is_deleted = False
            self.is_parallel = is_parallel

    def __init__(self, latest_observations_to_keep=5):
        self.thoughts: list[EnhancedCOT.Action] = []
        self.latest_observations_to_keep = latest_observations_to_keep
        self.repeated_thoughts = 0

    def add_action(self, action: EnhancedCOT.Action) -> bool:  # don't add if thought is repeated
        self.thoughts.append(action)
        return True

    def is_thought_repeated(self) -> bool:
        if len(self.thoughts) < 2:
            self.repeated_thoughts = 0
            return False
        last = self.thoughts[-1]
        prev = self.thoughts[-2]
        if last.next_tool_name == prev.next_tool_name and last.next_tool_args == prev.next_tool_args:
            self.repeated_thoughts += 1
            return True
        self.repeated_thoughts = 0
        return False

    def to_str(self):
        messages = []
        for i, thought in enumerate(self.thoughts):
            if thought.is_deleted:
                continue

            # 标准化工具列表（向后兼容：处理旧的字符串格式）
            tool_names = (
                thought.next_tool_name if isinstance(thought.next_tool_name, list) else [thought.next_tool_name]
            )
            tool_args_list = (
                thought.next_tool_args if isinstance(thought.next_tool_args, list) else [thought.next_tool_args]
            )
            observations = thought.observation

            # 确保长度一致
            num_tools = max(len(tool_names), len(tool_args_list), len(observations) if observations else 0)
            if len(tool_args_list) < num_tools:
                tool_args_list = tool_args_list + [{}] * (num_tools - len(tool_args_list))
                logger.info(f"tool_args_list is too short {len(tool_args_list)}<{num_tools}, padding with empty dict.")
            if len(observations) < num_tools:
                observations = observations + [("", "")] * (num_tools - len(observations))
                logger.info(f"observations is too short {len(observations)}<{num_tools}, padding with empty string.")
            # 判断是否需要省略详细信息
            is_omitted = i < len(self.thoughts) - self.latest_observations_to_keep

            # 构建assistant消息：包含thought和所有工具调用
            assistant_parts = [f"next_thought:{thought.next_thought}"]

            # 为每个工具构建章节
            tool_sections = []
            for tool_idx in range(num_tools):
                tool_name = tool_names[tool_idx] if tool_idx < len(tool_names) else ""
                tool_args = tool_args_list[tool_idx] if tool_idx < len(tool_args_list) else {}

                # 格式化工具参数
                try:
                    args_str = json.dumps(tool_args, ensure_ascii=False) if tool_args else "{}"
                except Exception:
                    args_str = str(tool_args)

                tool_section = f"tool_name: {tool_name}\ntool_args: {args_str}"
                tool_sections.append(tool_section)

            # 用分隔符连接所有工具章节
            if tool_sections:
                assistant_parts.append("\n\n---\n\n".join(tool_sections))

            assistant_str = "\n".join(assistant_parts)

            # 构建user消息：包含所有观察结果
            if is_omitted:
                # 省略详细信息
                total_obs_lines = 0
                for tool_idx, obs in observations:
                    total_obs_lines += len(str(obs).splitlines())
                user_str = f"observation: {'error ocurred. ' if thought.is_error else ''}output omitted ({total_obs_lines} lines)"
            else:
                # 显示详细信息
                obs_sections = []
                for tool_idx, obs in observations:
                    obs_render = str(obs)

                    tool_name = tool_names[tool_idx] if tool_idx < len(tool_names) else f"Tool {tool_idx + 1}"
                    obs_sections.append(f"[{tool_name}]\n{obs_render}")

                if obs_sections:
                    user_str = "observation:\n\n" + "\n\n---\n\n".join(obs_sections)
                else:
                    user_str = "observation: (no observations)"

            messages.append({"role": "assistant", "content": assistant_str})
            messages.append({"role": "user", "content": user_str})
        return messages


class EnhancedToolManager:
    logs, TOOL_LIST = [], {}

    class Error(Exception):
        class ErrorType(Enum):
            SYNTAX_ERROR = 1
            RUNTIME_ERROR = 2
            TIMEOUT = 3
            FILE_NOT_FOUND = 4
            SEARCH_TERM_NOT_FOUND = 5
            UNKNOWN = 6
            THIRD_PARTY_DEPENDENCIES = 7
            MULTIPLE_SEARCH_RESULTS_FOUND = 8
            BUG_REPORT_REQUIRED = 9
            INVALID_RESPONSE_FORMAT = 10
            INVALID_TOOL_NAME = 11
            INVALID_FILE_PATH = 12
            INVALID_TOOL_CALL = 13
            IMPORT_ERROR = 14

        def __init__(self, error_type: ErrorType, message: str):
            self.error_type = error_type
            self.message = message

    def tool(fn):
        def wrapper(self, *args, **kwargs):
            self.tool_invocations[fn.__name__] += 1
            try:
                return fn(self, *args, **kwargs)
            except EnhancedToolManager.Error as e:
                self.tool_failure[fn.__name__][e.error_type] += 1
                return e.message

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        wrapper.__signature__ = inspect.signature(fn)
        wrapper.__annotations__ = fn.__annotations__.copy()
        wrapper.is_tool = True
        return wrapper

    @classmethod
    def tool_parsing(cls, fn):
        tool_schemas = None
        name = fn.__name__
        doc_fn = fn.__doc__ or ""
        doc = doc_fn.split("Arguments:")[0]
        output_description = doc_fn.split("Output:")
        if len(output_description) > 1:
            output_description = "Output: " + output_description[1].strip()
            doc = doc + "\n\n" + output_description
        sig = inspect.signature(fn)
        properties = {}
        required = []
        for param in sig.parameters.values():
            if param.name == "self":
                continue
            if param.default is param.empty and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                required.append(param.name)
            type_hint = str(param.annotation) if param.annotation != param.empty else "string"
            param_description = re.search(f"{param.name}:([^\n]+)", doc_fn)
            if param_description:
                param_description = param_description.group(1)
            else:
                raise ValueError(f"Parameter description not found for {param.name} in {doc_fn}: tool name: {name}")
            if ("list" in type_hint.lower()) and ("str" in type_hint):
                properties[param.name] = {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": param_description,
                }
                continue
            elif "str" in type_hint:
                json_type = "string"
            elif "int" in type_hint:
                json_type = "integer"
            elif "float" in type_hint:
                json_type = "number"
            elif "bool" in type_hint:
                json_type = "boolean"
            else:
                json_type = "string"
            properties[param.name] = {"type": json_type, "description": param_description}
        parameters = {"type": "object", "properties": properties, "required": required}
        tool_schemas = {"name": name, "description": doc.strip(), "input_schema": parameters}
        return tool_schemas

    @classmethod
    def get_tool_args_for_tool(self, tool_name: str, required_only: bool = False) -> list[str]:
        if tool_name not in self.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        if not required_only:
            return list(self.TOOL_LIST[tool_name]["input_schema"]["properties"].keys())
        else:
            return self.TOOL_LIST[tool_name]["input_schema"]["required"]

    @classmethod
    def _check_dependency_errors(self, output: str) -> bool:
        text = output.lower()
        return any(
            sig.lower() in text
            for sig in [
                "ModuleNotFoundError",
                "No module named",
                "ImportError",
                "pkg_resources.DistributionNotFound",
                "pkg_resources.VersionConflict",
                "INTERNALERROR",
                "Could not find a version that satisfies the requirement",
                "No matching distribution found for",
                "not configured",
                "missing module named",
                "missing dependency",
                "Failed to import",
                "Could not import",
                "cannot import",
                "cannot open shared object file",
                "undefined symbol",
                "bad magic number",
                "incompatible library",
            ]
        )

    def get_tool_docs(self) -> str:
        return "\n\n".join(
            [json.dumps(tool_metadata, ensure_ascii=False) for _, tool_metadata in self.TOOL_LIST.items()]
        )

    def get_tool(self, tool_name: str):
        if tool_name not in self.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        tool_method = getattr(self, tool_name, None)
        if tool_method is None or not callable(tool_method):
            return f"Error: tool '{tool_name}' does not exist. Please use one of the following tools: {', '.join(self.TOOL_LIST.keys())}"
        return tool_method

    def get_final_git_patch(self) -> str:
        try:
            command = """
            shopt -s globstar

            cp .gitignore .gitignore.backup 2>/dev/null || true
            echo 'src/agent.py' >> .gitignore
            echo 'src/agent_runner.py' >> .gitignore

            git add **/*.py 2>/dev/null || true
            git add **/*.toml 2>/dev/null || true
            git add **/*.cfg 2>/dev/null || true
            git add **/*.txt 2>/dev/null || true

            git diff --cached > .patch.txt
            cat .patch.txt

            mv .gitignore.backup .gitignore 2>/dev/null || true
            """
            print("Generating git patch...")
            output = subprocess.run(["bash", "-c", command], timeout=30, capture_output=True, text=True)
            return output.stdout
        except Exception as e:
            logger.error(f"Error generating git patch: {e}")
            return f"Error generating git patch: {e}"


class EnhancedNetwork:
    class ErrorType(Enum):
        EMPTY_RESPONSE = 1
        RESERVED_TOKEN_PRESENT = 2
        RATE_LIMIT_EXCEEDED = 3
        INVALID_RESPONSE_FORMAT = 4
        TIMEOUT = 5
        UNKNOWN = 6
        NETWORK_ERROR = 7
        AUTHENTICATION_ERROR = 8
        RESOURCE_EXHAUSTED = 9

    @classmethod
    def is_valid_response(cls, raw_text: str) -> bool:
        if type(raw_text) is dict and raw_text.get("error", None) is not None and raw_text.get("error") != "":
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        if not raw_text.strip().endswith("}") and not raw_text.strip().endswith("}]"):
            return False, "Incomplete response, your response must be shorter to fit within context limit"
        if len(raw_text) == 0:
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        if "<|reserved_token_" in raw_text:
            return False, cls.ErrorType.RESERVED_TOKEN_PRESENT.name
        if "API request failed with status 429" in raw_text:
            return False, cls.ErrorType.RATE_LIMIT_EXCEEDED.name
        if "Read timed out" in raw_text:
            return False, cls.ErrorType.TIMEOUT.name
        if "Network unreachable" in raw_text or "Connection refused" in raw_text:
            return False, cls.ErrorType.NETWORK_ERROR.name
        return True, None

    @classmethod
    def get_error_counter(cls) -> dict[str, int]:
        return {k: 0 for k in cls.ErrorType.__members__}

    @classmethod
    def fix_json_string_with_llm(cls, json_string: str, attempt: int = 0) -> dict:
        messages = [
            {
                "role": "system",
                "content": "Fix the json string sent by the user.  Reply only with the json string and nothing else.",
            },
            {"role": "user", "content": json_string},
        ]
        response = cls.make_request(messages, model=DEEPSEEK_MODEL_NAME)
        try:
            response = response.replace("```json", "").strip("```")
            response = json.loads(response)
            return response
        except JSONDecodeError as e:
            logger.error(f"Error fixing json string: {e},trying again..")
            logger.error(f"json string is :{json_string}")
            logger.error(f"LLM response is :{response}")
            return None

    @classmethod
    def make_request(cls, messages: list, model: str, attempt: int = 0, temperature: float = 0.0) -> str:
        global run_id
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
        print("[REQUEST] run_id:", run_id)
        request_data = {"run_id": run_id if run_id else str(uuid4()), "messages": messages, "temperature": temperature}
        headers = {"Content-Type": "application/json"}
        request_data["model"] = model
        try:
            response = requests.post(url, json=request_data, timeout=120, headers=headers)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout after 120 seconds for model {model}")
            return f"ERROR: Request timeout for model {model}"
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error for model {model}: {e}")
            return f"ERROR: Connection failed for model {model}"
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for model {model}: {e}")
            return f"ERROR: HTTP error {e.response.status_code} for model {model}"
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for model {model}: {e}")
            return f"ERROR: Request failed for model {model}"
        try:
            response_json = response.json()
        except JSONDecodeError as e:
            logger.error(f"Invalid JSON response for model {model}: {e}")
            logger.error(f"Response content: {response.text[:500]}...")
            return f"ERROR: Invalid JSON response for model {model}"
        try:
            is_oai_interface = (
                type(response_json) is dict
                and response_json.get("choices") is not None
                and len(response_json.get("choices")) > 0
                and response_json.get("choices")[0].get("message") is not None
            )
            if is_oai_interface:
                raw_text = response_json["choices"][0]["message"]["content"]
            else:
                if type(response_json) is str:
                    raw_text = response_json.strip("\n").strip()
                else:
                    raw_text = response_json
            if type(raw_text) is not dict:
                raw_text = raw_text.lstrip()
            return raw_text
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error parsing response structure for model {model}: {e}")
            logger.error(f"Response JSON: {response_json}")
            return f"ERROR: Invalid response structure for model {model}"
        except Exception as e:
            logger.error(f"Unexpected error processing response for model {model}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"ERROR: Unexpected error for model {model}"

    @classmethod
    def _request_next_action_with_retry(
        cls, messages: dict, model: str, max_retries: int = 5, base_delay: float = 1.0, temperature: float = 0.0
    ) -> str:
        raw_text = "not defined"
        error_counter = cls.get_error_counter()
        next_thought, next_tool_name, next_tool_args = None, None, None
        total_attempts = 0
        for attempt in range(max_retries):
            try:
                total_attempts += 1
                index = AGENT_MODELS.index(model) if model in AGENT_MODELS else -1
                raw_text = cls.make_request(
                    messages, model=AGENT_MODELS[(index + attempt) % len(AGENT_MODELS)], temperature=temperature
                )
                is_valid, error_msg = cls.is_valid_response(raw_text)
                if not (is_valid):
                    raise Exception(error_msg)
                next_thought, next_tool_name, next_tool_args, error_msg = cls.parse_response(raw_text)
                if error_msg:
                    raise Exception(error_msg)
                break
            except Exception as e:
                error_body = str(e)
                logger.error(f"Error: {error_body}")
                if attempt < max_retries:
                    delay = base_delay
                    logger.info(error_body)
                    logger.error("--------------------------------")
                    logger.error(f"response: {raw_text}")
                    logger.error("--------------------------------")
                    logger.info(f"[agent] Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    if (
                        "RATE_LIMIT_EXCEEDED" not in error_body
                        and "RESERVED_TOKEN_PRESENT" not in error_body
                        and "EMPTY_RESPONSE" not in error_body
                        and "TIMEOUT" not in error_body
                    ):
                        messages.append({"role": "assistant", "content": raw_text})
                        messages.append({"role": "user", "content": "observation: " + error_body})
                    time.sleep(random.uniform(1.2 * delay, 1.5 * delay))
                    continue
                else:
                    error_counter[cls.ErrorType.TIMEOUT.name] += 1
                    raise RuntimeError(error_body)
        return next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages

    @classmethod
    def parse_malformed_json(cls, arguments: list[str], json_string: str) -> dict | str:
        pattern = ""
        for i, k in enumerate(arguments):
            pattern += f'"{k}": (.*)'
            if i != len(arguments) - 1:
                pattern += r",\s*"
        match = re.search(pattern, json_string)
        if not match:
            return f"Error: {json_string} can not match pattern {pattern}"
        result_json = {}
        for i in range(len(arguments)):
            value = match.group(i + 1)
            value = value.strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            value = value.replace("\\n", "\n")
            result_json[arguments[i]] = value
        return result_json

    @classmethod
    def parse_next_tool_args(cls, tool_name: str, next_tool_args: str) -> dict | str:
        """
        parse string to json, fix unecaped " in values like this: '{"a": "text "text2" text3 "text4"", "b": "text3"}'
        returns json or error message
        """

        next_tool_args = next_tool_args.replace("```json", "").strip("```")
        error_msg = ""
        try:
            next_tool_args = Utils.load_json(next_tool_args.strip())
        except JSONDecodeError:
            error_msg = f"Invalid JSON: {next_tool_args}"
            try:
                next_tool_args = cls.parse_malformed_json(
                    EnhancedToolManager.get_tool_args_for_tool(tool_name, required=True), next_tool_args
                )
            except EnhancedToolManager.Error as e:
                raise Exception(e.message)
            except Exception:
                raise Exception(error_msg)
        return next_tool_args

    @classmethod
    def inference(
        cls, messages: List[Dict[str, Any]], model: str, run_id: str = str(uuid4()), temperature: float = 0.0
    ) -> dict:
        """Prod inference with caching"""
        cleaned_msgs: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            content = m.get("content", "")
            if role == "assistant" and not content.strip():
                continue
            cleaned_msgs.append({"role": role, "content": content})
        if not cleaned_msgs:
            raise RuntimeError("No valid messages to send to proxy.")
        next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages = (
            cls._request_next_action_with_retry(cleaned_msgs, model=model, temperature=temperature)
        )
        return next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages

    @classmethod
    def sanitise_text_resp(cls, text_resp: str) -> str:
        text_resp = re.sub("['\"]*next_thought['\"]*:", "next_thought:", text_resp)
        text_resp = re.sub("['\"]*next_tool_name['\"]*:", "next_tool_name:", text_resp)
        text_resp = re.sub("['\"]*next_tool_args['\"]*:", "next_tool_args:", text_resp)
        text_resp = re.sub("['\"]*observation['\"]*:", "observation:", text_resp)
        if (
            "next_thought" not in text_resp
            and "next_tool_name:" in text_resp
            and "next_tool_args:" in text_resp
            and text_resp.find("next_tool_name:") < text_resp.find("next_tool_args:")
            and text_resp.find("next_tool_name:") > 10
        ):
            logger.info(f"next_thought not found in {text_resp[:50]}, adding it")
            text_resp = "next_thought: " + text_resp
        if (
            "next_tool_name:" in text_resp
            and "next_tool_args:" in text_resp
            and text_resp.find("next_tool_name:") < text_resp.find("next_tool_args:")
        ):
            next_tool_name = (
                text_resp.split("next_tool_name:")[1]
                .split("next_tool_args:")[0]
                .strip()
                .strip("\n")
                .strip("'")
                .strip('"')
                .strip()
            )
            text_resp = re.sub(
                f"next_tool_name:['\" ]*{next_tool_name}['\" ]*", "next_tool_name: " + next_tool_name, text_resp
            )
        return text_resp

    @classmethod
    def parse_response(cls, text_resp: str) -> tuple[str, Any, Any]:
        error_msg = None
        text_resp = text_resp.strip()
        text_resp = text_resp.split("observation:")[0]
        text_resp = text_resp.strip().strip("\n")
        text_resp = cls.sanitise_text_resp(text_resp)
        if (
            "next_thought:" in text_resp
            and "next_tool_name:" in text_resp
            and "next_tool_args:" in text_resp
            and text_resp.find("next_thought:") < text_resp.find("next_tool_name:")
            and text_resp.find("next_tool_name:") < text_resp.find("next_tool_args:")
        ):
            next_thought = text_resp.split("next_thought:")[1].split("next_tool_name:")[0].strip().strip("\n")
            next_tool_name_raw = text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip("\n")
            next_tool_args_raw = (
                text_resp.split("next_tool_args:")[1].strip().split("next_thought:")[0].strip().strip("\n")
            )
            try:
                if next_tool_name_raw.startswith("["):
                    next_tool_name = Utils.load_json(next_tool_name_raw)
                else:
                    next_tool_name = [next_tool_name_raw]
                parsed_args = cls.parse_next_tool_args(next_tool_name, next_tool_args_raw)
                if isinstance(parsed_args, list):
                    next_tool_args = parsed_args
                else:
                    next_tool_args = [parsed_args for _ in next_tool_name]
            except JSONDecodeError as e:
                error_msg = f"Invalid JSON: {str(e)}"
        else:
            if "next_thought:" not in text_resp:
                error_msg = "Invalid response. next_thought not found"
            elif "next_tool_name:" not in text_resp:
                error_msg = "Invalid response. next_tool_name not found"
            elif "next_tool_args:" not in text_resp:
                error_msg = "Invalid response. next_tool_args not found"
            elif text_resp.find("next_thought:") > text_resp.find("next_tool_name:"):
                error_msg = "Invalid response. next_thought is after next_tool_name"
            elif text_resp.find("next_tool_name:") > text_resp.find("next_tool_args:"):
                error_msg = "Invalid response. next_tool_name is after next_tool_args"
            else:
                logger.error(f"We have no clue why parsing failed. Please check this \n{text_resp}\n")
            return None, None, None, error_msg
        if len(next_tool_name) == 1:
            return next_thought, next_tool_name[0], next_tool_args[0], error_msg
        return next_thought, next_tool_name, next_tool_args, error_msg


class FunctionVisitor(ast.NodeVisitor):
    def __init__(self, file_content: str):
        self.functions = {}
        self.current_class = None
        self.class_hierarchy = []
        self.file_content = file_content

    def visit_ClassDef(self, node):
        self.class_hierarchy.append(node.name)
        self.current_class = "::".join(self.class_hierarchy)
        self.generic_visit(node)
        self.class_hierarchy.pop()
        self.current_class = "::".join(self.class_hierarchy) if self.class_hierarchy else None

    def _process_function(self, node):
        full_function_name = f"{self.current_class}::{node.name}" if self.current_class else node.name
        line_number = node.lineno
        if isinstance(node.decorator_list, list) and len(node.decorator_list) > 0:
            line_number = node.decorator_list[0].lineno
        end_line_number = line_number
        if isinstance(node.body, list) and len(node.body) > 0:
            end_line_number = node.body[-1].lineno
        lines = self.file_content.split("\n")
        body = "\n".join(lines[line_number - 1 : end_line_number])
        self.functions[full_function_name] = {"class": self.current_class, "body": body, "line_number": line_number}
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self._process_function(node)

    def visit_AsyncFunctionDef(self, node):
        self._process_function(node)

    def visit_Module(self, node):
        self.current_class = None
        self.generic_visit(node)
        self.current_class = None


class Utils:
    @classmethod
    def limit_strings(cls, strings: str, n=1000) -> str:
        strings_list = strings.split("\n")
        if len(strings_list) > n:
            return "\n".join(strings_list[:n]) + "\n..." + f"({len(strings_list) - n} more lines)"
        else:
            return strings

    @classmethod
    def load_json(cls, json_string: str) -> dict:
        try:
            return json.loads(json_string)
        except Exception:
            try:
                return eval(json_string)
            except Exception:
                logger.info("unable to fix manually, trying with llm")
                fixed_json = EnhancedNetwork.fix_json_string_with_llm(json_string)
                if fixed_json:
                    return fixed_json
                else:
                    raise JSONDecodeError("Invalid JSON", json_string, 0)


class FixTaskEnhancedToolManager(EnhancedToolManager):
    def __init__(
        self,
        available_tools: Optional[list[str]] = [],
        test_runner: str = "pytest",
        test_runner_mode: str = "FILE",
        initial_checkpoint=None,
    ):
        self.new_files_created = []
        self.is_solution_approved = False
        self.test_runner = test_runner
        self.test_runner_mode = test_runner_mode
        self.generated_test_files = []
        self.initial_checkpoint = initial_checkpoint
        for cls in self.__class__.__mro__:
            for name, attr in cls.__dict__.items():
                if getattr(attr, "is_tool", False) and name not in self.TOOL_LIST:
                    if available_tools is not None and name not in available_tools:
                        continue
                    self.TOOL_LIST[name] = self.__class__.tool_parsing(attr)
        self.tool_failure = {k: {j: 0 for j in self.Error.ErrorType.__members__} for k in self.TOOL_LIST.keys()}
        self.tool_invocations = {k: 0 for k in self.TOOL_LIST.keys()}

    def check_syntax_error(self, content: str, file_path: str = "<unknown>") -> bool:
        try:
            ast.parse(content, filename=file_path)
            return False, None
        except SyntaxError as e:
            logger.error(f"Syntax error: {e}")
            return True, f"Syntax error. {str(e)}"

    def _get_file_content(
        self,
        file_path: str,
        search_start_line: int = None,
        search_end_line: int = None,
        search_term: str = None,
        limit: int = 5000,
    ) -> str:
        if search_term:
            logger.debug(f"search_term specified: {search_term}, searching in v2")
            return self.search_in_specified_file_v2(file_path, search_term)
        if search_start_line or search_end_line:
            for start, end, name in self.get_function_ranges(file_path):
                if search_start_line and start <= search_start_line <= end < search_start_line:
                    logger.debug(
                        f"search start line {search_start_line} is between function {start}-{end} ({name}), setting to {start}"
                    )
                    search_start_line = start
                if search_end_line and start <= search_end_line < end:
                    logger.debug(
                        f"search end line {search_end_line} is between function {start}-{end} ({name}), setting to {end}"
                    )
                    search_end_line = end
            logger.debug(f"search start line: {search_start_line}, search end line: {search_end_line}")
            with open(file_path, "r") as f:
                lines = f.readlines()
                s, e = max(0, (search_start_line or 1) - 1), min(len(lines), search_end_line or len(lines))
                content = f"Lines {s + 1}-{e} of {file_path}:\n{''.join(lines[s:e])}"
        else:
            with open(file_path, "r") as f:
                content = f.read()
        return Utils.limit_strings(content, n=limit) if limit != -1 else content

    def _save(self, file_path: str, content: str) -> str:
        is_syntax_error, error = self.check_syntax_error(content)
        if not is_syntax_error:
            with open(file_path, "w") as file:
                file.write(content)
            self.new_files_created.append(file_path)
            return f"File {file_path} saved successfully"
        else:
            return f"Error saving file: {error}"

    def get_function_ranges(self, file_path: str) -> list[tuple[int, int, str]]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            return f"Error reading '{file_path}': {e}"
        try:
            tree = ast.parse("\n".join(source_lines), filename=file_path)
        except SyntaxError as e:
            return f"Error parsing '{file_path}': {e}, {traceback.format_exc()}"
        func_ranges: list[tuple[int, int, str]] = []  # (start, end, name)
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = getattr(node, "lineno", None)
                    end = getattr(node, "end_lineno", None)
                    if start is not None and end is not None:
                        func_ranges.append((start, end, node.name))
        return func_ranges

    def _extract_function_matches(self, file_path: str, search_term: str, *, max_output_lines: int = 1000) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            return f"Error reading '{file_path}': {e}"
        match_lines = [idx + 1 for idx, line in enumerate(source_lines) if search_term in line]
        if not match_lines:
            return f"'{search_term}' not found in file '{file_path}'"
        func_ranges = self.get_function_ranges(file_path)

        def _containing_function(line_no: int):
            for start, end, name in func_ranges:
                if start <= line_no <= end:
                    return (start, end, name)
            return None

        functions_to_return: list[tuple[int, int, str]] = []
        standalone_lines: list[int] = []
        for ln in match_lines:
            info = _containing_function(ln)
            if info and info not in functions_to_return:
                functions_to_return.append(info)
            elif not info:
                standalone_lines.append(ln)
        chunks: list[str] = []
        for start, end, name in functions_to_return:
            func_src = "\n".join(source_lines[start - 1 : end])
            chunks.append(f"(lines {start}-{end}):\n{func_src}")
        for ln in standalone_lines:
            chunks.append(f"{ln}:{source_lines[ln - 1]}")
        return Utils.limit_strings("\n\n".join(chunks), n=max_output_lines)

    def _truncate_output(self, output: str, max_first_lines: int = 500, max_last_lines: int = 500) -> str:
        lines = output.split("\n")
        total_lines = len(lines)
        if total_lines <= max_first_lines + max_last_lines:
            return output
        first_lines = lines[:max_first_lines]
        last_lines = lines[-max_last_lines:]
        omitted_lines = total_lines - max_first_lines - max_last_lines
        truncated = "\n".join(first_lines)
        truncated += f"\n\n... ({omitted_lines} lines omitted) ...\n\n"
        truncated += "\n".join(last_lines)
        return truncated

    def get_final_git_patch(self) -> str:
        try:
            exts = (".py", ".ini", ".cfg", ".toml")
            exclude = {"src/agent.py", "src/agent_runner.py"}
            try:
                for _p in getattr(self, "generated_test_files", []):
                    exclude.add(os.path.relpath(_p))
            except Exception:
                pass
            ls = subprocess.run(
                ["git", "ls-files", "-m", "-o", "--exclude-standard"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout.splitlines()
            to_add = [f for f in ls if f.endswith(exts) and f not in exclude]
            if to_add:
                subprocess.run(["git", "add", "--"] + to_add, check=True, timeout=30)
            diff = subprocess.run(
                ["git", "diff", "--cached", "--no-color", "--unified=3"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            patch_text = diff.stdout or ""
            return patch_text
        except Exception as e:
            return f"Error generating git patch: {e}"

    @EnhancedToolManager.tool
    def get_file_content(
        self, file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None
    ) -> str:
        """
        Retrieves file contents with optional filtering based on search term and line numbers
        Arguments:
            file_path: filesystem path to target file. This file must be python file.
            search_start_line: optional start line number to begin extraction (1-indexed)
            search_end_line: optional end line number to end extraction (1-indexed)
            search_term: optional text pattern to filter matching lines
        """
        return self._get_file_content(file_path, search_start_line, search_end_line, search_term, limit=5000)

    @EnhancedToolManager.tool
    def save_file(self, file_path: str, content: str) -> str:
        """
        Writes text content to specified filesystem location. If there are any syntax errors in the code, it rejects the edit with an error message. Do not use this tool to create test or files to reproduce the error.
        Arguments:
            file_path: target filesystem path
            content: text data to write
        """
        if "test" in file_path.lower() or "reproduce" in file_path.lower():
            return "Error: You cannot use this tool to create test or files to reproduce the error."
        return self._save(file_path, content)

    @EnhancedToolManager.tool
    def get_approval_for_solution(self, solutions: list[str], selected_solution: int, reason_for_selection: str) -> str:
        """
        This tool is used to get approval for your proposed solution. You need to propose at least 2 meaningfully different and elegant solutions to the problem. While all the solutions proposed needs to be accurate, but following are guidelines for selecting the best solution:

        Arguments:
            solutions: list of solutions proposed by you. Here each solution individually should be very detailed and then must explain why they are better than the other solutions.
            selected_solution: Index of the solution you think is the best.
            reason_for_selection: Reason for selecting the solution over other solutions.
        """
        logger.info(f"solutions: {solutions}")
        logger.info(f"selected_solution: {selected_solution}")
        logger.info(f"reason_for_selection: {reason_for_selection}")
        parsed_solutions = []
        for solution in solutions:
            sols = re.split(r"(Solution \d+:)", solution)
            sols = [f"{sols[i]}{sols[i + 1]}" for i in range(1, len(sols), 2)]  # Combine the split parts correctly
            parsed_solutions.extend(sols)
        solutions = parsed_solutions
        if type(solutions) is not list or len(solutions) < 2:
            return "Error: solutions must be a list with length at least 2."
        self.is_solution_approved = True
        return "Approved"

    @EnhancedToolManager.tool
    def search_in_all_files_content(self, search_term: str, case_sensitive: bool = False) -> str:
        """
        Search for a text pattern across all .py files in the project, excluding any file with "test" in its path.
        Use at the beginning of the workflow to locate all possible references to a function, class, or variable.

        Arguments:
            search_term: text pattern to locate (e.g., "def test_function", "*SomeClass*")
            case_sensitive: flag to determine if the search should be case-sensitive
        """
        output = []
        search_flags = 0 if case_sensitive else re.IGNORECASE
        for root, _, files in os.walk("."):
            if ".git" in root or "docs" in root:
                continue
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    if re.search(search_term, file_path, search_flags):
                        output.append(f"{file_path} | Filename match")
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        if not re.search(search_term, content, search_flags):
                            continue
                        tree = ast.parse(content, filename=file_path)
                        visitor = FunctionVisitor(content)
                        visitor.visit(tree)
                        for function_name, function_info in visitor.functions.items():
                            body = function_info["body"]
                            if re.search(search_term, body, search_flags):
                                lines = body.split("\n")
                                for idx, line in enumerate(lines):
                                    if re.search(search_term, line, search_flags):
                                        line_number = function_info["line_number"] + idx
                                        output.append(f"{file_path}:{line_number} | {function_name} | {line.rstrip()}")
                    except Exception as e:
                        logger.error(f"Error searching in file {file_path} with search term {search_term}: {e}")
        output = Utils.limit_strings("\n".join(output), n=100)
        if not output:
            return f"'{search_term}' not found in the codebase."
        return output

    @EnhancedToolManager.tool
    def search_in_specified_file_v2(self, file_path: str, search_term: str) -> str:
        """
        Locates text patterns within a specific file
        Arguments:
            file_path: target file for pattern matching. This file must be python file.
            search_term: text pattern to find (e.g., "def test_function", "*SomeClass*")
        """
        if not file_path.endswith(".py"):
            return f"Error: file '{file_path}' is not a python file."
        return self._extract_function_matches(file_path, search_term)

    @EnhancedToolManager.tool
    def generate_tests(self, file_path: str, test_code: str, position: str = "append") -> str:
        """
        Create or append tests to the specified test file. Supports inserting entire test classes, standalone test functions, and other supporting code blocks (imports, helpers). Generated tests are excluded from the final patch automatically. For unittest files, automatically detects TestCase classes and can merge new test methods into existing classes with proper indentation.
        Arguments:
            file_path: path to the test file to create or modify
            test_code: the full test code block to insert (class(es), function(s), and/or helpers)
            position: where to place the code: "append", "top", "after_imports", "before_main", or "auto"
                     - "auto" (recommended): tries class merge, then unittest method insertion, then before_main,
                       then after_imports, finally append
                     - "append": tries unittest class insertion first, then appends to end of file
        """
        if not file_path.endswith(".py"):
            return f"Error: file '{file_path}' is not a python file."
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        test_fn = (test_code or "").strip()
        if not test_fn:
            return "Error: test_function_code cannot be empty."
        is_new_file = not os.path.exists(file_path)

        def _parse_classes_and_functions(block: str):
            try:
                tree = ast.parse(block)
                lines = block.splitlines()
                classes = []
                functions = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        start = node.lineno - 1
                        end = (node.end_lineno or node.lineno) - 1
                        classes.append((node, "\n".join(lines[start : end + 1])))
                    elif isinstance(node, ast.FunctionDef):
                        start = node.lineno - 1
                        end = (node.end_lineno or node.lineno) - 1
                        functions.append((node, "\n".join(lines[start : end + 1])))
                return classes, functions
            except Exception:
                return [], []

        def _insert_into_unittest_class(content: str, block: str) -> str:
            try:
                tree = ast.parse(content)
                lines = content.splitlines()
                test_classes = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        for base in node.bases:
                            base_name = ""
                            if isinstance(base, ast.Name):
                                base_name = base.id
                            elif isinstance(base, ast.Attribute):
                                base_name = base.attr
                            if "TestCase" in base_name:
                                test_classes.append(node)
                                break
                if not test_classes:
                    return None  # No unittest class found
                test_class = test_classes[0]
                class_end_line = test_class.end_lineno - 1  # 0-indexed
                last_method_line = test_class.lineno  # Start of class
                for node in test_class.body:
                    if isinstance(node, ast.FunctionDef):
                        last_method_line = max(last_method_line, node.end_lineno - 1)
                class_indent = ""
                for i in range(test_class.lineno - 1, len(lines)):
                    line = lines[i]
                    if line.strip() and line.strip().startswith("def "):
                        class_indent = line[: len(line) - len(line.lstrip())]
                        break
                if not class_indent:
                    class_indent = "    "  # Default to 4 spaces
                dedented_block = textwrap.dedent(block)
                test_method_lines = dedented_block.split("\n")
                indented_method = []
                for line in test_method_lines:
                    if line.strip():
                        indented_method.append(class_indent + line)
                    else:
                        indented_method.append("")  # Keep empty lines
                new_lines = lines[: class_end_line + 1]
                new_lines.append("")  # Blank line before new method
                new_lines.extend(indented_method)
                new_lines.extend(lines[class_end_line + 1 :])
                return "\n".join(new_lines)
            except Exception as e:
                print(f"Error inserting into unittest class: {e}")
                return None

        def _merge_classes_into_existing(content: str, block: str) -> str:
            try:
                existing_tree = ast.parse(content)
                content_lines = content.splitlines()
                new_classes, _ = _parse_classes_and_functions(block)
                if not new_classes:
                    return None
                existing_classes = {}
                for node in ast.walk(existing_tree):
                    if isinstance(node, ast.ClassDef):
                        existing_classes[node.name] = node
                updated = content
                for cls_node, cls_src in new_classes:
                    is_test_case = False
                    for base in cls_node.bases:
                        base_name = getattr(base, "id", None) or getattr(base, "attr", None) or ""
                        if "TestCase" in str(base_name):
                            is_test_case = True
                            break
                    if not is_test_case:
                        continue
                    if cls_node.name not in existing_classes:
                        continue
                    block_lines = cls_src.splitlines()
                    new_methods_sources = []
                    for member in cls_node.body:
                        if isinstance(member, ast.FunctionDef):
                            m_start = member.lineno - 1
                            m_end = (member.end_lineno or member.lineno) - 1
                            method_src = "\n".join(block_lines[m_start : m_end + 1])
                            new_methods_sources.append(method_src)
                    if not new_methods_sources:
                        continue
                    target_cls = existing_classes[cls_node.name]
                    class_end_line = target_cls.end_lineno - 1
                    last_method_line = target_cls.lineno
                    for n in target_cls.body:
                        if isinstance(n, ast.FunctionDef):
                            last_method_line = max(last_method_line, n.end_lineno - 1)
                    class_indent = "    "
                    for i in range(target_cls.lineno - 1, len(content_lines)):
                        line = content_lines[i]
                        if line.strip().startswith("def "):
                            class_indent = line[: len(line) - len(line.lstrip())]
                            break
                    indented_methods = []
                    for m_src in new_methods_sources:
                        dedented = textwrap.dedent(m_src)
                        for ln in dedented.splitlines():
                            indented_methods.append((class_indent + ln) if ln.strip() else "")
                        indented_methods.append("")
                    lines = updated.splitlines()
                    new_lines = lines[: class_end_line + 1] + [""] + indented_methods + lines[class_end_line + 1 :]
                    updated = "\n".join(new_lines)
                if updated != content:
                    return updated
                return None
            except Exception as e:
                print(f"Error merging classes: {e}")
                return None

        def _insert_after_imports(content: str, block: str) -> str:
            lines = content.splitlines()
            insert_idx = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    insert_idx = i + 1
                elif stripped == "" or stripped.startswith("#"):
                    insert_idx = max(insert_idx, i + 1)
                else:
                    break
            lines = (
                lines[:insert_idx] + (["", block, ""] if insert_idx < len(lines) else ["", block]) + lines[insert_idx:]
            )
            return "\n".join(lines).rstrip() + "\n"

        def _insert_before_main(content: str, block: str) -> str:
            marker = 'if __name__ == "__main__":'
            idx = content.find(marker)
            if idx == -1:
                return None
            return content[:idx].rstrip() + "\n\n" + block + "\n\n" + content[idx:]

        if is_new_file:
            new_content = test_fn + "\n"
            is_err, err = self.check_syntax_error(new_content)
            if is_err:
                return f"Error: generated test function has syntax error: {err}"
        else:
            original = self._get_file_content(file_path, limit=-1)
            if test_fn in original:
                rel = os.path.relpath(file_path)
                if rel not in self.generated_test_files:
                    self.generated_test_files.append(rel)
                return f"Test already present in '{rel}', no changes made."
            candidates = []
            if position == "append":
                candidates = [
                    lambda src: _merge_classes_into_existing(src, test_fn) or _insert_into_unittest_class(src, test_fn),
                    lambda src: src.rstrip() + "\n\n" + test_fn + "\n",
                ]
            elif position == "top":
                candidates = [lambda src: test_fn + "\n\n" + src]
            elif position == "after_imports":
                candidates = [lambda src: _insert_after_imports(src, test_fn)]
            elif position == "before_main":
                candidates = [lambda src: (_insert_before_main(src, test_fn) or src.rstrip() + "\n\n" + test_fn + "\n")]
            elif position == "auto":
                candidates = [
                    lambda src: _merge_classes_into_existing(
                        src, test_fn
                    ),  # FIRST: Try to merge into existing TestCase classes
                    lambda src: _insert_into_unittest_class(src, test_fn),  # THEN: Try to insert as test method
                    lambda src: (_insert_before_main(src, test_fn) or _insert_after_imports(src, test_fn)),
                    lambda src: src.rstrip() + "\n\n" + test_fn + "\n",
                    lambda src: test_fn + "\n\n" + src,
                ]
            else:
                return f"Error: invalid position '{position}'. Use 'append', 'top', 'after_imports', 'before_main', or 'auto'."
            new_content = None
            first_error = None
            for builder in candidates:
                try:
                    candidate = builder(original)
                    is_err, err = self.check_syntax_error(candidate)
                    if not is_err:
                        new_content = candidate
                        break
                    if first_error is None:
                        first_error = err
                except Exception as e:
                    if first_error is None:
                        first_error = e
                    continue
            if new_content is None:
                return f"Error: inserting test caused syntax error. First error: {first_error}"
        self._save(file_path, new_content)
        rel = os.path.relpath(file_path)
        if rel not in self.generated_test_files:
            self.generated_test_files.append(rel)

        return f"Tests {'created' if is_new_file else 'updated'} in '{rel}' (position={position})."

    @EnhancedToolManager.tool
    def run_repo_tests(self, file_paths: List[str]) -> str:
        """
        Runs the tests for the repository. This tool will only run the tests for the files provided.
        Arguments:
            file_paths: path of the files to run the tests for.
        Output:
            Returns the stdout/stderr from the executed files.
        """
        if self.test_runner == "pytest":
            file_paths_str = ", ".join([f"'{f}'" for f in file_paths])
            cmd = PYTEST_COMMAND_TEMPLATE.format(file_paths=file_paths_str)
            print(f"Running command: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=150)
            output = (result.stdout or "") + (result.stderr or "")
        elif self.test_runner == "unittest":
            output = ""
            for file_path in file_paths:
                result = subprocess.run(["python", file_path], capture_output=True, text=True, timeout=60)
                current_output = (result.stdout or "") + (result.stderr or "")
                output += current_output
        else:
            if self.test_runner_mode == "MODULE":
                modules = [filepath_to_module(f, os.getcwd(), self.test_runner) for f in file_paths]
                cmd = f"{self.test_runner} {' '.join(modules)}"
                print(f"Running command: {cmd}")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=150)
                output = (result.stdout or "") + (result.stderr or "")
            else:
                files_to_test = [clean_filepath(f, os.getcwd(), self.test_runner) for f in file_paths]
                cmd = f"{self.test_runner} {' '.join(files_to_test)}"
                print(f"Running command: {cmd}")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=150)
                output = (result.stdout or "") + (result.stderr or "")
            if self._check_dependency_errors(output):
                file_paths_str = ", ".join([f"'{f}'" for f in file_paths])
                cmd = PYTEST_COMMAND_TEMPLATE.format(file_paths=file_paths_str)
                print(f"Running command: {cmd}")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=150)
                output = (result.stdout or "") + (result.stderr or "")
        return self._truncate_output(output)

    @EnhancedToolManager.tool
    def run_code(self, content: str, file_path: str) -> str:
        """
        Runs any python code. You can use this tool directly to run any test code or bug reproduction code.
        Saves the code at the given file_path and then runs it. Do not use this tool to create test or files to reproduce the error unless user has specifically asked you to create test files as part of problem statement.

        Arguments:
            content: text code to write in file
            file_path: path of the file to save the code in. This file should always be in the current working directory.
        """
        self._save(file_path, content)
        self.generated_test_files.append(file_path)
        result = subprocess.run(["python", file_path], capture_output=True, text=True, check=False, timeout=60)
        if result.returncode != 0:
            lines = content.split("\n")
            numbered_lines = [f"{i + 1:6}|{line}" for i, line in enumerate(lines)]
            code_with_lines = "\n".join(numbered_lines)
            return f"Error running code: {result.stderr}\n\nCode with line numbers:\n{code_with_lines}"
        observation = f"{result.stdout}\n"
        return observation

    @EnhancedToolManager.tool
    def apply_code_edit(self, file_path: str, search: str, replace: str) -> str:
        """
        Performs targeted text replacement within source files. If there are any syntax errors in the code, it rejects the edit with an error message. Please note use you can only use this tool after you have approval from user on your proposed solution.
        Arguments:
            file_path: target file for modification
            search: exact text pattern to locate and replace
            replace: new text content to substitute
        Output:
            operation status - success confirmation or detailed error with guidance
        """

        def add_context_to_similar_match(original_content: str, formatted_match: str, context_lines: int = 2) -> str:
            """Add context lines around a similar match for better understanding."""
            lines = original_content.split("\n")
            match_lines = formatted_match.split("\n")
            if len(match_lines) < 2:
                return formatted_match
            actual_content_lines = match_lines[1:]
            actual_content = "\n".join(actual_content_lines)
            best_match_start = -1
            best_similarity = 0
            for i in range(len(lines) - len(actual_content_lines) + 1):
                candidate_lines = lines[i : i + len(actual_content_lines)]
                candidate_content = "\n".join(candidate_lines)
                import difflib

                similarity = difflib.SequenceMatcher(None, actual_content.strip(), candidate_content.strip()).ratio()
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_start = i
            if best_match_start == -1:
                return formatted_match
            start_line = max(0, best_match_start - context_lines)
            end_line = min(len(lines), best_match_start + len(actual_content_lines) + context_lines)
            context_lines_list = []
            for i in range(start_line, end_line):
                line_num = i + 1
                prefix = ">>> " if best_match_start <= i < best_match_start + len(actual_content_lines) else "    "
                context_lines_list.append(f"{prefix}{line_num:4}| {lines[i]}")
            description = (
                match_lines[0]
                if match_lines
                else f"Match found at lines {best_match_start + 1}-{best_match_start + len(actual_content_lines)}"
            )
            return f"{description}\n" + "\n".join(context_lines_list)

        def find_most_similar_content(
            original_content: str, search_string: str, max_results: int = 3
        ) -> list[tuple[float, str]]:
            """Find the most similar content chunks to the search string."""
            import difflib

            lines = original_content.split("\n")
            chunks = []
            for i, line in enumerate(lines):
                if line.strip():  # Skip empty lines
                    chunks.append((f"Line {i + 1}: {line.strip()}", line.strip()))
            search_lines = search_string.split("\n")
            target_chunk_size = max(3, len(search_lines))

            for i in range(len(lines) - target_chunk_size + 1):
                chunk_lines = lines[i : i + target_chunk_size]
                chunk_content = "\n".join(chunk_lines).strip()
                if chunk_content:
                    chunks.append((f"Lines {i + 1}-{i + target_chunk_size}: ...", chunk_content))
            similarities = []
            for chunk_desc, chunk_content in chunks:
                ratio = difflib.SequenceMatcher(None, search_string.strip(), chunk_content).ratio()
                if ratio > 0.3:  # Only include reasonably similar content
                    similarities.append((ratio, chunk_desc, chunk_content))
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [(ratio, f"{desc}\n{content}") for ratio, desc, content in similarities[:max_results]]

        if search == replace:
            return "ERROR: search and replace are the same. Please provide a different search and replace."
        if not self.is_solution_approved:
            return "Error: You cannot use this tool before you have approval from user on your proposed solution. Please call get_approval_for_solution tool first with list of proposed solutions."
        if not os.path.exists(file_path):
            return f"Error: file '{file_path}' does not exist."

        original = self._get_file_content(file_path, limit=-1)

        match original.count(search):
            case 0:
                # Find most similar content to help LLM correct the search string
                similar_matches = find_most_similar_content(original, search, 1)

                error_msg = f"Error: search string not found in file {file_path}."

                if similar_matches:
                    error_msg += "\n\nMost similar snippet found (you may need to adjust your search string):"
                    for i, (ratio, content) in enumerate(similar_matches, 1):
                        similarity_pct = int(ratio * 100)
                        # Add context lines around the match for better understanding
                        content_with_context = add_context_to_similar_match(original, content, context_lines=2)
                        error_msg += f"\n\n{i}. Similarity: {similarity_pct}%\n{content_with_context}"
                else:
                    error_msg += " No similar content found. Please check the file content and provide the exact code you want to replace."

                return error_msg
            case 1:
                new_content = original.replace(search, replace)
                try:
                    is_error, error = self.check_syntax_error(new_content)
                    if not is_error:
                        self._save(file_path, new_content)
                        replace_pos = new_content.find(replace)
                        if replace_pos != -1:
                            lines = new_content.split("\n")
                            chars_so_far = 0
                            replace_line_start = 0
                            for i, line in enumerate(lines):
                                if chars_so_far + len(line) >= replace_pos:
                                    replace_line_start = i
                                    break
                                chars_so_far += len(line) + 1  # +1 for newline
                            replace_lines_count = replace.count("\n") + 1
                            replace_line_end = replace_line_start + replace_lines_count - 1
                            start_line = max(0, replace_line_start - 20)
                            end_line = min(len(lines), replace_line_start + 20)

                            context_lines = []
                            for i in range(start_line, end_line):
                                line_num = i + 1
                                if replace_line_start <= i <= replace_line_end:
                                    prefix = ">>> "
                                else:
                                    prefix = "    "
                                context_lines.append(f"{prefix}{line_num:4}| {lines[i]}")
                            context = "\n".join(context_lines)
                            return f"ok, code edit applied successfully. Here is the edited code (lines {start_line + 1}-{end_line}):\n\n{context}"
                        else:
                            return "ok, code edit applied successfully"
                    else:
                        error.message = "code edit failed. " + error.message
                        raise error
                except Exception as e:
                    return f"Error: syntax error in file {file_path}. {str(e)}"
            case num_hits:
                return f"Error: search string found {num_hits} times in file '{file_path}'.\nPlease reformulate your search and replace to apply only one change."

    @EnhancedToolManager.tool
    def finish(self):
        """
        Signals completion of the current workflow execution
        Arguments:
            None
        """
        return "finish"


def set_env_for_agent():
    if os.getcwd() not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + os.getcwd()
    if Path(os.getcwd() + "/lib").exists() and os.getcwd() + "/lib" not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"] + ":" + os.getcwd() + "/lib"

    work_dir = os.getcwd()
    original_cwd = os.getcwd()

    try:
        os.chdir(work_dir)
        if not os.path.exists(".git"):
            subprocess.run(["git", "init"], check=True)
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
            subprocess.run(["git", "config", "--global", "user.email", "agent@sandbox.local"], check=True)
            subprocess.run(["git", "config", "--global", "user.name", "sandbox_agent"], check=True)
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], check=False, capture_output=True, text=True)
        else:
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
    except Exception as e:
        logger.error(f"ERROR: Could not initialize git repository: {e}")
    finally:
        os.chdir(original_cwd)


def clean_code_response(response: str) -> str:
    return response.strip().removeprefix("```python").removeprefix("```").removesuffix("```").strip()


def is_all_tests_passed(output: str) -> bool:
    has_failures = any(k in output.upper() for k in ["FAILED", "ERROR:", "FAIL:", "TRACEBACK"])
    if not has_failures:
        lines = output.split("\n")
        for line in lines:
            stripped = line.strip()
            if len(stripped) > 0 and all(c in ".FE" for c in stripped) and ("F" in stripped or "E" in stripped):
                return False
    return not has_failures


def extract_and_write_files(initial_solution: str, base_dir: str = ".") -> list:
    import os

    if not initial_solution.strip():
        return []
    created_files = []
    current_file, content = None, []

    def write_file():
        if current_file and content:
            path = os.path.join(base_dir, current_file)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(content).strip())
            created_files.append(path)
            print(f"Created file: {path}")

    for line in initial_solution.split("\n"):
        stripped = line.strip()
        if stripped.endswith(".py") and " " not in stripped and len(stripped) > 3 and not stripped.startswith("#"):
            write_file()
            current_file, content = stripped, []
        elif current_file:
            content.append(line)

    write_file()
    return created_files


def filepath_to_module(file_path: str, repo_path: str, test_runner: str) -> str:
    path = os.path.splitext(os.path.abspath(file_path))[0].removeprefix(os.path.abspath(repo_path)).lstrip(os.path.sep)
    path = path.removeprefix(os.path.dirname(test_runner)).lstrip(os.path.sep) if os.path.dirname(test_runner) else path
    return path.replace(os.path.sep, ".")


def clean_filepath(file_path: str, repo_path: str, test_runner: str) -> str:
    path = os.path.splitext(os.path.abspath(file_path))[0].removeprefix(os.path.abspath(repo_path)).lstrip(os.path.sep)
    return path.removeprefix(os.path.dirname(test_runner)).lstrip(os.path.sep) if os.path.dirname(test_runner) else path


def get_directory_tree(start_path: str = ".") -> str:
    try:
        result = subprocess.run(
            ["find", start_path, "-name", ".*", "-prune", "-o", "-print"], capture_output=True, text=True, timeout=10
        )
        return result.stdout
    except:
        return "\n".join(
            str(p) for p in Path(start_path).rglob("*") if not any(part.startswith(".") for part in p.parts)
        )


def check_problem_type(problem_statement: str) -> str:
    PROBLEM_TYPE_CHECK_PROMPT = textwrap.dedent(
        """
        You are the problem type checker that will categories problem type into:

        1. CREATE: If the problem statement is about creating a new functionality from scratch.
        2. FIX: If the problem statement is about fixing a bug, creating a new functionality or improving the existing codebase.

        Only respond with the "FIX" or "CREATE".
        """
    )
    retry = 0
    while retry < 10:
        try:
            messages = [
                {"role": "system", "content": PROBLEM_TYPE_CHECK_PROMPT},
                {"role": "user", "content": f"{problem_statement}\n# Project Tree Structure: \n{get_directory_tree()}"},
            ]
            response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)
            if response not in [PROBLEM_TYPE_CREATE, PROBLEM_TYPE_FIX]:
                retry += 1
            else:
                break
        except Exception as e:
            logger.error(f"Error: {e}")
            retry += 1
        time.sleep(2)
    return response


def get_test_runner_and_mode(repo_dir: str = "."):
    def count_test_cases(file_path: str) -> int:
        """Count the number of test cases (functions starting with 'test_') in a Python file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            import re

            test_functions = re.findall(r"^\s*def\s+test_\w+", content, re.MULTILINE)
            return len(test_functions)

        except (FileNotFoundError, UnicodeDecodeError):
            return 0

    def find_readme(file_path: str, repo_path: str) -> Optional[str]:
        current_dir = os.path.dirname(file_path)
        while True:
            for readme_name in ["README.md", "README.rst"]:
                readme_path = os.path.join(current_dir, readme_name)
                if os.path.exists(readme_path):
                    return readme_path
            if current_dir == repo_path:
                break
            current_dir = os.path.dirname(current_dir)

        return None

    def find_test_runner(readme_file_path: Optional[str] = None):
        FIND_TEST_RUNNER_PROMPT = textwrap.dedent(
            """
            You are a helpful assistant that can find the test runner for a given repository.
            - The test runner is the file that can run the individual test files and test cases. (e.g. pytest, unittest, etc.)
            - Do not use the test runner to run test for whole repository or test setup.
            - Read the README file and find the test runner. If there is no test runner, return pytest.
            - Output format should be as the following. No other texts are allowed.
            abc/test.py
            """
        )
        if not readme_file_path:
            return "pytest"
        try:
            with open(readme_file_path, "r", encoding="utf-8") as f:
                readme_content = f.read()

            response = EnhancedNetwork.make_request(
                [{"role": "system", "content": FIND_TEST_RUNNER_PROMPT}, {"role": "user", "content": readme_content}],
                model=DEEPSEEK_MODEL_NAME,
            )
            return response.strip() or "pytest"
        except Exception as e:
            logger.error(f"Error finding test runner: {e}")
            return "pytest"

    def get_test_runner_mode(test_runner: str):
        TEST_RUNNER_MODE_PROMPT = textwrap.dedent(
            """
            You are a helpful assistant that determines the mode of the test runner.
            Read the test runner file and determine if it requires a module or a file path to run the test.
            Output should be one of MODULE or FILE, No other texts are allowed.
            - MODULE: When the test runner requires a module path to run the test.
            - FILE: When the test runner requires a file path to run the test (e.g. pytest, unittest, py.test, etc.).
            """
        )

        if test_runner == "pytest":
            return "FILE"

        try:
            with open(test_runner, "r", encoding="utf-8") as f:
                runner_content = f.read()

            response = EnhancedNetwork.make_request(
                [{"role": "system", "content": TEST_RUNNER_MODE_PROMPT}, {"role": "user", "content": runner_content}],
                model=DEEPSEEK_MODEL_NAME,
            )
            return response.strip() or "FILE"
        except Exception as e:
            logger.error(f"Error determining test runner mode: {e}")
            return "FILE"

    test_runner = "pytest"
    test_runner_mode = "FILE"
    test_files = []  # Initialize the test_files list
    test_file_path = None
    for root, _, files in os.walk("."):
        for file in files:
            if "test_" in file and file.endswith(".py"):
                test_files.append(os.path.join(root, file))
    test_files.sort(key=len)

    for path in test_files:
        if count_test_cases(path) > 5:
            test_file_path = path
            break

    if not test_file_path:
        print("no test file found")
        return "pytest", "FILE"

    print(f"test_file_path: {test_file_path}")
    readme_file_path = find_readme(test_file_path, ".")
    if readme_file_path:
        print(f"README found: {readme_file_path}")
        test_runner = find_test_runner(readme_file_path)
        test_runner_mode = get_test_runner_mode(test_runner)
    else:
        print("No README found, using default pytest")
    file_paths = [test_file_path]
    logger.info(f"test_runner: {test_runner}, test_runner_mode: {test_runner_mode}")
    try:
        tool_manager = EnhancedToolManager()
        if test_runner != "pytest":
            if test_runner_mode == "MODULE":
                _file_paths = [filepath_to_module(f, repo_dir, test_runner) for f in file_paths]
            else:
                _file_paths = [clean_filepath(f, repo_dir, test_runner) for f in file_paths]
            cmd = f"{test_runner} {' '.join(_file_paths)}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            output = (result.stdout or "") + (result.stderr or "")

            has_dependency_error = tool_manager._check_dependency_errors(output)
            if not has_dependency_error:
                return test_runner, test_runner_mode
    except Exception as e:
        logger.error(f"Error getting test runner and mode: {e}")

    return "pytest", "FILE"


INFINITE_LOOP_CHECK_PROMPT = textwrap.dedent(
    """
    You are an expert code reviewer specializing in infinite loop detection and prevention. Your task is to analyze the generated Python code for potential infinite loops and provide a corrected version if issues are found.

    CRITICAL INFINITE LOOP DETECTION:
    1. Check for while True: loops without guaranteed exit conditions
    2. Verify all while loops have clear termination conditions
    3. Ensure recursive functions have proper base cases
    4. Look for loops that depend on external state that might never change
    5. Check for patterns that could lead to infinite iteration

    If you find potential infinite loops:
    - Provide a corrected version of the code
    - Ensure all loops have finite termination conditions
    - Add reasonable iteration limits or timeout mechanisms where appropriate

    If no infinite loops are detected:
    - Return the original code unchanged

    STRICT REQUIREMENT: Return the final Python code along with file names. Do not include any explanations, comments, or additional text.

    example:
    ```python
    a.py
    contents of a.py

    b.py
    contents of b.py
    ```
    """
)


def generate_initial_solution(problem_statement: str, code_skeleton: str, temperature: float = 0.7) -> str:
    GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT = textwrap.dedent(
        """
        You are an expert Python developer. Your task is to generate a complete, working Python solution for the given problem statement.
        
        Strict Requirements:
        1. Output the full content of Python files along with their file names. You **MUST** output the **file name** along with file content.
        2. Do not include explanations, comments, or markdown formatting in the main code.
        3. Use only standard Python (no external libraries).
        4. Implement all required classes and functions exactly with the same names as in the initial code stub.
        5. You may add helper functions or classes if needed, but do not remove or rename the original ones.
        6. Ensure the solution handles all edge cases, validates inputs, and produces correct outputs.
        7. The solution must be executable as-is with no placeholders or TODOs.
        8. If problem statement doesn't explicitely requires a list of strings as a response, do not use list of strings for multiline text problems, just use raw string format.
        9. **IMPORTANT**: Add clear comments above each edge case handling section to identify which specific edge case is being addressed. Use the format: `# Edge Case: [description of the edge case]`
        10. **IMPORTANT**: Add a comment at the end of each function/class that lists all edge cases handled, using the format: `# Handled Edge Cases: [list of edge cases]`

        Return only the final Python code.
        
        Response Examples:
        ```python
        a.py
        {{content}}
        
        b.py
        {{content}}
        ```
        """
    )
    retry = 0
    code_generation_messages = [
        {"role": "system", "content": GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT},
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n\nInitial python files:\n{code_skeleton}\nGenerate the complete and correct implementation in python files.\n\nSTRICT REQUIREMENT: - You **MUST** output the **file name** along with file content.\nexample:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```",
        },
    ]
    while retry < 10:
        try:
            code_response = EnhancedNetwork.make_request(
                code_generation_messages, model=QWEN_MODEL_NAME, temperature=temperature
            )
            loop_check_messages = [
                {"role": "system", "content": INFINITE_LOOP_CHECK_PROMPT},
                {
                    "role": "user",
                    "content": f"Generated Code:\n{code_response}\n\nAnalyze this code for potential infinite loops and provide a corrected version if any issues are found. Return ONLY the final Python code.",
                },
            ]
            loop_check_response = EnhancedNetwork.make_request(loop_check_messages, model=QWEN_MODEL_NAME)
            # Clean up the final response (use compat response as it's the final validated version)
            solution = clean_code_response(loop_check_response)
            lines = solution.split("\n")
            if lines[0].endswith(".py") == False:
                retry += 1
                code_generation_messages.append({"role": "assistant", "content": loop_check_response})
                code_generation_messages.append(
                    {
                        "role": "user",
                        "content": "Include file name in the response. example:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```",
                    }
                )
                continue
            return solution
        except Exception:
            retry += 1
            time.sleep(2)
    if retry >= 10:
        return ""
    return ""


def generate_single_testset(
    problem_statement: str, files_to_test: str, code_skeleton: str, temperature: float = 0.0
) -> tuple[str, set]:
    """Generate a single test set and return (testcode, function_names)"""

    GENERATE_TESTCASES_PROMPT = textwrap.dedent(
        """
        You are an expert Python unittest testcase developer. 
            Important points:-
            - you have generation limit of 2048 tokens. Hence you must stop generating more test cases when you are near the limit.
            - If you get syntax error, check if last assistant response was truncated. If yes, then skip last couple of test cases to fit in.
            
            You must respond directly with the test cases in the following format. 
            =========TEST_CASES
            <<test cases>>
            Do not include anything else. For Example:
            =========TEST_CASES

            import unittest
            from main_module import (
                main_func
            )

            class TestFuncA(unittest.TestCase):
                def test_main_func(self):
                    self.assertEqual(main_func(), "expected_output")

            if __name__ == "__main__":
                unittest.main()
        """
    )

    retry = 0
    test_generation_messages = [
        {"role": "system", "content": GENERATE_TESTCASES_PROMPT},
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n\nFiles To Test: {files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerate the complete and correct testcases in python files.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```",
        },
    ]

    while retry < 10:
        try:
            testcode_response = EnhancedNetwork.make_request(
                test_generation_messages, model=QWEN_MODEL_NAME, temperature=temperature
            )
            logger.info("Step 1 - Testcase Generation completed")

            testcases = clean_code_response(testcode_response)

            lines = testcases.split("\n")
            if lines[0].endswith(".py") == False:
                retry += 1
                test_generation_messages.append({"role": "assistant", "content": testcode_response})
                test_generation_messages.append(
                    {
                        "role": "user",
                        "content": "Include file name in the response. example:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```",
                    }
                )
                print(f"Retrying because the first line is not a python test file name:\n {testcases}")
                continue

            return testcases

        except Exception as e:
            retry += 1
            print(f"Exception in generate_single_testset: {e}")
            time.sleep(2)

    return "", set()


def extract_semantic_patterns_from_failures(solution_code: str, test_output: str, real_test_code: str = "") -> str:
    PATTERN_EXTRACTION_PROMPT = textwrap.dedent(
        """
        You are an expert at analyzing code failures and extracting semantic patterns.
        
        Your task: Analyze test failures and extract semantic patterns that reveal what the problem expects.
        
        Focus on discovering patterns from the failures themselves, not from assumptions:
        1. **Expected vs Actual Behavior**: What did tests expect vs what code produced?
        2. **Error Type Patterns**: What error types are expected for different failure modes?
        3. **Validation Order Patterns**: What order of checks do tests expect?
        4. **Edge Case Patterns**: What edge cases do the tests reveal are important?
        5. **Semantic Patterns**: What underlying concepts/patterns do the failures reveal?
        
        Output patterns as discoverable facts, not prescriptive rules.
        Let the test failures teach you what the problem expects.
        Be language-agnostic and problem-agnostic - focus on semantic patterns.
        """
    )
    try:
        content = f"""Solution Code:
{solution_code}

Test Output (showing failures):
{test_output}"""

        if real_test_code:
            content += f"""

Real Test Code (for context):
{real_test_code}"""
        content += """

Analyze the failures and extract semantic patterns that reveal what the problem expects.
Focus on what the tests teach us, not on assumptions."""
        response = EnhancedNetwork.make_request(
            [{"role": "system", "content": PATTERN_EXTRACTION_PROMPT}, {"role": "user", "content": content}],
            model=DEEPSEEK_MODEL_NAME,
            temperature=0.0,
        )
        return response
    except Exception as e:
        logger.error(f"Error extracting patterns: {e}")
        return ""


def basic_approach(code_skeleton: str, problem_statement: str, temperature: float = 0.0) -> tuple[Optional[str], str]:
    initial_solution = generate_initial_solution(problem_statement, code_skeleton, temperature)
    created_files = extract_and_write_files(initial_solution)
    test_cases = generate_single_testset(problem_statement, created_files, code_skeleton, temperature)
    test_files = extract_and_write_files(test_cases)
    for file in test_files:
        result = subprocess.run(
            ["python", file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120
        )
        print("=" * 80, result.stdout, "=" * 80, sep="\n")
        if result.returncode != 0 or not is_all_tests_passed(result.stdout):
            patterns = extract_semantic_patterns_from_failures(initial_solution, result.stdout)
            return None, patterns
    return initial_solution, ""


def advanced_approach(code_skeleton: str, problem_statement: str, timeout: int, accumulated_patterns: str = "") -> str:
    tool_manager = EnhancedToolManager()
    if accumulated_patterns:
        logger.info("Using accumulated patterns from basic approach attempts")
        initial_solution = generate_initial_solution_with_patterns(
            problem_statement, code_skeleton, accumulated_patterns
        )
    else:
        initial_solution = generate_initial_solution(problem_statement, code_skeleton)
    extract_and_write_files(initial_solution)
    patch = tool_manager.get_final_git_patch()
    return patch


def generate_initial_solution_with_patterns(
    problem_statement: str, code_skeleton: str, accumulated_patterns: str, temperature: float = 0.7
) -> str:
    GENERATE_SOLUTION_WITH_PATTERNS_PROMPT = textwrap.dedent("""
        You are an expert developer. Your task is to generate a complete, working solution for the given problem statement.
        
        **CRITICAL**: You have access to patterns discovered from previous failed attempts. These patterns reveal what the problem expects.
        
        Strict Requirements:
        1. Output the full content of files along with their file names. You **MUST** output the **file name** along with file content.
        2. Do not include explanations, comments, or markdown formatting in the main code.
        3. Use only standard libraries (no external dependencies unless specified).
        4. Implement all required classes and functions exactly with the same names as in the initial code stub.
        5. You may add helper functions or classes if needed, but do not remove or rename the original ones.
        6. Ensure the solution handles all edge cases, validates inputs, and produces correct outputs.
        7. The solution must be executable as-is with no placeholders or TODOs.
        8. Analyze the problem statement and code skeleton to understand what is required.
        9. Preserve any constants, types, or structures defined in the code skeleton.
        10. Discover patterns from the problem statement and code skeleton - let them guide your implementation.
        11. **Structured Input Validation**: When processing structured inputs that can represent multiple types:
            - **Calculate Minimum Requirement**: Analyze all type definitions in the code skeleton to determine the minimum structural requirement. For each type, count: type identifier (1 element) + number of required elements for that type. Find the minimum across all types. This minimum is typically 2 or 3, NOT 1. Store this calculated minimum in a variable.
            - **Completeness Before Type Determination**: Before identifying an item's type, check if the item's size is less than the calculated minimum. Items shorter than this minimum cannot have their type determined and should raise structural errors (indicating incompleteness), not semantic errors (indicating malformed data). This check must come BEFORE accessing any element to determine the type.
            - **Error Semantics**: Distinguish between structural errors (for items too incomplete to identify type) and semantic errors (for items that fail type-specific validation). Use the appropriate error type based on whether the issue is structural incompleteness or semantic malformation.
            - **Dynamic Inference**: Calculate all validation requirements from the code skeleton. Do not hardcode numeric values like 1, type names, or write explanatory comments - just implement the validation logic.
        
        **DISCOVERED PATTERNS FROM PREVIOUS ATTEMPTS**:
        {accumulated_patterns}
        
        These patterns were discovered from test failures. Use them to understand what the problem expects.
        Let these patterns guide your implementation, but discover the solution from the problem context.
        
        Return only the final code.
        
        Response Examples:
        ```
        a.ext
        {{content}}
        
        b.ext
        {{content}}
        ```
        """)
    retry = 0
    code_generation_messages = [
        {
            "role": "system",
            "content": GENERATE_SOLUTION_WITH_PATTERNS_PROMPT.format(accumulated_patterns=accumulated_patterns),
        },
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n\nInitial files:\n{code_skeleton}\n\n**DISCOVERED PATTERNS FROM PREVIOUS ATTEMPTS**:\n{accumulated_patterns}\n\nGenerate the complete and correct implementation.\n\nSTRICT REQUIREMENT: - You **MUST** output the **file name** along with file content.\nexample:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```",
        },
    ]
    while retry < MAX_INFERENCE_RETRIES:
        try:
            code_response = EnhancedNetwork.make_request(
                code_generation_messages, model=QWEN_MODEL_NAME, temperature=temperature
            )
            loop_check_messages = [
                {"role": "system", "content": INFINITE_LOOP_CHECK_PROMPT},
                {
                    "role": "user",
                    "content": f"Analyze this code for infinite loops:\n{code_response}",
                },
            ]
            loop_check_response = EnhancedNetwork.make_request(loop_check_messages, model=QWEN_MODEL_NAME)
            solution = clean_code_response(loop_check_response)
            lines = solution.split("\n")
            if lines[0].endswith(".py") == False:
                retry += 1
                code_generation_messages.append({"role": "assistant", "content": loop_check_response})
                code_generation_messages.append(
                    {
                        "role": "user",
                        "content": "Include file name in the response. example:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```",
                    }
                )
                continue
            return solution
        except Exception:
            retry += 1
            time.sleep(RETRY_BASE_DELAY)
    if retry >= MAX_INFERENCE_RETRIES:
        logger.error(f"Failed to generate initial solution after {MAX_INFERENCE_RETRIES} retries")
        return ""
    return ""


def execute_parallel_tools(tool_manager, tool_names: list, tool_args_list: list) -> list[tuple[int, str]]:
    """
    并行执行多个工具调用

    Args:
        tool_manager: 工具管理器实例
        tool_names: 工具名称列表
        tool_args_list: 工具参数列表，每个元素对应一个工具的参数

    Returns:
        工具执行结果列表，按输入顺序返回
    """
    if len(tool_names) != len(tool_args_list):
        return [f"Error: Mismatch between tool names ({len(tool_names)}) and args ({len(tool_args_list)})"]

    def execute_single_tool(tool_name: str, tool_args: dict) -> tuple[int, str]:
        """执行单个工具并返回结果"""
        try:
            tool = tool_manager.get_tool(tool_name)
            if isinstance(tool, str):  # Error message
                return (tool_names.index(tool_name), tool)
            if tool_args:
                result = tool(**tool_args)
            else:
                result = tool()
            return (tool_names.index(tool_name), result)
        except EnhancedToolManager.Error as e:
            return (tool_names.index(tool_name), f"observation: {e.message}")
        except Exception as e:
            import traceback

            error_traceback = traceback.format_exc()
            if isinstance(e, TypeError):
                error_msg = f"observation: {str(e)}"
            else:
                error_msg = f"observation: {repr(e)} {error_traceback}"
            return (tool_names.index(tool_name), error_msg)

    # 使用线程池并行执行
    results = [None] * len(tool_names)
    with ThreadPoolExecutor(max_workers=min(len(tool_names), 10)) as executor:
        futures = {
            executor.submit(execute_single_tool, tool_name, tool_args): i
            for i, (tool_name, tool_args) in enumerate(zip(tool_names, tool_args_list))
        }
        for future in as_completed(futures):
            try:
                idx, result = future.result()
                results[idx] = result
            except Exception as e:
                idx = futures[future]
                results[idx] = f"Error executing tool: {str(e)}"

    return results


def fix_task_solve_workflow(
    problem_statement: str,
    *,
    timeout: int,
    run_id_1: str,
    test_runner: str = "pytest",
    test_runner_mode: str = "FILE",
    n_max_steps=MAX_FIX_TASK_STEPS,
    extra_fix_request: str = "",
    initial_checkpoint=None,
) -> tuple[str, List[str], List[str]]:
    global run_id
    run_id = run_id_1
    cot = EnhancedCOT(latest_observations_to_keep=30)
    tool_manager = FixTaskEnhancedToolManager(
        available_tools=[
            "get_file_content",
            "save_file",
            "get_approval_for_solution",
            "search_in_all_files_content",
            "search_in_specified_file_v2",
            "run_repo_tests",
            "run_code",
            "apply_code_edit",
            "generate_tests",
            "finish",
        ],
        initial_checkpoint=initial_checkpoint,
        test_runner=test_runner,
        test_runner_mode=test_runner_mode,
    )
    logger.info("Starting main agent execution...")
    system_prompt = FIX_TASK_SYSTEM_PROMPT.format(
        tools_docs=tool_manager.get_tool_docs(), format_prompt=FORMAT_PROMPT_V0, extra_fix_request=extra_fix_request
    )
    instance_prompt = FIX_TASK_INSTANCE_PROMPT_TEMPLATE.format(problem_statement=problem_statement)
    start_time = time.time()
    last_test_result = "success"
    logs: List[str] = []
    for step in range(n_max_steps):
        logger.info(f"Execution step {step + 1}/{n_max_steps}")
        if time.time() - start_time > timeout:
            cot.add_action(
                EnhancedCOT.Action(
                    next_thought="global timeout reached",
                    next_tool_name="",
                    next_tool_args={},
                    observation="",
                    is_error=True,
                    inference_error_counter={},
                    request_data=[],
                )
            )
            break
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
        ]
        messages.extend(cot.to_str())
        messages.append({"role": "system", "content": STOP_INSTRUCTION})
        temperature = 0
        selected_model = GLM_MODEL_NAME
        if cot.is_thought_repeated():
            logger.info("[TEST_PATCH_FIND] Thought repeated, adding DO NOT REPEAT TOOL CALLS instruction")
            last_thought = cot.thoughts[-1]
            messages.append(
                {
                    "role": "user",
                    "content": DO_NOT_REPEAT_TOOL_CALLS.format(
                        previous_response=f"next_tool_name:{last_thought.next_tool_name}\n next_tool_args:{last_thought.next_tool_args}"
                    ),
                }
            )
            if cot.repeated_thoughts > 1:
                temperature = min(cot.repeated_thoughts / 10, 0.7)
                selected_model = (
                    AGENT_MODELS[random.randint(0, len(AGENT_MODELS) - 1)]
                    if cot.repeated_thoughts > 2
                    else GLM_MODEL_NAME
                )
                logger.info(
                    f"Use temperature={temperature} and model={selected_model} for higher cot.repeated_thoughts={cot.repeated_thoughts}"
                )
        try:
            next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages = (
                EnhancedNetwork.inference(messages, model=selected_model, run_id=run_id, temperature=temperature)
            )
        except Exception as e:
            import traceback  # Ensure traceback is accessible

            error_msg = f"\n\nERROR: {repr(e)} {traceback.format_exc()}"
            logger.error(f"Inference error: {error_msg}")
            cot.add_action(
                EnhancedCOT.Action(
                    next_thought=error_msg,
                    next_tool_name="",
                    next_tool_args={},
                    observation="",
                    is_error=True,
                    raw_response=raw_text,
                    total_attempts=total_attempts,
                ),
                inference_error_counter=error_counter,
                request_data=messages,
            )
            break
        # 统一处理：将单个工具调用转换为列表格式，统一使用并行执行逻辑
        # 单个工具调用可以看作是并行调用的特例（只有一个工具）
        is_parallel = isinstance(next_tool_name, list)

        # 标准化为列表格式
        if not is_parallel:
            # 单个工具调用：转换为列表格式
            if '"' in next_tool_name or "'" in next_tool_name:
                next_tool_name = next_tool_name.replace('"', "").replace("'", "")
            tool_names = [next_tool_name]
            tool_args_list = [next_tool_args] if next_tool_args else [{}]
        else:
            # 已经是并行调用：清理工具名称中的引号
            tool_names = []
            for tool_name in next_tool_name:
                if isinstance(tool_name, str):
                    cleaned_name = tool_name.replace('"', "").replace("'", "")
                    tool_names.append(cleaned_name)
                else:
                    tool_names.append(tool_name)

            # 确保参数是列表格式
            if not isinstance(next_tool_args, list):
                tool_args_list = [next_tool_args] * len(tool_names)
            else:
                tool_args_list = next_tool_args

        # 统一执行（单个或并行都使用相同的执行逻辑）
        num_tools = len(tool_names)
        execution_type = "parallel" if num_tools > 1 else "single"
        logger.info(f"About to execute {num_tools} tool(s) ({execution_type}): {tool_names}")

        logger.info(f"next_thought: {next_thought}\nnext_tool_name: {tool_names}\nnext_tool_args: {tool_args_list}\n")

        # 执行工具（单个工具也会通过并行执行函数处理，但只有一个线程）
        execution_results = execute_parallel_tools(tool_manager, tool_names, tool_args_list)

        # 统一使用列表格式（单个和并行都使用列表，简化逻辑）
        logger.info(f"{'Single' if num_tools == 1 else 'Parallel'} execution completed. {num_tools} tool(s) executed")

        # 统一创建 Action（单个和并行使用相同的逻辑，都使用列表格式）
        cot.add_action(
            EnhancedCOT.Action(
                next_thought=next_thought,
                next_tool_name=tool_names,  # 统一使用列表格式
                next_tool_args=tool_args_list,  # 统一使用列表格式
                observation=execution_results,  # 统一使用列表格式
                is_error=False,
                raw_response=raw_text,
                total_attempts=total_attempts,
                inference_error_counter=error_counter,
                request_data=messages,
                is_parallel=(num_tools > 1),  # 仅用于标识语义，不影响格式
            )
        )

        # 统一检查测试结果（单个和并行都支持）
        if "run_repo_tests" in tool_names:
            test_idx = tool_names.index("run_repo_tests")
            test_result = execution_results[test_idx]
            if "failed" in str(test_result).lower():
                last_test_result = "failed"
            else:
                last_test_result = "success"

        # 检查finish工具（统一处理单个和并行调用）
        if "finish" in tool_names:
            # 如果包含finish，需要检查测试结果
            if last_test_result == "failed":
                messages.append({"role": "user", "content": "The tests failed. Please fix the code and try again."})
                continue
            logger.info("[CRITICAL] Workflow called finish operation")
            break
        print(f"[CRITICAL] Completed step {step + 1}, continuing to next step")
    else:
        cot.add_action(
            EnhancedCOT.Action(
                next_thought="global timeout reached",
                next_tool_name="",
                next_tool_args={},
                observation="",
                is_error=True,
            )
        )
        logger.info(f"[CRITICAL] Workflow completed after reaching MAX_STEPS ({n_max_steps})")
        if n_max_steps < MAX_FIX_TASK_STEPS:
            return None
    logger.info(f"[CRITICAL] Workflow execution completed after {step + 1} steps")
    logger.info("[CRITICAL] About to generate final patch...")
    patch = tool_manager.get_final_git_patch()
    logger.info(f"Final Patch Generated..: Length: {len(patch)}")

    return patch


def process_create_task(input_dict):
    def get_code_skeleton() -> str:
        return "\n\n".join(
            f"{f}\n{{\n{open(os.path.join(r, f)).read()}\n}}"
            for r, _, files in os.walk(".")
            for f in files
            if f.endswith(".py")
        )

    problem_statement = input_dict.get("problem_statement", "")
    tool_manager = EnhancedToolManager()
    code_skeleton = get_code_skeleton()
    timeout = DEFAULT_TIMEOUT
    BASIC_APPROACH_RETRY = 20
    min_temperature = 0.1
    max_temperature = 1.2
    temperature_schedule, accumulated_patterns = [], []
    warmup_steps = 2
    plateau_steps = 4
    high_steps = BASIC_APPROACH_RETRY - (warmup_steps + plateau_steps)
    if high_steps < 0:
        warmup_steps = 1
        plateau_steps = BASIC_APPROACH_RETRY // 2
        high_steps = BASIC_APPROACH_RETRY - (warmup_steps + plateau_steps)
    mid_temperature = 0.6
    for i in range(warmup_steps):
        t = min_temperature + (mid_temperature - min_temperature) * (i / max(1, warmup_steps - 1))
        temperature_schedule.append(round(t, 3))
    for _ in range(plateau_steps):
        t = mid_temperature + random.uniform(-0.09, 0.09)
        t = max(min_temperature, min(max_temperature, t))
        temperature_schedule.append(round(t, 3))
    for i in range(high_steps):
        if high_steps > 1:
            t = mid_temperature + (max_temperature - mid_temperature) * (i / (high_steps - 1))
        else:
            t = max_temperature
        temperature_schedule.append(round(t, 3))
    for attempt, temperature in enumerate(temperature_schedule):
        os.system("git reset --hard")
        logger.info(f"[CREATE] Attempt {attempt + 1}/{BASIC_APPROACH_RETRY} using temperature={temperature}")
        initial_solution, patterns = basic_approach(code_skeleton, problem_statement, temperature=temperature)
        if patterns:
            accumulated_patterns.append(f"Attempt {attempt + 1} patterns:\n{patterns}")
            logger.info(f"Extracted patterns from attempt {attempt + 1}")
        if initial_solution is not None:
            extract_and_write_files(initial_solution)
            return tool_manager.get_final_git_patch()
        sleep_time = 1 + 0.5 * (temperature - min_temperature)
        time.sleep(sleep_time)

    logger.info(
        f"[CREATE] Basic approach failed after {BASIC_APPROACH_RETRY} retries with advanced temperature control, trying advanced approach"
    )
    combined_patterns = "\n\n".join(accumulated_patterns) if accumulated_patterns else ""
    if combined_patterns:
        logger.info(f"Passing {len(accumulated_patterns)} accumulated patterns to advanced approach")
    return advanced_approach(code_skeleton, problem_statement, timeout, combined_patterns)


def process_fix_task(input_dict: Dict[str, Any]):
    global run_id
    problem_text = input_dict.get("problem_statement")
    if not problem_text:
        raise ValueError("input_dict must contain 'problem_statement'.")
    timeout = int(os.getenv("AGENT_TIMEOUT", str(DEFAULT_TIMEOUT)))
    logs = []
    patch_text = ""  # Initialize to avoid UnboundLocalError
    repo_path = os.getenv("REPO_PATH", "/sandbox/repo")
    repod_dir = repo_path.split("/")[-1]
    if os.path.exists(repod_dir):
        os.chdir(repod_dir)
    set_env_for_agent()
    cwd = os.getcwd()
    logger.info(f"Current working directory: {cwd} and environ:{os.environ}")
    test_runner, test_runner_mode = get_test_runner_and_mode()
    print(f"test_runner: {test_runner}, test_runner_mode: {test_runner_mode}")

    try:
        patch_text = fix_task_solve_workflow(
            problem_text, timeout=timeout, run_id_1=run_id, test_runner=test_runner, test_runner_mode=test_runner_mode
        )
        os.system("git reset --hard")
    except Exception as e:
        import traceback  # Ensure traceback is accessible

        error_info = f"Error: {e}, {traceback.format_exc()}"
    finally:
        os.chdir(cwd)
    return patch_text


def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo"):
    global DEFAULT_PROXY_URL, DEFAULT_TIMEOUT, run_id
    run_id = os.getenv("RUN_ID", "")
    repo_dir = os.path.abspath(repo_dir)
    sys.path.insert(0, repo_dir)
    if os.path.exists(repo_dir):
        os.chdir(repo_dir)
    set_env_for_agent()
    try:
        problem_type = check_problem_type(input_dict.get("problem_statement"))
        if problem_type == PROBLEM_TYPE_FIX:
            result = process_fix_task(input_dict)
        else:
            result = process_create_task(input_dict)
    except Exception as e:
        logger.error(f"Error in agent_main: {e}")
        result = process_fix_task(input_dict)
    print(f"result: {result}")
    return result
