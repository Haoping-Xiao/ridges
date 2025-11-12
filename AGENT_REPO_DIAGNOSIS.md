# Agent 仓库初始化和运行诊断文档

本文档详细说明代理（Agent）如何初始化仓库和运行仓库的完整流程。

## 一、整体流程概览

代理的执行流程分为以下几个阶段：

1. **Sandbox 初始化阶段** - 在 Docker 容器中准备环境
2. **Agent 代码加载阶段** - 加载并执行 agent.py
3. **仓库初始化阶段** - 设置工作目录和 Git 仓库
4. **问题处理阶段** - 根据问题类型执行相应的工作流
5. **测试运行阶段** - 运行仓库测试验证修复

---

## 二、详细流程

### 阶段 1: Sandbox 初始化（外部调用）

**位置**: `evaluator/problem_suites/problem_suite.py::initialize_agent_sandbox()`

```89:128:evaluator/problem_suites/problem_suite.py
    def initialize_agent_sandbox(
        self,
        sandbox_manager: SandboxManager,
        problem: Problem,
        evaluation_run_id: UUID,
        agent_code: str,
        *,
        include_solution: bool = False
    ) -> Sandbox:
        try:
            def _on_mount(temp_dir: str):
                # Create /sandbox/agent.py
                with open(os.path.join(temp_dir, "agent.py"), "w") as f:
                    f.write(agent_code)
                
                # Create /sandbox/repo directory
                sandbox_repo_dir = os.path.join(temp_dir, "repo")
                os.mkdir(sandbox_repo_dir)

                # Copy problem files to /sandbox/repo
                self.copy_problem_files_to_directory(problem, sandbox_repo_dir)

                if include_solution:
                    # Create /sandbox/solution.diff
                    with open(os.path.join(temp_dir, "solution.diff"), "w") as f:
                        f.write(problem.solution_diff)



            return sandbox_manager.initialize_sandbox(
                name=f"agent-sandbox-{problem.name}-{evaluation_run_id}",
                python_script_path=os.path.join(os.path.dirname(__file__), "AGENT_RUNNER.py"),
                input_data={
                    "problem_statement": problem.problem_statement
                },
                env_vars={
                    "RUN_ID": evaluation_run_id
                },
                on_mount=_on_mount,
            )
```

**关键步骤**：
1. 创建 `/sandbox/agent.py` - 写入代理代码
2. 创建 `/sandbox/repo/` 目录
3. 复制问题文件到 `/sandbox/repo/`（通过 `copy_problem_files_to_directory()`）
4. 可选：如果 `include_solution=True`，创建 `/sandbox/solution.diff`
5. 创建 Sandbox 容器，指定运行 `AGENT_RUNNER.py`

---

### 阶段 2: Agent Runner 执行

**位置**: `evaluator/problem_suites/AGENT_RUNNER.py`

```8:36:evaluator/problem_suites/AGENT_RUNNER.py
def main():
    print("[AGENT_RUNNER] Entered main()")

    time.sleep(3)

    try:
        # Read input.json
        print("[AGENT_RUNNER] Reading input.json")
        with open("/sandbox/input.json", "r") as f:
            input_data = json.load(f)
        print("[AGENT_RUNNER] Read input.json")
        
        # Import agent module
        print("[AGENT_RUNNER] Loading /sandbox/agent.py")
        spec = importlib.util.spec_from_file_location("agent", "/sandbox/agent.py")
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)
        print("[AGENT_RUNNER] Loaded /sandbox/agent.py")
        
        # Check for the agent_main() function in /sandbox/agent.py
        if hasattr(agent_module, "agent_main"):
            print("[AGENT_RUNNER] agent_main() function found in /sandbox/agent.py")
        else:
            print("[AGENT_RUNNER] agent_main() function not found in /sandbox/agent.py")
            raise Exception("agent_main() function not found in /sandbox/agent.py")
        
        # Invoke agent_main function
        print("[AGENT_RUNNER] Entering agent's agent_main()")
        agent_main_return_value = agent_module.agent_main(input_data)
        print("[AGENT_RUNNER] Exited agent's agent_main()")
```

**关键步骤**：
1. 读取 `/sandbox/input.json` 获取输入数据（包含 `problem_statement`）
2. 动态加载 `/sandbox/agent.py` 模块
3. 验证 `agent_main()` 函数存在
4. 调用 `agent_main(input_data)` 开始处理问题

---

### 阶段 3: Agent Main 入口

**位置**: `agent.py::agent_main()`

```2404:2422:agent.py
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
```

**关键步骤**：
1. 获取 `RUN_ID` 环境变量
2. 设置 `repo_dir` 为绝对路径（默认为 `"repo"`，即 `/sandbox/repo`）
3. 将 `repo_dir` 添加到 `sys.path`
4. 如果 `repo_dir` 存在，切换到该目录
5. **调用 `set_env_for_agent()` 初始化仓库**（见阶段 4）
6. 根据问题类型调用相应处理函数（`process_fix_task` 或 `process_create_task`）

---

### 阶段 4: 仓库初始化（核心）

**位置**: `agent.py::set_env_for_agent()`

```1511:1534:agent.py
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
```

**关键步骤**：

#### 4.1 设置 Python 路径
- 将当前工作目录添加到 `PYTHONPATH`
- 如果存在 `lib` 目录，也添加到 `PYTHONPATH`

#### 4.2 初始化 Git 仓库
- **如果 `.git` 不存在**：
  1. 运行 `git init` 初始化仓库
  2. 配置 Git 安全目录：`git config --global --add safe.directory <work_dir>`
  3. 设置 Git 用户邮箱：`agent@sandbox.local`
  4. 设置 Git 用户名：`sandbox_agent`
  5. 添加所有文件：`git add .`
  6. 创建初始提交：`git commit -m "Initial commit"`（允许失败）
- **如果 `.git` 已存在**：
  - 仅配置安全目录

**注意**：此函数会保存并恢复当前工作目录，确保不影响调用者。

---

### 阶段 5: 处理修复任务

**位置**: `agent.py::process_fix_task()`

```2372:2401:agent.py
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
```

**关键步骤**：
1. 获取 `REPO_PATH` 环境变量（默认 `/sandbox/repo`）
2. 提取目录名并切换到该目录（如果存在）
3. **再次调用 `set_env_for_agent()`** 确保 Git 初始化
4. 获取测试运行器配置（`get_test_runner_and_mode()`）
5. 调用 `fix_task_solve_workflow()` 执行修复工作流
6. 完成后执行 `git reset --hard` 重置更改
7. 恢复原始工作目录

---

### 阶段 6: 修复工作流执行

**位置**: `agent.py::fix_task_solve_workflow()`

```2104:2134:agent.py
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
```

**关键步骤**：
1. 创建 `EnhancedCOT`（思维链）对象
2. 创建 `FixTaskEnhancedToolManager`，包含以下工具：
   - `run_repo_tests` - 运行仓库测试
   - `run_code` - 运行 Python 代码
   - `apply_code_edit` - 应用代码编辑
   - 其他文件操作和搜索工具
3. 进入循环，Agent 使用工具逐步解决问题

---

### 阶段 7: 运行仓库测试

**位置**: `agent.py::FixTaskEnhancedToolManager.run_repo_tests()`

```1297:1336:agent.py
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
```

**测试运行器类型**：

1. **pytest**：
   - 使用 `PYTEST_COMMAND_TEMPLATE` 格式化命令
   - 超时：150 秒

2. **unittest**：
   - 直接使用 `python <file_path>` 运行每个测试文件
   - 超时：60 秒/文件

3. **其他测试运行器**：
   - **MODULE 模式**：将文件路径转换为模块路径
   - **FILE 模式**：直接使用文件路径
   - 如果检测到依赖错误，回退到 pytest

---

### 阶段 8: 运行代码

**位置**: `agent.py::FixTaskEnhancedToolManager.run_code()`

```1339:1357:agent.py
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
```

**关键步骤**：
1. 保存代码到指定文件路径
2. 将文件添加到 `generated_test_files` 列表（这些文件会被排除在最终补丁之外）
3. 使用 `python <file_path>` 运行代码
4. 超时：60 秒
5. 如果失败，返回带行号的错误信息

---

## 三、关键环境变量

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `RUN_ID` | `""` | 评估运行 ID |
| `REPO_PATH` | `"/sandbox/repo"` | 仓库路径 |
| `AGENT_TIMEOUT` | `DEFAULT_TIMEOUT` | Agent 超时时间（秒） |

---

## 四、工作目录变化流程

```
初始状态: /sandbox
  ↓
AGENT_RUNNER.py 执行
  ↓
agent_main(repo_dir="repo")
  ↓
切换到: /sandbox/repo (如果存在)
  ↓
set_env_for_agent() 初始化 Git
  ↓
process_fix_task() 可能再次切换目录
  ↓
fix_task_solve_workflow() 执行修复
  ↓
所有工具在 /sandbox/repo 目录下运行
```

---

## 五、潜在问题和诊断点

### 5.1 Git 初始化失败
- **位置**: `set_env_for_agent()` 中的 Git 命令
- **可能原因**：
  - Git 未安装
  - 权限问题
  - 目录不存在
- **诊断**：检查日志中的 `"ERROR: Could not initialize git repository"`

### 5.2 目录切换问题
- **位置**: `process_fix_task()` 中的 `os.chdir()`
- **可能原因**：
  - `REPO_PATH` 环境变量设置错误
  - 目录不存在
- **诊断**：检查日志中的 `"Current working directory"`

### 5.3 测试运行器检测失败
- **位置**: `get_test_runner_and_mode()`
- **可能原因**：
  - README 文件不存在或无法读取
  - LLM API 调用失败
- **诊断**：检查 `test_runner` 和 `test_runner_mode` 的输出

### 5.4 测试执行超时
- **位置**: `run_repo_tests()` 和 `run_code()`
- **可能原因**：
  - 测试执行时间过长
  - 死锁或无限循环
- **诊断**：检查测试输出和超时设置（pytest: 150s, unittest: 60s, run_code: 60s）

---

## 六、调试建议

1. **检查工作目录**：
   ```python
   logger.info(f"Current working directory: {cwd} and environ:{os.environ}")
   ```

2. **检查 Git 状态**：
   ```python
   subprocess.run(["git", "status"], check=False)
   ```

3. **检查测试运行器配置**：
   ```python
   print(f"test_runner: {test_runner}, test_runner_mode: {test_runner_mode}")
   ```

4. **检查文件路径**：
   - 确保所有文件路径相对于当前工作目录
   - 检查 `PYTHONPATH` 设置

5. **查看 Agent 日志**：
   - 所有 `print()` 和 `logger.info()` 输出都会记录到 `agent_logs`

---

## 七、总结

代理的仓库初始化和运行流程可以概括为：

1. **初始化**：在 Docker Sandbox 中准备环境，复制代码到 `/sandbox/repo`
2. **Git 设置**：在仓库目录中初始化 Git，创建初始提交
3. **环境配置**：设置 `PYTHONPATH`，检测测试运行器
4. **执行修复**：使用工具逐步修复问题，运行测试验证
5. **生成补丁**：最终生成 Git diff 格式的补丁

整个过程在隔离的 Docker 容器中执行，确保安全性和可重复性。

