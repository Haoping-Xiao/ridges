# SWE-bench 评测流程详解

## 📋 概述

SWE-bench (Software Engineering Benchmark) 是一个用于评估 AI 代码代理在真实软件工程任务上表现的基准测试。本系统实现了完整的 SWE-bench 评测流程。

---

## 🔑 关键概念：两种补丁的区别

### `patch` - 源代码修复补丁

**作用**: 修复bug的源代码变更

**内容**: 修改项目源代码文件，修复问题

**示例** (django__django-10554):
```diff
diff --git a/django/db/models/sql/compiler.py b/django/db/models/sql/compiler.py
--- a/django/db/models/sql/compiler.py
+++ b/django/db/models/sql/compiler.py
@@ -356,7 +356,12 @@ def get_order_by(self):
                 else:
-                    raise DatabaseError('ORDER BY term does not match any column in the result set.')
+                    if col_alias:
+                        raise DatabaseError('ORDER BY term does not match any column in the result set.')
+                    # Add column used in ORDER BY clause without an alias to
+                    # the selected columns.
+                    self.query.add_select_col(src)
+                    resolved.set_source_expressions([RawSQL('%d' % len(self.query.select), ())])
```

**用途**: 
- ✅ Agent 需要生成这个补丁来修复问题
- ✅ 评测时会应用这个补丁到代码库
- ✅ 这是评测的核心目标

---

### `test_patch` - 测试用例补丁

**作用**: 验证修复的测试代码

**内容**: 添加新的测试用例，用于验证bug是否被正确修复

**示例** (django__django-10554):
```diff
diff --git a/tests/queries/test_qs_combinators.py b/tests/queries/test_qs_combinators.py
--- a/tests/queries/test_qs_combinators.py
+++ b/tests/queries/test_qs_combinators.py
@@ -153,6 +153,29 @@ def test_union_with_values_list_on_annotated_and_unannotated(self):
+    def test_union_with_values_list_and_order(self):
+        ReservedName.objects.bulk_create([...])
+        qs1 = ReservedName.objects.filter(order__gte=6)
+        qs2 = ReservedName.objects.filter(order__lte=5)
+        union_qs = qs1.union(qs2)
+        # ... 测试代码 ...
```

**用途**:
- ❌ Agent **不需要**生成这个补丁
- ✅ 评测系统会自动应用这个补丁（如果问题需要）
- ✅ 用于验证 Agent 生成的 `patch` 是否正确修复了问题
- ✅ 这些测试会被添加到 `FAIL_TO_PASS` 列表中

---

### 对比总结

| 特性 | `patch` | `test_patch` |
|------|---------|-------------|
| **目标** | 修复源代码bug | 添加验证测试 |
| **Agent生成** | ✅ 必须生成 | ❌ 不需要生成 |
| **评测应用** | ✅ 应用到代码库 | ✅ 应用到测试文件 |
| **作用** | 修复问题 | 验证修复 |
| **文件类型** | 源代码文件 | 测试文件 |

**重要**: Agent 只需要生成 `patch`（源代码修复），不需要生成 `test_patch`（测试用例）。评测系统会使用 `test_patch` 来验证 Agent 的修复是否正确。

---

## 🔄 完整评测流程

### 阶段1: 问题加载 (Problem Loading)

**位置**: `SWEBenchVerifiedSuite._load_problems()`

1. **加载数据集**
   - 从 `swebench_verified.json` 加载所有问题
   - 每个问题包含：
     - `instance_id`: 问题唯一标识（如 `django__django-10554`）
     - `repo`: 仓库名称（如 `django/django`）
     - `base_commit`: 基准提交哈希
     - `problem_statement`: 问题描述
     - `patch`: **源代码修复补丁** - 修复bug的代码变更（Agent需要生成的）
     - `test_patch`: **测试用例补丁** - 验证修复的测试代码（用于评测，Agent不需要生成）
     - `FAIL_TO_PASS`: 需要从失败变为通过的测试列表
     - `PASS_TO_PASS`: 必须保持通过的测试列表

2. **克隆仓库**
   - 检查 `repos/` 目录下是否存在对应仓库
   - 如果不存在，从 GitHub 克隆仓库
   - 仓库格式：`owner/name` → `owner_name/`

3. **验证提交**
   - 验证 `base_commit` 是否存在于仓库中
   - 确保可以访问到问题所需的代码状态

4. **构建问题对象**
   ```python
   Problem(
       name=instance_id,
       problem_statement=problem_statement,
       tests=[FAIL_TO_PASS + PASS_TO_PASS],
       solution_diff=patch,
       userdata=problem  # 存储完整的 SWE-bench 数据
   )
   ```

---

### 阶段2: Agent 运行 (Agent Execution)

**位置**: `ProblemSuite.run_agent_sandbox()`

#### 2.1 初始化 Agent Sandbox

**位置**: `ProblemSuite.initialize_agent_sandbox()`

1. **创建临时目录**
   - 准备 Docker 容器挂载点

2. **准备 Agent 代码**
   - 将 `agent_code` 写入 `/sandbox/agent.py`

3. **复制问题文件**
   - 调用 `copy_problem_files_to_directory()`
   - 将仓库在 `base_commit` 状态下的代码复制到 `/sandbox/repo/`
   - 使用 `clone_local_repo_at_commit()` 克隆到指定提交

4. **可选：包含解决方案**
   - 如果 `include_solution=True`，将 `solution.diff` 写入 `/sandbox/solution.diff`

5. **创建 Sandbox**
   ```python
   Sandbox(
       name=f"agent-sandbox-{problem_name}-{evaluation_run_id}",
       python_script_path="AGENT_RUNNER.py",
       input_data={"problem_statement": problem_statement},
       env_vars={"RUN_ID": evaluation_run_id}
   )
   ```

#### 2.2 运行 Agent

**位置**: `ProblemSuite.run_agent_sandbox()`

1. **执行 Agent**
   - 在 Docker 容器中运行 `AGENT_RUNNER.py`
   - `AGENT_RUNNER.py` 会：
     - 读取 `/sandbox/agent.py`
     - 读取 `/sandbox/repo/` 中的代码
     - 调用 `agent_main()` 处理问题
     - 返回 Git patch 格式的代码变更

2. **获取结果**
   - `patch`: Agent 生成的代码补丁（Git diff 格式）
   - `agent_logs`: Agent 执行过程中的日志

3. **错误处理**
   - 超时：`AGENT_TIMEOUT_RUNNING_AGENT`
   - 异常：`AGENT_EXCEPTION_RUNNING_AGENT`
   - 其他错误：`VALIDATOR_FAILED_RUNNING_AGENT`

---

### 阶段3: 评测初始化 (Evaluation Initialization)

**位置**: `SWEBenchVerifiedSuite.initialize_eval_sandbox()`

1. **创建临时目录**
   - 用于验证补丁

2. **复制问题文件（包含测试）**
   - 调用 `copy_problem_files_to_directory(include_tests=True)`
   - 这次包含测试文件

3. **验证补丁**
   - 调用 `validate_diff_for_local_repo(patch, temp_dir)`
   - 检查补丁是否可以在目标代码库上应用
   - 如果无效，抛出 `AGENT_INVALID_PATCH` 错误

4. **创建 TestSpec**
   - 使用 `make_test_spec(SWEbenchInstance(**swebench_instance))`
   - `TestSpec` 包含：
     - 测试环境配置
     - 测试运行命令
     - Docker 镜像信息

5. **构建预测对象**
   ```python
   pred = {
       "model_name_or_path": str(evaluation_run_id),
       "model_patch": patch,  # Agent 生成的补丁
       "instance_id": problem_name
   }
   ```

6. **返回评测 Sandbox**
   ```python
   SWEBenchVerifiedEvaluationSandbox(
       evaluation_run_id=evaluation_run_id,
       test_spec=test_spec,
       pred=pred
   )
   ```

---

### 阶段4: 评测执行 (Evaluation Execution)

**位置**: `SWEBenchVerifiedSuite.run_eval_sandbox()`

#### 4.1 运行测试

1. **调用 SWE-bench Harness**
   ```python
   instance_id, report = run_instance(
       test_spec=eval_sandbox.test_spec,
       pred=eval_sandbox.pred,
       rm_image=False,
       force_rebuild=False,
       client=get_docker_client(),
       run_id=str(evaluation_run_id),
       timeout=timeout_seconds
   )
   ```

2. **`run_instance()` 内部流程**：
   - **构建 Docker 镜像**：
     - 环境镜像（包含依赖）
     - 实例镜像（包含代码和补丁）
   - **应用补丁**：
     - 将 `model_patch` 应用到代码库
   - **运行测试**：
     - 执行 `FAIL_TO_PASS` 测试（应该从失败变为通过）
     - 执行 `PASS_TO_PASS` 测试（应该保持通过）
   - **收集结果**：
     - 记录每个测试的通过/失败状态

#### 4.2 解析测试结果

**位置**: `SWEBenchVerifiedSuite.run_eval_sandbox()`

```python
tests_status = report[instance_id]["tests_status"]

# FAIL_TO_PASS 测试结果
for test_name in tests_status["FAIL_TO_PASS"]["success"]:
    # 这些测试从失败变为通过 ✅
    test_results.append(ProblemTestResult(
        name=test_name,
        category=ProblemTestCategory.fail_to_pass,
        status=ProblemTestResultStatus.PASS
    ))

for test_name in tests_status["FAIL_TO_PASS"]["failure"]:
    # 这些测试仍然失败 ❌
    test_results.append(ProblemTestResult(
        name=test_name,
        category=ProblemTestCategory.fail_to_pass,
        status=ProblemTestResultStatus.FAIL
    ))

# PASS_TO_PASS 测试结果
for test_name in tests_status["PASS_TO_PASS"]["success"]:
    # 这些测试保持通过 ✅
    test_results.append(ProblemTestResult(
        name=test_name,
        category=ProblemTestCategory.pass_to_pass,
        status=ProblemTestResultStatus.PASS
    ))

for test_name in tests_status["PASS_TO_PASS"]["failure"]:
    # 这些测试被破坏了 ❌
    test_results.append(ProblemTestResult(
        name=test_name,
        category=ProblemTestCategory.pass_to_pass,
        status=ProblemTestResultStatus.FAIL
    ))
```

---

## 📊 评测状态流转

```
pending
  ↓
initializing_agent  (初始化 Agent Sandbox)
  ↓
running_agent       (运行 Agent，生成补丁)
  ↓
initializing_eval   (初始化评测 Sandbox，验证补丁)
  ↓
running_eval        (运行测试，收集结果)
  ↓
finished           (完成) 或 error (错误)
```

---

## 🎯 评测指标

### 成功标准

一个问题的评测被认为是**成功**的，当且仅当：

1. ✅ **所有 FAIL_TO_PASS 测试通过**
   - 这些是原本失败的测试，修复后应该通过

2. ✅ **所有 PASS_TO_PASS 测试通过**
   - 这些是原本通过的测试，修复后不应该被破坏

### 评测报告

```python
{
    "instance_id": "django__django-10554",
    "tests_status": {
        "FAIL_TO_PASS": {
            "success": ["test_union_with_values_list_and_order", ...],
            "failure": []
        },
        "PASS_TO_PASS": {
            "success": ["test_simple_union", "test_count_union", ...],
            "failure": []
        }
    }
}
```

---

## 🐳 Docker 镜像构建

### 预构建镜像（可选优化）

**位置**: `SWEBenchVerifiedSuite.prebuild_problem_images()`

在运行评测前，可以预构建 Docker 镜像以加速后续评测：

1. **构建环境镜像** (`build_env_images`)
   - 包含项目依赖
   - 每个仓库一个镜像

2. **构建实例镜像** (`build_instance_images`)
   - 基于环境镜像
   - 包含特定提交的代码
   - 每个问题一个镜像

**优势**：
- 避免每次评测都重新构建镜像
- 显著减少评测时间

---

## 🔍 关键组件说明

### 1. TestSpec

`TestSpec` 定义了测试运行的环境和配置：
- 仓库信息
- 提交哈希
- 测试命令
- Docker 镜像名称
- 架构要求（arm64/x86_64）

### 2. SWE-bench Harness

`run_instance()` 是 SWE-bench 官方提供的评测函数：
- 管理 Docker 容器生命周期
- 应用补丁到代码库
- 运行测试套件
- 收集测试结果

### 3. 补丁验证

`validate_diff_for_local_repo()` 确保：
- 补丁格式正确
- 补丁可以应用到目标代码库
- 不会产生冲突

---

## 📝 示例：django__django-10554

### 问题描述
- **问题**: Union queryset with ordering breaks on ordering with derived querysets
- **类型**: FIX（修复bug）
- **难度**: 1-4 小时

### 评测流程

1. **加载问题**
   - 从 `swebench_verified.json` 加载
   - 克隆 `django/django` 仓库
   - 检出到 `base_commit: 14d026cccb144c6877294ba4cd4e03ebf0842498`

2. **Agent 运行**
   - Agent 分析问题
   - 生成补丁修复 `compiler.py` 和 `query.py`
   - 返回 Git patch

3. **评测执行**
   - 应用 Agent 生成的补丁
   - 运行测试：
     - `test_union_with_values_list_and_order` (FAIL_TO_PASS)
     - `test_union_with_values_list_on_annotated_and_unannotated` (FAIL_TO_PASS)
     - 其他 PASS_TO_PASS 测试

4. **结果判定**
   - ✅ 如果所有 FAIL_TO_PASS 和 PASS_TO_PASS 都通过 → 成功
   - ❌ 如果有任何测试失败 → 失败

---

## 🚀 性能优化

1. **镜像预构建**
   - 在评测前预构建所有 Docker 镜像
   - 减少评测时间

2. **并行评测**
   - 可以同时运行多个问题的评测
   - 使用 `asyncio.gather()` 并行执行

3. **缓存机制**
   - 缓存已构建的 Docker 镜像
   - 避免重复构建

---

## ⚠️ 错误处理

### 常见错误类型

1. **AGENT_INVALID_PATCH**
   - 补丁格式错误或无法应用

2. **AGENT_TIMEOUT_RUNNING_AGENT**
   - Agent 执行超时

3. **AGENT_EXCEPTION_RUNNING_AGENT**
   - Agent 执行过程中抛出异常

4. **VALIDATOR_FAILED_INIT_EVAL**
   - 初始化评测环境失败

5. **VALIDATOR_FAILED_RUNNING_EVAL**
   - 运行评测失败

---

## 📚 相关文件

- `evaluator/problem_suites/swebench_verified/swebench_verified_suite.py` - SWE-bench 评测套件实现
- `evaluator/datasets/swebench_verified/swebench_verified.json` - 问题数据集
- `validator/main.py` - 评测运行主流程
- `evaluator/problem_suites/problem_suite.py` - 问题套件基类
- `evaluator/sandbox/sandbox_manager.py` - Sandbox 管理器

---

*最后更新: 2025年*

