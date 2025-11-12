# Django Issue #10554 - 正确答案

## 问题描述
Union queryset with ordering breaks on ordering with derived querysets

## 修复方案 (Patch)

### 文件 1: `django/db/models/sql/compiler.py`

**修改位置**: `get_order_by()` 方法 (第 356-363 行)

**原始代码**:
```python
                else:
                    raise DatabaseError('ORDER BY term does not match any column in the result set.')
```

**修复后的代码**:
```python
                else:
                    if col_alias:
                        raise DatabaseError('ORDER BY term does not match any column in the result set.')
                    # Add column used in ORDER BY clause without an alias to
                    # the selected columns.
                    self.query.add_select_col(src)
                    resolved.set_source_expressions([RawSQL('%d' % len(self.query.select), ())])
```

**关键改动**:
- 添加了 `col_alias` 检查：只有当列有别名时才抛出错误
- 如果没有别名，自动将 ORDER BY 中使用的列添加到 SELECT 列表中
- 使用 `add_select_col()` 方法添加列
- 更新 `resolved` 的源表达式为新的列索引

### 文件 2: `django/db/models/sql/query.py`

**新增方法**: `add_select_col()` (第 1777-1780 行)

```python
def add_select_col(self, col):
    self.select += col,
    self.values_select += col.output_field.name,
```

**作用**: 
- 将列添加到 `select` 元组中
- 同时更新 `values_select` 以包含字段名

---

## 测试用例 (Test Patch)

### 文件: `tests/queries/test_qs_combinators.py`

**新增测试方法**: `test_union_with_values_list_and_order()` (第 156-183 行)

```python
def test_union_with_values_list_and_order(self):
    ReservedName.objects.bulk_create([
        ReservedName(name='rn1', order=7),
        ReservedName(name='rn2', order=5),
        ReservedName(name='rn0', order=6),
        ReservedName(name='rn9', order=-1),
    ])
    qs1 = ReservedName.objects.filter(order__gte=6)
    qs2 = ReservedName.objects.filter(order__lte=5)
    union_qs = qs1.union(qs2)
    for qs, expected_result in (
        # Order by a single column.
        (union_qs.order_by('-pk').values_list('order', flat=True), [-1, 6, 5, 7]),
        (union_qs.order_by('pk').values_list('order', flat=True), [7, 5, 6, -1]),
        (union_qs.values_list('order', flat=True).order_by('-pk'), [-1, 6, 5, 7]),
        (union_qs.values_list('order', flat=True).order_by('pk'), [7, 5, 6, -1]),
        # Order by multiple columns.
        (union_qs.order_by('-name', 'pk').values_list('order', flat=True), [-1, 5, 7, 6]),
        (union_qs.values_list('order', flat=True).order_by('-name', 'pk'), [-1, 5, 7, 6]),
    ):
        with self.subTest(qs=qs):
            self.assertEqual(list(qs), expected_result)
```

**测试覆盖场景**:
1. ✅ 单列排序（升序和降序）
2. ✅ `values_list()` 前后调用 `order_by()` 的不同顺序
3. ✅ 多列排序
4. ✅ Union queryset 的排序功能

---

## 核心修复逻辑

**问题根源**: 
当使用 `union()` 和 `order_by()` 时，如果 ORDER BY 中的列不在 SELECT 列表中，会抛出 `DatabaseError`。

**解决方案**:
1. 检查列是否有别名（`col_alias`）
2. 如果有别名但不在结果集中 → 抛出错误（保持原有行为）
3. 如果没有别名 → 自动将该列添加到 SELECT 列表中，避免错误

这样既保持了向后兼容性，又解决了 union queryset 排序的问题。

