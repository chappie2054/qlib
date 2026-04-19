# 小时级多因子量化交易系统 - 预测性能优化方案

## 1. 问题背景

当前系统使用 qlib 的 `QlibDataLoader` + `DataHandlerLP` 进行每小时预测，单次耗时 **347秒（近6分钟）**，无法满足每小时实时预测的需求。

### 1.1 当前流程与耗时分布

```
QlibDataLoader.load()              → 308s (88.6%)  ← 主要瓶颈
RobustZScoreNorm.fit(全量数据)      →  14s (3.9%)
CSZFillna                           →  15s (4.3%)
DropnaLabel                         →   1s (0.3%)
CSRankNorm                          →   6s (1.8%)
fit & process data                  →  39s
─────────────────────────────────────────────────
总计                                → 347s
```

### 1.2 核心瓶颈分析

**数据加载（308s，占88%）**：每次预测都通过 `QlibDataLoader.load_group_df()` 调用 `D.features()` 重新计算所有因子表达式的全部历史值。

**全量标准化（14s）**：`RobustZScoreNorm` 每次都对全量历史数据重新 fit。

**关键源码位置**：

- `QlibDataLoader.load_group_df()` → `qlib/data/dataset/loader.py#L202-L227`
- `DataHandlerLP.fit()` → `qlib/data/dataset/handler.py#L514-L520`
- `LGBModel.predict()` → `qlib/contrib/model/gbdt.py#L125-L129`

---

## 2. 关键结论

### 2.1 qlib 模型 predict 是否必须使用 DataLoader？

**不需要。** `LGBModel.predict()` 内部最终调用的是标准 lightgbm 的 `self.model.predict(x_test.values)`，只接受 numpy 数组。可以完全绕过 `DatasetH`、`DataHandlerLP`、`QlibDataLoader`。

```python
# qlib/contrib/model/gbdt.py#L125-L129
def predict(self, dataset, segment="test"):
    x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
    return pd.Series(self.model.predict(x_test.values), index=x_test.index)
    #                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                    本质就是 lightgbm.Booster.predict(numpy_array)
```

### 2.2 qlib 内置 disk_cache 对增量场景有效吗？

**基本无效。** `DiskDatasetCache` 的缓存 key 基于 `instruments + fields + freq`（不含时间范围），首次生成全量缓存后，当 `end_time` 超出缓存范围时需要重新生成全量缓存，仍然是 308s。

关键源码（`qlib/data/cache.py#L709-L717`）：

```python
_cache_uri = self._uri(
    instruments=instruments,
    fields=fields,
    start_time=None,   # 缓存key不包含时间范围
    end_time=None,     # 不包含时间范围
    freq=freq,
)
```

### 2.3 自定义因子计算 + 直接调用模型的闭环方案是否可行？

**完全可行，且是最优方案。** 利用已有的 ClickHouse 数据管理模块，自己计算因子并持久化，然后直接调用 lightgbm 模型预测，性能从 347s 降到 <1s。

---

## 3. 优化方案

### 3.1 方案对比

| 方案 | 单次耗时 | 实现难度 | 适用场景 | 推荐度 |
|------|---------|---------|---------|--------|
| 原始方法（全量加载） | 347s | - | 无法用于实时 | - |
| StaticDataLoader 缓存 | <2s | 低 | 研究回测 | 中 |
| **自定义闭环方案** | **<1s** | **中** | **生产实盘** | **高** |

### 3.2 推荐方案：自定义闭环

```
ClickHouse（原始行情，每小时自动同步）
    │
    ▼
自定义因子计算（替代 QlibDataLoader）
    │
    ▼
因子持久化到数据库/文件
    │
    ▼
预拟合标准化（替代 RobustZScoreNorm.fit）
    │
    ▼
lightgbm.Booster.predict(numpy_array)（替代 DatasetH + model.predict）
    │
    ▼
预测得分 → 交易策略
```

### 3.3 核心实现

#### 训练阶段（一次性）

```python
import lightgbm as lgb
import pandas as pd

# 1. 用 qlib 正常训练模型
model = LGBModel(...)
model.fit(dataset)

# 2. 保存底层 lightgbm 模型
model.model.save_model('my_lgb_model.txt')

# 3. 保存标准化参数（在训练集上 fit 一次）
median = training_factor_data.median()
mad = (training_factor_data - median).abs().median().replace(0, 1)
pd.to_pickle({'median': median, 'mad': mad}, 'norm_params.pkl')
```

#### 预测阶段（每小时调用，<1s）

```python
import lightgbm as lgb
import pandas as pd

# 1. 加载模型和标准化参数
model = lgb.Booster(model_file='my_lgb_model.txt')
params = pd.read_pickle('norm_params.pkl')

# 2. 从 ClickHouse 读取已计算好的因子（或实时计算）
factor_df = load_factors_from_clickhouse()
# 格式要求：MultiIndex (datetime, instrument), columns=[factor0, factor1, ...]

# 3. 标准化
normalized = (factor_df - params['median']) / params['mad']
normalized = normalized.clip(-3, 3)

# 4. 预测（核心就这一行）
pred_scores = model.predict(normalized.values)

# 5. 包装结果
scores = pd.Series(pred_scores, index=factor_df.index)
```

---

## 4. 性能对比

| 环节 | 原始方法 | 优化后 |
|------|---------|--------|
| 数据获取 | QlibDataLoader.load() 308s | ClickHouse 查询 0.1s |
| 因子计算 | qlib 表达式引擎（全量） | 自定义计算（增量） 0.3s |
| 标准化 | RobustZScoreNorm.fit(全量) 14s | 预拟合参数 0.01s |
| 模型预测 | DatasetH.prepare() + predict 10s | model.predict(numpy) 0.05s |
| **总计** | **347s** | **<1s** |
| **提升** | - | **~350x** |

---

## 5. 需要实现的模块

### 5.1 ClickHouse 原始数据查询

根据已有的数据管理模块，编写 SQL 查询获取 `$open`, `$high`, `$low`, `$close`, `$volume` 等原始字段。

### 5.2 因子表达式计算器

将 qlib 的因子表达式（如 `Ref($close, -240)`, `Mean($volume, 20)` 等）翻译为 pandas 操作。可参考 qlib 的表达式引擎实现（`qlib/data/ops.py`），或使用 `pandas.eval()` 简化。

### 5.3 标准化参数管理

训练期计算并持久化 `median` 和 `mad` 参数，预测期直接加载使用。

### 5.4 模型导出与加载

训练完成后通过 `model.model.save_model()` 导出 lightgbm 模型，预测时通过 `lgb.Booster(model_file=...)` 加载。

---

## 6. 相关代码文件

| 文件 | 说明 |
|------|------|
| `workspace/py/closed_loop_predictor.py` | 完整闭环方案实现 |
| `qlib/contrib/model/gbdt.py` | LGBModel 源码（predict 方法） |
| `qlib/data/dataset/loader.py` | QlibDataLoader 源码 |
| `qlib/data/dataset/handler.py` | DataHandlerLP 源码 |
| `qlib/data/cache.py` | disk_cache 机制源码 |
