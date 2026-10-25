# 1X + 8 池两层模型使用指南

##  概述

这是**完全独立于现有 9 池模型**的新预测系统，采用两层架构：

- **Layer 1**: LightGBM 二分类 → P_1X（主队不败概率）
- **Layer 2**: 固定条件概率 0.865 → P_8pool|1X
- **最终**: P_8pool = P_1X × 0.865

### 动态选场策略
- **P_1X > 0.75** → 推荐 **1X（双选胜平）**
- **P_8pool > 0.65** → 推荐 **8 池比分**
- **否则** → **空仓**

### 理论优势
- **基础率提升**: 从 61.7%（8 池）→ 71.3%（主队不败）
- **模型命中率目标**: 70-75%（vs 现有 55-60%）
- **2 串 1 理论命中率**: 49-56%（vs 现有 30-36%）

---

## 🚀 快速开始

### Step 1: 训练模型

```bash
# 全量数据训练（70/15/15 时间切分）
python -u train_1x.py

# 或用指定日期前的数据（回测用）
python -u train_1x.py --end 20260601
```

**产物**（`models/` 目录）:
- `lgb_1x.txt` - LightGBM 模型
- `meta_1x.json` - 元信息（特征列、阈值、基线等）

### Step 2: 每日预测

```bash
# 预测当天比赛
python -u predict_1x.py

# 预测指定日期
python -u predict_1x.py --date 20260912

# 预测全部未开赛比赛
python -u predict_1x.py --all-unfinished

# 跳过邮件发送
python -u predict_1x.py --skip-email
```

**输出**:
- 控制台打印推荐列表
- 写入 `match.prediction`（兼容现有格式）
- 追加到 `predictions_1x.jsonl`
- 邮件发送（受 `config.py` 的 `ENABLE_EMAIL` 控制）

### Step 3: 回测验证

```bash
# 全量回测（默认用测试集后 15%）
python -u backtest_1x.py

# 指定起始日期
python -u backtest_1x.py --start 20260601
```

**报告**（`backtest_1x_report.json`）:
- Layer 1: 1X 命中率 vs 基础率 71.3%
- Layer 2: 8 池命中率 vs 基础率 61.7%
- 月度稳定性统计
- 期望收益（EV）计算

---

## 📁 文件清单

| 文件 | 功能 | 依赖 |
|------|------|------|
| `features_1x.py` | 特征工程（复用 features.py + 新增 home_unbeaten 标签） | `features.py` |
| `train_1x.py` | Layer 1 模型训练 | `features_1x.py` |
| `predict_1x.py` | 两层模型预测 | `train_1x.py`, `features_1x.py` |
| `backtest_1x.py` | 回测验证 | `train_1x.py`, `features_1x.py` |
| `features_1x_cache.csv` | 特征缓存（自动生成） | - |
| `predictions_1x.jsonl` | 预测记录日志（自动生成） | - |
| `backtest_1x_report.json` | 回测报告（自动生成） | - |

---

## 🔧 配置参数

### 阈值调整（`predict_1x.py` / `backtest_1x.py`）

```python
P_8POOL_GIVEN_1X = 0.865    # Layer 2 固定条件概率（历史统计 61.7%/71.3%）
THRESHOLD_1X = 0.75         # 1X 推荐阈值（可调: 0.70/0.75/0.80）
THRESHOLD_8POOL = 0.65      # 8 池推荐阈值（可调: 0.60/0.65/0.70）
```

### 训练超参（`train_1x.py`）

```python
TRAIN_RATIO = 0.70          # 训练集占比
VALID_RATIO = 0.15          # 验证集占比
LGB_1X_PARAMS = {...}       # LightGBM 参数（与现有 train_model.py 类似）
```

---

## 📊 预期效果对比

| 指标 | 现有 9 池模型 | 1X + 8 池两层模型（目标） |
|------|-------------|-------------------------|
| **基础率** | 61.7% (8 池) | 71.3% (1X) / 61.7% (8 池) |
| **模型命中率** | 55-60% | 70-75% (1X) / 65-70% (8 池) |
| **2 串 1 理论命中率** | 30-36% | 49-56% (1X) / 42-49% (8 池) |
| **平均赔率** | 2.0-3.0 | 1.3-1.6 (1X) / 2.0-3.0 (8 池) |
| **推荐场次/月** | 20-30 场 | 15-40 场（灵活） |

---

## ⚠️ 注意事项

1. **数据库连接**: 首次运行需确保能连接 RDS（白名单或本地库同步）
2. **特征缓存**: `features_1x_cache.csv` 会自动生成，后续运行优先读取缓存
3. **模型版本**: 每次重训会生成新模型文件（带时间戳），旧模型保留
4. **邮件开关**: 受 `config.py` 的 `ENABLE_EMAIL` 控制，可通过 `--skip-email` 临时跳过
5. **与现有模型并存**: 本系统与 `predict_ml.py`（9 池模型）完全独立，可并行使用

---

## 🔍 故障排查

### 问题 1: 数据库连接超时
**原因**: RDS 白名单未配置或网络不通  
**解决**: 
- 检查 `config.py` 的 `DB_CONFIG.host` 是否正确
- 添加服务器 IP 到 RDS 白名单
- 或使用本地同步库（修改 `config.py` 的 `host='localhost'`）

### 问题 2: 模型文件不存在
**原因**: 未先运行 `train_1x.py`  
**解决**: 
```bash
python -u train_1x.py
```

### 问题 3: 特征缓存损坏
**原因**: CSV 文件不完整  
**解决**: 
```bash
rm features_1x_cache.csv
python -u train_1x.py  # 重建缓存
```

### 问题 4: 命中率低于预期
**原因**: 阈值过高或模型过拟合  
**解决**: 
- 降低 `THRESHOLD_1X`（如 0.75 → 0.70）
- 检查 `backtest_1x_report.json` 的月度稳定性
- 重新训练模型（增加正则化或减少特征）

---

## 📈 后续优化方向

1. **Layer 2 升级**: 从固定值 0.865 升级为第二模型（学习"哪些 1X 更可能是 8 池"）
2. **新增特征**: 添加主队强度特征（主场胜率、主场进球等）
3. **滚动重训**: 每月用过去 12 个月数据重训（类似现有 `train_model.py`）
4. **联赛黑名单**: 集成 `backtest_result.json` 的联赛过滤
5. **大小球交叉**: 结合大小球盘口进一步筛选高置信场次

---

## 📞 技术支持

如有问题，请检查：
1. `backtest_1x_report.json` 的回测指标
2. `predictions_1x.jsonl` 的预测记录
3. 控制台输出的详细日志

或与现有 9 池模型对比：
```bash
# 对比两个模型的推荐差异
python -u predict_ml.py --date 20260912
python -u predict_1x.py --date 20260912
```
