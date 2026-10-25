# UPSIDE + 9 池两层预测模型

## 概述

完全独立于 OVER 八比分池模型的**预测系统**，采用两层架构，专攻亚盘平手/平半/半球/受让平半盘口下的“主队不败 + 9 比分池”预测。

### 核心洞察

在亚盘（初盘+终盘）为 `['半球', '平手/半球', '平手', '受让平手/半球']` 且机构数 > 12 的比赛中：
- **主队不败（UPSIDE）基础率**: ~69-71%（主胜或平局）
- **9 池命中率**: ~64.7%（8 种主队不败低比分 + 0-1）

UPSIDE 层用 LightGBM 二分类预测主队不败；9 池分支采用与 UNDER 同构的三阶段混合架构预测池概率。

### 两层架构

```
┌──────────────────────────────────────────────────────┐
│  Layer 1: LightGBM 二分类 → P_UPSIDE（主队不败概率）       │
│  标签: home_unbeaten（主胜或平局=1, 客胜=0）            │
│  特征: 复用 features_upside.py 全部特征                        │
│  输出: P_UPSIDE ∈ [0, 1]                                 │
└──────────────────────┬───────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────┐
│  9 池分支: 三阶段混合(同UNDER)                          │
│  池内筛选门 LightGBM(in_pool_9) → P_lgb                │
│  泊松回归×2 → λ_home/λ_away → Dixon-Coles矩阵 → P_dc   │
│  α融合 → Isotonic校准 → P_underool                       │
└──────────────────────┬───────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────┐
│  动态选场策略                                          │
│  P_UPSIDE > 0.75  → 推荐 UPSIDE（双选胜平）                    │
│  P_underool >= 阈值(meta EV选择) → 推荐 9 池比分(9注/场)   │
│  否则 → 空仓                                          │
└──────────────────────────────────────────────────────┘
```

### 9 池比分列表

```
POOL_UPSIDE = ['1-0', '2-0', '2-1', '3-1', '3-2', '0-0', '1-1', '2-2', '0-1']
```

即 8 种主队不败低比分（主胜 5 种 + 平局 3 种）+ 客胜 0-1（2026-10-12 起扩池，池不再是主队不败子集）。

---

## 项目结构

```
live/
├── features_upside_ext.py       # 特征工程（复用 features_upside.py + 新增 UPSIDE/9池标签）
├── train_upside.py          # Layer 1 + 9池三阶段混合训练
├── predict_upside.py        # 两层模型每日预测
├── backtest_upside.py       # 回测验证
├── models/
│   ├── lgb_upside.txt       # LightGBM UPSIDE 模型
│   ├── lgb_upside_pool.joblib      # 9池池内筛选门 LightGBM
│   ├── calib_upside.joblib         # 9池概率 Isotonic 校准器
│   ├── lgb_upside_goals_home.joblib  # 泊松主队进球模型
│   ├── lgb_upside_goals_away.joblib  # 泊松客队进球模型
│   └── meta_upside.json     # 元信息（特征列、阈值、alpha、rho等）
├── features_upside_ext_cache.csv  # 特征缓存（首次构建后复用）
├── predictions_upside.jsonl   # 每日预测推荐记录
└── backtest_upside_report.json # 回测报告
```

> **完全独立**: 与 OVER 八比分池模型各自独立，使用不同的标签、模型架构和预测流程。

---

## 快速开始

### Step 1: 构建特征

```bash
# 首次构建（从数据库读取，约 27000+ 场）
python -u features_upside_ext.py

# 后续调用自动读取缓存 features_upside_ext_cache.csv
# 强制重建缓存
python -u features_upside_ext.py  # 在代码中设置 rebuild=True
```

特征构建输出示例：
```
[UPSIDE] 样本数: 29052, 完场: 28984
[UPSIDE] 主队不败率: 69.2%
[UPSIDE] 9 池命中率: 64.7%
```

### Step 2: 训练模型

```bash
# 全量数据训练（70/15/15 时间切分）
python -u train_upside.py

# 只用指定日期前的数据（回测用）
python -u train_upside.py --end 20260601
```

产物（`models/` 目录）：
- `lgb_upside.txt` — LightGBM UPSIDE 模型
- `lgb_upside_pool.joblib` / `calib_upside.joblib` — 9池筛选门 / Isotonic 校准器
- `lgb_upside_goals_home/away.joblib` — 泊松进球模型
- `meta_upside.json` — 元信息（特征列、阈值、alpha、rho、AUC 等）

训练输出示例：
```
总样本: 25862 场
训练集: 18103 场 (2024112101 ~ 2026051630)
验证集: 3879 场 (2026051631 ~ 2026100103)
测试集: 3880 场 (2026100104 ~ 20261011124)
特征数: 104

UPSIDE + 9 池模型训练完成
最佳迭代: 285
验证集 AUC: 0.6854
基础率（主队不败占比）: 69.2%
模型推荐率（P_UPSIDE>0.75）: 38.6%
模型命中率: 82.6%

9池三阶段: 池内门AUC=0.6803, 融合α=0.0, rho=-0.02
9池投注阈值: P_underool >= 0.65 (valid集EV选择)
  P>=0.50:  3280场 池内率 67.6% 每场EV +1.27
  P>=0.65:  2314场 池内率 73.4% 每场EV +1.76
  ...
```

### Step 3: 每日预测

```bash
# 预测当天比赛（自动回顾昨天）
python -u predict_upside.py

# 指定日期
python -u predict_upside.py --date 20260912

# 预测全部未开赛比赛（不限日期）
python -u predict_upside.py --all-unfinished

# 跳过邮件发送
python -u predict_upside.py --skip-email

# 回顾昨天预测结果（对照实际比分，含 SP 值和准确率统计）
python -u predict_upside.py --review

# 回顾指定日期
python -u predict_upside.py --review 20260912

# 补录历史日期预测记录（供回顾使用）
python -u predict_upside.py --backfill 20260911
```

预测输出示例：
```
======================================================================
UPSIDE + 9 池两层模型预测
======================================================================
模型版本: 1x_v2_20261012_2042
训练时间: 2026-10-12 20:42:15
Layer 1 阈值: P_UPSIDE > 0.75 → UPSIDE
9 池阈值: P_underool >= 0.65 → 9 池 (9注/场)
9池三阶段: 池内门+泊松+DC, α=0.0
======================================================================

编号        联赛      对阵                      P_UPSIDE  P_underool      推荐
------------------------------------------------------------------------------------------
★ 2026101201  西乙    巴拉多利德 vs 阿尔巴塞特    81.0%   72.7%  UPSIDE（双选胜平）
★ 2026101224  阿甲    拉努斯 vs 萨斯菲尔德      73.5%   81.3%      9 池比分
      9池具体比分: 1-1:0.135  1-0:0.126  0-0:0.107
  2026101216  波兰超  GKS卡托威斯 vs KS莫摩斯   70.7%   50.1%        空仓

======================================================================
推荐统计:
  UPSIDE（双选胜平）: 2 场
  9 池比分：       12 场
  空仓：           5 场
======================================================================
```

### Step 3.5: 每日回顾（自动集成）

每日预测时**自动回顾**前一天的预测结果，对照实际比分统计命中率。回顾结果显示在预测前面，邮件也合并发送。

**输出示例**：
```
======================================================================
【第一部分：回顾前一天预测结果】
======================================================================
==============================================================================
UPSIDE+9 池模型回顾 20261011  (来源：predictions_upside.jsonl)
命中口径：UPSIDE=主队不败（主胜或平局），9 池=9 种比分 1-0/2-0/2-1/3-1/3-2/0-0/1-1/2-2/0-1
==============================================================================
2026101104 [单场 255] 日职联 千叶市原 vs 长崎成功丸
  预测：空仓 (P_UPSIDE=61.1%, P_underool=52.9%)
  实际：1-1 SP6.91  -> UPSIDE:-  9池:-  参考: UPSIDE✓ 9池✓(空仓未下注)

------------------------------------------------------------------------------
【准确率统计】
共 41 场预测，已完场 41 场
  UPSIDE 命中率：4/6 (66.7%)
  9池命中率：0/0
------------------------------------------------------------------------------

======================================================================
【第二部分：今日预测】
======================================================================

======================================================================
UPSIDE + 9 池两层模型预测
======================================================================
...
```

**回顾功能说明**：
- **单场 ID**：优先从 match 表查询 `single_match` 字段，格式为 `[单场 XX]`；无值时显示 `[-]`
- **SP 值**：从 `result_sp` 表查询开奖 SP 值，显示在实际比分后，如 `2-1 SP10.19`；无值时显示 `SP-`
- **准确率统计**：显示总预测场次、已完赛场次、UPSIDE 命中率和 9 池命中率
- **数据来源**：读取 `predictions_upside.jsonl` 对照 match 表实际比分

**定时任务**（服务器实际配置）：
```bash
# UPSIDE+9 池模型每日预测（22:35，自动包含前一天回顾）
35 22 * * * cd /coding/live && .venv/bin/python -u predict_upside.py >> /coding/live/logs/predict_upside.log 2>&1

# UPSIDE+9 池模型每周一重训（10:30）
30 10 * * 1 cd /coding/live && .venv/bin/python -u train_upside.py >> /coding/live/logs/train_upside.log 2>&1
```

**单独回顾**（可选）：
```bash
# 单独回顾昨天（不预测）
python -u predict_upside.py --review

# 回顾指定日期
python -u predict_upside.py --review 20260912
```

### Step 3.6: 历史数据补录（--backfill）

当需要回顾历史日期但 `predictions_upside.jsonl` 中无记录时（如爬虫失败、任务未执行），可使用补录功能：

```bash
# 补录指定日期的预测记录（含已完场比赛）
python -u predict_upside.py --backfill 20260911

# 补录后再回顾
python -u predict_upside.py --review 20260911
```

**补录说明**：
- **适用场景**：目标日期因爬虫失败、任务未执行等原因导致 jsonl 中无记录
- **行为逻辑**：对指定日期全部比赛（含已完场）使用赛前特征进行预测，按阈值和盘口过滤生成推荐并追加到 `predictions_upside.jsonl`
- **记录标记**：每条补录记录标记 `"backfill": true`
- **无副作用**：不更新 `match.prediction`、不发送邮件
- **数据覆盖**：同 `match_id` 记录按时间戳去重，新补录记录会覆盖旧记录
- **特征安全**：使用赛前特征（不含比分/标签），等价回测口径，无数据泄漏

### Step 4: 回测验证

```bash
# 全量回测（用测试集，后 15% 数据）
python -u backtest_upside.py

# 指定起始日期回测
python -u backtest_upside.py --start 20260601
```

回测输出示例：
```
======================================================================
【Layer 1: UPSIDE（主队不败）回测】
======================================================================
推荐场次: xxx 场
基础率（全体测试集）: 69.2%
模型命中率: xx.x%
提升幅度: +x.x%
AUC: 0.6880
Brier Score: 0.xxxx

月度稳定性:
  202608: 83.3% (50/60)
  202609: 83.7% (87/104)
  202610: 84.8% (190/224)

======================================================================
【9 池比分回测（三阶段混合: 池内门+泊松+DC+校准）】
======================================================================
推荐场次: 2682 场
基础率（全体测试集）: 65.1%
模型命中率: 69.5%
提升幅度: +6.8%

实现盈亏(有SP的 2564 场, 9注/场):
  总盈亏: +2539.4 元  每场EV: +0.99 元  ROI: +11.0%

重叠场次: xxx 场（同时满足 UPSIDE 和 9 池条件）

======================================================================
【综合评估】
======================================================================
UPSIDE 2串1理论命中率: xx.x%
9 池 2串1理论命中率: xx.x%
UPSIDE 期望收益（赔率1.4）: +0.170 ✓盈利
9 池实现每场EV（9注/场）: +0.99 ✓盈利
```

---

## 模型架构详解

### Layer 1: LightGBM 二分类

| 项目 | 配置 |
|------|------|
| 模型 | `lgb.LGBMClassifier` |
| 目标 | `binary`（主队不败=1, 客胜=0） |
| 迭代数 | 600（早停 60 轮） |
| 学习率 | 0.05 |
| 叶子数 | 31 |
| 最小样本 | 60 |
| 子采样 | 0.9（每轮） |
| 列采样 | 0.8 |
| 正则化 | alpha=0.1, lambda=1.0 |

**训练策略**：
1. 70/15/15 时间切分（train/valid/test）
2. 验证集早停（patience=60）
3. 用 train+valid 重训最终模型（best_iteration 轮）
4. 防泄漏检查：特征列表混入标签列直接报错

### 9 池分支: 三阶段混合（同 UNDER）

| 组件 | 说明 |
|------|------|
| 池内筛选门 | LightGBM 二分类（标签 `in_pool_9`）→ P_lgb |
| 泊松回归×2 | LightGBM `objective='poisson'` 分别估主/客期望进球 λ_home / λ_away |
| Dixon-Coles 矩阵 | 用 λ + 全局 rho 生成比分概率矩阵（tau 低比分修正），9 池概率和 = P_dc |
| α 融合 | P = α·P_lgb + (1-α)·P_dc，α 在验证集按 Brier 最小选取 |
| Isotonic 校准 | 融合概率校准为真实命中率 → P_underool |
| EV 阈值 | 每场 9 注各 1 元：命中=SP-9，未中=-9，验证集选 EV 最优阈值写入 `meta_upside.json['threshold_pool']` |
| 9 池取 Top | 在 `POOL_UPSIDE` 内按矩阵概率降序取 Top N（默认 `TOP_SCORES_POOL=3`）作为具体展示比分 |

- rho 在训练集上按 DC 对数似然网格搜索拟合，存入 `meta_upside.json['rho']`
- 产物：`models/lgb_upside_pool.joblib` / `calib_upside.joblib` / `lgb_upside_goals_home.joblib` / `lgb_upside_goals_away.joblib`
- 旧模型无门/校准器产物时自动降级：P_underool 用 DC 原值
- 具体比分写入 jsonl 的 `top_picks` 字段，预测/邮件中以"9池具体比分: 1-1:0.135  1-0:0.126  0-0:0.107"形式输出
- 2026-10-12 重训（tag 1x_v2）：融合 α=0.0（DC 池概率主导）、阈值 0.65

### 动态选场策略

| 条件 | 推荐 | 适用场景 |
|------|------|----------|
| P_UPSIDE ≥ 0.75 | **UPSIDE（双选胜平）** | 高信心主队不败，投注胜平双选 |
| P_underool ≥ 阈值(meta) | **9 池比分** | 9 池全部比分各 1 注（9注/场） |
| 否则 | **空仓** | 信心不足，不投注 |

> 两层推荐互斥（UPSIDE 优先）；9 池含 0-1，与 UPSIDE 方向不完全一致，池推荐独立于 UPSIDE 概率。

### 杯赛过滤（参考 UNDER，联赛名含"杯"即排除）

- 判定：`features_upside.is_cup_league()`，联赛名含"杯"字即视为杯赛（如欧冠、足总杯、巴西杯）
- 应用位置：训练 `prepare_data()`、每日预测、`--backfill` 补录、`backtest_upside.py` 回测，全链路一致排除
- 原因：杯赛赛制/战意与联赛差异大，历史分布不稳定，纳入会拉低命中率

---

## 特征工程（features_upside_ext.py）

### 设计原则

- **完全复用** `features_upside.py` 的特征（亚盘、大小球、欧赔、近 10 场、往绩、积分榜、技术统计等）
- **仅替换标签列**：`in_pool` → `home_unbeaten` + `in_pool_9`
- **无未来函数**：只用赛前信息

### 特征组成（~130个）

| 特征组 | 数量 | 函数 |
|--------|------|------|
| 亚盘 | 14 | `asian_features_upside()` |
| 大小球 | 8 | `ou_features_upside()` |
| 欧赔 | 11 | `euro_features_upside()` |
| 期望进球衍生 | 3 | `goal_expectation()` |
| 近10场战绩(主/客+差值) | ~54 | `_recent_team_features_upside()` |
| 对战往绩 | 11 | `h2h_features_upside()` |
| P0: 积分榜详细(主/客+差值) | ~31 | `_standings_features_upside()` |
| 联赛先验 | 1 | `league_pool_prior` |

**P0 新增特征**: `h_ls_*`/`a_ls_*` 各 11 个（ played/won/drawn/lost/goals_for/goals_against/goal_diff/gf_avg/ga_avg/ppg/win_rate）+ 差值 6 个（ls_gf_diff/ls_ga_diff/ls_goal_diff_diff/ls_win_rate_diff/ls_ppg_diff/ls_home_advantage）

> 技术统计特征已集成到 `_recent_team_features_upside()` 中：射正率、控球率、场均角球、危险进攻等 8 个 + 差值 6 个。

### 标签定义

| 标签 | 含义 | 取值 |
|------|------|------|
| `home_unbeaten` | 主队不败 | 1=主胜或平局, 0=客胜, None=未完场 |
| `in_pool_9` | 9 池命中 | 1=在 9 池内, 0=不在, None=未完场 |

### 特征缓存

- 缓存文件：`features_upside_ext_cache.csv`
- 优先读缓存，`rebuild=True` 或缓存不存在时重跑
- 修改特征/标签后需删除缓存强制重建

### 防数据泄漏

- `feature_columns_upside(df)` 显式排除标签/辅助列：`home_unbeaten`, `in_pool_9`, `in_pool_8`, `in_pool`, `sp`, `score`
- `train_upside_model()` 入口护栏：特征混入标签列直接 `ValueError` 拒绝训练

---

## 与 OVER 模型对比

| 维度 | OVER 八比分池模型 | UPSIDE + 9 池两层模型 |
|------|-------------------|-------------------|
| 目标 | 11 比分池命中（BTTS+中低比分） | 主队不败 + 9 池比分 |
| 基础率 | ~55-60% | ~69-71%（UPSIDE）/ ~64.7%（9 池） |
| 模型架构 | LightGBM + 泊松回归 + Dixon-Coles | LightGBM 二分类 + 三阶段混合（同UNDER） |
| 特征数 | 108 | ~130（复用） |
| 复杂度 | 高（三阶段混合） | 高（UPSIDE层 + 三阶段混合） |
| 选场策略 | 固定阈值 0.70 | UPSIDE 固定 0.75 / 9池 meta EV 阈值 |
| 输出 | 11 比分概率 + 投注推荐 | P_UPSIDE + P_underool + 推荐类型 |
| 文件 | `train_over.py` / `predict_over.py` | `train_upside.py` / `predict_upside.py` |

> 两个模型完全独立，使用不同的标签（in_pool vs home_unbeaten/in_pool_9）、不同的模型架构，可并行运行、对比效果。

---

## 配置参数

### 训练超参（train_upside.py）

```python
TRAIN_RATIO = 0.70          # 训练集占比
VALID_RATIO = 0.15          # 验证集占比
THRESHOLD_UPSIDE = 0.75         # UPSIDE 推荐阈值
TOP_SCORES_POOL = 3         # 9池推荐输出的具体比分个数
BETS_PER_MATCH = 9          # 每场投注注数(9池全部比分各1注)
# LGB_POOL_PARAMS: 9池池内筛选门超参(objective='binary')
# LGB_POISSON_PARAMS: 泊松进球模型超参(objective='poisson')
# RHO_GRID: Dixon-Coles rho 网格搜索范围(-0.30~0.30, 步长0.02)
```

### 预测配置（predict_upside.py）

```python
THRESHOLD_UPSIDE = 0.75             # UPSIDE 推荐阈值
THRESHOLD_9POOL_FALLBACK = 0.65 # 9 池阈值回退值(优先用 meta 的 threshold_pool)
HANDICAP_TARGETS = {'半球', '平手/半球', '平手', '受让平手/半球'}  # 盘口过滤(机构数>12)
```

### 回测配置（backtest_upside.py）

```python
THRESHOLD_UPSIDE = 0.75
# 9 池阈值读 meta_upside.json['threshold_pool'], 回退 THRESHOLD_9POOL_FALLBACK
```

---

### 定时部署配置

**服务器实际 crontab**：
```bash
# UPSIDE+9 池模型每日预测（22:35，自动包含前一天回顾）
35 22 * * * cd /coding/live && .venv/bin/python -u predict_upside.py >> /coding/live/logs/predict_upside.log 2>&1

# UPSIDE+9 池模型每周一重训（10:30）
30 10 * * 1 cd /coding/live && .venv/bin/python -u train_upside.py >> /coding/live/logs/train_upside.log 2>&1
```

**日志文件**：
- 预测日志：`logs/predict_upside.log`
- 训练日志：`logs/train_upside.log`
- 推荐记录：`predictions_upside.jsonl`
- 回测报告：`backtest_upside_report.json`
- 特征缓存：`features_upside_ext_cache.csv`

---

## 邮件通知

预测结果邮件发送受 `config.py` 的 `ENABLE_EMAIL` 开关控制（默认 True）。

### 每日预测邮件（22:35 定时任务）

邮件内容包含两部分：
1. **回顾部分**：前一天预测结果对照实际比分（含 SP 值和准确率统计）
2. **预测部分**：今日比赛预测推荐

**邮件主题格式**：
- 预测：`UPSIDE+9池预测 YYYYMMDD (UPSIDE:N 9池:N 空仓:N)`
- 回顾：`UPSIDE+9池模型回顾 YYYYMMDD 预测 N 场 完场 N UPSIDE 命中 N (XX.X%) 9池命中 N (XX.X%)`

**邮件内容包含**：
- 模型版本与训练时间
- 推荐统计（UPSIDE 场数 / 9 池场数 / 空仓场数）
- 每场比赛的 P_UPSIDE、P_underool、推荐类型
- 回顾部分含实际比分、SP 值、命中状态

```bash
# 跳过邮件发送
python -u predict_upside.py --skip-email
```

---

## 常见问题

### Q: 修改特征后需要做什么？
A: 删除 `features_upside_ext_cache.csv` 缓存文件，重跑 `train_upside.py` 重建特征并训练。

### Q: 如何调整推荐阈值？
A: UPSIDE 层修改 `predict_upside.py` 中的 `THRESHOLD_UPSIDE`；9 池阈值由训练时在验证集按 EV 自动选择，存于 `meta_upside.json['threshold_pool']`（缺失时回退 `THRESHOLD_9POOL_FALLBACK=0.65`）。

### Q: 模型效果不如预期怎么办？
A: 1) 检查 `backtest_upside.py` 回测报告，确认 AUC 和命中率；2) 尝试调整阈值；3) 检查特征是否有泄漏。

### Q: 与 OVER 模型冲突吗？
A: 完全不冲突。两个模型使用不同的特征缓存、模型文件、预测日志，可并行运行。

### Q: 为什么 9 池分支用三阶段混合而不是固定条件概率？
A: 旧版 P_8pool = P_UPSIDE × 0.865 依赖"8池是主队不败子集"的条件关系；2026-10-12 扩池加入 0-1（客胜）后该关系不再成立，改为与 UNDER 同构的三阶段混合（池内门+泊松+DC+α融合+Isotonic），阈值由验证集 EV 自动选择。

---

## 技术栈

- **Python 3.11**
- **LightGBM** — 梯度提升二分类模型
- **pandas** — 特征表构建与数据处理
- **scikit-learn** — AUC / Brier Score 评估
- **joblib** — 模型序列化存储
- **pymysql** — MySQL 数据库连接

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1x_v1 | 2026-09-15 | 初始版本：LightGBM 二分类 + 固定条件概率 0.865 |
| 1x_v1.1 | 2026-09-18 | 新增：回顾功能增强（SP 值显示、准确率统计、单场 ID）、--backfill 补录模式、邮件主题格式优化 |
| 1x_v1.2 | 2026-09 | 参考 UNDER：8 池接入泊松回归×2 + Dixon-Coles 输出具体投注比分（top_picks）；文档补充杯赛过滤说明 |
| 1x_v2 | 2026-10-12 | 比分池扩为 9 种（+0-1）；盘口过滤确认含受让平半（4 种, 机构数>12）；9 池概率改三阶段混合（池内门+泊松+DC+α融合+Isotonic），阈值 valid 集 EV 自动选择写入 meta；投注 9 注/场（命中=SP-9）；回顾兼容旧 8 池记录 |
| 1x_v3 | 2026-10-14 | 欧赔特征集成：新增 `euro_features_upside()` 函数（11个特征：公司数、初/即时隐含胜平负概率、公司间分歧度、返还率、主胜概率变动），数据从 `european_odds` 表批量加载。三模型全部重训 tag=20261014 |
| 1x_v4 | 2026-10-21 | P0/P1/P2特征工程：积分榜详细特征扩展（+22个）、技术统计特征集成（+14个）。特征 104→~130。三模型全部重训 |
