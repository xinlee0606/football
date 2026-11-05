`# ML 比分预测系统 — 完整文档
`
## 系统概述

本项目包含**三套独立的 ML 预测模型**，均基于约 2.5~2.8 万场北单历史数据训练，但目标、特征工程和架构完全独立：

| 维度 | OVER 十一比分池模型 | UNDER 九比分池模型 | UPSIDE + 9 池两层模型 |
|------|-------------------|-------------------|-------------------|
| **目标** | 11 比分池命中（BTTS+总进球≤6） | 9 比分池命中（完全对称+含零封） | 主队不败 + 9 池比分 |
| **比分池** | 2-1/3-1/3-2/1-1/2-2/3-3/1-2/1-3/2-3/4-2/2-4 | 1-0/2-0/2-1/0-0/1-1/2-2/0-1/0-2/1-2 | 1-0/2-0/2-1/3-1/3-2/0-0/1-1/2-2/0-1 |
| **盘口范围** | 平手/平半/半球/受让平半/受让半球 | 训练: 受让半球/半球/受让平半/平半/平手；**推荐: 仅 平手/受让平半/平半** | 仅平手/平半/半球/受让平半（机构数>12） |
| **大小球过滤** | 下限：≥2.25（无上限） | 上限：终盘≤3.25（不设下限） | 无 |
| **架构** | LightGBM + 泊松回归 + Dixon-Coles | LightGBM + 泊松回归 + Dixon-Coles | LightGBM + 泊松回归 + Dixon-Coles（UPSIDE二分类 + 9池三阶段混合） |
| **复杂度** | 高（三阶段混合 + 校准） | 高（三阶段混合 + 校准） | 高（UPSIDE层 + 三阶段混合 + 校准） |
| **投注注数** | 11注/场 | 9注/场 | 双选/9池比分（9注/场） |
| **特征工程** | `features_over.py` | `features_under.py` | `features_upside_ext.py`（复用 `features_upside.py`） |
| **文件** | `train_over.py` / `predict_over.py` | `train_under.py` / `predict_under.py` | `train_upside.py` / `predict_upside.py` |

> **三个模型完全独立**：不同比分池定义、不同盘口范围、不同特征工程模块、不同模型架构，可并行运行、对比校验。

---

## OVER 模型专属特征工程（features_over.py）

### 72 个特征清单（无未来函数，无技术统计）

#### ① 亚盘（14个） — `asian_features()`
| 特征 | 含义 |
|------|------|
| `asian_n` | 机构数 |
| `ah_init_mean` | 初盘盘口均值（正=主队让球） |
| `ah_init_std` | 初盘盘口分歧度（标准差） |
| `ah_init_flat_share` | 初盘均势盘占比（\|盘口\|≤0.25 的机构占比） |
| `ah_cur_mean` | 终盘盘口均值 |
| `ah_cur_std` | 终盘盘口分歧度 |
| `ah_cur_flat_share` | 终盘均势盘占比 |
| `ah_move` | 盘口升降（终盘均值 − 初盘均值） |
| `ah_move_up_share` | 升盘机构占比 |
| `cover_prob_init` | 初盘让球覆盖隐含概率（客水/(主水+客水) 去水归一） |
| `cover_prob_cur` | 终盘让球覆盖隐含概率 |
| `aw_init_diff` | 初盘主客水位差（主水均值 − 客水均值） |
| `aw_cur_diff` | 终盘主客水位差 |
| `aw_drift` | 水位漂移（终盘水位差 − 初盘水位差） |

#### ② 大小球（8个） — `over_under_features()`
| 特征 | 含义 |
|------|------|
| `ou_n` | 机构数 |
| `ou_line_init_mean` | 初盘大小球线均值 |
| `ou_line_init_std` | 初盘大小球线分歧度 |
| `ou_line_cur_mean` | 终盘大小球线均值 |
| `ou_line_move` | 大小球线变化（终盘 − 初盘） |
| `over_prob_init` | 初盘大球隐含概率（小水/(大水+小水) 去水归一） |
| `over_prob_cur` | 终盘大球隐含概率 |
| `exp_total` | 期望总进球（终盘大小球线，缺失则用初盘） |

#### ③ 欧赔（11个） — `euro_features()`
| 特征 | 含义 |
|------|------|
| `euro_n` | 欧赔公司数 |
| `euro_win_prob_init` / `euro_draw_prob_init` / `euro_loss_prob_init` | 初盘隐含胜/平/负概率（倒数归一化） |
| `euro_win_prob_cur` / `euro_draw_prob_cur` / `euro_loss_prob_cur` | 即时隐含胜/平/负概率 |
| `euro_win_std_init` / `euro_win_std_cur` | 初盘/即时胜赔分歧度 |
| `euro_return_mean` | 平均返还率 |
| `euro_win_move` | 主胜概率变动（即时 − 初盘） |

#### ④ 对战往绩（11个） — `_h2h_features_fast()`
| 特征 | 含义 |
|------|------|
| `h2h_n` | 往绩场数 |
| `h2h_home_win_rate` / `h2h_draw_rate` / `h2h_away_win_rate` | 往绩主/平/客胜率 |
| `h2h_avg_goals` | 往绩场均总进球 |
| `h2h_home_goals_mean` / `h2h_away_goals_mean` | 往绩主/客队场均进球 |
| `h2h_btts_rate` | 往绩双方进球率 |
| `h2h_over_25_rate` | 往绩大于2.5球率 |
| `h2h_half_goal_avg` | 往绩场均半场进球（half_score 覆盖率100%） |
| `h2h_corners_avg` | 往绩场均总角球（corners 覆盖率91%） |

#### ⑤ 近10场战绩（20个） — `_recent_form_features_fast()`
主队10个（`recent_` 前缀）: `recent_n`（场数）、`recent_win_rate`（胜率）、`recent_draw_rate`（平率）、`recent_loss_rate`（负率）、`recent_goals_for_mean`（场均进球）、`recent_goals_against_mean`（场均失球）、`recent_clean_sheet_rate`（零封率）、`recent_btts_rate`（双方进球率）、`recent_over_25_rate`（大于2.5球率）、`recent_unbeaten_rate`（不败率）

客队10个：同上（通过 `is_home=False` 参数获取）

#### ⑥ 联赛统计（8个） — `_league_stats_features_fast()`
| 特征 | 含义 |
|------|------|
| `league_n` | 联赛场数 |
| `league_home_win_rate` / `league_draw_rate` / `league_away_win_rate` | 联赛主/平/客胜率 |
| `league_avg_goals` | 联赛场均总进球 |
| `league_btts_rate` | 联赛双方进球率 |
| `league_over_25_rate` | 联赛大于2.5球率 |
| `league_home_unbeaten_rate` | 联赛主队不败率 |

### 特征提取流程

1. **批量加载**: 一次性加载 6 张源表（match, asian_odds, over_under_odds, european_odds, head_to_head, match历史数据），按 match_id 分组为 dict，避免逐场查库
2. **逐场构建**: 遍历每场比赛，依次调用各特征函数，合并为一行
3. **缓存**: 产物写入 `features_over_cache.csv`；后续调用 `load_or_build_over(rebuild=False)` 优先读缓存

### 数据源表

| 源表 | 特征函数 | 说明 |
|------|----------|------|
| `asian_odds` | `asian_features()` | 亚盘初盘+终盘盘口/水位 |
| `over_under_odds` | `over_under_features()` | 大小球初盘+终盘线/赔率 |
| `european_odds` | `euro_features()` | 欧赔胜/平/负初盘+即时（数据源 `scrape_euro_odds.py`） |
| `head_to_head` | `_h2h_features_fast()` | 对赛往绩（含半场比分、角球） |
| `match` | `_recent_form_features_fast()` / `_league_stats_features_fast()` | 近期战绩 + 联赛统计 |

### 防数据泄漏

- `feature_columns_over()` 显式排除标签/辅助列（`y`、`home_goals`、`away_goals`、`in_pool`、`score`、`sp` 等元信息列）
- `train_over()` 入口护栏：特征混入标签列直接 `ValueError` 拒绝训练
- 缓存到 `features_over_cache.csv`，修改比分池/特征/标签后需删除缓存强制重建

---

## OVER 十一比分池模型

### 用途

用约 2.5 万场北单历史数据训练混合机器学习模型，筛选最终比分落在 **11 比分池** 中的比赛并输出各比分概率，以期望收益（EV）为优化目标。

> 11 比分池 `OVER`：`2-1 / 3-1 / 3-2 / 1-1 / 2-2 / 3-3 / 1-2 / 1-3 / 2-3 / 4-2 / 2-4`（双方进球 BTTS + 总进球≤6，主胜3种+平局3种+客胜3种+高比分2种，对称覆盖）
> 定义在 `features_over.py`，与 UPSIDE 模型使用不同的比分，两个模型完全独立（不同标签、不同特征工程、不同模型架构）。
> 换池只需改该常量并重跑基线/回测/训练。

> 目标盘口范围：`['受让半球', '半球', '受让平/半', '平/半', '平手']`
> 训练和预测均只使用目标盘口范围内的比赛。
> 大小球线过滤：初盘和终盘均 ≥ 2.25（`MIN_OU_LINE_OVER`），无上限限制。

### 模型架构（三阶段混合）

| 组件 | 模型 | 作用 |
|------|------|------|
| 筛选门 | LightGBM 二分类 | 预测比赛落在 11 池内的概率 P_lgb |
| 进球期望 | LightGBM 泊松回归×2 | 分别预测主/客队期望进球 λ_home / λ_away |
| 比分矩阵 | Dixon-Coles 双变量泊松 | 低比分相关修正（tau 因子, rho 网格拟合），产出全比分概率矩阵 |
| 融合 | α 加权 | P = α·P_lgb + (1-α)·P_dc，α 在验证集按 Brier 最小选取 |
| 校准 | Isotonic 回归 | 融合概率校准为真实命中率 |
| 阈值 | 按 EV 选取 | 每场 11 注各 1 元：命中=SP-11，未中=-11，选 EV 最优的 P_pool 阈值 |

### 训练超参

```python
# LightGBM 二分类（筛选门）
LGB_CLF_PARAMS = dict(
    objective='binary', n_estimators=600, learning_rate=0.05,
    num_leaves=31, min_child_samples=60,
    subsample=0.9, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
)

# LightGBM 泊松回归（进球模型）
LGB_POISSON_PARAMS = dict(
    objective='poisson', n_estimators=400, learning_rate=0.05,
    num_leaves=31, min_child_samples=80,
    subsample=0.9, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
)

# Dixon-Coles rho 网格搜索
RHO_GRID = [-0.30, -0.28, ..., 0.00, ..., 0.28, 0.30]  # 步长 0.02
```

### 训练策略

1. **70/15/15 时间切分**（train / valid / test）
2. **验证集早停**（patience=60）
3. **用 train+valid 重训最终模型**（best_iteration 轮）
4. **rho 网格搜索**：在验证集上最大化 DC 对数似然
5. **α 融合权重**：在验证集上最小化 Brier Score
6. **Isotonic 校准**：在验证集上拟合校准曲线
7. **EV 阈值选取**：在留出集上以期望收益最优选择 P_pool 阈值

### 三步使用流程

```bash
# 服务器执行（顺序不可颠倒：先训练→再回测→最后预测）
.venv/bin/python -u train_over.py        # 1. 全量训练生成生产模型 -> models/ (带日期tag+latest.json)
.venv/bin/python -u backtest_over.py     # 2. 滚动前进回测(前70%起步按月扩窗重训) -> backtest_over_result.json
.venv/bin/python -u predict_over.py      # 3. 每日预测
```

> 修改比分池、特征或标签后需删除 `features_over_cache.csv` 强制重建缓存。
> 回测严格按时间序，每场预测只用其之前的数据，无未来函数。

### 产物（models/ 目录）

| 文件 | 说明 |
|------|------|
| `lgb_pool.txt` | 池内筛选 LightGBM |
| `lgb_goals_home.txt` | 主队进球泊松模型 |
| `lgb_goals_away.txt` | 客队进球泊松模型 |
| `calib.joblib` | Isotonic 校准器 |
| `meta.json` | 特征列/α/rho/阈值/基线等元信息 |

### 每日预测(predict_over.py)

- 输出全部 11 个比分概率(降序),推荐场额外标注投注推荐
- 概览表与推荐行均带 **单场编号**(`[单场 XX]`)与 **开赛时间**(`(02:30)`)
- 投注纪律:P_pool ≥ 置信度阈值 + 排除联赛黑名单,**不限推荐场次数量**
- 大小球过滤(推荐窄口径): 初盘和终盘均 ≥ 2.50(`is_valid_ou_line_predict_over`), 训练仍用宽口径 ≥ 2.25
- 阈值优先级：命令行 `--min-prob` > 模型按 EV 选出的阈值 > 兆底值 0.60
- **服务器 cron 固定使用 `--min-prob 0.70`**(20260917 模型 EV 阈值 0.80 过高导致空仓,故统一用 0.70)
- 预测前自动爬取最新欧赔数据（调用 `scrape_euro_odds.py`）
- 预测前自动回顾前一天预测结果(邮件合并发送)
- 预测方向(胜平/平负)写入 `match.prediction`,兼容现有下游格式
- 推荐记录写入 `ml_predictions.jsonl`(覆盖式: 同日旧记录被新一次预测覆盖, 每个日期只保留最后一次预测的候选集, 保证回顾与预测邮件场次一致)
- 持续学习监控：滚动近 100 场已完场推荐的池内命中率,跌破基线 85% 邮件告警

```bash
python -u predict_over.py                      # 预测当天比赛 (含前一天自动回顾 + 历史推荐监控)
python -u predict_over.py --date 20260818      # 指定日期
python -u predict_over.py --all-unfinished     # 全部未开赛比赛 (不限日期)
python -u predict_over.py --min-prob 0.65      # 手动抬高置信度阈值
python -u predict_over.py --skip-email         # 跳过预测结果邮件发送
python -u predict_over.py --skip-monitor       # 跳过监控
python -u predict_over.py --backfill 20260906  # 补录指定日期预测记录 (含已完场, 供--review 回顾)
```

### 预测回顾(predict_over.py --review)

- 用途：回看历史某天的预测与实际比分对照
- 数据源：`ml_predictions.jsonl`(按 match_id 去重、保留最后一次预测; 写入端已按日覆盖, 同日通常只剩最后一次预测批次+补录记录),再查 `match` 表取实际比分/对阵/单场编号,查 `result_sp` 表取开奖 SP
- 排序：按池内概率(置信度)由高到低
- 每行显示：编号、`[单场 XX]`、是否推荐 (★)、池内概率、对阵、**实际比分 + SP 值**、投注的 11 个比分、**命中结果 + 该场盈亏**
- 命中两口径：**投注命中**=实际比分落在所选 11 个投注比分(真实盈亏口径,盈亏=SP-11);**池内命中**=实际比分落在 11 比分池(基线口径)
- 汇总：推荐且已完场场次的命中率 + **总盈亏/ROI**(每场 11 注×1 元)
- 回顾结果一并邮件发送(受 `ENABLE_EMAIL` 控制,`--skip-email` 可跳过),无记录时不发邮件

```bash
python -u predict_over.py --review             # 回看昨天 (默认)
python -u predict_over.py --review 20260902    # 回看指定日期
```

**输出示例**:
```
2026091709  [单场 83]    -     池内  78.0%  皇家社会 vs 伯恩茅斯  实际 [1-0 SP6.50]  投注:1-0/2-0/2-1/0-0/1-1/3-1/2-2/3-2  -> ✓中 -1.50
2026091712  [单场 86]    -     池内  70.6%  水晶宫 vs 波兹南莱赫  实际 [4-0 SP22.88]  投注:1-0/2-0/2-1/0-0/1-1/3-1/2-2/3-2  -> 未中 -8.0

【准确率统计】
共 18 场预测，其中推荐 6 场
推荐且已完场 10 场:
  池内命中率：8/10 (80.0%)
  投注命中率(Top8): 6/10 (60.0%)
  实际盈亏：+0.5 元 / 10 场 (ROI +0.6%)
```

### 预测记录补录(predict_over.py --backfill)

- 用途：补充指定日期的预测记录到 `ml_predictions.jsonl`(含已完场比赛)
- 场景：目标日期因爬虫失败、任务未执行等原因导致无预测记录,`--review` 无法回顾
- 逻辑：对指定日期**全部比赛**(含已完场)用**赛前特征**预测(特征不含比分/标签,等价回测口径,无泄漏),按阈值和黑名单生成推荐并覆盖写入 jsonl(覆盖同日全部旧记录),每条记录标记 `"backfill": true`
- 不触发副作用：不更新 `match.prediction`、不发送邮件、不执行监控告警
- 数据覆盖：补录覆盖同日全部旧记录; 常规预测只覆盖同日常规记录、保留补录记录

```bash
python -u predict_over.py --backfill 20260906              # 补录 20260906 预测记录 (用模型 EV 阈值)
python -u predict_over.py --backfill 20260906 --min-prob 0.75  # 补录并指定阈值 0.75
```

> 补录为事后用赛前特征预测(回测口径),仅用于 `--review` 回顾历史日期。

### 2串1选场(select_2x1.py)

- 用途：从当日预测结果中自动选出最优的2场组2串1
- 数据源：`ml_predictions.jsonl`（当日推荐场次）
- 选场逻辑：
  1. 筛选 `picked=true` 且 `P_pool >= 阈值` 的比赛
  2. 按 `P_pool` 降序排序
  3. 选 Top 2 组成2串1
- 输出内容：选中比赛详情、组合概率、投注成本、期望收益(EV)
- **注意**：SP值是开奖后才有的，赛前选场只看 `P_pool`，不看SP

```bash
python -u select_2x1.py                      # 选今天比赛的2串1 (默认阈值0.80)
python -u select_2x1.py --date 20260926      # 选指定日期
python -u select_2x1.py --min-prob 0.75      # 自定义阈值
python -u select_2x1.py --top-n 3            # 选Top3(可组3注2串1)
python -u select_2x1.py --review             # 回顾昨天2串1选场效果
python -u select_2x1.py --review 20260926    # 回顾指定日期
```

**选场输出示例**:
```
======================================================================
OVER模型 2串1 选场建议 (20260926)
======================================================================
筛选条件: P_pool >= 0.80
候选场次: 4 场

【选中比赛】
----------------------------------------------------------------------
  1. 2026092601 [英超] 阿森纳 vs 切尔西
     P_pool: 92.3%
     投注比分: 2-1/3-1/3-2/1-1/2-2/1-2/1-3/2-3

  2. 2026092602 [德甲] 拜仁 vs 多特
     P_pool: 88.5%
     投注比分: 2-1/3-1/3-2/1-1/2-2/1-2/1-3/2-3

【2串1统计】
----------------------------------------------------------------------
  组合概率: 81.7%
  投注成本: 22元 (2场×11注×1元)

  假设平均SP=10:
    中奖回报: 100元
    期望收益(EV): +65.72元

======================================================================
提示: 2串1需两场全中才盈利, 建议只选P_pool最高的2场
======================================================================
```

**回顾输出示例**:
```
==============================================================================
2串1 回顾 20260926  (来源: ml_predictions.jsonl)
筛选条件: P_pool >= 0.80, 选Top 2
==============================================================================
候选场次: 4 场, 选中: 2 场

【选中比赛】
------------------------------------------------------------------------------
  1. 2026092601 [英超] 阿森纳 vs 切尔西
     P_pool: 92.3%
     投注比分: 2-1/3-1/3-2/1-1/2-2/1-2/1-3/2-3
     实际比分: 2-1 SP10.19 -> ✓中

  2. 2026092602 [德甲] 拜仁 vs 多特
     P_pool: 88.5%
     投注比分: 2-1/3-1/3-2/1-1/2-2/1-2/1-3/2-3
     实际比分: 3-1 SP12.50 -> ✓中

【2串1结果】
------------------------------------------------------------------------------
  已完场: 2/2 场
  结果: ✓ 两场全中!
  SP乘积: 127.38
  中奖回报: 127.38元
  投注成本: 22元 (2场×11注×1元)
  实际盈亏: +111.38元

==============================================================================
```

**使用流程**:
```bash
# 1. 先运行预测(生成ml_predictions.jsonl)
python -u predict_over.py --min-prob 0.70

# 2. 再运行选场(从预测结果中选Top2)
python -u select_2x1.py --min-prob 0.80

# 3. 赛后回顾(查实际比分+SP值,计算盈亏)
python -u select_2x1.py --review 20260926
```

> **2串1数学**：组合概率 = P1 × P2，例如 0.92 × 0.88 = 81.0%
> **投注成本**：2场 × 11注 × 1元 = 22元
> **盈亏公式**：两场都中 → SP1×SP2 - 22元；任一场不中 → -22元

### 定时部署

```
30 21 * * * 每日 21:30 predict_over.py --min-prob 0.70  # 当日预测 (阈值 0.70, 在 18:00 筛选之后, 含前一天自动回顾)
40 12 * * * 每日 12:40 predict_over.py --review          # 回顾昨天预测命中 (结果邮件发送)
0 10 * * 1  每周一 10:00 train_over.py                # 全量重训 (持续学习)
```

> **阈值说明**: 服务器 cron 固定使用 `--min-prob 0.70` 覆盖模型 EV 阈值。原因:20260917 模型算出阈值 0.80 过高导致空仓,故统一用 0.70 保证每日有推荐。如需调整,修改 crontab 即可。

---

## UNDER 九比分池模型

### 用途

用北单历史数据训练混合机器学习模型，筛选最终比分落在 **9 比分池** 中的比赛并输出各比分概率，以期望收益（EV）为优化目标。

> 9 比分池 `UNDER`：`1-0 / 2-0 / 2-1 / 0-0 / 1-1 / 2-2 / 0-1 / 0-2 / 1-2`
> 完全对称：主胜3种 + 平局3种 + 客胜3种，含零封比分（1-0、0-1、0-0）
> 2026-10-12 按全库赛果频率 Top9 重构（换入高频 2-0/0-2，换出冷门 3-2/2-3）。
> 定义在 `features_under.py`，与 OVER 模型使用不同的 9 种比分，两个模型完全独立。

> 训练盘口范围（宽口径）：`['受让半球', '半球', '受让平/半', '平/半', '平手']`
> **预测推荐盘口范围（窄口径）：`['平手', '受让平/半', '平/半']`**（2026-10-10 起，详见 WIKI_9P.md）
> **大小球线上限过滤（2026-10-13 起）**：终盘大小球线 ≤ 3.25（`MAX_OU_LINE_9P`）。
> 全库分档显示 UNDER 池内命中率随大小球线单调下降，OU>3.25 的"进球大战"档 ROI 崩至~0（低分池难命中），故训练/回测/预测统一剔除终盘 > 3.25 的比赛；**不设下限**（9 池含 0-0、1-0、0-1 等低比分，低球档反而是其主场）。实现见 `is_valid_ou_line_under`。

### 模型架构（三阶段混合）

与 OVER 模型完全相同的三阶段混合架构：

| 组件 | 模型 | 作用 |
|------|------|------|
| 筛选门 | LightGBM 二分类 | 预测比赛落在 9 池内的概率 P_lgb |
| 进球期望 | LightGBM 泊松回归×2 | 分别预测主/客队期望进球 λ_home / λ_away |
| 比分矩阵 | Dixon-Coles 双变量泊松 | 低比分相关修正（tau 因子, rho 网格拟合），产出全比分概率矩阵 |
| 融合 | α 加权 | P = α·P_lgb + (1-α)·P_dc，α 在验证集按 Brier 最小选取 |
| 校准 | Isotonic 回归 | 融合概率校准为真实命中率 |
| 阈值 | 按 EV 选取 | 每场 9 注各 1 元：命中=SP-9，未中=-9，选 EV 最优的 P_pool 阈值 |

### 与 OVER 的关键差异

| 差异点 | OVER | UNDER |
|--------|-------|-------|
| 比分池 | 11种(BTTS, 含高比分) | 9种(完全对称, 含零封) |
| 零封比分 | 无 | 1-0, 0-1, 0-0 |
| 大小球过滤 | 下限 ≥2.25（无上限） | 上限 终盘≤3.25（不设下限） |
| 盘口范围 | 训练=推荐(五种) | 训练五种 / **推荐仅 平手・受让平半・平半** |
| 投注注数 | 11注/场 | 9注/场 |
| 盈亏公式 | 命中=SP-11, 未中=-11 | 命中=SP-9, 未中=-9 |
| 特征缓存 | `features_over_cache.csv` | `features_under_cache.csv` |
| 模型文件 | `lgb_pool_*.txt`, `latest.json` | `lgb_pool_under_*.txt`, `latest_under.json` |
| 预测日志 | `ml_predictions.jsonl` | `predictions_under.jsonl` |
| 回测结果 | `backtest_result.json` | `backtest_under_result.json` |

### 使用流程

```bash
# 服务器执行
.venv/bin/python -u backtest_under.py     # 1. 滚动前进回测 -> backtest_under_result.json
.venv/bin/python -u train_under.py        # 2. 全量训练 -> models/ (带日期tag+latest_under.json)
.venv/bin/python -u predict_under.py      # 3. 每日预测
```

### 每日预测(predict_under.py)

```bash
python -u predict_under.py                      # 预测当天比赛
python -u predict_under.py --date 20260818      # 指定日期
python -u predict_under.py --all-unfinished     # 全部未开赛比赛
python -u predict_under.py --min-prob 0.65      # 手动指定置信度阈值
python -u predict_under.py --review             # 回看昨天预测结果
python -u predict_under.py --backfill 20260906  # 补录指定日期预测记录
```

### 定时部署

```
30 21 * * * 每日 21:30 predict_under.py --min-prob 0.65  # 当日9池预测
40 12 * * * 每日 12:40 predict_under.py --review          # 回顾昨天9池预测命中
0 10 * * 1  每周一 10:00 train_under.py                   # 全量重训
```

> 详细文档见 [WIKI_9P.md](WIKI_9P.md)。

---

## UPSIDE + 9 池两层模型

### 核心洞察

在亚盘（初盘+终盘）为 `['半球', '平手/半球', '平手', '受让平手/半球']` 且机构数 > 12 的比赛中：
- **主队不败（UPSIDE）基础率**: ~69-71%（主胜或平局）
- **9 池命中率**: ~64.7%（8 种主队不败低比分 + 0-1）

UPSIDE 层用 LightGBM 二分类预测主队不败；9 池分支与 UNDER 同构的三阶段混合预测池概率。

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
│  池内筛选门 LightGBM(in_pool_9) + 泊松回归λ×2 +         │
│  Dixon-Coles矩阵池概率 → α融合 → Isotonic校准 → P_underool │
└──────────────────────┬───────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────┐
│  动态选场策略                                          │
│  P_UPSIDE > 0.75  → 推荐 UPSIDE（双选胜平）                    │
│  P_underool >= 阈值(meta EV选择) → 推荐 9 池比分(9注/场)   │
│  否则 → 空仓                                          │
└──────────────────────────────────────────────────────
```

### 9 池比分列表

```
POOL_UPSIDE = ['1-0', '2-0', '2-1', '3-1', '3-2', '0-0', '1-1', '2-2', '0-1']
```

即 8 种主队不败低比分（主胜 5 种 + 平局 3 种）+ 客胜 0-1（2026-10-12 起扩池）。

### 模型架构详解

#### Layer 1: LightGBM 二分类

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

#### 9 池分支: 三阶段混合（同 UNDER）

| 组件 | 模型 | 作用 |
|------|------|------|
| 池内筛选门 | LightGBM 二分类 | 预测比赛落在 9 池内的概率 P_lgb（标签 `in_pool_9`） |
| 进球期望 | LightGBM 泊松回归×2 | 分别预测主/客队期望进球 λ_home / λ_away |
| 比分矩阵 | Dixon-Coles 双变量泊松 | 低比分相关修正（tau 因子, rho 网格拟合），9 池概率和 = P_dc |
| 融合 | α 加权 | P = α·P_lgb + (1-α)·P_dc，α 在验证集按 Brier 最小选取 |
| 校准 | Isotonic 回归 | 融合概率校准为真实命中率 → P_underool |
| 阈值 | 按 EV 选取 | 每场 9 注各 1 元：命中=SP-9，未中=-9，验证集选 EV 最优阈值写入 `meta_upside.json['threshold_pool']` |

> 2026-10-12 重训（tag 1x_v2）选出融合 α=0.0（DC 池概率主导）、阈值 0.65；测试集 9 池命中 69.5% vs 基础率 65.1%，实现 ROI +11.0%。

### 标签定义（features_upside_ext.py）

| 标签 | 含义 | 取值 |
|------|------|------|
| `home_unbeaten` | 主队不败 | 1=主胜或平局, 0=客胜, None=未完场 |
| `in_pool_9` | 9 池命中 | 1=在 9 池内, 0=不在, None=未完场 |

### 四步使用流程

```bash
# 1. 构建特征（复用 features_upside.py + 新增标签）
python -u features_upside_ext.py

# 2. 训练模型
python -u train_upside.py

# 3. 每日预测
python -u predict_upside.py

# 4. 回测验证
python -u backtest_upside.py
```

### 产物

| 文件 | 说明 |
|------|------|
| `features_upside_ext_cache.csv` | 特征缓存（首次构建后复用） |
| `models/lgb_upside.txt` | LightGBM UPSIDE 模型 |
| `models/lgb_upside_pool.joblib` | 9 池池内筛选门 LightGBM |
| `models/calib_upside.joblib` | 9 池概率 Isotonic 校准器 |
| `models/lgb_upside_goals_home/away.joblib` | 泊松进球模型×2 |
| `models/meta_upside.json` | 元信息（特征列、阈值、alpha、rho 等） |
| `predictions_upside.jsonl` | 每日预测推荐记录 |
| `backtest_upside_report.json` | 回测报告 |

### 每日预测（predict_upside.py）

- 输出每场比赛的 P_UPSIDE、P_underool、推荐类型
- 按 P_UPSIDE 降序排序
- 预测方向（胜平）写入 `match.prediction`，兼容现有下游格式
- 推荐记录写入 `predictions_upside.jsonl`(覆盖式: 同日旧记录被覆盖)
- 邮件发送受 `ENABLE_EMAIL` 开关控制

```bash
python -u predict_upside.py                      # 预测当天比赛
python -u predict_upside.py --date 20260912      # 指定日期
python -u predict_upside.py --all-unfinished     # 全部未开赛比赛
python -u predict_upside.py --skip-email         # 跳过邮件发送
```

### 回测验证（backtest_upside.py）

- 用测试集（后 15% 数据）回测
- 统计指标：
  - UPSIDE 命中率（P_UPSIDE > 0.75 的场次中，实际主队不败占比）
  - 9 池命中率（P_underool >= 阈值 的场次中，实际 9 池占比）
  - 对比基础率（~69% 和 ~65%）
  - 9 池实现盈亏（9注/场: 命中=SP-9, 未中=-9）
  - 月度稳定性
  - 2 串 1 理论命中率
  - 期望收益（EV）

```bash
python -u backtest_upside.py                     # 全量回测
python -u backtest_upside.py --start 20260601    # 指定起始日期
```

### 定时部署

```
35 22 * * * 每日 22:35 predict_upside.py             # 当日预测 (在 21:30 OVER 预测之后)
30 10 * * 1  每周一 10:30 train_upside.py            # 全量重训
```

---

## 模型选择指南

### 何时使用 OVER 模型

- 需要**具体比分预测**（11 种比分概率分布）
- 盘口为**平手/平半/半球/受让平半/受让半球**
- 追求**双方进球+中低比分场景**（主胜3种+平局3种+客胜3种+高比分2种）
- 有 SP 值数据，以**期望收益（EV）**为优化目标

### 何时使用 UPSIDE + 9 池模型

- 盘口为**平手/平半/半球/受让平半**且机构数 > 12
- 追求**高命中率**（UPSIDE 基础率 ~70%、9 池 ~65%）
- 投注**双选胜平**（UPSIDE）或**9 池比分（9注/场）**

### 组合使用

两个模型可并行运行，互不冲突：
- OVER 模型覆盖平手/平半/半球/受让平半/受让半球盘口，输出 11 比分概率
- UPSIDE 模型专攻平手/平半/半球/受让平半盘口，输出主队不败概率 + 9 池概率
- 同一场比赛可能同时被两个模型推荐，可交叉验证

---

## 技术栈

- **Python 3.11**
- **LightGBM** — 梯度提升分类/回归模型
- **pandas** — 特征表构建与数据处理
- **scikit-learn** — AUC / Brier Score / Isotonic 校准
- **joblib** — 模型序列化存储
- **pymysql** — MySQL 数据库连接

---

## 常见问题

### Q: 修改特征后需要做什么？
A: 删除 `features_over_cache.csv`（OVER）、`features_under_cache.csv`（UNDER）或 `features_upside_ext_cache.csv`（UPSIDE）缓存文件，重跑训练脚本。

### Q: 两个模型会冲突吗？
A: 完全不冲突。使用不同的特征缓存、模型文件、预测日志，可并行运行。

### Q: 如何调整推荐阈值?
A: OVER 模型用 `--min-prob` 命令行参数(服务器 cron 已固定为 0.70);UPSIDE 模型 UPSIDE 层阈值改 `predict_upside.py` 的 `THRESHOLD_UPSIDE`，9 池阈值由训练时 EV 自动选择写入 `meta_upside.json['threshold_pool']`。

### Q: 为什么服务器用固定阈值 0.70 而不是模型 EV 阈值?
A: 20260917 模型 EV 算出阈值 0.80,导致当日所有比赛 p_pool<0.80,全部 `picked=False`,邮件显示“空仓”。为保证每日有推荐,服务器 cron 固定使用 `--min-prob 0.70` 覆盖模型阈值。如需调整,修改 crontab 即可。

### Q: 模型效果不如预期怎么办？
A: 1) 运行回测脚本确认 AUC 和命中率；2) 检查特征是否有泄漏；3) 尝试调整阈值；4) 考虑重训模型。

### Q: 为什么 UPSIDE 模型 9 池分支用三阶段混合？
A: 2026-10-12 起（1x_v2）废弃固定条件概率 0.865（扩池含 0-1 后条件关系不再成立），改为与 UNDER 同构的三阶段混合：池内门 + 泊松 + DC + α融合 + Isotonic 校准，阈值由验证集 EV 自动选择。

### Q: 如何定时运行？
A: 配置 crontab 定时任务，见各模型的"定时部署"章节。

---

## 版本历史

| 版本 | 日期 | 模型 | 变更 |
|------|------|------|------|
| v1 | 2026-07 | UNDER | 初始版本：LightGBM + 泊松回归 + Dixon-Coles 混合模型 |
| v2 | 2026-08 | UNDER | 新增 --review 回看模式、持续学习监控、邮件告警 |
| v3 | 2026-09 | UNDER | 新增--backfill 补录模式、review 输出 SP 值 + 盈亏统计、预测前自动回顾前一天、阈值固定 0.70 |
| 9p_v1 | 2026-08-23 | UNDER | 初始版本：参照OVER模型架构创建9池预测系统，含特征工程/训练/回测/预测全流程 |
| v4 | 2026-09-23 | OVER | 比分池改为 8 种主队不败低比分，新增盘口筛选+大小球筛选+排除杯赛，投注 8 注全投 |
| 1x_v1 | 2026-09-15 | UPSIDE+8 池 | 初始版本：LightGBM 二分类 + 固定条件概率 0.865 |
| v5 | 2026-10-10 | OVER/UNDER | 预测推荐口径收窄：ML 大小球只推 ≥2.75（`MIN_OU_LINE_PREDICT_ML`）；9P 盘口只推 平手/受让平半/平半（`PREDICT_HANDICAPS_9P`）；训练口径均不变 |
| 9p_v4 | 2026-10-12 | UNDER | 比分池按全库赛果频率 Top9 重构（换入高频 2-0/0-2，换出冷门 3-2/2-3）；滚动回测覆盖 32.6%→53.7%，ROI +36.5%→+41.3% |
| 9p_v5 | 2026-10-13 | UNDER | 新增大小球线上限过滤：终盘 > 3.25（`MAX_OU_LINE_9P`）的"进球大战"死区剔除，不设下限（`is_valid_ou_line_under`）；训练/回测/预测统一接入 |
| v6 | 2026-10-11 | OVER/UNDER/UPSIDE | jsonl 改覆盖式写入（`replace_prediction_log`）：每个日期只保留最后一次预测候选集，修复同日多次预测时回顾与预测邮件场次不一致 |
| 1x_v2 | 2026-10-12 | UPSIDE+9 池 | 比分池扩为 9 种（+0-1）；盘口增加受让平半；9 池概率改三阶段混合（池内门+泊松+DC+α融合+Isotonic），阈值 valid 集 EV 自动选择；投注 9 注/场 |
| v7 | 2026-10-13 | OVER/UNDER/UPSIDE | h2h 往绩特征+2：新增 `h2h_half_goal_avg`（场均半场进球）+`h2h_corners_avg`（场均角球）；数据源从 match 表切换到 head_to_head 表。9P 特征 49→51、ML 特征 54→56。三模型全部重训 tag=20261013 |
| v8 | 2026-10-13 | 全部 | 新增 `european_odds` 表 + `scrape_euro_odds.py` 欧赔数据补录脚本（数据源 `1x2d.titan007.com/{id}.js`，每场~130家公司胜平负初盘+即时赔率）；后台全量补录中 |
| v9 | 2026-10-14 | OVER/UNDER/UPSIDE | 欧赔特征集成：新增 `euro_features_upside()` 函数（11个特征：公司数、初/即时隐含胜平负概率、公司间分歧度、返还率、主胜概率变动），数据从 `european_odds` 表批量加载。9P 特征 51→62、ML 特征 56→67、UPSIDE 同步新增。三模型全部重训 tag=20261014 |
| v10 | 2026-10-21 | OVER/UNDER/UPSIDE | P0/P1/P2特征工程：新增联赛积分榜详细特征(28个/模型)、交锋初始盘口特征(5个/模型)、页面盘口/排名特征(5个/模型)；技术统计特征(30个/模型)。OVER比分池 8→11（+3-3/4-2/2-4），投注 11注/场。特征文件重命名：`features_upside_over.py`→`features_over.py`、`features_upside_under.py`→`features_under.py`。训练文件：`train_model.py`→`train_over.py`。三模型全部重训 |
| v11 | 2026-10-25 | OVER/UNDER | 移除技术统计特征（30个）：覆盖率不足（全量60%）、缺失值导致模型效果变差。OVER/UNDER 特征从 ~108/~98 降至 **62个**。添加多进程并行优化（`multiprocessing.Pool`），训练提速 4-6倍 |

---

# 数据处理与特征工程关键逻辑

> 本文档记录 OVER/UNDER 模型的数据处理流程、特征工程关键逻辑和最新代码实现细节，便于后续回顾、分析和调整。

## 一、数据预处理流程

### 1.1 数据源

| 数据表 | 用途 | 关键字段 |
|--------|------|----------|
| `match` | 比赛基本信息 | match_id, league, home_team, away_team, score, match_time |
| `asian_odds` | 亚盘赔率 | initial_handicap, current_handicap, initial_home_odds, initial_away_odds |
| `over_under_odds` | 大小球赔率 | initial_line, current_line, initial_over_odds, initial_under_odds |
| `european_odds` | 欧赔 | initial_win_odds, initial_draw_odds, initial_loss_odds |
| `match_history` | 历史交锋 | home_team, away_team, score, half_score, corners |
| `team_recent` | 近期战绩 | team, score, shots, shots_on_target, corners, dangerous_attacks, possession, fouls, yellow_cards |

### 1.2 数据清洗规则

```python
# 比分解析
def parse_score(score):
    try:
        if pd.isna(score) or score == '' or score == 'nan':
            return None, None
        parts = str(score).split('-')
        if len(parts) == 2:
            return int(parts[0].strip()), int(parts[1].strip())
    except:
        pass
    return None, None

# 数值转换
def to_float(v):
    try:
        if v is None or v == '' or v == 'nan':
            return None
        return float(v)
    except:
        return None

# 盘口转换
def handicap_to_num(h):
    """将盘口字符串转换为数值"""
    mapping = {
        '受让球半': 1.5, '受让一球/球半': 1.25, '受让一球': 1.0,
        '受让半球/一球': 0.75, '受让半球': 0.5, '受让平/半': 0.25,
        '平手': 0.0, '平/半': -0.25, '半球': -0.5,
        '半球/一球': -0.75, '一球': -1.0, '一球/球半': -1.25,
        '球半': -1.5
    }
    return mapping.get(str(h).strip(), None)
```

---

## 二、特征工程关键逻辑

### 2.1 时间衰减机制

**所有近期战绩特征（含技术统计）都应用了时间衰减权重。**

```python
# 时间衰减参数
TIME_DECAY_LAMBDA = 0.08  # 指数衰减率

# 权重计算公式
# w_i = exp(-λ × (n-1-i))
# 其中：n=总场数, i=索引(0=最旧, n-1=最新)
# 最新比赛权重=1.0，越旧的比赛权重越低

weights = [np.exp(-TIME_DECAY_LAMBDA * (n - 1 - i)) for i in range(n)]
total_weight = sum(weights)
```

**权重示例（10场比赛）：**

| 比赛位置 | 索引 i | 权重计算 | 权重值 |
|---------|--------|---------|--------|
| 最新（第10场） | 9 | exp(-0.08×0) | **1.000** |
| 第9场 | 8 | exp(-0.08×1) | **0.923** |
| 第8场 | 7 | exp(-0.08×2) | **0.852** |
| 第7场 | 6 | exp(-0.08×3) | **0.787** |
| 第6场 | 5 | exp(-0.08×4) | **0.726** |
| 第5场 | 4 | exp(-0.08×5) | **0.670** |
| 第4场 | 3 | exp(-0.08×6) | **0.619** |
| 第3场 | 2 | exp(-0.08×7) | **0.571** |
| 第2场 | 1 | exp(-0.08×8) | **0.527** |
| 最旧（第1场） | 0 | exp(-0.08×9) | **0.487** |

**应用范围：**
- ✅ 近期战绩特征（胜负率、进球数等）
- ✅ 技术统计特征（射门、射正、角球等7项）
- ✅ 主客场分离（h_rf_*, a_rf_*）

---

### 2.2 技术统计特征（19个）

**OVER 和 UNDER 模型都使用了相同的19个技术统计特征：**

#### ① 交锋技术统计（1个）
- `h2h_corners_avg` - 历史交锋角球平均数

#### ② 主队近期技术统计（7个）
- `h_rf_shots_mean` - 主队近期射门平均数（时间衰减加权）
- `h_rf_sot_mean` - 主队近期射正平均数
- `h_rf_corners_mean` - 主队近期角球平均数
- `h_rf_danger_mean` - 主队近期危险进攻平均数
- `h_rf_poss_mean` - 主队近期控球率平均数
- `h_rf_fouls_mean` - 主队近期犯规平均数
- `h_rf_yellow_mean` - 主队近期黄牌平均数

#### ③ 客队近期技术统计（7个）
- `a_rf_shots_mean` - 客队近期射门平均数
- `a_rf_sot_mean` - 客队近期射正平均数
- `a_rf_corners_mean` - 客队近期角球平均数
- `a_rf_danger_mean` - 客队近期危险进攻平均数
- `a_rf_poss_mean` - 客队近期控球率平均数
- `a_rf_fouls_mean` - 客队近期犯规平均数
- `a_rf_yellow_mean` - 客队近期黄牌平均数

#### ④ 组合技术统计特征（4个）
- `possession_diff` - 控球率差值（主-客）
- `corners_diff` - 角球差值
- `fouls_diff` - 犯规差值
- `yellow_diff` - 黄牌差值

**特征质量：**
- 缺失率：约0.03%（极低）
- 数据源：team_recent 表
- 更新频率：每场比赛后更新

---

### 2.3 盘口特征缺失值处理

**亚盘特征（14个）：**

```python
def asian_features(rows):
    feats = {k: None for k in keys}  # 初始化为 None
    
    for r in rows:
        v_i = handicap_to_num(r['initial_handicap'])
        v_c = handicap_to_num(r['current_handicap'])
        
        # ✅ 即时盘缺失 → 用初盘填充
        if v_c is None:
            v_c = v_i
        
        if v_i is not None:
            ih.append(v_i)
        if v_c is not None:
            ch.append(v_c)
    
    # ✅ 仅当有数据时才计算
    if ih:
        feats['ah_init_mean'] = _mean(ih)
        feats['ah_init_std'] = _std(ih)
    if ch:
        feats['ah_cur_mean'] = _mean(ch)
    if ih and ch:  # 两者都有才计算变动
        feats['ah_move'] = _mean(ch) - _mean(ih)
```

**处理逻辑：**
- ✅ 即时盘缺失 → 用初盘填充
- ✅ 部分数据缺失 → 仅计算有数据的部分
- ✅ 完全无数据 → 返回全 None（后续由中位数填充）

---

### 2.4 大小球特征缺失值处理

**大小球特征（8个）：**

```python
def over_under_features(rows):
    feats = {k: None for k in keys}
    
    for r in rows:
        v_i = to_float(r['initial_line'])
        v_c = to_float(r['current_line'])
        
        # ✅ 即时大小球线缺失 → 用初盘填充
        if v_c is None:
            v_c = v_i
    
    # ✅ 仅当有数据时才计算
    if li:
        feats['ou_line_init_mean'] = _mean(li)
    if lc:
        feats['ou_line_cur_mean'] = _mean(lc)
    if li and lc:  # 两者都有才计算变动
        feats['ou_line_move'] = _mean(lc) - _mean(li)
    
    # ✅ 隐含概率需要大球和小球赔率都存在
    if lo_i and uo_i:
        feats['over_prob_init'] = _mean([u / (o + u) for o, u in zip(lo_i, uo_i)])
```

**处理逻辑：**
- ✅ 即时盘缺失 → 用初盘填充
- ✅ 赔率缺失 → 不计算该特征（保持 None）
- ✅ 隐含概率计算 → 需要大球和小球赔率都存在

---

### 2.5 欧赔特征缺失值处理

**欧赔特征（11个）：**

```python
def euro_features(rows):
    feats = {k: None for k in keys}
    
    for r in rows:
        iw = to_float(r['initial_win_odds'])
        id_ = to_float(r['initial_draw_odds'])
        il = to_float(r['initial_loss_odds'])
        
        # ✅ 严格校验：三个赔率都必须存在且 > 1
        if iw and id_ and il and iw > 1 and id_ > 1 and il > 1:
            inv_w, inv_d, inv_l = 1/iw, 1/id_, 1/il
            total = inv_w + inv_d + inv_l
            
            # ✅ 去水归一化
            pw_i.append(inv_w / total)
            pd_i.append(inv_d / total)
            pl_i.append(inv_l / total)
```

**处理逻辑：**
- ✅ 严格校验 → 三个赔率（胜/平/负）都必须存在且 > 1
- ✅ 去水归一化 → `P = (1/odds) / sum(1/odds)`
- ✅ 任一赔率缺失 → 不计算该场比赛的隐含概率

---

### 2.6 交锋特征缺失值处理

**交锋特征（11个）：**

```python
def _h2h_features_fast(home_team, away_team, h2h_data, max_h2h=10):
    feats = {k: None for k in keys}
    
    h2h_rows = h2h_data[:max_h2h]
    n = len(h2h_rows)
    
    # ✅ 无交锋记录 → 返回全 None
    if n == 0:
        return feats
    
    for match in h2h_rows:
        hg, ag = parse_score(match['score'])
        
        # ✅ 比分解析失败 → 跳过该场
        if hg is None:
            continue
    
    # ✅ 仅当有有效数据时才计算
    if goals_list:
        feats['h2h_n'] = n
        feats['h2h_home_win_rate'] = home_wins / n
        feats['h2h_avg_goals'] = _mean(goals_list)
    
    # ✅ 可选特征（半场、角球）→ 有数据才计算
    if half_goals_list:
        feats['h2h_half_goal_avg'] = _mean(half_goals_list)
    if corners_list:
        feats['h2h_corners_avg'] = _mean(corners_list)
```

**处理逻辑：**
- ✅ 无交锋记录 → 返回全 None
- ✅ 比分解析失败 → 跳过该场比赛
- ✅ 可选特征（半场、角球）→ 有数据才计算，无数据保持 None

---

### 2.7 联赛特征缺失值处理

**联赛特征（8个）：**

```python
def _league_stats_features_fast(league, league_data):
    feats = {k: None for k in keys}
    
    n = len(league_data)
    
    # ✅ 无联赛数据 → 返回全 None
    if n == 0:
        return feats
    
    for match in league_data:
        hg, ag = parse_score(match['score'])
        
        # ✅ 比分解析失败 → 跳过
        if hg is None:
            continue
    
    # ✅ 仅当有进球数据时才计算
    if goals_list:
        feats['league_n'] = n
        feats['league_home_win_rate'] = home_wins / n
        feats['league_avg_goals'] = _mean(goals_list)
```

**处理逻辑：**
- ✅ 无联赛数据 → 返回全 None
- ✅ 比分解析失败 → 跳过该场比赛
- ✅ 仅计算有数据的部分

---

### 2.8 近期战绩特征缺失值处理

**近期战绩特征（20个，含技术统计）：**

```python
def _recent_form_features_fast(team_name, team_data, is_home=True, league_avgs=None):
    feats = {k: None for k in keys}
    
    n = len(team_data)
    
    # ✅ 无近期比赛 → 返回全 None
    if n == 0:
        return feats
    
    # 计算胜负率等（仅当有有效比分时）
    if goals_for_count > 0:
        feats['recent_n'] = n
        feats['recent_win_rate'] = wins / total_weight
        feats['recent_goals_for_mean'] = goals_for_w / total_weight
    
    # ✅ 技术统计缺失 → 用联赛平均值填充
    feats['rf_shots_mean'] = (shots_w / shots_count) if shots_count > 0 else league_avgs.get('shots')
    feats['rf_sot_mean'] = (sot_w / sot_count) if sot_count > 0 else league_avgs.get('shots_on_target')
    feats['rf_corners_mean'] = (corners_w / corners_count) if corners_count > 0 else league_avgs.get('corners')
    feats['rf_danger_mean'] = (danger_w / danger_count) if danger_count > 0 else league_avgs.get('dangerous_attacks')
    feats['rf_poss_mean'] = (poss_w / poss_count) if poss_count > 0 else league_avgs.get('possession')
    feats['rf_fouls_mean'] = (fouls_w / fouls_count) if fouls_count > 0 else league_avgs.get('fouls')
    feats['rf_yellow_mean'] = (yellow_w / yellow_count) if yellow_count > 0 else league_avgs.get('yellow_cards')
```

**处理逻辑：**
- ✅ 无近期比赛 → 返回全 None
- ✅ 比分解析失败 → 跳过该场比赛
- ✅ 技术统计缺失 → 用联赛平均值填充

---

## 三、缺失值处理三层机制

### 3.1 第一层：特征生成阶段

| 特征类型 | 缺失处理策略 |
|---------|-------------|
| 盘口特征 | 即时盘缺失 → 用初盘填充 |
| 大小球特征 | 即时盘缺失 → 用初盘填充 |
| 欧赔特征 | 严格校验，任一赔率缺失 → 不计算 |
| 交锋特征 | 无数据 → 返回全 None |
| 联赛特征 | 无数据 → 返回全 None |
| 近期战绩 | 无数据 → 返回全 None |
| 技术统计 | 球队无数据 → 用联赛平均值填充 |

### 3.2 第二层：训练阶段

```python
def _add_missing_indicators(df, feats):
    """为所有特征列添加缺失指示器(1=缺失, 0=非缺失)"""
    indicator_cols = []
    # 只为原始特征创建指示器，跳过已经是指示器的列
    original_feats = [f for f in feats if not f.endswith('_missing')]
    
    for col in original_feats:
        if col not in df.columns:
            continue
        indicator_name = f'{col}_missing'
        df[indicator_name] = df[col].isnull().astype(int)  # 1=缺失, 0=非缺失
        indicator_cols.append(indicator_name)
    
    new_feats = feats + indicator_cols
    return df, new_feats

# 训练时填充
df_train, feats = _add_missing_indicators(df_train, feats)
medians = df_train[feats].median().to_dict()
X_tr = df_train[feats].fillna(df_train[feats].median())
X_va = df_valid[feats].fillna(df_valid[feats].median())

# 确保数值类型，再次填充
for col in X_tr.columns:
    X_tr[col] = pd.to_numeric(X_tr[col], errors='coerce')
    X_va[col] = pd.to_numeric(X_va[col], errors='coerce')
X_tr = X_tr.fillna(0)
X_va = X_va.fillna(0)
```

**处理逻辑：**
- ✅ 为每个特征创建**缺失指示器**（90个原始特征 → 180个特征）
- ✅ 使用**训练集中位数**填充缺失值
- ✅ 强制转换为数值类型，剩余NaN填充为0
- ✅ 缺失指示器让模型知道哪些值是填充的

### 3.3 第三层：预测阶段

```python
# 使用训练时保存的中位数填充
fill_medians = model['medians']
X = X.fillna(fill_medians)

# 确保数值类型
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.fillna(0)
```

**处理逻辑：**
- ✅ 使用**训练时保存的中位数**（存储在模型meta中）
- ✅ 保持训练和预测的填充逻辑一致
- ✅ 强制转换数值类型，剩余NaN填充为0

---

## 四、完整处理流程

```
┌─────────────────────────────────────────────────────────────┐
│ 第一层：特征生成                                              │
│                                                             │
│ 1. 盘口特征：                                                │
│    - 即时盘缺失 → 用初盘填充                                 │
│    - 部分数据缺失 → 仅计算有数据的部分                        │
│                                                             │
│ 2. 大小球特征：                                              │
│    - 即时盘缺失 → 用初盘填充                                 │
│    - 赔率缺失 → 不计算隐含概率                               │
│                                                             │
│ 3. 欧赔特征：                                                │
│    - 严格校验：三个赔率都必须存在且 > 1                      │
│    - 去水归一化计算隐含概率                                  │
│                                                             │
│ 4. 交锋/联赛/近期特征：                                      │
│    - 无数据 → 返回全 None                                    │
│    - 比分解析失败 → 跳过该场                                 │
│    - 可选特征（半场/角球）→ 有数据才计算                     │
│                                                             │
│ 5. 技术统计特征：                                            │
│    - 球队无数据 → 用联赛平均值填充                           │
│    - 应用时间衰减权重（λ=0.08）                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 第二层：训练阶段                                              │
│ - 创建缺失指示器（90个 → 180个特征）                          │
│ - 使用训练集中位数填充所有 None 值                            │
│ - 强制数值类型转换，剩余 NaN 填充为 0                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 第三层：预测阶段                                              │
│ - 使用训练时保存的中位数填充                                  │
│ - 保持与训练一致的填充逻辑                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 五、关键设计原则

### 5.1 保守策略
- 宁可缺失也不使用不可靠数据
- 严格校验数据质量（如欧赔三个赔率都必须存在）

### 5.2 分层处理
- 特征生成 → 缺失指示器 → 中位数填充
- 每一层都有明确的职责

### 5.3 信息保留
- 缺失指示器让模型知道哪些值是填充的
- 避免信息丢失，提高模型鲁棒性

### 5.4 一致性
- 训练和预测使用相同的填充逻辑
- 确保模型在生产环境中的表现与训练时一致

### 5.5 时间敏感性
- 所有近期特征都应用时间衰减
- 确保模型更关注球队近期状态

---

## 六、特征统计

### 6.1 OVER 模型特征清单（90个）

| 类别 | 数量 | 特征名 |
|------|------|--------|
| 亚盘 | 14 | asian_n, ah_init_mean, ah_init_std, ah_init_flat_share, ah_cur_mean, ah_cur_std, ah_cur_flat_share, ah_move, ah_move_up_share, cover_prob_init, aw_init_diff, aw_cur_diff, aw_drift |
| 大小球 | 8 | ou_n, ou_line_init_mean, ou_line_init_std, ou_line_cur_mean, ou_line_move, over_prob_init, over_prob_cur |
| 欧赔 | 11 | euro_n, euro_win_prob_init, euro_draw_prob_init, euro_loss_prob_init, euro_win_prob_cur, euro_draw_prob_cur, euro_loss_prob_cur, euro_win_std_init, euro_win_std_cur, euro_return_mean, euro_win_move |
| 交锋 | 11 | h2h_n, h2h_home_win_rate, h2h_draw_rate, h2h_away_win_rate, h2h_avg_goals, h2h_home_goals_mean, h2h_away_goals_mean, h2h_btts_rate, h2h_over_25_rate, h2h_half_goal_avg, h2h_corners_avg |
| 近期战绩 | 20 | recent_n, recent_win_rate, recent_draw_rate, recent_goals_for_mean, recent_goals_against_mean, recent_clean_sheet_rate, recent_btts_rate, recent_over_25_rate, recent_unbeaten_rate, h_rf_shots_mean, h_rf_sot_mean, h_rf_corners_mean, h_rf_danger_mean, h_rf_poss_mean, h_rf_fouls_mean, h_rf_yellow_mean, a_rf_shots_mean, a_rf_sot_mean, a_rf_corners_mean, a_rf_danger_mean, a_rf_poss_mean, a_rf_fouls_mean, a_rf_yellow_mean |
| 联赛统计 | 8 | league_n, league_home_win_rate, league_draw_rate, league_away_win_rate, league_avg_goals, league_btts_rate, league_over_25_rate, league_home_unbeaten_rate, league_goal_strength |
| 组合特征 | 4 | attack_eff_diff, possession_diff, corners_diff, fouls_diff, yellow_diff, form_momentum_h, form_momentum_a |
| **总计** | **90** | + 90个缺失指示器 = **180个特征** |

### 6.2 UNDER 模型特征清单

与 OVER 模型相同，共90个原始特征 + 90个缺失指示器 = 180个特征。

---

## 七、版本更新记录

### 2026-11-04
- ✅ 添加技术统计特征（19个）
- ✅ 应用时间衰减机制（λ=0.08）
- ✅ 完善缺失值处理三层机制
- ✅ 调整比分池：
  - OVER: 2-1, 1-2, 2-2, 3-1, 3-0, 0-3, 3-2, 1-3, 2-3
  - UNDER: 1-1, 1-0, 2-1, 0-1, 1-2, 0-0, 2-0, 0-2, 3-0
- ✅ 调整大小球线过滤：
  - OVER: 2.5 <= ou_line <= 3.25
  - UNDER: ou_line <= 2.5

### 2026-11-02
- ✅ 移除技术统计指标特征（后于11-04恢复）

### 2026-11-01
- ✅ 初始版本

---

## 八、常见问题

### Q1: 为什么技术统计特征缺失率这么低（0.03%）？
A: 数据源（team_recent表）覆盖率高，大部分比赛都有完整的技术统计数据。

### Q2: 时间衰减系数λ=0.08是如何选择的？
A: 通过网格搜索和回测验证，λ=0.08在命中率和稳定性之间取得最佳平衡。

### Q3: 为什么使用联赛平均值填充技术统计缺失值？
A: 联赛平均值代表该联赛的典型水平，比全局平均值更合理，比中位数更能反映球队实力。

### Q4: 缺失指示器会增加模型复杂度吗？
A: 是的，特征数从90增加到180。但实验表明，缺失指示器能提供额外信息，提升模型性能（AUC提升约0.004）。

### Q5: 训练和预测的填充逻辑为什么必须一致？
A: 如果不一致，会导致分布偏移，模型在生产环境中的表现会与训练时不同。

---

## 九、参考资料

- `features_over.py` - OVER模型特征工程
- `features_under.py` - UNDER模型特征工程
- `train_over.py` - OVER模型训练脚本
- `train_under.py` - UNDER模型训练脚本
- `predict_over.py` - OVER模型预测脚本
- `predict_under.py` - UNDER模型预测脚本


---

# 回测与回测后处理逻辑

> 本文档记录 OVER/UNDER 模型的回测方法、预测回顾流程和补录机制，便于理解模型评估和监控体系。

## 一、回测（Backtest）逻辑

### 1.1 回测方法：滚动前进（Walk-Forward）

**时间轴示意：**

```
|----训练集(70%)----|--调参集(15%)--|--回测窗口1--|--回测窗口2--|--回测窗口3--|
                     ↑                              ↑
                  不参与调参                    按月滚动预测
```

**核心流程：**

```python
# 1. 数据切分
训练集 = 前70%数据
调参集 = 中间15%数据（用于选择超参）
验收集 = 后15%数据（不参与任何调参，最终评估用）

# 2. 按月滚动回测
for 每个月 in 回测期:
    # 用该月之前的所有数据训练
    训练数据 = 该月之前的全部数据
    调参数据 = 训练数据的15%
    
    # 训练模型
    model = train_over(训练数据, 调参数据)
    
    # 预测该月比赛
    预测结果 = predict_rows(model, 该月比赛)
    
    # 记录每场预测结果
    records.append({
        'match_id': 比赛ID,
        'month': 月份,
        'score': 实际比分,
        'in_pool': 是否在比分池内,
        'p_pool': 模型预测概率,
        'hit_top9': 是否命中Top9,
        'hit_top3': 是否命中Top3,
        'sp': 开奖SP值,
        'ev_model': 模型期望EV,
    })
```

### 1.2 回测评估指标

| 指标 | 说明 | 计算方式 | 优秀标准 |
|------|------|----------|----------|
| **基线池内率** | 无筛选时的命中率 | `池内比赛数 / 总比赛数` | >40% |
| **Top9精确比分命中率** | 模型推荐9个比分的命中情况 | `命中数 / 总比赛数` | >50% |
| **Top3精确比分命中率** | 前3个比分的命中情况 | `命中数 / 总比赛数` | >25% |
| **Brier分数** | 概率校准度（越低越好） | `mean((预测概率-实际结果)²)` | <0.20 |
| **对数损失** | 概率准确性（越低越好） | `-mean(y*log(p) + (1-y)*log(1-p))` | <0.55 |
| **EV（期望值）** | 每场期望盈亏 | `命中率×平均SP - 1` | >+2.0 |

### 1.3 回测输出

**单场记录示例：**

```json
{
    "match_id": "20261105001",
    "month": "202611",
    "league": "英超",
    "score": "2-1",
    "in_pool": 1,
    "p_pool": 0.75,
    "hit_top9": 1,
    "hit_top3": 0,
    "sp": 15.5,
    "ev_model": 2.5
}
```

**汇总统计示例：**

```json
{
    "总场次": 1000,
    "基线池内率": "450/1000 = 45.0%",
    "Brier分数": 0.1850,
    "对数损失": 0.5230,
    "Top9命中率": "320/1000 = 32.0%",
    "Top3命中率": "180/1000 = 18.0%",
    "平均EV": "+2.5/场"
}
```

**输出文件：**
- `backtest_over_result.json` - OVER模型回测结果
- `backtest_under_result.json` - UNDER模型回测结果
- `backtest_upside_result.json` - UPSIDE模型回测结果

---

## 二、预测回顾（Review）逻辑

### 2.1 回顾流程

```
1. 读取预测记录（ml_predictions.jsonl）
   ↓
2. 筛选目标日期
   ↓
3. 查询实际比分（从match表）
   ↓
4. 查询开奖SP值（从result_sp表）
   ↓
5. 对照预测与实际结果
   ↓
6. 计算命中率和盈亏
   ↓
7. 生成回顾报告（邮件发送）
```

### 2.2 回顾核心逻辑

```python
def review(target_date, send_mail=True):
    """回看指定日期预测: 读取ml_predictions.jsonl, 与实际比分对照命中情况"""
    
    # 1. 读取预测记录
    logs = read_jsonl('ml_predictions.jsonl')
    
    # 2. 筛选目标日期（去重，保留最后一次预测）
    daily_logs = {}
    for l in logs:
        if l['match_id'].startswith(target_date):
            daily_logs[l['match_id']] = l
    
    # 3. 查询实际比分
    actual_scores = query_db(
        "SELECT match_id, home_team, away_team, score FROM match WHERE match_id IN (...)"
    )
    
    # 4. 查询开奖SP值
    sp_values = query_db(
        "SELECT match_id, sp_value FROM result_sp WHERE match_id IN (...)"
    )
    
    # 5. 对照分析
    for match_id, prediction in daily_logs.items():
        预测比分 = prediction['top_picks']  # 模型推荐的9个比分
        实际比分 = actual_scores[match_id]['score']
        开奖SP = sp_values.get(match_id)
        
        # 判断是否命中
        池内命中 = 实际比分 in OVER_POOL  # 实际比分是否在9比分池内
        投注命中 = 实际比分 in 预测比分   # 实际比分是否在推荐的9个比分内
        
        # 计算盈亏（每场投9注，每注1元）
        if 投注命中:
            盈亏 = 开奖SP - 9  # 中1注，获得SP值，损失9注成本
        else:
            盈亏 = -9  # 未命中，损失9注
    
    # 6. 生成报告
    报告 = f"""
    预测回顾 {target_date}
    ====================
    总预测场次: {n}
    池内命中率: {pool_hit_rate}%
    投注命中率: {bet_hit_rate}%
    总盈亏: {total_profit}元
    平均每场盈亏: {avg_profit}元
    """
    
    # 7. 邮件发送
    if send_mail:
        send_email(f"OVER预测回顾 {target_date}", 报告)
```

### 2.3 回顾报告示例

```
==============================================================================
预测回顾 20261104  (来源: ml_predictions.jsonl, 按池内概率由高到低)
命中口径: 池内=实际比分∈9比分池  ✓中=实际比分∈所选9个投注比分
==============================================================================

[1] 概率0.82 | 英超 | 阿森纳 vs 切尔西 | 预测: 2-1(18%),1-1(15%),2-0(12%)...
             实际比分: 2-1 | SP: 15.5 | ✓池内 ✓投注命中 | 盈亏: +6.5元

[2] 概率0.78 | 西甲 | 皇马 vs 巴萨 | 预测: 1-1(20%),2-1(18%),1-0(15%)...
             实际比分: 1-2 | SP: 18.0 | ✓池内 ✗未命中 | 盈亏: -9元

[3] 概率0.75 | 德甲 | 拜仁 vs 多特 | 预测: 3-1(16%),2-1(14%),2-0(12%)...
             实际比分: 3-1 | SP: 22.0 | ✓池内 ✓投注命中 | 盈亏: +13元

...

==============================================================================
汇总统计
==============================================================================
总预测场次: 15场
池内命中率: 12/15 = 80.0%
投注命中率: 8/15 = 53.3%
总盈亏: +45元
平均每场盈亏: +3.0元
==============================================================================
```

### 2.4 回顾命令

```bash
# 回顾昨天预测
python -u predict_over.py --review

# 回顾指定日期
python -u predict_over.py --review 20261104

# 回顾并发送邮件
python -u predict_over.py --review 20261104  # 默认send_mail=True
```

---

## 三、补录（Backfill）逻辑

### 3.1 补录用途

- 补录历史日期的预测记录
- 用于回顾分析历史预测效果
- 故障恢复：当预测脚本未执行时，手动补录

### 3.2 补录流程

```python
def backfill(target_date, min_prob=None):
    """补录指定日期预测记录到ml_predictions.jsonl(含已完场比赛)"""
    
    # 1. 加载历史数据
    df = load_features(target_date)
    
    # 2. 加载最新模型
    model = load_latest_model()
    
    # 3. 预测（含已完场比赛）
    pred = predict_rows(model, df)
    
    # 4. 筛选（可选）
    if min_prob:
        pred = pred[pred['p_pool'] >= min_prob]
    
    # 5. 写入预测记录（标记为backfill）
    for _, row in pred.iterrows():
        record = {
            'match_id': row['match_id'],
            'top_picks': row['top_picks'],
            'p_pool': row['p_pool'],
            'backfill': True,  # 标记为补录
            ...
        }
        append_to_jsonl('ml_predictions.jsonl', record)
    
    # 注意：不写match.prediction、不发邮件、不跑监控
```

### 3.3 补录命令

```bash
# 补录指定日期
python -u predict_over.py --backfill 20261104

# 补录并筛选概率
python -u predict_over.py --backfill 20261104 --min-prob 0.75
```

### 3.4 补录与常规预测的区别

| 特性 | 常规预测 | 补录 |
|------|---------|------|
| 执行时间 | 每天15:00自动 | 手动执行 |
| 数据范围 | 仅未开赛比赛 | 含已完场比赛 |
| 写入数据库 | ✅ 写match.prediction | ❌ 不写 |
| 发送邮件 | ✅ 发送预测邮件 | ❌ 不发送 |
| 运行监控 | ✅ 运行监控检查 | ❌ 不运行 |
| 记录标记 | 无标记 | `backfill: True` |
| 用途 | 当日预测 | 历史回顾 |

---

## 四、完整流程图

```
┌─────────────────────────────────────────────────────────────┐
│ 训练阶段（每周二）                                            │
│                                                             │
│ 1. 加载特征缓存                                              │
│ 2. 筛选目标盘口比赛                                           │
│ 3. 时间切分（训练70% / 调参15% / 验收15%）                    │
│ 4. 训练模型（LightGBM + Dixon-Coles）                         │
│ 5. 验收段评估（不参与调参）                                    │
│ 6. 保存模型到models/目录                                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 回测阶段（每周三凌晨）                                        │
│                                                             │
│ 1. 加载历史数据                                              │
│ 2. 按月滚动回测                                              │
│    - 用历史数据训练                                           │
│    - 预测当月比赛                                             │
│    - 记录每场预测结果                                         │
│ 3. 评估指标                                                  │
│    - 池内命中率                                               │
│    - Top9/Top3命中率                                         │
│    - Brier分数、对数损失                                      │
│    - EV曲线                                                  │
│ 4. 输出回测报告（backtest_over_result.json）                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 预测阶段（每天15:00）                                         │
│                                                             │
│ 1. 加载最新模型                                              │
│ 2. 构建当日特征                                              │
│ 3. 筛选目标盘口比赛                                           │
│ 4. 预测比分概率                                              │
│ 5. 写入预测记录（ml_predictions.jsonl）                       │
│ 6. 发送邮件通知                                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 回顾阶段（每天12:50或手动--review）                           │
│                                                             │
│ 1. 读取预测记录（ml_predictions.jsonl）                       │
│ 2. 查询实际比分（match表）                                    │
│ 3. 查询开奖SP值（result_sp表）                                │
│ 4. 对照预测与实际结果                                         │
│ 5. 计算命中率和盈亏                                           │
│ 6. 生成回顾报告（邮件发送）                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 五、定时任务对应关系

| 定时任务 | 脚本 | 频率 | 输出 |
|---------|------|------|------|
| **训练** | `train_over.py` | 每周二10:40 | 模型文件 + 验收报告 |
| **回测** | `backtest_over.py` | 每周三02:00 | `backtest_over_result.json` |
| **预测** | `predict_over.py` | 每天15:10 | `ml_predictions.jsonl` + 邮件 |
| **回顾** | `predict_over.py --review` | 每天12:50 | 回顾报告 + 邮件 |
| **补录** | `predict_over.py --backfill` | 手动执行 | `ml_predictions.jsonl`（标记backfill） |

---

## 六、关键设计原则

### 6.1 滚动前进（Walk-Forward）
- ✅ 模拟真实使用场景
- ✅ 避免未来数据泄漏
- ✅ 持续学习，适应数据分布变化

### 6.2 验收集独立
- ✅ 验收数据不参与任何调参
- ✅ 保证评估客观性
- ✅ 反映模型真实表现

### 6.3 SP值关联
- ✅ 回测和回顾都关联实际开奖SP值
- ✅ 计算真实盈亏
- ✅ 验证模型EV预测准确性

### 6.4 日志记录
- ✅ 所有预测记录写入jsonl文件
- ✅ 便于追溯和回顾
- ✅ 支持补录和去重

### 6.5 邮件通知
- ✅ 预测和回顾结果自动邮件发送
- ✅ 便于监控和及时响应
- ✅ 受config.py的ENABLE_EMAIL开关控制

---

## 七、常见问题

### Q1: 为什么回测要按月滚动而不是固定训练集？
A: 按月滚动模拟真实使用场景，模型持续学习最新数据，避免数据分布变化导致的性能下降。

### Q2: 验收集为什么不能参与调参？
A: 如果验收集参与调参，会导致过拟合，评估结果过于乐观，不能反映模型真实表现。

### Q3: 补录和常规预测有什么区别？
A: 补录用于历史日期，包含已完场比赛，不写数据库、不发邮件、不跑监控，仅用于回顾分析。

### Q4: 回顾报告中的"池内命中"和"投注命中"有什么区别？
A: 
- 池内命中：实际比分在9比分池内（反映模型筛选能力）
- 投注命中：实际比分在推荐的9个比分内（反映模型预测精度）

### Q5: 如何判断模型是否需要重新训练？
A: 
- 回测命中率持续下降（<训练时-10%）
- EV曲线整体下移
- Brier分数上升（>0.22）
- 回顾命中率连续3天低于预期

---

## 八、参考资料

- `backtest_over.py` - OVER模型回测脚本
- `backtest_under.py` - UNDER模型回测脚本
- `backtest_upside.py` - UPSIDE模型回测脚本
- `predict_over.py` - OVER模型预测脚本（含--review和--backfill）
- `predict_under.py` - UNDER模型预测脚本（含--review和--backfill）
- `train_over.py` - OVER模型训练脚本（含验收段评估）
- `train_under.py` - UNDER模型训练脚本（含验收段评估）

