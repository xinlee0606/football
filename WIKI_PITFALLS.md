# 数据时序与特征泄漏坑点（2026-11-18）

> 本文记录两个长期存在、会导致**离线指标虚高**的缺陷，以及如何识别它们。
> 两者都不是某次改动引入的，而是一开始就埋下的；修复前所有已发布的回测数字都应视为偏乐观。

---

## 坑 1：`match_id` 有两种长度，被推断成整数后时间排序完全错乱

### 现象

同一份特征缓存，用内存表和读 CSV 两种方式加载，回测得到的窗口数和结果不一样：

| 加载方式 | `match_id` dtype | 完场样本尾部 | 滚动窗口 |
|---|---|---|---|
| `--rebuild`（内存表，来自 MySQL varchar） | `str` | 2026111801 | 9 个（连续月份） |
| 读 `features_*_cache.csv` | `int64` | 20261108145 | 23 个（散乱跨全年） |

### 根因

期号 `match_id` 是 `YYYYMMDD` + 当日序号，序号可能占 2 位或 3 位：

```text
2026110801     ← 10 位（序号 < 100）
20261108145    ← 11 位（序号 ≥ 100）
```

库里 31170 行中 10 位占 27260 行、11 位占 3910 行。`pd.read_csv` 默认把它推断成 `int64`，于是：

1. **数值排序把所有 11 位 id 一律甩到所有 10 位 id 之后**（11 位数恒大于 10 位数），时间序被打乱；
2. `backtest_under.py` / `backtest_over.py` 恰好是**先 `sort_values('match_id')` 再 `astype(str)`**，排序用的是错的整数序；
3. `run_walkforward` 用 `df_hist = d[d['match_id'] < win['match_id'].min()]` 取历史集 —— 字符串与整数混比、且时间序已乱，导致**历史集里混进了未来的场次**。

后果是滚动前进回测的"测试段"不再是最后 8 个月，而是散落在整个时间轴上的样本，其中大量场次在训练段已经出现过。

### 正确做法

按**字符串**读入，让日期前缀决定顺序：

```python
# features_under.py / features_over.py 的 load_cache_*()
pd.read_csv(CACHE_FILE, dtype={'match_id': str})
```

`features_upside.py` 从一开始就是这么写的，OVER/UNDER 是漏了。补齐后：

| | 修复前 | 修复后 |
|---|---|---|
| 滚动窗口 | 23 个，202411~202611 散乱 | **9 个，202603~202611 连续** |
| 完场样本尾部 | 20261108 | **20261117** |
| UNDER 回测 Brier | 0.1812 | **0.1950** |
| UNDER 回测 Top3 | 37.3% | **35.0%** |

> 顺带：`train_over.py` 里从来没有 `astype(str)`，所以 OVER 的**训练**时间三段切分同样是错的（不只是回测）；`train_under.py` 是先 `astype(str)` 再排序，切分本身正确。

### 如何自查

```python
d = pd.read_csv('features_under_cache.csv')
print(d['match_id'].dtype)                        # 出现 int64 就有风险
print(d['match_id'].astype(str).str.len().value_counts())   # 长度不唯一就要显式 dtype=str
```

字符串序是否等于日期序，可用同一个断言验证：`sorted(unique_ids) == sorted(unique_ids, key=lambda x: x[:8])`。

---

## 坑 2：历史类特征按「索引列表末尾」取值 = 未来函数 + train/serve skew

### 现象

`features_under.py` 里球队近期战绩原本这样取：

```python
home_recent_data = home_team_matches.get(home, [])[-10:]     # ← 全库最后10场
```

`home_team_matches[home]` 是**整支球队的全部已完场历史**，`[-10:]` 取的是**建表时刻的最后 10 场**，与"当前这一场比赛"处于什么日期无关。

实证（伯明翰 / 斯托克港 等队，主场 50 场横跨近两年）：

| 特征 | 修复前取值个数 | 修复后 |
|---|---|---|
| `h_recent_win_rate` | **1**（恒 0.5） | 12 ~ 33 |
| `h_recent_goals_for_mean` | **1**（恒 2.1） | 随时间变化 |
| `h_recent_n` 按月均值 | 恒 10 | **1.04 → 9.6** 逐月爬升 |
| `league_n` 按月均值 | 恒为该联赛总量 | **9.5 → 500** 逐月累积 |

### 三处同类错误

| 位置 | 原实现 | 问题 |
|---|---|---|
| `_recent_form_features_upside_fast` | `team_matches[team][-10:]` | 取全库最后 10 场，含未来；且**包含本场自身** |
| `_league_stats_features_upside_fast` | 传入整个 `league_matches[league]` | 全量联赛历史，含未来；且包含本场 |
| `_h2h_features_upside_fast` | `h2h_data[:10]` | 不但没截断，还取**最老的 10 次交锋**（语义上应为最近 10 次）；且 `head_to_head` 查询没 `SELECT match_id`，无日期可截 |

### 为什么线上没暴露 —— train/serve skew

`history_df` 只含**已完场**比赛。线上预测时今天的比赛还没完场，所以"列表最后 10 场"**恰好就是**真实最近 10 场 —— 行为是对的。错的只有**历史训练行**：它们用的是"截至建表时刻"的末尾 10 场。

于是模型学到的是「截至建表时刻的球队强度静态编码」，线上喂的是「真实近期状态」，两边口径不同。

> ⚠️ 这类缺陷最难发现的地方就在这：修完之后**离线指标会明显下降**（原本虚高），而线上表现应当改善。如果没预期这一点，很容易误判成"改坏了"。

### 修复方式

统一用 `match_id` 前 8 位推导日期序数，配合 `bisect` 做严格截断：

```python
def _history_before(data_list, ord_list, cur_ord, tail_n=None):
    """取严格早于当前比赛日期的场次; tail_n 不为空时再取其中最近 tail_n 场"""
    if cur_ord is None or not ord_list:
        return []
    pos = bisect.bisect_left(ord_list, cur_ord)     # 同日的其他场次也排除, 保证口径唯一
    if pos == 0:
        return []
    return data_list[:pos] if tail_n is None else data_list[max(0, pos - tail_n):pos]
```

要点：

1. **日期来自 `match_id` 前缀，不能用 `match.match_time`** —— 后者是 `varchar(50)`，只存 `'08:30'` 这类时刻，没有日期。
2. 每个历史索引都要配一条**逐条对齐**的日期序数列表（`home_team_ords` / `league_ords` / `h2h_ords`），否则 bisect 出来的位置对不上数据。
3. 日期解析不出来的行**整行不入索引**（`continue`），而不是退化成"全部历史" —— 宁缺勿漏。实测库中期号 100% 合法，跳过数为 0。
4. 同日场次用 `bisect_left` 排除，保证"训练行"和"预测行"看到同一套口径。

---

## 附：本次踩到的运维/噪音坑

| 坑 | 说明 |
|---|---|
| `pkill -f 'xxx'` 自匹配 | 远程命令字符串本身含 `xxx`，`bash -c` 的 cmdline 会被 pkill 匹配到，**把自己的会话杀掉**，表现为命令无输出。用 `pkill -f '[_]xxx'` 括号技巧规避。 |
| paramiko 启动后台任务卡住 | `exec_command` 后 `o.read()` 会一直阻塞，因为 stdout 通道被子进程持有。要么重定向所有 fd 并 `setsid`，要么开裸通道 `sleep` 几秒后直接 `chan.close()`，不等退出码。 |
| `⚠ 模型期望180个特征，数据框只有90个` | **不是 bug**，是 `predict_rows` 的兼容分支噪音：它先取 `model['feats']` 与 df 列的交集（此时 90 个 `*_missing` 指示器还没生成，于是报"缺少90个"），随后自己补齐指示器再预测，列序与训练一致。 |
| SSH 主机密钥变更 | 服务器重装/换实例后 `known_hosts` 会报 `REMOTE HOST IDENTIFICATION HAS CHANGED`。先确认原因（`uptime` 能看出机器是否刚重建过），再决定更新指纹；不要习惯性绕过校验。 |
| 修改特征构建逻辑后 | 必须删除 `features_*_cache.csv` 强制重建，否则读的还是旧缓存，改动完全不生效。 |
