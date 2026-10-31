# 新球体育北单比赛数据爬虫 - 项目WIKI

## 项目简介

本项目是一个针对 [titan007.com](https://live.titan007.com) (新球体育) 的体育赛事数据爬虫系统，专门用于爬取北京单场（北单）比赛的完整数据，包括比赛信息、联赛积分、对赛往绩、近期战绩、亚盘数据、大小球数据、赛季统计和技术统计等。

### 核心特性

- **Selenium + BeautifulSoup** 架构，支持动态JS渲染页面爬取
- **Edge浏览器** 自动化操作（可配置为无头模式）
- **8张数据表** 完整覆盖比赛各维度数据 + 开奖SP值
- **智能去重** 基于 `site_match_id` 防止同一场比赛重复爬取
- **数据清洗** 自动清理球队名称中的排名数字、"(中)"标记等
- **MySQL存储** 结构化数据入库，支持后续分析
- **OVER/UNDER/UPSIDE比分预测系统** LightGBM筛选门 + Dixon-Coles比分矩阵，三模型独立训练与预测（OVER:11比分池BTTS+总进球≤6, UNDER:9比分池完全对称, UPSIDE:两层模型）

---

## 项目结构

```
live/
├── config.py              # 配置文件（数据库连接、爬虫参数、目标URL、浏览器路径）
├── init_db.py             # 数据库初始化脚本（创建数据库和所有表）
├── scraper.py             # 主爬虫程序（完整爬取流程 + 比赛状态更新方法）
├── history_scraper.py     # 历史数据回填爬虫（Over完场页入口, 无技术统计, 断点续传）
├── update_match_status.py # 比赛状态/比分更新入口（数据源: Over_*.htm）
├── filter_matches.py      # 比赛筛选脚本（9条件组合筛选 + 联赛概率统计 + 六比分预测）
├── predict_scores.py      # 六比分预测模块（历史频率Top6优先, 定方向→定进球数→锁比分回退）
├── backtest.py            # 六比分命中率时间序回测框架（无未来函数, 方法A/B/C对比）
├── sp_scraper.py          # 北单开奖SP值回填爬虫（数据源: 500彩票网, result_sp表）
├── features_over.py            # OVER特征工程（72特征，无技术统计，缓存features_over_cache.csv）
├── features_under.py           # UNDER特征工程（72特征，无技术统计，缓存features_under_cache.csv）
├── features_upside.py          # UPSIDE特征工程（74特征，缓存features_upside_cache.csv）
├── probe_data.py          # ML数据摸底（9池基线命中率/比分分布/覆盖率, 产出baseline_report.json）
├── train_over.py          # OVER模型训练（LightGBM+Dixon-Coles混合, 按EV选阈值, 模型存models/）
├── train_under.py         # UNDER模型训练（同OVER架构）
├── train_upside.py        # UPSIDE模型训练（同OVER架构）
├── backtest_over.py         # OVER滚动前进回测（前70%起步按月扩窗重训, 产出backtest_over_result.json）
├── backtest_under.py        # UNDER滚动前进回测
├── backtest_upside.py       # UPSIDE滚动前进回测
├── predict_over.py          # OVER每日预测+回顾（9比分概率+2:0/0:2分析+投注纪律+监控告警+--review回看）
├── predict_under.py         # UNDER每日预测+回顾
├── predict_upside.py        # UPSIDE每日预测+回顾
├── models/                # 三模型产物（OVER/UNDER/UPSIDE，带日期tag + latest.json指针）
├── verify_match.py        # 单场比赛数据验证脚本
├── verify_all.py          # 全量数据验证脚本
├── check_match.py         # 比赛数据检查工具
├── run_daily.sh           # 每日爬虫定时任务（Linux版）
├── run_history.sh         # 历史回填后台长跑脚本（Linux版, nohup+日志）
├── setup_linux_cron.sh    # Linux服务器一键部署定时任务（含ML预测/重训任务）
├── crontab_linux.txt      # Linux crontab 定时任务配置清单（7个任务，含ML预测/重训/回顾）
├── requirements.txt       # Python依赖清单（爬虫 + ML依赖）
└── .venv/                 # Python虚拟环境
```

---

## 环境要求

| 组件 | 版本要求 |
|------|----------|
| Python | 3.8+ |
| MySQL | 5.7+ / 8.0+ |
| 浏览器 | Windows: Microsoft Edge/Chrome；Linux: Chromium + chromedriver |

> Linux服务器浏览器安装（CentOS 8 需先切换vault源）：`sudo dnf install -y chromium chromium-driver`。
> 代码会自动检测浏览器与驱动路径，也可在 `config.py` 的 `CHROME_BINARY_PATH` 显式指定。

### 依赖安装

```bash
pip install -r requirements.txt
```

包含两部分依赖：
- **爬虫**: selenium、pymysql、beautifulsoup4、lxml、requests
- **ML预测系统**: numpy、pandas、scikit-learn、lightgbm、joblib

---

## 配置说明

编辑 `config.py`：

```python
# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '你的MySQL密码',    # ← 修改为实际密码
    'database': 'match_live',
    'charset': 'utf8mb4'
}

# 爬虫配置
CHROME_HEADLESS = False          # False=显示浏览器窗口, True=无头模式
CHROME_BINARY_PATH = ''          # 浏览器可执行文件路径(空=自动检测, Linux服务器用)
PAGE_LOAD_TIMEOUT = 30           # 页面加载超时(秒)
ELEMENT_WAIT_TIMEOUT = 15        # 元素等待超时(秒)
REQUEST_DELAY = 2                # 请求间隔(秒), 避免被封
MATCH_LIST_WAIT = 10             # 等待比赛列表加载时间(秒)

# 目标URL
MAIN_URL = 'https://live.titan007.com/oldIndexall.aspx'
OVER_URL_TEMPLATE = 'https://bf.titan007.com/football/Over_{date}.htm'  # 完场比分页面模板（{date}=YYYYMMDD）
```

---

## 快速开始

### 1. 初始化数据库

```bash
python init_db.py
```

这会自动创建 `match_live` 数据库和 8 张数据表。

### 2. 运行爬虫

```bash
# 爬取所有比赛
python scraper.py

# 调试模式：限制爬取N场比赛（快速验证流程）
python scraper.py 1    # 仅爬取1场
python scraper.py 3    # 仅爬取3场
```

爬虫会自动执行以下流程：
1. 打开Edge浏览器
2. 访问赛事列表页面
3. 筛选"单场"比赛
4. 解析比赛列表
5. 逐场爬取详细数据（含技术统计）
6. 保存到MySQL数据库

### 3. 验证数据

```bash
python verify_match.py   # 验证单场比赛所有关联表数据
```

---

## 数据库设计

### 表结构概览

| 表名 | 说明 | 关键字段 |
|------|------|----------|
| `match` | 比赛基本信息 | match_id, site_match_id, league, home_team, away_team, score |
| `league_standings` | 联赛积分排名 | match_id, team_type, scope(总/主/客/近6), points, ranking |
| `head_to_head` | 对赛往绩 | match_id, match_date, home_team, score, away_team, 盘口数据 |
| `recent_form` | 近期战绩 + 技术统计 | match_id, team_type, is_current_match(0=历史/1=技术统计) |
| `asian_odds` | 亚盘数据 | match_id, company, 初盘/即时 水位盘口 |
| `over_under_odds` | 大小球数据 | match_id, company, 初盘/即时 大小球线水位 |
| `season_stats` | 赛季数据统计 | match_id, team_type, section, scope, stat_name, stat_value |
| `result_sp` | 北单开奖SP值（500彩票网） | issue_no, match_no, score, sp_value, match_id(回填) |

### 关键设计说明

#### match 表
- `match_id`: 系统生成的比赛编号，格式为 `日期+序号`，如 `2026072301`
- `site_match_id`: 网站原始比赛ID（从页面行ID提取，如 `tr1_3027598` → `3027598`），设有唯一索引，用于去重
- `page_handicap` / `page_goals`: 完场页（Over_*.htm）自带的亚盘（*号已转受让）与进球数线快照，仅历史回填时写入
- `score`: 完场比分，数据库实际存储为**横杠格式**（如 `1-0`，来自完场页抓取）；ML相关脚本（`features_upside.py` 等）已兼容 `1-0` 与 `1:0` 两种写法

#### recent_form 表（近期战绩 + 技术统计）
- `is_current_match=0`: 历史战绩记录（每支球队近10场比赛，主客各10场共20条）
- `is_current_match=1`: 当前比赛技术统计（角球、射门、控球率等）
- `team_name`: 该记录所属的球队名称（关注球队）
- `detail_url`: 该场历史比赛对应的现场分析页面URL
- 技术统计字段：corners, shots, shots_on_target, attacks, dangerous_attacks 等16项

#### result_sp 表（开奖SP值）
- 由 `sp_scraper.py` 从500彩票网开奖页回填，唯一键 `issue_no + match_no`
- `sp_value`: 比分玩法开奖SP值，用于ML系统的EV（期望收益）计算与回测盈亏统计
- `match_id`: 通过北单场次号/队名模糊匹配后回填，关联到 `match` 表

---

## 爬取流程详解

### 第一步：比赛列表爬取
- 访问 `https://live.titan007.com/oldIndexall.aspx`
- 自动点击筛选按钮，选择"单场"比赛
- 解析 `table_live` 表格，提取每场比赛的基本信息
- 通过 `site_match_id` 去重，已存在的比赛自动跳过

### 第二步：分析页面爬取
对每场新比赛，访问分析页面（如 `https://zq.titan007.com/analysis/3035507cn.htm`），依次爬取：

1. **联赛积分排名** - 从 `h2.fx_title2` 和后续表格解析（主客各4条：总/主/客/近6）
2. **对赛往绩** - 从 `h2#003` 历史对赛表格解析（含角球、36*初盘数据）
3. **近期战绩** - 从 `table_hn`(主队) 和 `table_an`(客队) 解析，同时提取比分链接URL
4. **近期战绩技术统计** - 依次访问每场近期战绩的现场分析页面，抓取关注球队技术统计
5. **赛季数据统计** - 从 `h2#017` 区域解析（战绩/得失球/净胜球）
6. **亚盘数据** - 点击"亚让"链接跳转到亚盘专页爬取（约13-16家机构）
7. **大小球数据** - 点击"进球数"链接跳转到大球专页爬取（约16家机构）

### 第三步：近期战绩技术统计爬取（核心功能）

对近期战绩中的每场比赛（主客各10场，共20场），执行以下流程：

1. 从比分单元格提取现场分析页面URL（`https://live.titan007.com/detail/{match_id}cn.htm`）
2. 访问该页面，定位技术统计区域（`div#teamTechDiv_detail > ul#teamTechDiv > li.lists`）
3. 使用**模糊球队名匹配**确定关注球队是主队还是客队
4. 提取关注球队的技术统计数据并UPDATE到对应的 `recent_form` 记录

**抓取的技术统计项（16项）：**

| 统计项 | 字段名 | 统计项 | 字段名 |
|--------|--------|--------|--------|
| 角球 | corners | 射门不中 | shots_off_target |
| 半场角球 | half_corners | 射门被挡 | shots_blocked |
| 射门 | shots | 任意球 | free_kicks |
| 射正 | shots_on_target | 控球率 | possession |
| 进攻 | attacks | 半场控球率 | half_possession |
| 危险进攻 | dangerous_attacks | 传球 | passes |
| 黄牌 | yellow_cards | 传球成功率 | pass_success_rate |
| 犯规 | fouls | 越位 | offsides |

**模糊球队名匹配策略：**
- 同一球队在不同页面可能使用不同汉字（如"索菲亚" vs "索非亚"）
- 匹配规则：子串包含 或 字符重叠度 ≥ 60%

---

## 比赛状态更新

### 用途
每天定时执行，从完场比分页面获取北单比赛的最新**状态**和**比分**，更新到 `match` 表。
- 仅更新已存在的比赛（按 `single_match` 匹配），**不爬取详情数据、不插入新比赛**。

### 数据源
- 完场比分页面：`https://bf.titan007.com/football/Over_YYYYMMDD.htm`（由 `config.py` 的 `OVER_URL_TEMPLATE` 配置）
- 北单过滤：页面 JS 变量 `beiDanSchedule = "sid1,sid2,...|单场A,单场B,..."`，sid 列表与单场编号按位置一一对应，等价于在页面上点击“选择 → 北单”。

### 匹配与更新逻辑
1. 解析 `beiDanSchedule` 建立 `sid → 单场编号` 映射
2. 解析比赛表格 `table_live`，每个 `tr1_` 行含 `sid` 属性，建立 `sid → 行` 映射
3. 对每场北单比赛，按 `single_match`（规范化去前导零，如页面`单场093`→数据库`单场93`）+ 日期前缀（`match_id LIKE 'YYYYMMDD%'`）匹配数据库
4. 仅更新 `status`（状态）和 `score`（比分）两个字段

### 用法
```bash
python update_match_status.py            # 默认更新昨天的比赛
python update_match_status.py 20260726   # 指定日期更新
```

### 定时部署
通过 crontab 每天中午执行（见 `crontab_linux.txt` 任务1+2）：
```
0 12 * * * /opt/live/run_daily.sh >> /opt/live/logs/daily.log 2>&1
```

---

## 历史数据回填（history_scraper.py）

### 用途
从完场页批量回填近1年北单比赛历史数据，供筛选规则回测与比分预测概率统计使用。

### 数据源与流程
- 入口：`https://bf.titan007.com/football/Over_YYYYMMDD.htm`（完场比分页），通过页面 JS 变量 `beiDanSchedule` 过滤出**仅北单比赛**
- 列表字段：联赛/时间/状态/主客队/比分/半场，另将页面"亚盘/进球数"两列存入 match 表 `page_handicap`（*号转受让）/ `page_goals`
- 详情流程与每日爬虫一致：分析页（联赛积分/对赛往绩/近期战绩）+ 亚盘专页 + 大小球专页
- **差异：不爬技术统计**（跳过 season_stats 和近期战绩逐场详情页，通过子类空实现覆盖父类方法）

### 去重与断点续传
- match 表按 `site_match_id` 去重，已存在比赛复用原 match_id；新比赛按"日期+当天顺序序号"从已有最大序号+1续编
- 详情阶段：`asian_odds` 或 `recent_form` 有该场记录即视为已爬，自动跳过 —— 中断后重跑同一命令即可续传
- 长跑保障：浏览器会话失效（invalid session id）自动重建并重试

### 用法
```bash
python -u history_scraper.py                              # 回填最近365天
python -u history_scraper.py 3                            # 回填最近3天
python -u history_scraper.py 20250806 20260806            # 指定日期区间(含首尾)
python -u history_scraper.py 20250806 20250806 --limit 2  # 测试: 每天最多2场详情
```

### Linux服务器后台长跑（run_history.sh）
```bash
chmod +x run_history.sh
./run_history.sh 20250806 20260806    # nohup后台启动, 日志写入 logs/history.log
tail -f logs/history.log              # 实时查看进度
pkill -f history_scraper.py           # 停止任务
```
脚本内置：防重复启动、UTF-8编码保护（防中文输出在非UTF-8 locale下崩溃）。

### 数据量与耗时参考
- 北单场次：周末 100~160 场/天，工作日 20~40 场/天，全年约 **3.5~4万场**
- 每场详情约 20 秒，全量约需 **200+小时**，建议服务器 nohup 长跑 + 断点续传分批完成

---

## 比赛筛选

### 用途
从数据库中筛选符合多个条件的比赛，每个条件可通过开关自由添加或组合。

### 用法
```bash
python filter_matches.py
```
通过 `MATCH_ID_PATTERN` 控制筛选范围（支持 `*` 通配符，如 `'20260727*'`；留空 `''` 表示不限制）。

### 筛选条件（9个，可自由组合）

| 开关 | 条件 | 判定标准 | 关键参数 |
|------|------|----------|----------|
| `ENABLE_COND1` | 亚盘机构数 | 初盘+即时盘归一化后均为目标盘口的机构数 > 阈值 | 目标盘口={受让平手/半球, 平手/半球, 平手}，阈值=10 |
| `ENABLE_COND2` | 主客积分差 | 主队主场积分 与 客队客场积分 分差 ≤ 阈值 | 阈值=9 |
| `ENABLE_COND3` | 近6场积分差 | 两队近6场积分分差 ≤ 阈值 | 阈值=6（默认关闭） |
| `ENABLE_COND4` | 对赛往绩胜场 | 近N场交锋主客胜场差 ≤ 阈值 | N=6，阈值=2（默认关闭） |
| `ENABLE_COND5` | 技术统计差异 | 近N场5项技术统计每项差异率 ≤ 阈值 | N=3，阈值=30%（默认关闭） |
| `ENABLE_COND6` | 正常比分概率 | 近N场目标盘口下正常比分合并概率 ≥ 阈值 | N=10，阈值=60% |
| `ENABLE_COND7` | 射正率+危险进攻 | 两队近6场平均射正率、危险进攻各自达标 | 射正率各>30%，危险进攻各>30 |
| `ENABLE_COND8` | 联赛黑名单 | 联赛不在回测低命中率黑名单内 | `COND8_LEAGUE_BLACKLIST`（12个联赛） |
| `ENABLE_COND9` | 覆盖概率 | 盘口档位历史频率Top6覆盖概率 ≥ 阈值 | 阈值=50% |

> 正常比分：`1-0 / 2-1 / 3-2 / 0-0 / 1-1 / 2-2 / 0-1 / 1-2 / 3-1`

#### 条件1盘口归一化
条件1的盘口匹配前先经 `_COND1_ALIASES` 别名表归一化，兼容 `受让平手/半球`/`受平手/半球`/`受让平/半`/`受平/半` 等变体写法。`asian_odds` 表实际存储的受让写法是 `受让平手/半球`；新写法只需在别名表加一行。

#### 条件8联赛黑名单（回测驱动）
黑名单来源于 `backtest.py` 回测：重点档位（平手/平半/受让平半）下历史频率Top6命中率 < 52% 且样本 ≥ 30场的联赛。当前12个：德甲、英超、法甲、荷甲、瑞士超、瑞典甲、奥乙、瑞士甲、捷甲、爱甲、法乙、苏冠。回填数据推进后重跑回测可刷新名单。

#### 条件9覆盖概率（置信度过滤）
覆盖概率 = 该比赛即时主线盘口档位下，历史完场比赛比分频率Top6的场次占比（即单场命中的经验概率）。回测显示：覆盖概率≥50%档命中率56.9%，40-50%档仅47.4%，故作为选场过滤条件。

### 附加信息：联赛级正常比分概率

开关 `ENABLE_LEAGUE_NORMAL_PROB`（默认开启）。对每场通过筛选的比赛，按其所在**联赛分组**，额外输出一行统计信息：

> 该联赛在数据库中**所有已完场**（`status='完'`，跨所有日期）且**满足条件1**的比赛里，赛果属于9种正常比分的概率。

- 概率 = 该联赛(已完场 ∩ 满足条件1 ∩ 正常比分)场数 / 该联赛(已完场 ∩ 满足条件1)场数
- 仅作为展示信息，**不参与筛选**；同一联赛的各场比赛显示相同概率值
- 样本量较少时（如仅1~2场）概率参考意义有限，随数据积累会更有价值

输出示例：
```
★ [2026072639] [单场138] 捷甲  赫拉德茨克拉洛韦 vs 帕尔杜比斯  (02:00)
    条件1-亚盘机构数: 12家机构(需>10)
    条件2-主客积分差: 主主场28分/客客场21分, 差7(需≤9)
    条件6-正常比分概率: 目标盘口12场, 正常比分9场, 概率75%(需≥60%)
    联赛正常比分概率: 捷甲 已完场且满足条件1共2场, 正常比分2场, 概率100%
```

---

## 六比分预测（predict_scores.py）

### 用途
为通过筛选的比赛预测6个比分，用于北单比分玩法 2串1 投注。由 `filter_matches.py` 的 `ENABLE_SCORE_PREDICT` 开关控制输出。

### 预测逻辑（两级）

**第一级：历史频率Top6（默认, `ENABLE_FREQ_POOL=True`）**
1. 取该场 `asian_odds` 即时盘机构众数主线（如 +0.25）→ 映射盘口档位（平手/平半/受让平半/半球等9档）
2. 统计 `match` 表该档位（`page_handicap`）所有已完场比赛的比分频次，取Top6直接作为预测
3. 覆盖概率 = Top6场次占比；同档位样本 < 50场时回退全体样本
4. 输出理由行：`历史频率Top6 (平/半档1354场, 覆盖概率62%) [即时主线+0.25 -> 平/半]`

**第二级：三步推演回退**（无即时盘口或档位不在映射内, 如±1.25以上深盘）
1. 定方向：预测结果+即时亚盘决定方向优先级（均势盘重防平局）
2. 定进球数：大小球主线≤2.5小球、≥3大球、中间线用两队近10场场均总进球裁决
3. 锁比分：按方向从比分池按3/2/1名额抽取，小球模式优先总进球≤2

### 独立调试
```bash
python predict_scores.py 2026080903   # 查看单场比赛预测6比分及理由
```

---

## 命中率回测（backtest.py）

### 用途
无未来函数的时间序回测框架，量化评估六比分预测策略的命中率（暂不涉及SP收益），指导选场条件与比分池的调优。

### 回测机制
- 样本：`match` 表 status='完' 且有比分+完场页盘口（`page_handicap`）的比赛，按 match_id 时间序逐场回放
- 无未来函数：频率表边回放边累积，每场预测只用该场之前的完场数据；样本不足自动回退更大分组
- 三方法对比：
  - 方法A 固定比分池（复刻旧版三步推演逻辑）
  - 方法B 历史频率Top6按盘口档位分组
  - 方法C 盘口档位×大小球分档双维分组
- 输出维度：总体/重点档位（平手、平/半、受让平/半）/按盘口/按联赛（≥30场）命中率 + 覆盖概率分档

### 用法
```bash
python -u backtest.py                        # 全量回测
python -u backtest.py 20250101 20250331      # 指定日期范围
python -u backtest.py --detail               # 导出逐场明细 backtest_detail.csv
```

### 关键结论（2025-01~05, 6763场）
| 方法 | 总体命中率 | 重点档位命中率 |
|------|-----------|---------------|
| A 固定比分池 | 51.3% | 54.1% |
| B 频率Top6（单盘口档） | 55.3% | 59.1% |
| C 盘口×大小球细分 | 54.3% | 57.8% |

- 频率法显著优于固定池；双维细分反而因样本被切碎而下降，故预测维持单盘口档位分组
- 联赛差异巨大（如阿甲73% vs 德甲47%）→ 产出联赛黑名单（条件8）
- 覆盖概率≥50%档命中率56.9% vs 40-50%档47.4% → 产出条件9阈值

---

## 开奖SP回填（sp_scraper.py）

### 用途
从500彩票网北单开奖页回填比分玩法开奖SP值到 `result_sp` 表，为ML系统提供EV（期望收益）计算与回测盈亏统计依据。
- 数据页：`https://zx.500.com/zqdc/kaijiang.php?playid=3&expect={期号}`，服务端渲染，requests+BeautifulSoup 直接抓取，无需Selenium
- 期号格式 `YYMMN`：2位年份+2位月份+1位月内期序（如 `25041`=2025年4月第1期）
- 断点续传：期号+场次唯一键去重，已有数据且SP一致则整期跳过
- 已集成到每日流程：`update_match_status.py` 每日串联调用 `update_by_date()` 增量更新并回填 `match_id`

### 匹配策略（--match）
两级匹配把 `result_sp` 记录关联到 `match` 表：
1. 北单场次号（`single_match`）与 match 表精确对应（首选）
2. 队名模糊+比分一致（兜底，日期±1天窗口）

### 用法
```bash
python -u sp_scraper.py                       # 默认爬取近365天
python -u sp_scraper.py 25041 26077           # 指定期号区间(含首尾)
python -u sp_scraper.py 20250401 20250430     # 指定日期区间(按期过滤)
python -u sp_scraper.py --match               # 只做match_id匹配(不爬取)
```

---

## ML比分预测系统（UNDER 9比分池）

### 用途
用约2.5万场北单历史数据训练混合机器学习模型，筛选最终比分落在 **9比分池** 中的比赛并输出各比分概率，以期望收益（EV）为优化目标。与六比分预测（规则法）并行，是独立的 ML 方案。

> 9比分池 `UNDER`：`1:0 / 2:0 / 2:1 / 0:0 / 1:1 / 2:2 / 0:1 / 0:2 / 1:2`（主胜低比分 + 平局 + 客胜低比分，对称覆盖）
> 11比分池 `OVER`：`2:1 / 3:1 / 3:2 / 1:1 / 2:2 / 3:3 / 1:2 / 1:3 / 2:3 / 4:2 / 2:4`（双方进球BTTS + 总进球≤6）
> 定义分别在 `features_under.py` 和 `features_over.py`，换池只需改常量并重跑基线/回测/训练。

### 模型架构（两阶段混合）
| 组件 | 模型 | 作用 |
|------|------|------|
| 筛选门 | LightGBM 二分类 | 预测比赛落在9池内的概率 P_lgb |
| 进球期望 | LightGBM 泊松回归×2 | 分别预测主/客队期望进球 λ |
| 比分矩阵 | Dixon-Coles 双变量泊松 | 低比分相关修正（tau因子, rho网格拟合），产出全比分概率矩阵 |
| 融合 | α加权 | P = α·P_lgb + (1-α)·P_dc，α在验证集按Brier最小选取 |
| 校准 | Isotonic回归 | 融合概率校准为真实命中率 |
| 阈值 | 按EV选取 | 每场6注各1元：命中=SP-6，未中=-6，选EV最优的P_pool阈值 |

### 特征工程（72个特征，无未来函数）

#### 提取流程
1. **`load_raw(cursor)`**: 一次性全量查询 6 张源表（`match`/`asian_odds`/`over_under_odds`/`recent_form`/`head_to_head`/`league_standings`/`result_sp`），按 `match_id` 分组为 dict，避免逐场查库
2. **逐场构建**: 遍历 `match` 表每场比赛，依次调用各分组特征函数，合并为一行
3. **联赛先验**: 按 `match_id` 升序（天然时间序）累积该联赛之前的池内命中数，平滑后写入 `league_pool_prior`（只用该场之前的完场数据，无未来函数）
4. **缓存**: 产物写入 `features_upside_cache.csv`；后续调用 `load_or_build(rebuild=False)` 优先读缓存，`rebuild=True` 或缓存不存在时重跑

#### 数据源表
| 源表 | 特征函数 | 说明 |
|------|----------|------|
| `asian_odds` | `asian_features()` | 亚盘初盘+终盘盘口/水位 |
| `over_under_odds` | `over_under_features()` | 大小球初盘+终盘线/赔率 |
| `european_odds` | `euro_features()` | 欧赔胜/平/负初盘+即时 |
| `head_to_head` | `_h2h_features_fast()` | 对赛往绩（含半场比分、角球） |
| `match` | `_recent_form_features_fast()` / `_league_stats_features_fast()` | 近期战绩 + 联赛统计 | |

#### 全部62个特征清单

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

**② 大小球（8个）** — `over_under_features()`
| 特征 | 含义 |
|------|------|
| `ou_n` | 机构数 |
| `ou_line_init_mean` | 初盘大小球线均值 |
| `ou_line_init_std` | 初盘大小球线分歧度 |
| `ou_line_cur_mean` | 终盘大小球线均值 |
| `ou_line_move` | 大小球线变化（终盘 − 初盘） |
| `over_prob_init` | 初盘大球隐含概率（小水/(大水+小水) 去水归一） |
| `over_prob_cur` | 终盘大球隐含概率 |
| `exp_total` | 期望总进球（取终盘线，无则回退初盘线） |

**④ 期望进球先验（3个）** — 由亚盘+大小球推导

**③ 欧赔（11个）** — `euro_features()`
| 特征 | 含义 |
|------|------|
| `home_adv` | 主队让球优势（= `ah_cur_mean`，正=主队占优） |
| `exp_home` | 主队期望进球（总进球/2 + 让球×0.55） |
| `exp_away` | 客队期望进球（总进球/2 − 让球×0.55） |

**⑤ 近10场战绩（20个）** — `_recent_form_features_fast()` 双队各10个
主队10个（`recent_` 前缀）: `recent_n`（场数）、`recent_win_rate`（胜率）、`recent_draw_rate`（平率）、`recent_loss_rate`（负率）、`recent_goals_for_mean`（场均进球）、`recent_goals_against_mean`（场均失球）、`recent_clean_sheet_rate`（零封率）、`recent_btts_rate`（双方进球率）、`recent_over_25_rate`（大于2.5球率）、`recent_unbeaten_rate`（不败率）

客队10个（同上，通过 `is_home=False` 参数获取）

**⑥ 对战往绩（11个）** — `_h2h_features_fast()`
| 特征 | 含义 |
|------|------|
| `h2h_n` | 往绩场数 |
| `h2h_home_win_rate` / `h2h_draw_rate` / `h2h_away_win_rate` | 往绩主/平/客胜率 |
| `h2h_avg_goals` | 往绩场均总进球 |
| `h2h_home_goals_mean` / `h2h_away_goals_mean` | 往绩主/客队场均进球 |
| `h2h_btts_rate` | 往绩双方进球率 |
| `h2h_over_25_rate` | 往绩大于2.5球率 |
| `h2h_half_goal_avg` | 往绩场均半场进球 |
| `h2h_corners_avg` | 往绩场均总角球 |

**⑦ 联赛统计（8个）** — `_league_stats_features_fast()`
| 特征 | 含义 |
|------|------|
| `league_n` | 联赛场数 |
| `league_home_win_rate` / `league_draw_rate` / `league_away_win_rate` | 联赛主/平/客胜率 |
| `league_avg_goals` | 联赛场均总进球 |
| `league_btts_rate` | 联赛双方进球率 |
| `league_over_25_rate` | 联赛大于2.5球率 |
| `league_home_unbeaten_rate` | 联赛主队不败率 |

#### 防数据泄漏
- `feature_columns()` 显式排除标签/辅助列（`y`、`home_goals`、`away_goals`、`in_pool`、`score`、`sp` 等元信息列），`train_models()` 入口护栏发现特征混入标签列直接报错拒绝训练
- 比分格式兼容：`is_finished()` 同时识别 `1-0`（数据库实际存储，完场页抓取）与 `1:0` 两种写法；`norm_score()` 统一转为横杠格式
- 缓存到 `features_upside_cache.csv`，修改比分池/特征/标签后需删除缓存强制重建

### 四步使用流程（服务器执行，本机连不上阿里云RDS）
```bash
./venv/bin/python -u probe_data.py      # 1. 数据摸底: 9池基线命中率/比分分布/覆盖率 -> baseline_report.json
./venv/bin/python -u backtest_over.py     # 2. 滚动前进回测(前70%起步按月扩窗重训) -> backtest_over_result.json
./venv/bin/python -u train_model.py     # 3. 全量训练生成生产模型 -> models/ (带日期tag+latest.json)
./venv/bin/python -u predict_over.py      # 4. 每日预测
```
> 修改比分池、特征或标签后需删除 `features_upside_cache.csv` 强制重建缓存。
> 回测严格按时间序，每场预测只用其之前的数据，无未来函数。

### 每日预测（predict_over.py）
- 输出全部9个比分概率（降序），推荐场额外标注 **2:0/0:2 概率分析**（如"偏向2:0(主队净胜2球), 差X个百分点"）
- 概览表与推荐行均带 **单场编号**（`[单场XX]`）与 **开赛时间**（`(02:30)`），格式与 filter_matches.py 一致
- 投注纪律：P_pool ≥ 置信度阈值 + 排除联赛黑名单（复用 backtest.py 产出），**不限推荐场次数量**；阈值优先级：命令行 `--min-prob` > 模型按EV选出的阈值 > 兜底值0.60
- 预测方向（胜平/平负）写入 `match.prediction`，兼容现有下游格式；推荐记录追加到 `ml_predictions.jsonl`
- 预测结果邮件发送：推荐报告（含模型版本、每场9比分概率与2:0/0:2分析）邮件推送，受 `ENABLE_EMAIL` 开关控制（config.py 默认 True），`--skip-email` 可跳过
- 持续学习监控：滚动近100场已完场推荐的池内命中率，跌破基线85%邮件告警（受 `ENABLE_EMAIL` 开关控制）
- 持续学习策略：每周一 cron 全量重训，新完场数据自然累积进训练集，模型版本带日期可回滚
```bash
python -u predict_over.py                 # 预测当天比赛(含历史推荐监控)
python -u predict_over.py --date 20260818 # 指定日期
python -u predict_over.py --all-unfinished # 全部未开赛比赛(不限日期)
python -u predict_over.py --min-prob 0.65 # 手动抬高置信度阈值(只推高置信场次)
python -u predict_over.py --skip-email    # 跳过预测结果邮件发送
python -u predict_over.py --skip-monitor  # 跳过监控
```

### 预测回顾（predict_over.py --review）
- 用途：回看历史某天的预测与实际比分对照。完场比赛已不属于"未开赛"样本，重跑 `--date` 无法回看，故从 `ml_predictions.jsonl` 读取
- 数据源：`ml_predictions.jsonl`（按 match_id 去重、保留最后一次预测），再查 `match` 表取实际比分/对阵/单场编号
- 排序：按池内概率（置信度）由高到低
- 每行显示：编号、`[单场XX]`、是否推荐(★)、池内概率、对阵、实际比分、投注的6个比分、命中结果
- 命中两口径：**投注命中**=实际比分落在所选6个投注比分（真实盈亏口径）；**池内命中**=实际比分落在9比分池（基线口径）
- 汇总只统计"推荐且已完场"场次的命中率；回顾结果一并邮件发送（受 `ENABLE_EMAIL` 控制，`--skip-email` 可跳过），无记录时不发邮件
```bash
python -u predict_over.py --review          # 回看昨天(默认)
python -u predict_over.py --review 20260902 # 回看指定日期
```

### 定时部署（见 `crontab_linux.txt` 任务4/5/7）
```
30 18 * * * 每日18:30 predict_over.py          # 当日预测(在18:00筛选之后)
40 12 * * * 每日12:40 predict_over.py --review  # 回顾昨天预测命中(结果邮件发送)
0 10 * * 1  每周一10:00 train_model.py         # 全量重训(持续学习)
```

---

## 数据清洗规则

### 球队名称清理 (`clean_team_name`)

| 原始名称 | 清理后 | 规则 |
|----------|--------|------|
| `1济州SK` | `济州SK` | 去除前导排名数字 |
| `米阿尔比(中)` | `米阿尔比` | 去除"(中)"标记 |
| `迪弗当日2` | `迪弗当日` | 去除尾部数字 |
| `克鲁比04` | `克鲁比04` | 队名固有数字保留 |

---

## 去重机制

### 比赛去重
- 使用 `site_match_id`（网站原始ID）作为唯一标识
- `save_match()` 方法先查询是否已存在相同 `site_match_id`
- 已存在的比赛会更新基本信息，但不会重复爬取详细数据
- 即使不同日期运行爬虫，同一场比赛也会复用原有记录

### 技术统计去重
- 近期战绩记录采用"先删后插"策略（`DELETE WHERE match_id AND is_current_match=0`）
- 技术统计随近期战绩一起重新爬取，确保数据最新
- `is_current_match=1` 的记录通过 `save_tech_stats_to_recent_form()` 检查去重

---

## 常见问题

### Q: 浏览器窗口自动打开但没有操作？
A: 可能是页面加载超时，检查网络连接。可以调整 `config.py` 中的 `PAGE_LOAD_TIMEOUT`。

### Q: 提示找不到比赛表格？
A: 网站页面结构可能已变化，需要更新 `scraper.py` 中的解析逻辑。

### Q: 如何只爬取特定联赛？
A: 修改 `scrape_match_list()` 中的筛选逻辑，或在数据库层面过滤。

### Q: 技术统计数据为NULL？
A: 可能原因：1) 该场近期战绩的比分单元格无链接（detail_url为空）；2) 现场分析页面无技术统计区域（比赛太旧或未收录）。已完赛的比赛通常都有技术统计。

### Q: 如何定时运行？
A: 可以使用 Windows 任务计划程序或 cron 定时执行 `python scraper.py`。

---

## 技术栈

- **Python 3.8+**
- **Selenium** - 浏览器自动化，处理JS动态渲染页面（自动检测浏览器与chromedriver路径）
- **BeautifulSoup4** - HTML解析和数据提取（含500彩票网开奖页）
- **PyMySQL** - MySQL数据库连接
- **ML预测** - numpy / pandas / scikit-learn / LightGBM / joblib（特征工程、模型训练与存取）
- **浏览器引擎** - Windows: Edge/Chrome；Linux服务器: Chromium + chromedriver（无头模式）

---

## 数据量参考

### 单场比赛数据量（实测）

| 表 | 记录数 | 说明 |
|----|--------|------|
| match | 1 | 比赛基本信息 |
| league_standings | 8 | 主客各4条（总/主/客/近6） |
| head_to_head | 3~10 | 历史交锋记录 |
| recent_form (历史) | 20 | 主客各10场，含技术统计 |
| asian_odds | 13~16 | 各博彩公司亚盘数据 |
| over_under_odds | 16 | 各博彩公司大小球数据 |
| season_stats | 46 | 赛季战绩/得失球/净胜球 |

### 批量爬取数据量（约39场单场比赛）

| 表 | 记录数 |
|----|--------|
| match | ~39 |
| league_standings | ~312 |
| head_to_head | ~200+ |
| recent_form (历史+技术统计) | ~780 |
| asian_odds | ~500+ |
| over_under_odds | ~600+ |
| season_stats | ~1800 |
