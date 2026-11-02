# ML模型训练与回测指南（OVER/UNDER/UPSIDE）

## 三模型架构概览

| 模型 | 脚本 | 比分池 | 用途 |
|------|------|--------|------|
| **OVER** | `train_over.py` | 11比分池（大球向） | 预测总进球≥3球的比赛 |
| **UNDER** | `train_under.py` | 9比分池（小球向） | 预测总进球≤2球的比赛 |
| **UPSIDE** | `train_upside.py` | 两层模型（8+9池） | 预测上盘（胜平）方向 |

---

## 训练流程（推荐方法）

### 前置准备

```bash
# 1. 清理旧缓存释放内存（必须！避免OOM）
rm -f /coding/live/features_*_cache.csv

# 2. 检查当前进程状态
ps aux | grep train | grep -v grep
```

### 训练执行（逐个执行，禁止并发）

```bash
cd /coding/live

# OVER 模型训练
setsid bash train_over_only.sh > logs/train_over.log 2>&1 &
tail -f logs/train_over.log

# UNDER 模型训练（等待OVER完成后）
setsid bash train_under_only.sh > logs/train_under.log 2>&1 &
tail -f logs/train_under.log

# UPSIDE 模型训练（等待UNDER完成后）
setsid bash train_upside_only.sh > logs/train_upside.log 2>&1 &
tail -f logs/train_upside.log
```

> **重要**：
> - 使用 `setsid` 确保SSH断开后进程继续运行
> - 日志输出到文件，避免终端依赖
> - 每个模型训练时间：3-5小时（实际可能更快）
> - 必须逐个训练，不能同时运行多个模型

### 进度监控

```bash
# 检查是否有进程在运行
ps aux | grep train | grep -v grep

# 查看最新日志
tail -n 50 /coding/live/logs/train_over.log

# 检查是否被OOM Kill
dmesg | grep -i oom | tail -5

# 检查内存使用
free -m
```

### 验证新特征生效

- **缓存文件大小增加**: 
  - OVER/UNDER: ~25MB → ~30MB
  - UPSIDE: ~40MB → ~45MB
- **训练日志显示更多特征列**
- **模型重要性分析**（训练完成后）

---

## 回测流程（推荐方法）

### 回测脚本

| 脚本 | 用途 | 输出文件 |
|------|------|----------|
| `backtest_over.py` | OVER模型滚动回测 | `backtest_over_result.json` |
| `backtest_under.py` | UNDER模型滚动回测 | `backtest_under_result.json` |
| `backtest_upside.py` | UPSIDE模型滚动回测 | `backtest_upside_result.json` |

### 回测原理

**滚动前进验证（Walk-Forward Validation）**：

```
时间轴: |----训练集(85%)----|调参(15%)| 预测下月 | 预测下月 | ...
        2024-11            2026-08   2026-09   2026-10
```

1. **按月滚动**：每个自然月为一个回测窗口
2. **持续学习**：用该月之前全部数据训练，预测该月比赛
3. **评估指标**：
   - 池内命中率（核心指标）
   - Top6/Top3比分命中率
   - Brier分数、对数损失
   - **实现盈亏**（关联开奖SP）
   - **模型EV**（期望价值）

### 执行回测

```bash
cd /coding/live

# 方法1：逐个回测（推荐）
setsid bash -c '.venv/bin/python -u backtest_over.py > logs/backtest_over.log 2>&1' &
tail -f logs/backtest_over.log

# 等待OVER完成后，执行UNDER回测
setsid bash -c '.venv/bin/python -u backtest_under.py > logs/backtest_under.log 2>&1' &
tail -f logs/backtest_under.log

# 等待UNDER完成后，执行UPSIDE回测
setsid bash -c '.venv/bin/python -u backtest_upside.py > logs/backtest_upside.log 2>&1' &
tail -f logs/backtest_upside.log

# 方法2：一次性回测全部
cat > backtest_all.sh << 'EOF'
#!/bin/bash
echo "开始OVER回测..."
.venv/bin/python -u backtest_over.py 2>&1 | tee logs/backtest_over.log

echo "开始UNDER回测..."
.venv/bin/python -u backtest_under.py 2>&1 | tee logs/backtest_under.log

echo "开始UPSIDE回测..."
.venv/bin/python -u backtest_upside.py 2>&1 | tee logs/backtest_upside.log

echo "全部回测完成!"
EOF

chmod +x backtest_all.sh
setsid bash backtest_all.sh > logs/backtest_all.log 2>&1 &
tail -f logs/backtest_all.log
```

### 回测输出示例

```
==================================================================
【总体(全部滚动窗口)】
==================================================================
回测场次: 2414
基线池内率(无筛选): 52.6%
Brier分数: 0.2303  对数损失: 0.6506

  阈值     场数   覆盖率   池内命中率   Top6命中   实现盈亏   模型EV
  >=0.50   1424   59.0%    60.7%      29.8%     +2.67    +2.67
  >=0.60    968   40.1%    64.6%      29.8%     +4.23    +4.23
  >=0.70     94    3.9%    76.6%      29.8%    +10.46   +10.46  ← 推荐阈值
  >=0.80     13    0.5%    84.6%      29.8%    +26.34   +26.34

==================================================================
【按月明细(阈值0.60)】
==================================================================
  月份      场数   基线池内率   筛选后场数   筛选后池内率
  202608     800      52.1%        320        63.5%
  202609     850      52.8%        340        64.2%
  202610     764      53.0%        308        65.1%  ← 本月表现
```

### 关键指标解读

**✅ 合格标准**：
- 阈值0.70时，池内命中率 ≥ 65%
- 实现盈亏 > 0（正EV）
- 月度表现稳定（无大幅波动）

**⚠️ 需要关注**：
- 如果回测命中率 **低于** 训练验收命中率 → 可能过拟合
- 如果某月表现特别差 → 检查该月数据质量
- 如果实现盈亏为负 → 模型不可用

---

## 训练 vs 回测的区别

| 对比项 | 训练验收 | 独立回测 |
|--------|----------|----------|
| **数据范围** | 固定验收段（最近10%） | 全部历史数据滚动验证 |
| **训练方式** | 单次训练 | 每月重新训练（模拟真实） |
| **评估维度** | 单一命中率 | 命中率+盈亏+EV+月度趋势 |
| **耗时** | 包含在训练中 | 额外30-60分钟/模型 |
| **可靠性** | 初步验证 | **更可靠**（避免过拟合） |

---

## 完整工作流

```bash
# 第1步：清理缓存
rm -f /coding/live/features_*_cache.csv

# 第2步：训练OVER模型
setsid bash train_over_only.sh > logs/train_over.log 2>&1 &
tail -f logs/train_over.log

# 第3步：训练UNDER模型（等待OVER完成）
setsid bash train_under_only.sh > logs/train_under.log 2>&1 &
tail -f logs/train_under.log

# 第4步：训练UPSIDE模型（等待UNDER完成）
setsid bash train_upside_only.sh > logs/train_upside.log 2>&1 &
tail -f logs/train_upside.log

# 第5步：回测OVER模型
setsid bash -c '.venv/bin/python -u backtest_over.py > logs/backtest_over.log 2>&1' &
tail -f logs/backtest_over.log

# 第6步：回测UNDER模型（等待OVER回测完成）
setsid bash -c '.venv/bin/python -u backtest_under.py > logs/backtest_under.log 2>&1' &
tail -f logs/backtest_under.log

# 第7步：回测UPSIDE模型（等待UNDER回测完成）
setsid bash -c '.venv/bin/python -u backtest_upside.py > logs/backtest_upside.log 2>&1' &
tail -f logs/backtest_upside.log

# 第8步：查看回测结果
cat backtest_over_result.json | python -m json.tool
cat backtest_under_result.json | python -m json.tool
cat backtest_upside_result.json | python -m json.tool
```

---

## 本地环境（Windows）

### 环境配置

| 项目 | 本地（Windows） | 服务器（Linux） |
|------|-----------------|-----------------|
| Python 路径 | `D:\Program Files (x86)\python311\python.exe` | `/coding/live/.venv/bin/python` |
| 虚拟环境 | `D:\Coding\QiuSport\live\.venv` | `/coding/live/.venv` |
| 内存 | 充足（16GB+） | 受限（3.5GB，易 OOM） |
| 数据库 | 同一阿里云 RDS（直连） | 同一阿里云 RDS（内网） |
| 后台执行 | 不需要 | `setsid` + `nohup` |
| 浏览器 | Edge/Chrome | Chromium + chromedriver |

### 本地依赖安装

```powershell
cd D:\Coding\QiuSport\live
.venv\Scripts\activate
pip install -r requirements.txt
```

> 如果国内 pip 下载慢，使用镜像源：
> ```powershell
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

---

## 本地训练

### 前置准备

```powershell
cd D:\Coding\QiuSport\live

# 清理旧缓存（必须！避免加载旧特征）
Remove-Item features_*_cache.csv -ErrorAction SilentlyContinue

# 检查是否有残留进程
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like '*train*' }
```

### 训练执行（逐个执行，禁止并发）

```powershell
cd D:\Coding\QiuSport\live

# OVER 模型训练
& ".venv\Scripts\python.exe" -u train_over.py

# UNDER 模型训练（等待 OVER 完成后）
& ".venv\Scripts\python.exe" -u train_under.py

# UPSIDE 模型训练（等待 UNDER 完成后）
& ".venv\Scripts\python.exe" -u train_upside.py
```

> **本地 vs 服务器区别**：
> - 本地内存充足，不需要 `setsid` 后台执行
> - 本地可以直接看到实时输出，不需要 `tail -f` 查看日志
> - 本地训练速度通常比服务器快（CPU 性能更好）
> - 模型产物保存在 `models/` 目录，带日期标签（如 `lgb_pool_20261025.txt`）

### 训练输出示例

```
========================================
  OVER 模型训练
========================================
加载数据: 42,156 场比赛
特征工程: 62 个特征
  亚盘特征: 14 个
  大小球特征: 8 个
  近期战绩: 29 个
  对战往绩: 4 个
  联赛积分: 4 个
  其他: 3 个
训练集: 35,833 场 (85%)
验证集: 6,323 场 (15%)

LightGBM 筛选门:
  AUC: 0.7234  Brier: 0.2156
  最优阈值: 0.68 (EV=+3.45)

Dixon-Coles 比分矩阵:
  rho: 0.0823

融合模型 (α=0.65):
  验证集池内命中率: 64.2%
  验证集实现盈亏: +4.12

模型已保存: models/lgb_pool_20261025.txt
校准模型: models/calib_20261025.joblib
元信息: models/meta_20261025.json
========================================
```

### 本地训练注意事项

1. **内存充足**：本地 16GB+ 内存，不会触发 OOM Kill
2. **可以直接运行**：不需要 `setsid` 或 `nohup`
3. **实时查看进度**：终端直接输出训练日志
4. **模型文件位置**：训练产物在 `models/` 目录
5. **缓存文件**：特征缓存在 `features_*_cache.csv`，下次训练可复用（`rebuild=False`）

---

## 本地回测

### 回测执行

```powershell
cd D:\Coding\QiuSport\live

# OVER 模型回测
& ".venv\Scripts\python.exe" -u backtest_over.py

# UNDER 模型回测
& ".venv\Scripts\python.exe" -u backtest_under.py

# UPSIDE 模型回测
& ".venv\Scripts\python.exe" -u backtest_upside.py
```

### 查看回测结果

```powershell
# 格式化查看 JSON 结果
Get-Content backtest_over_result.json | ConvertFrom-Json | ConvertTo-Json -Depth 10

# 或直接用 Python 查看
& ".venv\Scripts\python.exe" -c "import json; print(json.dumps(json.load(open('backtest_over_result.json')), indent=2, ensure_ascii=False))"
```

### 回测结果文件

| 模型 | 输出文件 | 说明 |
|------|----------|------|
| OVER | `backtest_over_result.json` | 滚动回测结果（命中率/盈亏/EV） |
| UNDER | `backtest_under_result.json` | 同上 |
| UPSIDE | `backtest_upside_result.json` | 同上 |

---

## 本地预测

### 预测执行

```powershell
cd D:\Coding\QiuSport\live

# 预测今天比赛（默认）
& ".venv\Scripts\python.exe" -u predict_over.py

# 指定日期预测
& ".venv\Scripts\python.exe" -u predict_over.py --date 20261025

# 预测全部未开赛比赛
& ".venv\Scripts\python.exe" -u predict_over.py --all-unfinished

# 手动抬高置信度阈值
& ".venv\Scripts\python.exe" -u predict_over.py --min-prob 0.70

# 跳过邮件发送（本地调试用）
& ".venv\Scripts\python.exe" -u predict_over.py --skip-email

# 回看历史预测
& ".venv\Scripts\python.exe" -u predict_over.py --review 20261024
```

### 预测输出示例

```
========================================
  OVER 模型预测 - 2026-10-25
========================================
模型版本: lgb_pool_20261025
特征版本: features_over (62 特征)
预测比赛: 8 场

--- 概览 ---
[单场093] 捷甲  赫拉德茨 vs 帕尔杜比斯  (02:00)
  池内概率: 72.3%  推荐: ★
  Top3比分: 1:0(18.2%) 2:1(15.6%) 1:1(12.4%)

[单场095] 德甲  拜仁 vs 多特  (03:30)
  池内概率: 45.1%  推荐: ✗ (低于阈值0.60)
  Top3比分: 2:1(14.3%) 3:1(11.8%) 2:0(10.2%)

--- 汇总 ---
推荐场次: 3 场 (阈值 0.60)
预计命中率: 64.2% (基于回测)
========================================
```

### 本地预测注意事项

1. **数据库连接**：本地直连阿里云 RDS，需确保网络畅通
2. **欧赔数据**：预测前会自动爬取最新欧赔数据（`scrape_euro_odds.py`）
3. **特征重建**：预测脚本强制 `rebuild=True`，每次重建全部特征（约 3 万场）
4. **邮件通知**：默认开启邮件发送，本地调试可用 `--skip-email` 跳过
5. **预测结果**：同时写入 `ml_predictions.jsonl` 供后续回顾

---

## 部署到服务器

### 完整部署流程

```
本地开发 → 本地测试 → Git 提交 → 推送到 GitHub → 服务器拉取 → 服务器运行
```

### 步骤 1：本地修改并测试

```powershell
cd D:\Coding\QiuSport\live

# 修改代码后，先在本地运行验证
& ".venv\Scripts\python.exe" -u train_over.py
& ".venv\Scripts\python.exe" -u backtest_over.py
& ".venv\Scripts\python.exe" -u predict_over.py --skip-email
```

### 步骤 2：Git 提交并推送

```powershell
cd D:\Coding\QiuSport\live

git add .
git commit -m "feat: 描述修改内容"
git push origin main
```

### 步骤 3：同步代码到服务器

**方式 A：服务器 Git Pull（推荐）**

```bash
# SSH 登录服务器
ssh root@8.130.188.90

# 拉取最新代码
cd /coding/live
git pull origin main
```

**方式 B：本地批量上传**

```powershell
cd D:\Coding\QiuSport\live
& ".venv\Scripts\python.exe" _sync_to_server.py
```

> `_sync_to_server.py` 会自动上传所有核心文件到服务器 `/coding/live/`

### 步骤 4：服务器验证

```bash
# SSH 登录服务器
ssh root@8.130.188.90

cd /coding/live

# 检查文件是否更新
ls -la train_over.py predict_over.py features_over.py

# 检查语法
.venv/bin/python -c "import train_over; print('OK')"
.venv/bin/python -c "import predict_over; print('OK')"
.venv/bin/python -c "import features_over; print('OK')"
```

### 步骤 5：服务器执行训练/预测

```bash
cd /coding/live

# 清理缓存（如果修改了特征工程）
rm -f features_*_cache.csv

# 训练（后台执行）
setsid bash train_over_only.sh > logs/train_over.log 2>&1 &
tail -f logs/train_over.log

# 预测（每日定时执行，也可手动）
.venv/bin/python -u predict_over.py
```

### 步骤 6：检查定时任务

```bash
# 查看当前 crontab
crontab -l

# 编辑定时任务
crontab -e

# 参考配置（见 crontab_linux.txt）
# 0 12 * * *  /coding/live/run_daily.sh >> /coding/live/logs/daily.log 2>&1
# 30 18 * * *  /coding/live/.venv/bin/python -u /coding/live/predict_over.py >> /coding/live/logs/predict_over.log 2>&1
# 0 10 * * 1  /coding/live/.venv/bin/python -u /coding/live/train_over.py >> /coding/live/logs/train_over_weekly.log 2>&1
```

### 部署检查清单

- [ ] 本地代码修改并测试通过
- [ ] Git 提交并推送到 GitHub
- [ ] 服务器 `git pull` 或 `_sync_to_server.py` 上传
- [ ] 服务器文件验证（`ls -la` + 语法检查）
- [ ] 如修改特征工程：清理缓存 `rm -f features_*_cache.csv`
- [ ] 服务器执行训练/预测验证
- [ ] 检查定时任务是否需要更新
- [ ] 查看日志确认运行正常

---

## 本地 vs 服务器命令对照表

| 操作 | 本地（Windows PowerShell） | 服务器（Linux Bash） |
|------|---------------------------|---------------------|
| 激活虚拟环境 | `.venv\Scripts\activate` | `source .venv/bin/activate` |
| 运行脚本 | `& ".venv\Scripts\python.exe" -u train_over.py` | `.venv/bin/python -u train_over.py` |
| 后台执行 | 不需要 | `setsid bash train_over_only.sh > logs/train.log 2>&1 &` |
| 查看日志 | 终端直接输出 | `tail -f logs/train_over.log` |
| 清理缓存 | `Remove-Item features_*_cache.csv` | `rm -f features_*_cache.csv` |
| 查看进程 | `Get-Process python` | `ps aux \| grep train \| grep -v grep` |
| 检查内存 | 任务管理器 | `free -m` |
| 检查 OOM | 不适用 | `dmesg \| grep -i oom \| tail -5` |
| 同步代码 | `git push` + `_sync_to_server.py` | `git pull origin main` |

---

## 常见问题

### Q: 训练进程被OOM Kill怎么办？

A: 服务器内存3.5GB，Python进程易占用3.3GB触发OOM。解决方案：
1. 必须先清理缓存：`rm -f /coding/live/features_*_cache.csv`
2. 使用 `setsid` 后台执行，避免SSH断开
3. 检查内存：`free -m`，确保可用内存>2GB再启动

### Q: 训练完成后是否必须回测？

A: **是的，强烈建议回测**。原因：
1. 验证新特征效果（是否真正提升模型表现）
2. 确认模型稳定性（验收段命中率是否达标）
3. 对比历史表现（与之前的模型版本进行对比）
4. 训练验收只是初步验证，回测更可靠（避免过拟合）

### Q: 回测命中率低于训练验收怎么办？

A: 可能原因：
1. **过拟合**：训练集表现好但泛化能力差
2. **数据泄漏**：特征使用了未来信息
3. **样本偏差**：验收段数据分布与整体不同

解决方案：
1. 检查特征工程是否有未来函数
2. 增加正则化参数（如LightGBM的`reg_alpha`）
3. 减少特征数量，保留最重要特征

### Q: 如何实现盈亏为正？

A: 关键指标：
1. **阈值选择**：使用EV最优的P_pool阈值（通常0.65-0.75）
2. **命中率**：阈值0.70时命中率需≥65%
3. **SP值**：关联开奖SP值计算真实盈亏

公式：
- 命中盈亏 = `2 * sp * 0.65`（每注2元，返奖率65%）
- 未中盈亏 = `-18`（9注×2元全损）
- ROI = 总盈亏 / (场次×18)

---

## 注意事项

1. **OOM Killer 风险**: 服务器内存 3.5GB，Python 进程易占用 3.3GB → 被杀掉
2. **必须逐个训练**: 不能同时运行多个模型
3. **后台执行**: 使用 `setsid` 确保 SSH 断开后继续运行
4. **新特征生效**: 删除旧缓存后自动检测代码变更，重新构建包含新特征的缓存
5. **预期效果**:
   - 每个模型训练时间：3-5小时
   - 总计：10-15小时
   - 预计性能提升：2-5%（需重新训练后验证）
