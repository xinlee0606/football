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
