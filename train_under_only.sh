#!/bin/bash
# train_under_only.sh - UNDER模型独立训练脚本
cd /coding/live

echo "=== 开始训练 UNDER 模型 ==="
echo "时间: $(date)"

# 清理旧缓存
rm -f features_under_cache.csv
echo "已清理 features_under_cache.csv"

# 训练UNDER模型
.venv/bin/python -u train_under.py --rebuild > logs/train_under.log 2>&1

echo "=== UNDER 模型训练完成 ==="
echo "退出码: $?"
echo "时间: $(date)"
