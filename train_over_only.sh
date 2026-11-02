#!/bin/bash
# train_over_only.sh - OVER模型独立训练脚本
cd /coding/live

echo "=== 开始训练 OVER 模型 ==="
echo "时间: $(date)"

# 清理旧缓存
rm -f features_over_cache.csv
echo "已清理 features_over_cache.csv"

# 训练OVER模型
.venv/bin/python -u train_over.py --rebuild > logs/train_over.log 2>&1

echo "=== OVER 模型训练完成 ==="
echo "退出码: $?"
echo "时间: $(date)"
