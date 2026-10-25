#!/bin/bash
# train_upside_only.sh - UPSIDE模型独立训练脚本
cd /coding/live

echo "=== 开始训练 UPSIDE 模型 ==="
echo "时间: $(date)"

# 清理旧缓存
rm -f features_upside_cache.csv
echo "已清理 features_upside_cache.csv"

# 训练UPSIDE模型（注意：train_upside.py不支持--rebuild参数）
.venv/bin/python -u train_upside.py > logs/train_upside.log 2>&1

echo "=== UPSIDE 模型训练完成 ==="
echo "退出码: $?"
echo "时间: $(date)"
