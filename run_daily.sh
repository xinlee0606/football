#!/bin/bash
# run_daily.sh - 每日爬虫定时任务（Linux版）
# 功能：每天中午12点执行
#   [1/2] 先更新前一天比赛状态和比分（只抓1个完场页，几分钟完成，不会被爬取卡死拖累）
#   [2/2] 再爬取当天比赛数据（断点续传，耗时数小时）
# 防重入：爬取步骤加flock排他锁，上一次爬取未结束时本次自动跳过，避免实例叠加越跑越慢
# 用法：chmod +x run_daily.sh && ./run_daily.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Python路径（优先使用虚拟环境）
if [ -d "$SCRIPT_DIR/.venv" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON="python3"
fi

# 日志目录
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# 爬取步骤防重入锁文件（状态更新不加锁，保证每天必跑）
LOCK_FILE="$LOG_DIR/scraper.lock"
exec 9>"$LOCK_FILE"

echo "============================================"
echo " 北单爬虫定时任务 - $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# 任务1：更新前一天比赛状态和比分（前置执行：仅依赖match表已入库的昨天比赛，与当天爬取无依赖）
YESTERDAY=$(date -d "yesterday" '+%Y%m%d')
echo ""
echo "[1/2] 更新前一天($YESTERDAY)比赛状态和比分..."
$PYTHON -u update_match_status.py "$YESTERDAY" >> "$LOG_DIR/update_status.log" 2>&1
if [ $? -ne 0 ]; then
    echo "[错误] update_match_status.py 执行失败"
fi

# 任务2：爬取比赛数据（断点续传；持锁运行，上一次未结束则跳过）
echo ""
echo "[2/2] 爬取比赛数据（断点续传）..."
if flock -n 9; then
    $PYTHON -u scraper.py --resume >> "$LOG_DIR/scraper.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "[错误] scraper.py 执行失败"
    fi
else
    echo "[跳过] 上一次爬取任务仍在运行，本次跳过爬取（状态更新已照常执行）"
fi

echo ""
echo "============================================"
echo " 全部完成 - $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
