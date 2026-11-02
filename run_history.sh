#!/bin/bash
# run_history.sh - Linux服务器运行历史数据回填(后台nohup长跑 + 日志 + 断点续传)
# 用法：
#   chmod +x run_history.sh
#   ./run_history.sh                       # 回填最近365天
#   ./run_history.sh 3                     # 回填最近3天(测试)
#   ./run_history.sh 20250806 20260805     # 回填指定日期区间
#   ./run_history.sh 20250806 20250806 --limit 2   # 测试: 每天限2场详情
# 中断后重跑同一命令即可断点续传(已爬比赛自动跳过)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Python路径（优先使用虚拟环境）
if [ -d "$SCRIPT_DIR/.venv" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON="python3"
fi

# 防止中文/特殊符号输出在非UTF-8 locale下报UnicodeEncodeError中断长跑
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/history.log"

# 防止重复启动
if pgrep -f "history_scraper.py" > /dev/null; then
    echo "⚠ history_scraper.py 已在运行中, 如需重启请先: pkill -f history_scraper.py"
    exit 1
fi

nohup "$PYTHON" -u history_scraper.py "$@" >> "$LOG_FILE" 2>&1 &
PID=$!

echo "=========================================="
echo "✓ 历史回填已启动 (PID=$PID)"
echo "  参数: $@"
echo "  日志: $LOG_FILE"
echo "=========================================="
echo "常用命令："
echo "  实时查看进度: tail -f $LOG_FILE"
echo "  查看最新状态: tail -n 50 $LOG_FILE"
echo "  停止任务:     pkill -f history_scraper.py"
echo "  查完成场次:   mysql查询 SELECT COUNT(DISTINCT match_id) FROM asian_odds;"
