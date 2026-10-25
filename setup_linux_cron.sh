#!/bin/bash
# setup_linux_cron.sh - Linux服务器一键部署定时任务
# 使用方法：chmod +x setup_linux_cron.sh && ./setup_linux_cron.sh

echo "=========================================="
echo " 北单爬虫项目 - Linux定时任务安装"
echo "=========================================="

# 获取当前脚本所在目录作为项目路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "项目目录: $SCRIPT_DIR"

# 检查Python虚拟环境
if [ -d "$SCRIPT_DIR/.venv" ]; then
    PYTHON_PATH="$SCRIPT_DIR/.venv/bin/python"
    echo "✓ 找到虚拟环境: $PYTHON_PATH"
else
    echo "⚠ 未找到虚拟环境，正在创建..."
    python3 -m venv "$SCRIPT_DIR/.venv"
    PYTHON_PATH="$SCRIPT_DIR/.venv/bin/python"
    echo "✓ 虚拟环境已创建: $PYTHON_PATH"
fi

# 安装Python依赖
echo ""
echo "安装Python依赖..."
$PYTHON_PATH -m pip install -r "$SCRIPT_DIR/requirements.txt" -q
echo "✓ 依赖安装完成"

# 创建日志目录
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
echo "✓ 日志目录: $LOG_DIR"

# 设置脚本执行权限
chmod +x "$SCRIPT_DIR/run_daily.sh"
echo "✓ run_daily.sh 已设置执行权限"

# 检查Chrome浏览器
if command -v google-chrome &> /dev/null || command -v chromium-browser &> /dev/null || command -v chromium &> /dev/null; then
    echo "✓ Chrome/Chromium 已安装"
else
    echo "⚠ 未检测到Chrome/Chromium，请手动安装："
    echo "  Ubuntu/Debian: sudo apt-get install -y chromium-browser"
    echo "  CentOS/RHEL:   sudo yum install -y chromium"
fi

# 检查chromedriver
if command -v chromedriver &> /dev/null; then
    echo "✓ chromedriver: $(which chromedriver)"
else
    echo "⚠ chromedriver未安装，请手动安装："
    echo "  Ubuntu/Debian: sudo apt-get install -y chromium-chromedriver"
    echo "  CentOS/RHEL:   sudo yum install -y chromedriver"
fi

# 安装crontab定时任务
echo ""
echo "安装定时任务..."

# 移除旧的相关任务（如果存在）
crontab -l 2>/dev/null | grep -v "run_daily.sh" | grep -v "filter_matches.py" | grep -v "filter_matches_bigqiu.py" | grep -v "predict_ml.py" | grep -v "train_model.py" | grep -v "history_scraper.py" > /tmp/crontab_tmp

# 添加新任务
cat >> /tmp/crontab_tmp << EOF
# 北单爬虫 - 每天中午12:00执行爬虫+更新状态
0 12 * * * $SCRIPT_DIR/run_daily.sh >> $LOG_DIR/daily.log 2>&1
# 北单爬虫 - 每天下午18:00执行筛选并发邮件
0 18 * * * cd $SCRIPT_DIR && $PYTHON_PATH filter_matches.py >> $LOG_DIR/filter.log 2>&1
# ML 比分预测 - 每天晚上 21:30 执行混合模型预测 (阈值 0.70)
30 21 * * * cd $SCRIPT_DIR && $PYTHON_PATH -u predict_ml.py --min-prob 0.70 >> $LOG_DIR/predict_ml.log 2>&1
# ML预测回顾 - 每天中午12:40回顾昨天预测命中(对照实际比分, 结果邮件发送)
40 12 * * * cd $SCRIPT_DIR && $PYTHON_PATH -u predict_ml.py --review >> $LOG_DIR/review.log 2>&1
# ML比分预测 - 每周一上午10:00全量重训模型(持续学习)
0 10 * * 1 cd $SCRIPT_DIR && $PYTHON_PATH -u train_model.py >> $LOG_DIR/train_ml.log 2>&1
# 完场页盘口回填 - 每天凌晨00:40回填昨天(page_handicap/page_goals, 回测与筛选条件9依赖)
40 0 * * * cd $SCRIPT_DIR && $PYTHON_PATH -u history_scraper.py 1 >> $LOG_DIR/history_daily.log 2>&1
# 筛选回顾 - 每天中午12:50回顾昨天常规版筛选命中(对照实际比分+SP, 结果邮件发送)
50 12 * * * cd $SCRIPT_DIR && $PYTHON_PATH -u filter_matches.py --review >> $LOG_DIR/review_filter.log 2>&1
# 筛选回顾 - 每天下午13:00回顾昨天大球版筛选命中(对照实际比分+SP, 结果邮件发送)
0 13 * * * cd $SCRIPT_DIR && $PYTHON_PATH -u filter_matches_bigqiu.py --review >> $LOG_DIR/review_filter_bigqiu.log 2>&1
EOF

crontab /tmp/crontab_tmp
rm -f /tmp/crontab_tmp

echo ""
echo "=========================================="
echo "✓ 部署完成！"
echo "=========================================="
echo ""
echo "定时任务："
echo "  [1] 每天 12:00 - scraper.py(断点续传) + update_match_status.py(前一天)"
echo "  [2] 每天 18:00 - filter_matches.py(筛选+邮件通知)"
echo "  [3] 每天 21:30 - predict_ml.py --min-prob 0.70(ML 比分预测，阈值 0.70)"
echo "  [4] 每周一 10:00 - train_model.py(全量重训, 持续学习)"
echo "  [5] 每天 00:40 - history_scraper.py 1(回填昨天完场页盘口)"
echo "  [6] 每天 12:40 - predict_ml.py --review(回顾昨天预测命中+邮件)"
echo "  [7] 每天 12:50 - filter_matches.py --review(回顾昨天常规版筛选命中+比分+SP+邮件)"
echo "  [8] 每天 13:00 - filter_matches_bigqiu.py --review(回顾昨天大球版筛选命中+比分+SP+邮件)"
echo ""
echo "首次使用ML预测前需手动执行:"
echo "  cd $SCRIPT_DIR && $PYTHON_PATH -u probe_data.py      # 数据摸底基线"
echo "  cd $SCRIPT_DIR && $PYTHON_PATH -u backtest_ml.py     # 滚动回测(约几分钟)"
echo "  cd $SCRIPT_DIR && $PYTHON_PATH -u train_model.py     # 训练并保存模型"
echo ""
echo "常用命令："
echo "  查看定时任务: crontab -l"
echo "  查看爬虫日志: tail -f $LOG_DIR/scraper.log"
echo "  查看筛选日志: tail -f $LOG_DIR/filter.log"
echo "  手动测试爬虫: cd $SCRIPT_DIR && $PYTHON_PATH scraper.py 2"
echo "  手动测试筛选: cd $SCRIPT_DIR && $PYTHON_PATH filter_matches.py"
echo ""
