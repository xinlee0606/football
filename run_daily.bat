@echo off
chcp 65001 >nul
echo ============================================
echo  北单爬虫定时任务 - %date% %time%
echo ============================================

cd /d D:\Coding\QiuSport\live
set PYTHON=D:\Coding\QiuSport\live\.venv\Scripts\python.exe

echo.
echo [1/2] 爬取比赛数据（断点续传）...
%PYTHON% scraper.py --resume
if %errorlevel% neq 0 (
    echo [错误] scraper.py 执行失败，错误码: %errorlevel%
)

echo.
echo [2/2] 更新前一天比赛状态和比分...
%PYTHON% update_match_status.py
if %errorlevel% neq 0 (
    echo [错误] update_match_status.py 执行失败，错误码: %errorlevel%
)

echo.
echo ============================================
echo  全部完成 - %date% %time%
echo ============================================
