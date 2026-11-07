"""批量补录 UNDER 模型预测记录 (2026-08-01 ~ 2026-09-30)

用法:
    python batch_backfill_under.py              # 补录全部日期
    python batch_backfill_under.py --start 20260815  # 从指定日期开始
    python batch_backfill_under.py --dry-run    # 模拟运行(不实际写入)
"""
import argparse
from datetime import datetime, timedelta
import os
import sys

# 添加 live 目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict_under import backfill


def generate_date_range(start_date, end_date):
    """生成日期范围列表 YYYYMMDD"""
    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)
    return dates


def main():
    ap = argparse.ArgumentParser(description='批量补录 UNDER 模型预测记录')
    ap.add_argument('--start', default='20260801', metavar='YYYYMMDD',
                    help='起始日期(默认 20260801)')
    ap.add_argument('--end', default='20260930', metavar='YYYYMMDD',
                    help='结束日期(默认 20260930)')
    ap.add_argument('--min-prob', type=float, default=None,
                    help='置信度阈值(默认使用模型元数据中的阈值)')
    ap.add_argument('--dry-run', action='store_true',
                    help='模拟运行(不实际写入)')
    args = ap.parse_args()

    dates = generate_date_range(args.start, args.end)
    print(f"计划补录 {len(dates)} 天: {args.start} ~ {args.end}\n")

    success_count = 0
    skip_count = 0
    fail_dates = []

    for i, date in enumerate(dates, 1):
        print(f"[{i}/{len(dates)}] {date}", end=" ... ")
        
        if args.dry_run:
            print("(DRY-RUN)")
            continue
        
        try:
            backfill(date, min_prob=args.min_prob)
            print("✓")
            success_count += 1
        except Exception as e:
            print(f"✗ {e}")
            fail_dates.append((date, str(e)))
    
    # 汇总
    print("\n" + "=" * 70)
    print("补录完成!")
    print(f"  成功: {success_count} 天")
    print(f"  失败: {len(fail_dates)} 天")
    if fail_dates:
        print("\n失败详情:")
        for date, err in fail_dates:
            print(f"  {date}: {err}")
    print("=" * 70)


if __name__ == '__main__':
    main()
