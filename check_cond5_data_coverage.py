#!/usr/bin/env python3
# check_cond5_data_coverage.py - 统计 Condition 5 数据覆盖率
# 用法：python check_cond5_data_coverage.py [start_date]
# 示例：python check_cond5_data_coverage.py 20260901
import sys
import pymysql
from config import DB_CONFIG

# Condition 5 配置（与 filter_matches.py 保持一致）
COND5_STATS = ['shots_on_target', 'dangerous_attacks', 'shots', 'corners', 'possession']
COND5_WINDOW_MIN_MATCHES = {'recent3': 2, 'recent6': 3, 'recent10': 5}
COND5_MIN_VALID_STATS = 2

WINDOWS = [
    ('recent3', 3, 2),
    ('recent6', 6, 3),
    ('recent10', 10, 5),
]


def to_float(val):
    if val is None:
        return None
    try:
        return float(str(val).replace('%', '').strip())
    except (ValueError, TypeError):
        return None


def check_cond5_data_available(cursor, match_id):
    """检查 Condition 5 是否有足够数据（不计算优势分，只检查数据可用性）"""
    # 获取主客队近10场技术统计
    rows_by_team = {}
    for team_type in ['home', 'away']:
        cursor.execute(
            f"SELECT {', '.join(COND5_STATS)} FROM recent_form "
            f"WHERE match_id = %s AND team_type = %s AND is_current_match = 0 "
            f"ORDER BY match_date DESC LIMIT 10",
            (match_id, team_type))
        rows = cursor.fetchall()
        if not rows:
            return False, f"缺少{team_type}近期战绩"
        
        # 预计算浮点值矩阵
        rows_float = []
        for row in rows:
            rows_float.append([to_float(row[i]) for i in range(len(COND5_STATS))])
        rows_by_team[team_type] = rows_float

    # 对每队每指标，计算各窗口均值（仅当有效场次≥min_valid_matches）
    avg = {}
    for team_type in ['home', 'away']:
        avg[team_type] = {}
        rf = rows_by_team[team_type]
        for i in range(len(COND5_STATS)):
            avg[team_type][i] = {}
            for w_name, w_limit, w_min in WINDOWS:
                values = [rf[j][i] for j in range(min(w_limit, len(rf))) if rf[j][i] is not None]
                if len(values) >= w_min:
                    avg[team_type][i][w_name] = sum(values) / len(values)

    # 对每指标，检查是否有有效窗口数据（主客队都有）
    valid_stat_count = 0
    for i in range(len(COND5_STATS)):
        has_valid_window = False
        for w_name, w_limit, w_min in WINDOWS:
            h_val = avg['home'][i].get(w_name)
            a_val = avg['away'][i].get(w_name)
            if h_val is not None and a_val is not None:
                has_valid_window = True
                break
        if has_valid_window:
            valid_stat_count += 1

    if valid_stat_count < COND5_MIN_VALID_STATS:
        return False, f"有效指标{valid_stat_count}<{COND5_MIN_VALID_STATS}"
    
    return True, f"有效指标{valid_stat_count}"


def main():
    start_date = sys.argv[1] if len(sys.argv) > 1 else '20260901'
    
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # 获取指定日期以来的所有比赛
    cursor.execute(
        "SELECT match_id, league, home_team, away_team FROM `match` "
        "WHERE match_id >= %s ORDER BY match_id",
        (start_date + '%',)
    )
    matches = cursor.fetchall()
    
    total = len(matches)
    has_data = 0
    no_data = 0
    no_data_reasons = {}
    
    print(f"统计范围：{start_date} 至今")
    print(f"总比赛数：{total}")
    print("=" * 70)
    
    for match_id, league, home_team, away_team in matches:
        ok, reason = check_cond5_data_available(cursor, match_id)
        if ok:
            has_data += 1
        else:
            no_data += 1
            # 统计失败原因
            if reason not in no_data_reasons:
                no_data_reasons[reason] = 0
            no_data_reasons[reason] += 1
    
    print(f"\n满足 Condition 5 数据要求：{has_data} ({has_data/total*100:.1f}%)")
    print(f"不满足数据要求：{no_data} ({no_data/total*100:.1f}%)")
    
    if no_data_reasons:
        print("\n数据不足原因分布：")
        for reason, count in sorted(no_data_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count} 场")
    
    cursor.close()
    conn.close()


if __name__ == '__main__':
    main()
