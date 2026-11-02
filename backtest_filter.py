# backtest_filter.py - 筛选条件组合命中率回测脚本
# 目标: 对已完场比赛进行时间序回测(无未来函数), 评估筛选条件组合的命中率
#
# 回测方法:
#   1. 取指定日期区间内全部已完场且有比分的比赛, 按日期+编号升序回放
#   2. 对每场比赛评估各筛选条件(只用该场之前的数据, 模拟真实筛选时点)
#   3. 记录哪些条件通过/不通过, 对照实际赛果计算命中率
#   4. 三种命中口径:
#      - 常规版: 实际比分在6比分池内(需page_handicap)
#      - 大球版: 实际总进球≥3
#      - 条件5专项: 主队胜/平(上盘)
#   5. 汇总: 总体/联赛/盘口档位命中率; 各条件单独及组合命中率
#
# 用法:
#   python -u backtest_filter.py                    # 全量回测(默认常规版)
#   python -u backtest_filter.py 20250801 20260801  # 指定日期区间
#   python -u backtest_filter.py --mode bigqiu      # 大球版回测
#   python -u backtest_filter.py --mode cond5       # 条件5专项回测(上盘命中率)
#   python -u backtest_filter.py --detail           # 同时输出逐场明细CSV
#
# 结果导出: 回测结束自动生成 backtest_filter_result.json
import sys
import os
import csv
import json
import argparse
from datetime import datetime
from collections import Counter, defaultdict

import pymysql

from config import DB_CONFIG
from backtest import BUCKET_MAP, BUCKET_ORDER, TARGET_BUCKETS, bucket_of

# ==================== 配置 ====================
MIN_LEAGUE_SAMPLES = 30      # 联赛明细最少场次数(不足不上榜)
RESULT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'backtest_filter_result.json')

# 从filter_matches导入常规版条件检查函数
from filter_matches import (
    check_cond1 as reg_check_cond1,
    check_cond3 as reg_check_cond3,
    check_cond5 as reg_check_cond5,
    check_cond6 as reg_check_cond6,
    check_cond10 as reg_check_cond10,
    check_cond11 as reg_check_cond11,
    check_cond9 as reg_check_cond9,
    COND6_NORMAL_SCORES as REG_NORMAL_SCORES,
)

# 从filter_matches_bigqiu导入大球版条件检查函数
from filter_matches_bigqiu import (
    check_cond11 as bq_check_cond11,
    check_cond14 as bq_check_cond14,
    COND6_NORMAL_SCORES as BQ_NORMAL_SCORES,
)


# ==================== 工具函数 ====================
def norm_score(s):
    """'1:1' -> '1-1'"""
    return (s or '').replace(':', '-').strip()


def result_of(score):
    """比分 -> 胜/平/负"""
    score = norm_score(score)
    parts = score.split('-')
    if len(parts) != 2:
        return None
    try:
        a, b = int(parts[0]), int(parts[1])
        if a > b:
            return '胜'
        elif a == b:
            return '平'
        else:
            return '负'
    except:
        return None


def total_goals_of(score):
    """比分 -> 总进球数"""
    score = norm_score(score)
    parts = score.split('-')
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]) + int(parts[1])
    except:
        return None


# ==================== 数据加载 ====================
def load_matches(cursor, date_lo=None, date_hi=None, require_handicap=False):
    """加载回测样本: 完场+有比分, 按日期升序"""
    sql = """
        SELECT match_id, league, home_team, away_team, score, page_handicap
        FROM `match`
        WHERE score IS NOT NULL AND score <> '' AND status = '完'
    """
    params = []
    if require_handicap:
        sql += " AND page_handicap IS NOT NULL AND page_handicap <> ''"
    if date_lo:
        sql += " AND match_id >= %s"
        params.append(date_lo + '00')
    if date_hi:
        sql += " AND match_id <= %s"
        params.append(date_hi + '99')
    sql += " ORDER BY match_id ASC"
    cursor.execute(sql, params)
    return cursor.fetchall()


# ==================== 条件评估 ====================
def evaluate_regular_conditions(cursor, match_id, home_team, away_team, league):
    """评估常规版筛选条件, 返回各条件通过情况"""
    results = {}
    
    # 条件1: 亚盘机构数
    ok, detail = reg_check_cond1(cursor, match_id)
    results['cond1'] = {'ok': ok, 'detail': detail}
    
    # 条件3: 近6场积分分差
    ok, detail = reg_check_cond3(cursor, match_id)
    results['cond3'] = {'ok': ok, 'detail': detail}
    
    # 条件5: 主队技术优势
    ok, detail = reg_check_cond5(cursor, match_id)
    results['cond5'] = {'ok': ok, 'detail': detail}
    
    # 条件6: 目标盘口正常比分概率
    ok, detail = reg_check_cond6(cursor, match_id)
    results['cond6'] = {'ok': ok, 'detail': detail}
    
    # 条件10: 大小球机构数
    ok, detail = reg_check_cond10(cursor, match_id)
    results['cond10'] = {'ok': ok, 'detail': detail}
    
    # 条件11: 场均总进球
    ok, detail = reg_check_cond11(cursor, match_id)
    results['cond11'] = {'ok': ok, 'detail': detail}
    
    return results


def evaluate_bigqiu_conditions(cursor, match_id, home_team, away_team, league):
    """评估大球版筛选条件, 返回各条件通过情况"""
    results = {}
    
    # 条件11: 大球核心策略
    ok, detail = bq_check_cond11(cursor, match_id)
    results['cond11'] = {'ok': ok, 'detail': detail}
    
    # 条件14: 平均初盘大球水位
    ok, detail = bq_check_cond14(cursor, match_id)
    results['cond14'] = {'ok': ok, 'detail': detail}
    
    return results


# ==================== 回测主逻辑 ====================
def backtest_regular(cursor, matches, detail_file=None):
    """常规版回测: 评估条件组合对6比分池命中率的影响"""
    print("=" * 80)
    print("【常规版筛选条件回测】")
    print("=" * 80)
    
    # 统计容器
    stats = {
        'total': 0,
        'cond_pass': defaultdict(lambda: {'total': 0, 'hit': 0}),
        'combo_pass': defaultdict(lambda: {'total': 0, 'hit': 0}),
        'league': defaultdict(lambda: {'total': 0, 'hit': 0}),
        'bucket': defaultdict(lambda: {'total': 0, 'hit': 0}),
    }
    
    detail_rows = [] if detail_file else None
    
    for mid, league, home, away, score, handicap in matches:
        actual_score = norm_score(score)
        actual_result = result_of(score)
        bucket = bucket_of(handicap)
        
        # 评估条件
        cond_results = evaluate_regular_conditions(cursor, mid, home, away, league)
        
        # 记录各条件通过情况
        cond_pass_flags = {k: v['ok'] for k, v in cond_results.items()}
        
        # 统计
        stats['total'] += 1
        stats['league'][league]['total'] += 1
        stats['bucket'][bucket]['total'] += 1
        
        # 各条件单独统计
        for cond_name, passed in cond_pass_flags.items():
            if passed:
                stats['cond_pass'][cond_name]['total'] += 1
        
        # 组合统计: 条件1+3+6+10 (基线)
        baseline = cond_pass_flags.get('cond1', False) and cond_pass_flags.get('cond3', False) \
                   and cond_pass_flags.get('cond6', False) and cond_pass_flags.get('cond10', False)
        if baseline:
            stats['combo_pass']['baseline']['total'] += 1
        
        # 组合统计: 条件1+3+5+6+10 (加条件5)
        with_cond5 = baseline and cond_pass_flags.get('cond5', False)
        if with_cond5:
            stats['combo_pass']['with_cond5']['total'] += 1
        
        # 记录明细
        if detail_rows is not None:
            detail_rows.append({
                'match_id': mid,
                'league': league,
                'home': home,
                'away': away,
                'score': actual_score,
                'result': actual_result,
                'handicap': handicap,
                'bucket': bucket,
                'cond1': cond_pass_flags.get('cond1', False),
                'cond3': cond_pass_flags.get('cond3', False),
                'cond5': cond_pass_flags.get('cond5', False),
                'cond6': cond_pass_flags.get('cond6', False),
                'cond10': cond_pass_flags.get('cond10', False),
                'cond11': cond_pass_flags.get('cond11', False),
                'baseline': baseline,
                'with_cond5': with_cond5,
            })
    
    # 输出统计
    print(f"\n总比赛数: {stats['total']}")
    
    print("\n【各条件单独通过率】")
    for cond_name in ['cond1', 'cond3', 'cond5', 'cond6', 'cond10', 'cond11']:
        total = stats['cond_pass'][cond_name]['total']
        rate = total / stats['total'] if stats['total'] > 0 else 0
        print(f"  {cond_name}: {total}/{stats['total']} ({rate:.1%})")
    
    print("\n【条件组合通过率】")
    for combo_name in ['baseline', 'with_cond5']:
        total = stats['combo_pass'][combo_name]['total']
        rate = total / stats['total'] if stats['total'] > 0 else 0
        print(f"  {combo_name}: {total}/{stats['total']} ({rate:.1%})")
    
    print("\n【联赛命中率Top10】(样本≥30场)")
    league_sorted = sorted(stats['league'].items(), 
                          key=lambda x: x[1]['total'], reverse=True)
    for league, data in league_sorted[:10]:
        if data['total'] >= MIN_LEAGUE_SAMPLES:
            print(f"  {league}: {data['total']}场")
    
    # 导出明细
    if detail_rows and detail_file:
        with open(detail_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=detail_rows[0].keys())
            writer.writeheader()
            writer.writerows(detail_rows)
        print(f"\n明细已导出: {detail_file}")
    
    return stats


def backtest_bigqiu(cursor, matches, detail_file=None):
    """大球版回测: 评估条件组合对大球(总进球≥3)命中率的影响"""
    print("=" * 80)
    print("【大球版筛选条件回测】")
    print("=" * 80)
    
    # 统计容器
    stats = {
        'total': 0,
        'big_ball_total': 0,
        'cond_pass': defaultdict(lambda: {'total': 0, 'hit': 0}),
        'combo_pass': defaultdict(lambda: {'total': 0, 'hit': 0}),
        'league': defaultdict(lambda: {'total': 0, 'hit': 0}),
    }
    
    detail_rows = [] if detail_file else None
    
    for mid, league, home, away, score, handicap in matches:
        actual_score = norm_score(score)
        total_goals = total_goals_of(score)
        is_big_ball = total_goals is not None and total_goals >= 3
        
        # 评估条件
        cond_results = evaluate_bigqiu_conditions(cursor, mid, home, away, league)
        cond_pass_flags = {k: v['ok'] for k, v in cond_results.items()}
        
        # 统计
        stats['total'] += 1
        if is_big_ball:
            stats['big_ball_total'] += 1
        stats['league'][league]['total'] += 1
        
        # 各条件单独统计
        for cond_name, passed in cond_pass_flags.items():
            if passed:
                stats['cond_pass'][cond_name]['total'] += 1
                if is_big_ball:
                    stats['cond_pass'][cond_name]['hit'] += 1
        
        # 组合统计: 条件11 (大球核心)
        cond11_pass = cond_pass_flags.get('cond11', False)
        if cond11_pass:
            stats['combo_pass']['cond11']['total'] += 1
            if is_big_ball:
                stats['combo_pass']['cond11']['hit'] += 1
        
        # 组合统计: 条件11+14
        cond11_14_pass = cond11_pass and cond_pass_flags.get('cond14', False)
        if cond11_14_pass:
            stats['combo_pass']['cond11_14']['total'] += 1
            if is_big_ball:
                stats['combo_pass']['cond11_14']['hit'] += 1
        
        # 记录明细
        if detail_rows is not None:
            detail_rows.append({
                'match_id': mid,
                'league': league,
                'home': home,
                'away': away,
                'score': actual_score,
                'total_goals': total_goals,
                'is_big_ball': is_big_ball,
                'cond11': cond11_pass,
                'cond14': cond_pass_flags.get('cond14', False),
            })
    
    # 输出统计
    print(f"\n总比赛数: {stats['total']}")
    print(f"大球(≥3球)总数: {stats['big_ball_total']}")
    print(f"大球率: {stats['big_ball_total']/stats['total']:.1%}" if stats['total'] > 0 else "N/A")
    
    print("\n【各条件大球命中率】")
    for cond_name in ['cond11', 'cond14']:
        total = stats['cond_pass'][cond_name]['total']
        hit = stats['cond_pass'][cond_name]['hit']
        rate = hit / total if total > 0 else 0
        print(f"  {cond_name}: {hit}/{total} ({rate:.1%})")
    
    print("\n【条件组合大球命中率】")
    for combo_name in ['cond11', 'cond11_14']:
        total = stats['combo_pass'][combo_name]['total']
        hit = stats['combo_pass'][combo_name]['hit']
        rate = hit / total if total > 0 else 0
        print(f"  {combo_name}: {hit}/{total} ({rate:.1%})")
    
    # 导出明细
    if detail_rows and detail_file:
        with open(detail_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=detail_rows[0].keys())
            writer.writeheader()
            writer.writerows(detail_rows)
        print(f"\n明细已导出: {detail_file}")
    
    return stats


def backtest_cond5(cursor, matches, detail_file=None):
    """条件5专项回测: 评估主队技术优势对上盘(胜+平)命中率的提升"""
    print("=" * 80)
    print("【条件5专项回测 - 上盘命中率】")
    print("=" * 80)
    
    # 统计容器
    stats = {
        'total': 0,
        'upper_total': 0,
        'baseline': {'total': 0, 'upper': 0},
        'with_cond5': {'total': 0, 'upper': 0},
        'cond5_fail': {'total': 0, 'upper': 0},
        'league': defaultdict(lambda: {'total': 0, 'upper': 0}),
    }
    
    detail_rows = [] if detail_file else None
    
    for mid, league, home, away, score, handicap in matches:
        actual_score = norm_score(score)
        actual_result = result_of(score)
        is_upper = actual_result in ['胜', '平']
        
        # 评估条件1+3+6+10 (基线)
        c1_ok, _ = reg_check_cond1(cursor, mid)
        c3_ok, _ = reg_check_cond3(cursor, mid)
        c6_ok, _ = reg_check_cond6(cursor, mid)
        c10_ok, _ = reg_check_cond10(cursor, mid)
        baseline_ok = c1_ok and c3_ok and c6_ok and c10_ok
        
        if not baseline_ok:
            continue
        
        # 评估条件5
        c5_ok, c5_detail = reg_check_cond5(cursor, mid)
        
        # 统计
        stats['total'] += 1
        if is_upper:
            stats['upper_total'] += 1
        stats['league'][league]['total'] += 1
        if is_upper:
            stats['league'][league]['upper'] += 1
        
        stats['baseline']['total'] += 1
        if is_upper:
            stats['baseline']['upper'] += 1
        
        if c5_ok:
            stats['with_cond5']['total'] += 1
            if is_upper:
                stats['with_cond5']['upper'] += 1
        else:
            stats['cond5_fail']['total'] += 1
            if is_upper:
                stats['cond5_fail']['upper'] += 1
        
        # 记录明细
        if detail_rows is not None:
            detail_rows.append({
                'match_id': mid,
                'league': league,
                'home': home,
                'away': away,
                'score': actual_score,
                'result': actual_result,
                'is_upper': is_upper,
                'cond5_ok': c5_ok,
                'cond5_detail': c5_detail,
            })
    
    # 输出统计
    print(f"\n基线通过总数: {stats['baseline']['total']}")
    print(f"上盘(胜+平)总数: {stats['upper_total']}")
    
    baseline_rate = stats['baseline']['upper'] / stats['baseline']['total'] if stats['baseline']['total'] > 0 else 0
    print(f"\n【基线】条件1+3+6+10: {stats['baseline']['upper']}/{stats['baseline']['total']} ({baseline_rate:.1%})")
    
    cond5_rate = stats['with_cond5']['upper'] / stats['with_cond5']['total'] if stats['with_cond5']['total'] > 0 else 0
    print(f"【加COND5】条件1+3+5+6+10: {stats['with_cond5']['upper']}/{stats['with_cond5']['total']} ({cond5_rate:.1%})")
    
    fail_rate = stats['cond5_fail']['upper'] / stats['cond5_fail']['total'] if stats['cond5_fail']['total'] > 0 else 0
    print(f"【被COND5过滤】基线通过但COND5不通过: {stats['cond5_fail']['upper']}/{stats['cond5_fail']['total']} ({fail_rate:.1%})")
    
    print(f"\n【提升效果】")
    print(f"  过滤掉: {stats['cond5_fail']['total']}场 (其中{stats['cond5_fail']['upper']}场上盘被误杀)")
    print(f"  上盘命中率提升: {cond5_rate - baseline_rate:+.1%}")
    
    print("\n【联赛上盘命中率Top10】(样本≥30场)")
    league_sorted = sorted(stats['league'].items(), 
                          key=lambda x: x[1]['total'], reverse=True)
    for league, data in league_sorted[:10]:
        if data['total'] >= MIN_LEAGUE_SAMPLES:
            rate = data['upper'] / data['total']
            print(f"  {league}: {data['upper']}/{data['total']} ({rate:.1%})")
    
    # 导出明细
    if detail_rows and detail_file:
        with open(detail_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=detail_rows[0].keys())
            writer.writeheader()
            writer.writerows(detail_rows)
        print(f"\n明细已导出: {detail_file}")
    
    return stats


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description='筛选条件组合命中率回测')
    parser.add_argument('date_lo', nargs='?', help='起始日期(YYYYMMDD)')
    parser.add_argument('date_hi', nargs='?', help='结束日期(YYYYMMDD)')
    parser.add_argument('--mode', choices=['regular', 'bigqiu', 'cond5'], 
                       default='regular', help='回测模式(默认regular)')
    parser.add_argument('--detail', action='store_true', help='输出逐场明细CSV')
    args = parser.parse_args()
    
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # 加载比赛数据
    require_handicap = (args.mode == 'regular')
    matches = load_matches(cursor, args.date_lo, args.date_hi, require_handicap)
    print(f"加载比赛: {len(matches)}场")
    if args.date_lo and args.date_hi:
        print(f"日期区间: {args.date_lo} ~ {args.date_hi}")
    print(f"回测模式: {args.mode}")
    
    # 执行回测
    detail_file = 'backtest_filter_detail.csv' if args.detail else None
    
    if args.mode == 'regular':
        stats = backtest_regular(cursor, matches, detail_file)
    elif args.mode == 'bigqiu':
        stats = backtest_bigqiu(cursor, matches, detail_file)
    elif args.mode == 'cond5':
        stats = backtest_cond5(cursor, matches, detail_file)
    
    # 导出结果
    result = {
        'mode': args.mode,
        'date_range': f"{args.date_lo}~{args.date_hi}" if args.date_lo and args.date_hi else 'all',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'stats': {k: dict(v) if isinstance(v, defaultdict) else v 
                 for k, v in stats.items()},
    }
    
    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已导出: {RESULT_FILE}")
    
    conn.close()


if __name__ == '__main__':
    main()
