# select_2x1.py - OVER模型2串1自动选场脚本
# 功能:
#   1. 读取当日预测结果(ml_predictions.jsonl)
#   2. 筛选 P_pool >= 阈值的推荐场次
#   3. 按 P_pool 降序选Top2组成2串1
#   4. 计算2串1理论命中率
#   5. 输出选场建议
#   6. --review: 回顾历史日期的2串1选场效果(查实际比分+SP值)
#
# 用法:
#   python -u select_2x1.py                 # 选今天比赛的2串1
#   python -u select_2x1.py --date 20260926 # 选指定日期
#   python -u select_2x1.py --min-prob 0.80 # 自定义阈值
#   python -u select_2x1.py --top-n 3       # 选Top3(可组3注2串1)
#   python -u select_2x1.py --review 20260926 # 回顾历史日期
import argparse
import json
import os
from datetime import datetime, timedelta
from itertools import combinations

import pymysql

from config import DB_CONFIG
from features_over import OVER_SET, is_finished, parse_score

PRED_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'ml_predictions.jsonl')

# 默认配置
DEFAULT_MIN_PROB = 0.80   # 最低池内概率阈值
DEFAULT_TOP_N = 2         # 选Top N场组2串1
TOP_PICKS = 11            # 每场投注11个比分


def load_predictions(target_date):
    """读取指定日期的预测记录"""
    if not os.path.exists(PRED_LOG_FILE):
        print(f"✗ 未找到预测记录文件: {PRED_LOG_FILE}")
        return []
    
    with open(PRED_LOG_FILE, encoding='utf-8') as fp:
        logs = [json.loads(line) for line in fp if line.strip()]
    
    # 按match_id去重(保留最后一次), 只取目标日期
    latest = {}
    for l in logs:
        mid = str(l.get('match_id', ''))
        if mid.startswith(target_date):
            latest[mid] = l
    
    return list(latest.values())


def select_2x1(predictions, min_prob=DEFAULT_MIN_PROB, top_n=DEFAULT_TOP_N):
    """选场逻辑: 筛选推荐场次, 按P_pool降序选Top N"""
    # 筛选推荐场次(picked=true 且 p_pool >= 阈值)
    candidates = [
        p for p in predictions
        if p.get('picked') and p.get('p_pool', 0) >= min_prob
    ]
    
    # 按P_pool降序排序
    candidates.sort(key=lambda x: x.get('p_pool', 0), reverse=True)
    
    # 选Top N
    selected = candidates[:top_n]
    
    return selected, candidates


def calc_2x1_stats(selected):
    """计算2串1统计"""
    if len(selected) < 2:
        return None
    
    # 2串1理论命中率 = P1 × P2 × ... × Pn
    probs = [p.get('p_pool', 0) for p in selected]
    combined_prob = 1.0
    for p in probs:
        combined_prob *= p
    
    # 成本: 2场 × 11注 × 1元 = 22元
    cost = len(selected) * 11
    
    return {
        'combined_prob': combined_prob,
        'cost': cost,
        'probs': probs,
    }


def format_output(selected, all_candidates, stats, target_date):
    """格式化输出"""
    lines = []
    
    lines.append("=" * 70)
    lines.append(f"OVER模型 2串1 选场建议 ({target_date})")
    lines.append("=" * 70)
    lines.append(f"筛选条件: P_pool >= {DEFAULT_MIN_PROB:.2f}")
    lines.append(f"候选场次: {len(all_candidates)} 场")
    lines.append("")
    
    if len(selected) < 2:
        lines.append(f"✗ 符合条件的场次不足2场 (当前{len(selected)}场)")
        lines.append("  建议: 降低阈值或等待更多比赛")
        return '\n'.join(lines)
    
    # 输出选中的比赛
    lines.append("【选中比赛】")
    lines.append("-" * 70)
    for i, p in enumerate(selected, 1):
        mid = p.get('match_id', '')
        league = p.get('league', '')
        home = p.get('home_team', '?')
        away = p.get('away_team', '?')
        p_pool = p.get('p_pool', 0)
        picks = p.get('top_picks', '')
        
        lines.append(f"  {i}. {mid} [{league}] {home} vs {away}")
        lines.append(f"     P_pool: {p_pool*100:.1f}%")
        lines.append(f"     投注比分: {picks}")
        lines.append("")
    
    # 2串1统计
    if stats:
        lines.append("【2串1统计】")
        lines.append("-" * 70)
        lines.append(f"  组合概率: {stats['combined_prob']*100:.1f}%")
        lines.append(f"  投注成本: {stats['cost']}元 ({len(selected)}场×11注×1元)")
        lines.append("")
        
        # 盈亏分析(假设平均SP=10)
        avg_sp = 10
        win_payout = avg_sp * avg_sp  # SP1 × SP2
        ev = stats['combined_prob'] * (win_payout - stats['cost']) + \
             (1 - stats['combined_prob']) * (-stats['cost'])
        
        lines.append(f"  假设平均SP={avg_sp}:")
        lines.append(f"    中奖回报: {win_payout}元")
        lines.append(f"    期望收益(EV): {ev:+.2f}元")
        lines.append("")
    
    # 所有候选场次(供参考)
    if len(all_candidates) > len(selected):
        lines.append("【其他候选场次】")
        lines.append("-" * 70)
        for i, p in enumerate(all_candidates[len(selected):], len(selected)+1):
            mid = p.get('match_id', '')
            league = p.get('league', '')
            home = p.get('home_team', '?')
            away = p.get('away_team', '?')
            p_pool = p.get('p_pool', 0)
            lines.append(f"  {i}. {mid} [{league}] {home} vs {away}  P_pool: {p_pool*100:.1f}%")
        lines.append("")
    
    lines.append("=" * 70)
    lines.append("提示: 2串1需两场全中才盈利, 建议只选P_pool最高的2场")
    lines.append("=" * 70)
    
    return '\n'.join(lines)


def review(target_date, min_prob=DEFAULT_MIN_PROB, top_n=DEFAULT_TOP_N):
    """回顾历史日期的2串1选场效果"""
    lines = []
    
    def out(s=''):
        lines.append(s)
        print(s)
    
    if not os.path.exists(PRED_LOG_FILE):
        print(f"✗ 未找到预测记录文件: {PRED_LOG_FILE}")
        return None
    
    with open(PRED_LOG_FILE, encoding='utf-8') as fp:
        logs = [json.loads(line) for line in fp if line.strip()]
    
    # 按match_id去重(保留最后一次), 只取目标日期
    latest = {}
    for l in logs:
        mid = str(l.get('match_id', ''))
        if mid.startswith(target_date):
            latest[mid] = l
    
    if not latest:
        print(f"✗ {target_date} 无预测记录")
        print("  请先运行: python -u predict_over.py 或 python -u predict_over.py --backfill")
        return None
    
    # 查询实际比分和SP值
    ids = list(latest.keys())
    conn = pymysql.connect(**DB_CONFIG)
    cur = conn.cursor()
    fmt = ','.join(['%s'] * len(ids))
    cur.execute(f"SELECT match_id, home_team, away_team, score FROM `match` "
                f"WHERE match_id IN ({fmt})", ids)
    match_info = {r[0]: {'home': r[1], 'away': r[2], 'score': r[3]} for r in cur.fetchall()}
    
    cur.execute(f"SELECT match_id, sp_value FROM result_sp WHERE CONVERT(match_id USING utf8mb4) COLLATE utf8mb4_general_ci IN ({fmt})", ids)
    sp_map = {r[0]: (float(r[1]) if r[1] is not None else None) for r in cur.fetchall()}
    cur.close()
    conn.close()
    
    # 筛选推荐场次
    candidates = [
        p for p in latest.values()
        if p.get('picked') and p.get('p_pool', 0) >= min_prob
    ]
    candidates.sort(key=lambda x: x.get('p_pool', 0), reverse=True)
    selected = candidates[:top_n]
    
    out("=" * 78)
    out(f"2串1 回顾 {target_date}  (来源: ml_predictions.jsonl)")
    out(f"筛选条件: P_pool >= {min_prob:.2f}, 选Top {top_n}")
    out("=" * 78)
    out(f"候选场次: {len(candidates)} 场, 选中: {len(selected)} 场")
    out("")
    
    if len(selected) < top_n:
        out(f"✗ 符合条件的场次不足{top_n}场 (当前{len(selected)}场)")
        out("  建议: 降低阈值或该日期无足够高置信场次")
        return '\n'.join(lines)
    
    # 输出选中比赛及实际结果
    out("【选中比赛】")
    out("-" * 78)
    
    finished_count = 0
    all_hit = True
    total_cost = top_n * TOP_PICKS  # 总成本
    total_payout = 0
    sp_product = 1.0
    
    for i, p in enumerate(selected, 1):
        mid = p.get('match_id', '')
        league = p.get('league', '')
        home = p.get('home_team', '?')
        away = p.get('away_team', '?')
        p_pool = p.get('p_pool', 0)
        picks_str = p.get('top_picks', '')
        picks = [s.split(':')[0] for s in str(picks_str).split(';') if s]
        
        mi = match_info.get(mid, {})
        sc = mi.get('score')
        sp = sp_map.get(mid)
        sp_str = f"SP{sp:.2f}" if sp is not None else "SP-"
        
        out(f"  {i}. {mid} [{league}] {home} vs {away}")
        out(f"     P_pool: {p_pool*100:.1f}%")
        out(f"     投注比分: {'/'.join(picks[:8])}")
        
        if is_finished(sc):
            h, a = parse_score(sc)
            norm = f"{h}-{a}"
            in_pool = norm in OVER_SET
            in_top = norm in picks
            hit_flag = '✓中' if in_top else ('池内' if in_pool else '未中')
            
            out(f"     实际比分: {norm} {sp_str} -> {hit_flag}")
            finished_count += 1
            
            if not in_top:
                all_hit = False
            
            if sp is not None:
                sp_product *= sp
        else:
            out(f"     实际比分: {str(sc) if sc else '-'} {sp_str} -> 未完场")
            all_hit = False
        out("")
    
    # 2串1结果统计
    out("【2串1结果】")
    out("-" * 78)
    out(f"  已完场: {finished_count}/{top_n} 场")
    
    if finished_count < top_n:
        out(f"  结果: 有{top_n - finished_count}场未完场, 暂无法结算")
    elif all_hit:
        # 两场都中
        payout = sp_product * 1  # 1元本金
        profit = payout - total_cost
        out(f"  结果: ✓ 两场全中!")
        out(f"  SP乘积: {sp_product:.2f}")
        out(f"  中奖回报: {payout:.2f}元")
        out(f"  投注成本: {total_cost}元 ({top_n}场×{TOP_PICKS}注×1元)")
        out(f"  实际盈亏: {profit:+.2f}元")
    else:
        # 至少一场不中
        out(f"  结果: ✗ 至少一场未中")
        out(f"  投注成本: {total_cost}元 ({top_n}场×{TOP_PICKS}注×1元)")
        out(f"  实际盈亏: {-total_cost:+.2f}元")
    
    out("")
    out("=" * 78)
    
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description='OVER模型2串1自动选场')
    ap.add_argument('--date', default=datetime.now().strftime('%Y%m%d'),
                    help='指定日期 YYYYMMDD (默认今天)')
    ap.add_argument('--min-prob', type=float, default=DEFAULT_MIN_PROB,
                    help=f'最低池内概率阈值 (默认{DEFAULT_MIN_PROB})')
    ap.add_argument('--top-n', type=int, default=DEFAULT_TOP_N,
                    help=f'选Top N场组2串1 (默认{DEFAULT_TOP_N})')
    ap.add_argument('--review', nargs='?', const='', default=None,
                    help='回顾模式: 对照历史选场结果与实际比分, 不带值默认昨天, 可指定YYYYMMDD')
    args = ap.parse_args()
    
    # 回顾模式
    if args.review is not None:
        target = args.review or (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
        review(target, args.min_prob, args.top_n)
        return
    
    print(f"读取 {args.date} 预测记录...")
    predictions = load_predictions(args.date)
    
    if not predictions:
        print(f"✗ {args.date} 无预测记录")
        print("  请先运行: python -u predict_over.py")
        return
    
    print(f"共 {len(predictions)} 场预测")
    
    selected, candidates = select_2x1(predictions, args.min_prob, args.top_n)
    stats = calc_2x1_stats(selected)
    output = format_output(selected, candidates, stats, args.date)
    
    print("\n" + output)


if __name__ == '__main__':
    main()
