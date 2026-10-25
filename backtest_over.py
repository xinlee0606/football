# backtest_over.py - 混合模型滚动前进回测脚本（Step 3）
# 方法:
#   1. 完场样本按时间排序, 前TRAIN_RATIO做首轮训练
#   2. 之后每个自然月为一个回测窗口: 用该月之前全部数据训练(内部再切15%调参),
#      预测该月比赛 -> 逐段滚动直到数据末端(模拟真实使用+持续学习)
#   3. 评估: 筛选后池内命中率 / Top6比分命中率 / Brier / 对数损失
#   4. EV: 关联result_sp开奖SP, 每场Top6比分各投1元的实现盈亏与模型期望EV
#   5. 阈值敏感性: 不同P_pool阈值下的命中率与EV曲线(定投注纪律用)
#
# 用法:
#   python -u backtest_over.py                # 全量滚动回测
#   python -u backtest_over.py --ratio 0.75   # 指定首轮训练占比
# 结果: 控制台汇总 + backtest_over_result.json
import argparse
import json
import math
import os
from datetime import datetime

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

from features_over import feature_columns_over, is_target_handicap_over, is_valid_ou_line_over, is_cup_league_over, load_or_build_over
from train_over import (TOP_PICKS, TRAIN_RATIO, choose_threshold,
                         predict_rows, split_by_time, train_over)

RESULT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'backtest_over_result.json')


def month_of(match_id):
    return str(match_id)[:6]


def run_walkforward(d, train_ratio=TRAIN_RATIO, verbose=True):
    """滚动前进回测, 返回逐场记录列表"""
    n = len(d)
    cutoff = int(n * train_ratio)
    d_test_all = d.iloc[cutoff:].copy()
    if d_test_all.empty:
        return []

    months = sorted(d_test_all['match_id'].map(month_of).unique())
    if verbose:
        print(f"回测样本: {len(d_test_all)}场, 按月滚动: {len(months)}个窗口 "
              f"({months[0]}~{months[-1]})")

    feats = feature_columns_over()
    records = []
    for i, mo in enumerate(months):
        win = d_test_all[d_test_all['match_id'].map(month_of) == mo]
        df_hist = d[d['match_id'] < win['match_id'].min()]
        if len(df_hist) < 500:
            continue
        tr, va, _ = split_by_time(df_hist, train_ratio=0.85, valid_ratio=0.15)
        if verbose:
            print(f"  [{i + 1}/{len(months)}] {mo}: 训练{len(tr)}+调参{len(va)} "
                  f"-> 预测{len(win)}场...", flush=True)
        model = train_over(tr, va, feats, verbose=False)
        pred = predict_rows(model, win)
        for _, r in pred.iterrows():
            picks = [s.split(':')[0] for s in r['top_picks'].split(';')]
            records.append({
                'match_id': r['match_id'], 'month': mo, 'league': r['league'],
                'score': r['score'], 'in_pool': int(r['y']),
                'p_pool': float(r['p_pool']),
                'hit_top6': int(r['score'] in picks),
                'hit_top3': int(r['score'] in picks[:3]),
                'sp': None if r['sp'] is None or (isinstance(r['sp'], float)
                                                   and math.isnan(r['sp'])) else float(r['sp']),
                'ev_model': None if r['ev_model'] is None or
                (isinstance(r['ev_model'], float) and math.isnan(r['ev_model']))
                else float(r['ev_model']),
            })
    return records


# ==================== 统计输出 ====================
def rate(hit, total):
    return f"{hit}/{total} = {hit / total * 100:.1f}%" if total else "0/0"


def summarize(records, title):
    n = len(records)
    pool_hit = sum(r['in_pool'] for r in records)
    top6_hit = sum(r['hit_top6'] for r in records)
    top3_hit = sum(r['hit_top3'] for r in records)
    p = np.array([r['p_pool'] for r in records])
    y = np.array([r['in_pool'] for r in records])
    brier = brier_score_loss(y, p)
    eps = 1e-9
    ll = log_loss(y, np.clip(p, eps, 1 - eps), labels=[0, 1])

    print(f"\n{'=' * 66}\n【{title}】\n{'=' * 66}")
    print(f"回测场次: {n}")
    print(f"基线池内率(无筛选): {rate(pool_hit, n)}")
    print(f"Brier分数: {brier:.4f}  对数损失: {ll:.4f} (越低概率越准)")
    print(f"Top{TOP_PICKS}精确比分命中率: {rate(top6_hit, n)}")
    print(f"Top3精确比分命中率: {rate(top3_hit, n)}")

    # 阈值敏感性
    print(f"\n  {'阈值':<7}{'场数':>7}{'覆盖率':>8}{'池内命中率':>12}"
          f"{'Top6命中':>12}{'实现盈亏':>10}{'模型EV':>9}")
    th_rows = []
    for th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        sel = [r for r in records if r['p_pool'] >= th]
        if not sel:
            continue
        s_hit = sum(r['in_pool'] for r in sel)
        s_top6 = sum(r['hit_top6'] for r in sel)
        # 实现盈亏: 每场8注各1元, 命中Top8 -> +SP-8, 否则 -8
        profits = []
        evs = []
        for r in sel:
            if r['sp'] is not None:
                profits.append((r['sp'] - TOP_PICKS) if r['hit_top6'] else -TOP_PICKS)
            if r['ev_model'] is not None:
                evs.append(r['ev_model'])
        profit_str = f"{np.mean(profits):+.2f}" if profits else '-'
        ev_str = f"{np.mean(evs):+.2f}" if evs else '-'
        print(f"  >={th:.2f} {len(sel):>6}{len(sel) / n * 100:>7.1f}%"
              f"{s_hit / len(sel) * 100:>11.1f}%"
              f"{s_top6 / len(sel) * 100:>11.1f}%"
              f"{profit_str:>10}{ev_str:>9}")
        th_rows.append({
            'threshold': th, 'n': len(sel),
            'coverage': round(len(sel) / n, 4),
            'pool_hit_rate': round(s_hit / len(sel), 4),
            'top6_hit_rate': round(s_top6 / len(sel), 4),
            'realized_profit_per_match': round(float(np.mean(profits)), 3) if profits else None,
            'model_ev_per_match': round(float(np.mean(evs)), 3) if evs else None,
        })
    return th_rows


def monthly_table(records):
    """按月汇总"""
    print(f"\n{'=' * 66}\n【按月明细(阈值0.60)】\n{'=' * 66}")
    print(f"  {'月份':<8}{'场数':>6}{'基线池内率':>12}{'筛选后场数':>10}{'筛选后池内率':>12}")
    by_month = {}
    for r in records:
        by_month.setdefault(r['month'], []).append(r)
    rows = []
    for mo in sorted(by_month):
        rs = by_month[mo]
        base = sum(r['in_pool'] for r in rs) / len(rs)
        sel = [r for r in rs if r['p_pool'] >= 0.60]
        sel_rate = sum(r['in_pool'] for r in sel) / len(sel) if sel else None
        sel_str = f"{sel_rate * 100:.1f}%" if sel is not None and sel else '-'
        print(f"  {mo:<8}{len(rs):>6}{base * 100:>11.1f}%{len(sel):>10}{sel_str:>14}")
        rows.append({'month': mo, 'n': len(rs), 'baseline_pool_rate': round(base, 4),
                     'selected': len(sel),
                     'selected_pool_rate': round(sel_rate, 4) if sel else None})
    return rows


# ==================== 主流程 ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ratio', type=float, default=TRAIN_RATIO,
                    help=f'首轮训练占比(默认{TRAIN_RATIO})')
    ap.add_argument('--rebuild', action='store_true', help='强制重建特征缓存')
    args = ap.parse_args()

    df = load_or_build_over(rebuild=args.rebuild)
    d = df[df['in_pool'].notna()].copy().sort_values('match_id').reset_index(drop=True)
    # 盘口筛选
    hc_mask = d.apply(
        lambda r: is_target_handicap_over(r.get('ah_cur_mean'), r.get('ah_init_mean')),
        axis=1)
    d = d[hc_mask]
    # 大小球线筛选
    ou_mask = d.apply(
        lambda r: is_valid_ou_line_over(r.get('ou_line_init_mean'), r.get('ou_line_cur_mean')),
        axis=1)
    d = d[ou_mask]
    # 排除杯赛
    d = d[~d['league'].apply(is_cup_league_over)]
    d['y'] = d['in_pool'].astype(int)
    from features_over import parse_score
    d['home_goals'], d['away_goals'] = zip(*d['score'].map(parse_score))
    print(f"完场样本: {len(d)}场 ({str(d['match_id'].iloc[0])[:8]} ~ "
          f"{str(d['match_id'].iloc[-1])[:8]}), 池内率{d['y'].mean() * 100:.1f}%")

    records = run_walkforward(d, train_ratio=args.ratio)
    if not records:
        print("无可回测样本")
        return

    print("\n" + "#" * 66)
    print("# 滚动前进回测结果总览")
    print("#" * 66)
    th_rows = summarize(records, "总体(全部滚动窗口)")
    m_rows = monthly_table(records)

    sp_n = sum(1 for r in records if r['sp'] is not None)
    data = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train_ratio': args.ratio,
        'samples': len(records),
        'baseline_pool_rate': round(sum(r['in_pool'] for r in records) / len(records), 4),
        'sp_coverage': round(sp_n / len(records), 4),
        'threshold_sensitivity': th_rows,
        'monthly': m_rows,
    }
    with open(RESULT_FILE, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, ensure_ascii=False, indent=1)
    print(f"\n回测结果已导出: {RESULT_FILE}")
    print(f"SP数据覆盖: {rate(sp_n, len(records))} (无SP场次盈亏按未中奖-6计)")


if __name__ == '__main__':
    main()
