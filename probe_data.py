# probe_data.py - 数据摸底脚本（Step 0）
# 功能: 统计8比分池基线命中率、比分分布、特征表覆盖率, 输出基线报告
# 用法: python -u probe_data.py
# 说明: 报告同时写入 baseline_report.json, 作为模型效果的对照锚点
import json
import os
from datetime import datetime

import pymysql

from config import DB_CONFIG
from features_upside import OVER, OVER_SET, is_finished, norm_score, is_target_handicap

REPORT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'baseline_report.json')


def main():
    conn = pymysql.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM `match`")
    total = cur.fetchone()[0]

    cur.execute("SELECT league, score FROM `match` ORDER BY match_id")
    rows = cur.fetchall()

    dist, league_stat = {}, {}
    finished, hit = 0, 0
    for league, sc in rows:
        if not is_finished(sc):
            continue
        finished += 1
        s = norm_score(sc)
        dist[s] = dist.get(s, 0) + 1
        ls = league_stat.setdefault(league, [0, 0])   # [池内命中, 完场数]
        ls[1] += 1
        if s in OVER_SET:
            hit += 1
            ls[0] += 1

    print("=" * 62)
    print("数据摸底报告")
    print("=" * 62)
    print(f"总场次: {total}, 有效完场: {finished}")
    if not finished:
        print("无有效完场数据")
        return

    base_rate = hit / finished
    print(f"8比分池命中: {hit} 场")
    print(f"★ 基线池内命中率(无任何筛选): {base_rate * 100:.2f}%")
    print(f"  8比分池: {'、'.join(OVER)}")

    print(f"\n比分分布 Top15 (共{len(dist)}种比分):")
    for sc, n in sorted(dist.items(), key=lambda x: -x[1])[:15]:
        mark = ' ★池内' if sc in OVER_SET else ''
        print(f"  {sc}: {n:>6} ({n / finished * 100:5.2f}%){mark}")

    # 时间跨度
    cur.execute("SELECT MIN(match_id), MAX(match_id) FROM `match`")
    lo, hi = cur.fetchone()
    span = f"{lo[:8] if lo else '?'} ~ {hi[:8] if hi else '?'}"
    print(f"\n数据时间跨度(match_id): {span}")

    # 特征表覆盖率
    print("\n特征表覆盖率:")
    coverage = {}
    for t in ['asian_odds', 'over_under_odds', 'head_to_head', 'recent_form',
              'league_standings', 'result_sp']:
        cur.execute(f"SELECT COUNT(DISTINCT match_id) FROM {t}")
        n = cur.fetchone()[0]
        coverage[t] = round(n / finished, 4) if finished else 0.0
        flag = '✓' if coverage[t] >= 0.8 else '⚠ 低于80%, 建议确认是否回填'
        print(f"  {t:<18} {n:>6} 场  {coverage[t] * 100:5.1f}%  {flag}")

    # 联赛池内命中率 Top/Bottom
    lg_list = [(lg, h, n, h / n) for lg, (h, n) in league_stat.items() if n >= 30]
    lg_list.sort(key=lambda x: -x[3])
    print(f"\n联赛池内命中率(样本>=30场, 共{len(lg_list)}个联赛):")
    print("  命中率最高5个:")
    for lg, h, n, r in lg_list[:5]:
        print(f"    {lg}: {r * 100:.1f}% ({h}/{n})")
    print("  命中率最低5个:")
    for lg, h, n, r in lg_list[-5:]:
        print(f"    {lg}: {r * 100:.1f}% ({h}/{n})")

    cur.close()
    conn.close()

    report = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_matches': total,
        'finished_matches': finished,
        'score_span': span,
        'pool8': OVER,
        'pool_hits': hit,
        'baseline_pool_rate': round(base_rate, 4),
        'score_distribution_top15': {
            sc: {'n': n, 'rate': round(n / finished, 4), 'in_pool': sc in OVER_SET}
            for sc, n in sorted(dist.items(), key=lambda x: -x[1])[:15]
        },
        'feature_coverage': coverage,
        'league_pool_rate': {
            lg: {'hits': h, 'n': n, 'rate': round(r, 4)}
            for lg, h, n, r in lg_list
        },
    }
    with open(REPORT_FILE, 'w', encoding='utf-8') as fp:
        json.dump(report, fp, ensure_ascii=False, indent=1)
    print(f"\n基线报告已写入: {REPORT_FILE}")


if __name__ == '__main__':
    main()
