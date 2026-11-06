# backtest.py - 六比分预测命中率回测脚本
# 目标: 对已完场比赛进行时间序回测(无未来函数), 评估6比分命中率
#       分组统计: 总体 / 联赛 / 盘口档位; 并对比"固定比分池"与"历史频率Top6"
#
# 回测方法:
#   1. 取全部已完场且带完场页盘口(page_handicap)的比赛, 按日期+编号升序回放
#   2. 对每场比赛生成6个比分预测(只用该场之前的完场数据, 模拟真实下注时点):
#      方法A(固定池): 复用 predict_scores.py 的方向+大小球逻辑与固定比分池
#      方法B(频率Top6): 按盘口档位分组统计此前全部完场比赛的比分频率取Top6
#      方法C(频率Top6细分): 盘口档位×大小球分档(page_goals)双维分组, 样本不足回退单盘口档
#   3. 实际比分在6个预测之内 = 命中
#   4. 汇总: 总体/联赛/盘口命中率; 频率法置信度(Top6历史覆盖概率)分档命中率
#
# 用法:
#   python -u backtest.py                    # 全量回测
#   python -u backtest.py 20250801 20260801  # 指定日期区间
#   python -u backtest.py --detail           # 同时输出逐场明细CSV到 backtest_detail.csv
#
# 结果导出: 回测结束自动生成 backtest_result.json(联赛黑名单+总体命中率),
#   filter_matches.py 每次运行时也会自动调用 update_backtest_result() 刷新该文件
import sys
import os
import csv
import json
from datetime import datetime
from collections import Counter, defaultdict

import pymysql

from config import DB_CONFIG
from predict_scores import (SCORE_POOLS, SCORE_ALLOC, SMALL_LINE, BIG_LINE,
                            AVG_TOTAL_SMALL, _handicap_to_num, _to_float,
                            _parse_score, _total_goals)

# ==================== 配置 ====================
TOP_N = 6                    # 预测比分个数
MIN_HISTORY = 50             # 频率法最少历史样本数(不足则回退全体样本)
MIN_LEAGUE_SAMPLES = 30      # 联赛明细最少场次数(不足不上榜)
CONF_THRESHOLDS = [0.30, 0.40, 0.50]   # 置信度(覆盖概率)分档阈值

# 盘口档位映射(page_handicap原始字符串 -> 档位名)
BUCKET_MAP = {
    '平手': '平手', '受平手': '平手',
    '平/半': '平/半', '平手/半球': '平/半',
    '受让平/半': '受让平/半', '受平/半': '受让平/半', '受平手/半球': '受让平/半',
    '半球': '半球', '受半球': '受让半球',
    '半/一': '半/一', '受半/一': '受让半/一',
    '一球': '一球', '受一球': '受让一球',
}
BUCKET_ORDER = ['平手', '平/半', '受让平/半', '半球', '受让半球',
                '半/一', '受让半/一', '一球', '受让一球', '其他']
TARGET_BUCKETS = ['平手', '平/半', '半球']   # 重点关注档位

# 联赛黑名单自动导出规则: 重点档位下频率Top6(方法B)命中率过低且样本足够
BLACKLIST_MAX_RATE = 0.51          # 命中率低于此值的联赛进黑名单
RESULT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'backtest_result.json')


def bucket_of(handicap):
    return BUCKET_MAP.get((handicap or '').strip(), '其他')


def goal_bucket(goals):
    """大小球线(page_goals)分档, 无法解析返回None"""
    v = _to_float(goals)
    if v is None:
        return None
    if v < 2.25:
        return '<=2'
    if v < 2.6:
        return '2.25-2.5'
    if v < 3.1:
        return '2.75-3'
    return '>=3.25'


def norm_score(s):
    """'1:1' -> '1-1'"""
    return (s or '').replace(':', '-').strip()


def result_of(score):
    """比分 -> 胜/平/负"""
    a, b = _parse_score(score)
    if a is None:
        return None
    return '胜' if a > b else ('平' if a == b else '负')


# ==================== 数据加载 ====================
def load_matches(cursor, date_lo=None, date_hi=None):
    """加载回测样本: 完场+有比分+有完场盘口, 按日期升序"""
    sql = """
        SELECT match_id, league, home_team, away_team, score,
               page_handicap, page_goals, prediction
        FROM `match`
        WHERE score IS NOT NULL AND score <> ''
          AND page_handicap IS NOT NULL AND page_handicap <> ''
          AND status = '完'
    """
    params = []
    if date_lo:
        sql += " AND match_id >= %s"
        params.append(date_lo + '00')
    if date_hi:
        sql += " AND match_id <= %s"
        params.append(date_hi + '99')
    sql += " ORDER BY match_id ASC"
    cursor.execute(sql, params)
    return cursor.fetchall()


def load_recent_form(cursor):
    """一次性加载近期战绩比分(用于方法A的大小球判断)
    返回 {match_id: {'home': [比分...], 'away': [比分...]}} (按页面原始顺序, 新在前)"""
    cursor.execute("""
        SELECT match_id, team_type, score FROM recent_form
        WHERE is_current_match = 0 AND score IS NOT NULL AND score <> ''
        ORDER BY match_id, match_date DESC
    """)
    data = {}
    for mid, tt, score in cursor.fetchall():
        data.setdefault(mid, {'home': [], 'away': []})[tt].append(score)
    return data


def load_ou_lines(cursor):
    """大小球即时主线(众数) {match_id: line}"""
    cursor.execute("""
        SELECT match_id, current_line, initial_line FROM over_under_odds
    """)
    cnt = {}
    for mid, cl, il in cursor.fetchall():
        v = _to_float(cl if cl is not None else il)
        if v is not None:
            cnt.setdefault(mid, Counter())[v] += 1
    return {mid: c.most_common(1)[0][0] for mid, c in cnt.items()}


# ==================== 方法A: 固定比分池(现有逻辑) ====================
def predict_fixed(match, ou_line, recent_scores):
    """复刻 predict_six_scores: 方向(prediction+盘口修正) + 大小球 + 固定池3/2/1分配"""
    mid, _lg, _ht, _at, _sc, handicap, goals, prediction = match
    hc = _handicap_to_num(handicap)

    if prediction == '胜平':
        dirs = ['主胜', '平局', '客胜']
    elif prediction == '平负':
        dirs = ['客胜', '平局', '主胜']
    else:
        dirs = ['平局', '主胜', '客胜']
    if hc is not None:
        if hc < 0:
            dirs = ['客胜', '平局', '主胜']
        elif abs(hc) <= 0.25 and dirs[0] != '平局':
            dirs.remove('平局')
            dirs.insert(0, '平局')

    # 定进球数: 优先大小球线, 其次page_goals, 再次近期场均总进球
    line = ou_line
    if line is None:
        line = _to_float(goals)
    avgs = []
    for tt in ['home', 'away']:
        totals = [_total_goals(s) for s in recent_scores.get(mid, {}).get(tt, [])[:10]]
        totals = [t for t in totals if t is not None]
        avgs.append(sum(totals) / len(totals) if totals else None)
    if line is not None:
        if line <= SMALL_LINE:
            small = True
        elif line >= BIG_LINE:
            small = False
        elif avgs[0] is not None and avgs[1] is not None:
            small = (avgs[0] + avgs[1]) / 2 <= AVG_TOTAL_SMALL
        else:
            small = True
    else:
        if avgs[0] is not None and avgs[1] is not None:
            small = (avgs[0] + avgs[1]) / 2 <= AVG_TOTAL_SMALL
        else:
            small = True

    # 3/2/1分配 + 小球优先低进球
    cands = []
    for d, quota in zip(dirs, SCORE_ALLOC):
        pool = SCORE_POOLS[d]
        if small:
            pool = sorted(pool, key=lambda s: _total_goals(s))
        cands.extend(pool[:quota + 1])
    seen, ordered = set(), []
    for s in cands:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    if small:
        low = [s for s in ordered if _total_goals(s) <= 2]
        high = [s for s in ordered if _total_goals(s) > 2]
        ordered = low + high
    picks = ordered[:6]
    if len(picks) < 6:
        for d in dirs:
            for s in SCORE_POOLS[d]:
                if s not in picks:
                    picks.append(s)
                if len(picks) >= 6:
                    break
            if len(picks) >= 6:
                break
    return picks[:6], None


# ==================== 方法B: 历史频率Top6 ====================
class FreqTable:
    """按任意分组键维护比分频率(边回放边累积, 天然无未来函数)"""

    def __init__(self):
        self.groups = defaultdict(Counter)   # group_key -> Counter(score)

    def add(self, key, score):
        self.groups[key][score] += 1

    def top_scores(self, key):
        """返回 (Top比分列表, 覆盖概率, 样本量)"""
        cnt = self.groups.get(key)
        total_n = sum(cnt.values()) if cnt else 0
        if total_n == 0:
            return [], 0.0, 0
        top = cnt.most_common(TOP_N)
        cover = sum(c for _, c in top) / total_n
        return [s for s, _ in top], cover, total_n

    def top_with_fallback(self, keys, min_history=MIN_HISTORY):
        """按给定分组键依次尝试, 返回首个样本>=min_history的结果
        返回: (Top比分列表, 覆盖概率, 样本量, 命中使用的分组键)"""
        for k in keys:
            picks, cover, n = self.top_scores(k)
            if n >= min_history:
                return picks, cover, n, k
        return [], 0.0, 0, None


# ==================== 时间序回放(核心, 供main与自动更新共用) ====================
def replay_matches(matches, recent_scores, ou_lines):
    """按日期升序回放全部比赛, 返回逐场记录列表(无未来函数)"""
    freq = FreqTable()
    records = []
    for m in matches:
        mid, league, _ht, _at, score, handicap, goals, _pred = m
        actual = norm_score(score)
        bkt = bucket_of(handicap)
        gbkt = goal_bucket(goals)

        # 先用"此前数据"预测, 再将本场加入频率表(无未来函数)
        picks_fixed, _ = predict_fixed(m, ou_lines.get(mid), recent_scores)
        picks_b, cover_b, n_b, _ = freq.top_with_fallback([f'B|{bkt}', 'ALL'])
        keys_c = ([f'C|{bkt}|{gbkt}', f'B|{bkt}', 'ALL']
                  if gbkt else [f'B|{bkt}', 'ALL'])
        picks_c, cover_c, n_c, key_c = freq.top_with_fallback(keys_c)
        freq.add(f'B|{bkt}', actual)
        if gbkt:
            freq.add(f'C|{bkt}|{gbkt}', actual)
        freq.add('ALL', actual)

        records.append({
            'match_id': mid, 'league': league, 'bucket': bkt,
            'actual': actual, 'hit_fixed': actual in picks_fixed,
            'hit_b': bool(picks_b) and actual in picks_b,
            'hit_c': bool(picks_c) and actual in picks_c,
            'cover': cover_b, 'cover_c': cover_c,
            'fallback_c': bool(gbkt) and (key_c != f'C|{bkt}|{gbkt}'),
        })
    return records


# ==================== 黑名单计算与结果导出 ====================
def compute_league_blacklist(records):
    """重点档位下频率Top6(方法B)命中率过低的联赛 -> (黑名单, 联赛统计)
    规则: 样本>=MIN_LEAGUE_SAMPLES 且 命中率<BLACKLIST_MAX_RATE"""
    by_league = defaultdict(list)
    for r in records:
        if r['bucket'] in TARGET_BUCKETS:
            by_league[r['league']].append(r['hit_b'])
    blacklist, stats = [], {}
    for lg, hits in by_league.items():
        n = len(hits)
        if n < MIN_LEAGUE_SAMPLES:
            continue
        hit = sum(hits)
        rate = hit / n
        stats[lg] = {'场数': n, '命中': hit, '命中率': round(rate, 4)}
        if rate < BLACKLIST_MAX_RATE:
            blacklist.append(lg)
    return sorted(blacklist), stats


def export_result(records, path=RESULT_FILE):
    """导出回测结果JSON(联赛黑名单+总体命中率), 供filter_matches.py自动加载"""
    blacklist, stats = compute_league_blacklist(records)
    n = len(records)
    data = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'samples': n,
        'overall_hit': {
            '固定池': round(sum(1 for r in records if r['hit_fixed']) / n, 4),
            'B盘口档': round(sum(1 for r in records if r['hit_b']) / n, 4),
            'C盘口×大小球': round(sum(1 for r in records if r['hit_c']) / n, 4),
        },
        'blacklist_rule': (f"重点档位({'/'.join(TARGET_BUCKETS)})下方法B命中率"
                           f'<{BLACKLIST_MAX_RATE:.0%}且样本>={MIN_LEAGUE_SAMPLES}场'),
        'league_blacklist': blacklist,
        'league_stats': stats,
    }
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, ensure_ascii=False, indent=1)
    return data


def update_backtest_result(conn=None, verbose=True):
    """全量回测并导出backtest_result.json, 返回联赛黑名单列表
    供filter_matches.py每次运行时自动调用(复用其数据库连接, 不重复建连)
    Args:
        conn: 可复用的数据库连接; None则自建并关闭
        verbose: 是否打印进度
    """
    own_conn = conn is None
    if own_conn:
        conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    if verbose:
        print("自动回测: 加载数据...")
    matches = load_matches(cursor)
    recent_scores = load_recent_form(cursor)
    ou_lines = load_ou_lines(cursor)
    cursor.close()
    if own_conn:
        conn.close()
    if verbose:
        print(f"自动回测: 时间序回放 {len(matches)} 场...")
    records = replay_matches(matches, recent_scores, ou_lines)
    data = export_result(records)
    if verbose:
        print(f"自动回测完成: {data['samples']}场, 黑名单{len(data['league_blacklist'])}个联赛, "
              f"结果已写入 {RESULT_FILE}")
    return data['league_blacklist']


# ==================== 统计输出 ====================
def rate(hit, total):
    return f"{hit}/{total} = {hit / total * 100:.1f}%" if total else "0/0"


def summarize(records, title, group_key, min_samples=1, top_k=None):
    print(f"\n{'=' * 62}\n【{title}】\n{'=' * 62}")
    print(f"{'分组':<12}{'场数':>6}  {'固定池':>12}  {'B盘口档':>12}  {'C盘口×大小球':>14}")
    groups = {}
    for r in records:
        groups.setdefault(r[group_key], []).append(r)
    items = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    if top_k:
        items = items[:top_k]
    for name, rs in items:
        if len(rs) < min_samples:
            continue
        f_hit = sum(1 for r in rs if r['hit_fixed'])
        b_hit = sum(1 for r in rs if r['hit_b'])
        c_hit = sum(1 for r in rs if r['hit_c'])
        print(f"{str(name):<12}{len(rs):>6}  {rate(f_hit, len(rs)):>12}  "
              f"{rate(b_hit, len(rs)):>12}  {rate(c_hit, len(rs)):>14}")


# ==================== 主流程 ====================
def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    detail = '--detail' in sys.argv[1:]
    date_lo = args[0] if len(args) >= 2 else None
    date_hi = args[1] if len(args) >= 2 else None

    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    print("加载数据...")
    matches = load_matches(cursor, date_lo, date_hi)
    recent_scores = load_recent_form(cursor)
    ou_lines = load_ou_lines(cursor)
    cursor.close()
    conn.close()
    print(f"回测样本: {len(matches)} 场 "
          f"({matches[0][0][:8]} ~ {matches[-1][0][:8]})" if matches else "无样本")

    records = replay_matches(matches, recent_scores, ou_lines)

    if not records:
        print("无可回测样本")
        return

    # ---- 总体 ----
    print("\n" + "#" * 62)
    print("# 回测结果总览")
    print("#" * 62)
    n = len(records)
    f_hit = sum(1 for r in records if r['hit_fixed'])
    b_hit = sum(1 for r in records if r['hit_b'])
    c_hit = sum(1 for r in records if r['hit_c'])
    print(f"样本总量: {n} 场")
    print(f"方法A 固定比分池      : {rate(f_hit, n)}")
    print(f"方法B 频率Top6(盘口档): {rate(b_hit, n)}")
    print(f"方法C 频率Top6(盘口×大小球): {rate(c_hit, n)}")

    # ---- 重点盘口档位 ----
    tgt = [r for r in records if r['bucket'] in TARGET_BUCKETS]
    if tgt:
        tf = sum(1 for r in tgt if r['hit_fixed'])
        tb = sum(1 for r in tgt if r['hit_b'])
        tc = sum(1 for r in tgt if r['hit_c'])
        print(f"\n重点档位({'/'.join(TARGET_BUCKETS)}) {len(tgt)}场: "
              f"固定池 {rate(tf, len(tgt))} | B盘口档 {rate(tb, len(tgt))} | "
              f"C盘口×大小球 {rate(tc, len(tgt))}")

    # ---- 按盘口 ----
    summarize(records, "按盘口档位命中率", 'bucket')

    # ---- 按联赛(仅重点档位 + 样本>=门槛) ----
    tgt_records = [r for r in records if r['bucket'] in TARGET_BUCKETS]
    summarize(tgt_records, "重点档位下按联赛命中率", 'league',
              min_samples=MIN_LEAGUE_SAMPLES)

    # ---- 方法C回退情况 ----
    fb = sum(1 for r in records if r['fallback_c'])
    print(f"\n方法C双维分组样本不足回退到单盘口档的场次: {fb} "
          f"({fb / n * 100:.1f}%)")

    # ---- 置信度分档(方法B) ----
    print(f"\n{'=' * 62}\n【方法B 置信度(覆盖概率)分档命中率】\n{'=' * 62}")
    print(f"{'置信度区间':<16}{'场数':>6}  {'命中率':>16}")
    prev = 0.0
    for th in CONF_THRESHOLDS + [9.9]:
        lo, hi = prev, min(th, 1.01)
        rs = [r for r in records if lo <= r['cover'] < hi and r['cover'] > 0]
        if rs:
            hit = sum(1 for r in rs if r['hit_b'])
            label = f"[{lo:.0%}, {hi:.0%})" if hi <= 1.01 else f">={lo:.0%}"
            print(f"{label:<16}{len(rs):>6}  {rate(hit, len(rs)):>16}")
        prev = th
    for th in CONF_THRESHOLDS:
        rs = [r for r in records if r['cover'] >= th]
        if rs:
            hit = sum(1 for r in rs if r['hit_b'])
            print(f">= {th:.0%} 累计:  {len(rs):>4}场  {rate(hit, len(rs))}")

    # ---- 逐场明细 ----
    if detail:
        out = 'backtest_detail.csv'
        with open(out, 'w', newline='', encoding='utf-8-sig') as fp:
            w = csv.writer(fp)
            w.writerow(['match_id', '联赛', '盘口档位', '实际比分',
                        '固定池命中', '方法B命中', '方法B覆盖概率',
                        '方法C命中', '方法C覆盖概率', '方法C回退'])
            for r in records:
                w.writerow([r['match_id'], r['league'], r['bucket'], r['actual'],
                            int(r['hit_fixed']), int(r['hit_b']),
                            f"{r['cover']:.3f}",
                            int(r['hit_c']), f"{r['cover_c']:.3f}",
                            int(r['fallback_c'])])
        print(f"\n逐场明细已写入: {out}")

    # ---- 导出结果JSON(联赛黑名单等, 供filter_matches.py自动加载) ----
    data = export_result(records)
    print(f"\n回测结果已导出: {RESULT_FILE}")
    print(f"联赛黑名单({len(data['league_blacklist'])}个): "
          f"{'、'.join(data['league_blacklist']) or '无'}")


if __name__ == '__main__':
    main()
