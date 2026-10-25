# predict_scores.py - 六比分预测模块
# 用法: 被 filter_matches.py 调用, predict_six_scores(cursor, match_id, prediction)
# 框架: 定方向(胜平负) -> 定进球数(大小球) -> 锁定6个比分
#
# 1. 定方向: 以 filter_matches.py 预测模块输出的 prediction(胜平/平负) 为主轴,
#    结合即时亚盘主线盘口修正:
#      - 主队受让(盘口为负) -> 客强主弱, 方向改为 [客胜, 平局, 主胜]
#      - 均势盘(平手/平手半球) -> 重防平局, 平局提到方向首位
#      - 主队让球(盘口>=半球) -> 维持预测方向
# 2. 定进球数: 大小球即时主线 <=2.5 判小球, >=3 判大球,
#    中间线(如2/2.5)由两队近10场场均总进球决定; 无盘口线时直接用场均总进球判断
# 3. 锁定6个比分: 优先"历史频率Top6"(按即时亚盘主线盘口档位分组统计已完场比赛
#    比分频率, 回测命中率优于固定池), 具体选分模式由POOL_MODE控制:
#    'pool8': 重点档位(平手/平半/半球)从固定8比分池内按频率动态选6个
#             (回测: 命中率56.5%与无条件Top6持平, 命中均SP更高约12.1);
#    'mixed': 方向内频率Top4 + 档位其余最高频2个补齐,
#             既保证多数比分与预测方向一致又把覆盖率损失压到最小;
#    样本不足时回退固定池3/2/1分配,
#    小球模式下优先总进球<=2的比分, 放弃2:2等大比分选项
from collections import Counter

# ==================== 配置 ====================
SCORE_POOLS = {
    '主胜': ['1-0', '2-0', '2-1', '3-1'],
    '平局': ['0-0', '1-1', '2-2'],
    '客胜': ['0-1', '0-2', '1-2'],
}
SCORE_ALLOC = [3, 2, 1]      # 三个方向的比分名额分配（按方向优先级，固定池回退用）
DIR_TOP_N = 4                # 混合选分: 方向内频率Top N个比分
FILL_N = 2                   # 混合选分: 方向外其余比分中最高频补齐名额
SMALL_LINE = 2.5             # 大小球线 <= 此值直接判小球
BIG_LINE = 3.0               # 大小球线 >= 此值直接判大球
RECENT_N_GOALS = 10          # 场均总进球统计取近N场
AVG_TOTAL_SMALL = 2.25       # 两队近N场场均总进球 <= 此值判小球

# 历史频率Top6配置（回测: 频率法命中率优于固定池）
ENABLE_FREQ_POOL = True      # 优先使用历史频率Top6
FREQ_MIN_HISTORY = 50        # 同档位最少历史样本数(不足回退全体样本)

# 选分模式: 'pool8' = 重点档位8比分池内按频率动态选6;
#           'mixed' = 方向内Top4+其余最高频2个补齐
POOL_MODE = 'pool8'
POOL8_SCORES = ['1-1', '1-0', '2-0', '2-1', '0-0', '0-1', '2-2', '3-1']
POOL8_BUCKETS = ['平手', '平/半', '半球']   # 仅这些档位启用8池(其余档位仍走mixed)

# 即时亚盘数值 -> 盘口档位（与完场页盘口page_handicap分组对齐）
_NUM_TO_BUCKET = {
    0.0: '平手', 0.25: '平/半', -0.25: '受让平/半',
    0.5: '半球', -0.5: '受让半球',
    0.75: '半/一', -0.75: '受让半/一',
    1.0: '一球', -1.0: '受让一球',
}
# 档位 -> 完场页盘口(page_handicap)原始字符串集合
_BUCKET_GROUPS = {
    '平手': ['平手', '受平手'],
    '平/半': ['平/半', '平手/半球'],
    '受让平/半': ['受让平/半', '受平/半', '受平手/半球'],
    '半球': ['半球'], '受让半球': ['受让半球'],
    '半/一': ['半/一'], '受让半/一': ['受让半/一'],
    '一球': ['一球'], '受让一球': ['受让一球'],
}

# 盘口字符串 -> 数值映射（正数=主队让球，负数=客队让球/主队受让）
_HANDICAP_MAP = {
    '受球半': -1.5, '受一/球半': -1.25, '受一球': -1.0, '受半/一': -0.75,
    '受半球': -0.5, '受平/半': -0.25, '受平手/半球': -0.25, '受平手': 0.0,
    '平手': 0.0, '平/半': 0.25, '平手/半球': 0.25,
    '半球': 0.5, '半/一': 0.75, '一球': 1.0,
    '一/球半': 1.25, '球半': 1.5, '球半/两球': 1.75, '两球': 2.0,
}


# ==================== 工具函数 ====================
def _handicap_to_num(h):
    """盘口字符串转数值，无法识别返回None"""
    if h is None:
        return None
    h = str(h).strip()
    if h in _HANDICAP_MAP:
        return _HANDICAP_MAP[h]
    try:
        return float(h)
    except (ValueError, TypeError):
        return None


def _to_float(val):
    """安全转浮点数（兼容'2/2.5'这类双线写法取均值）"""
    if val is None:
        return None
    s = str(val).strip()
    if '/' in s:
        parts = [p for p in s.split('/') if p.strip()]
        vals = []
        for p in parts:
            try:
                vals.append(float(p))
            except (ValueError, TypeError):
                pass
        return sum(vals) / len(vals) if vals else None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _parse_score(score):
    """'1-0'/'1:0' -> (1, 0), 解析失败返回 (None, None)"""
    try:
        a, b = str(score).replace(':', '-').split('-')
        return int(a), int(b)
    except (ValueError, TypeError, AttributeError):
        return None, None


def _total_goals(score):
    x, y = _parse_score(score)
    return x + y if x is not None else None


# ==================== 数据获取 ====================
def _main_current_handicap(cursor, match_id):
    """即时亚盘主线盘口数值 + 均势盘(平手/平手半球)机构数"""
    cursor.execute("""
        SELECT current_handicap, initial_handicap FROM asian_odds WHERE match_id = %s
    """, (match_id,))
    cnt = Counter()
    for curr_h, init_h in cursor.fetchall():
        v = _handicap_to_num(curr_h if curr_h is not None else init_h)
        if v is not None:
            cnt[v] += 1
    if not cnt:
        return None, 0
    main = cnt.most_common(1)[0][0]
    flat_count = sum(c for v, c in cnt.items() if abs(v) <= 0.25)
    return main, flat_count


def _main_ou_line(cursor, match_id):
    """大小球即时主线数值（众数），无数据返回None"""
    cursor.execute("""
        SELECT current_line, initial_line FROM over_under_odds WHERE match_id = %s
    """, (match_id,))
    cnt = Counter()
    for curr_l, init_l in cursor.fetchall():
        v = _to_float(curr_l if curr_l is not None else init_l)
        if v is not None:
            cnt[v] += 1
    return cnt.most_common(1)[0][0] if cnt else None


def _recent_avg_total(cursor, match_id):
    """两队近RECENT_N_GOALS场各自的场均总进球，返回 (主, 客)；无数据返回 (None, None)"""
    avgs = {}
    for tt in ['home', 'away']:
        cursor.execute("""
            SELECT score FROM recent_form
            WHERE match_id = %s AND team_type = %s AND is_current_match = 0
            ORDER BY match_date DESC LIMIT %s
        """, (match_id, tt, RECENT_N_GOALS))
        totals = []
        for (score,) in cursor.fetchall():
            t = _total_goals(score)
            if t is not None:
                totals.append(t)
        avgs[tt] = sum(totals) / len(totals) if totals else None
    return avgs['home'], avgs['away']


# ==================== 第一步: 定方向 ====================
def _determine_direction(cursor, match_id, prediction):
    """返回 (方向优先级列表, 理由列表)"""
    main_hc, flat_count = _main_current_handicap(cursor, match_id)

    if prediction == '胜平':
        dirs = ['主胜', '平局', '客胜']
    elif prediction == '平负':
        dirs = ['客胜', '平局', '主胜']
    else:
        dirs = ['平局', '主胜', '客胜']

    reasons = []
    if prediction:
        reasons.append(f"模型预测{prediction}")
    if main_hc is not None:
        if main_hc < 0:
            dirs = ['客胜', '平局', '主胜']
            reasons.append(f"即时盘主队受让({main_hc:+g}), 客队占优")
        elif abs(main_hc) <= 0.25:
            if dirs[0] != '平局':
                dirs.remove('平局')
                dirs.insert(0, '平局')
            reasons.append(f"即时均势盘({flat_count}家平手/平半), 重防平局")
        else:
            reasons.append(f"即时盘主队让{main_hc:g}球")
    if not reasons:
        reasons.append("无盘口数据, 按均衡格局处理")
    return dirs, reasons


# ==================== 第二步: 定进球数 ====================
def _determine_goals(cursor, match_id):
    """返回 (是否小球, 理由列表)"""
    line = _main_ou_line(cursor, match_id)
    h_avg, a_avg = _recent_avg_total(cursor, match_id)

    reasons = []
    if line is not None:
        if line <= SMALL_LINE:
            small = True
            reasons.append(f"大小球主线{line:g}≤{SMALL_LINE:g}, 小球")
        elif line >= BIG_LINE:
            small = False
            reasons.append(f"大小球主线{line:g}≥{BIG_LINE:g}, 大球")
        else:
            # 中间线由近期场均总进球裁决
            if h_avg is not None and a_avg is not None:
                small = (h_avg + a_avg) / 2 <= AVG_TOTAL_SMALL
                reasons.append(f"大小球主线{line:g}, 近{RECENT_N_GOALS}场场均总进球"
                               f"主{h_avg:.2f}/客{a_avg:.2f} -> {'小球' if small else '大球'}")
            else:
                small = True
                reasons.append(f"大小球主线{line:g}, 无近期数据按小球处理")
    else:
        if h_avg is not None and a_avg is not None:
            small = (h_avg + a_avg) / 2 <= AVG_TOTAL_SMALL
            reasons.append(f"无大小球数据, 近{RECENT_N_GOALS}场场均总进球"
                           f"主{h_avg:.2f}/客{a_avg:.2f} -> {'小球' if small else '大球'}")
        else:
            small = True
            reasons.append("无大小球及近期数据, 默认小球")
    return small, reasons


# ==================== 第三步: 锁定6个比分 ====================
def _pick_scores(dirs, small):
    """按方向优先级分配名额抽取6个比分, 小球模式优先总进球<=2"""
    cands = []
    for d, quota in zip(dirs, SCORE_ALLOC):
        pool = SCORE_POOLS[d]
        if small:
            pool = sorted(pool, key=lambda s: _total_goals(s))
        cands.extend(pool[:quota + 1])  # 每方向多取1个作候补

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
    # 不足6个时用剩余比分池补齐
    if len(picks) < 6:
        for d in dirs:
            for s in SCORE_POOLS[d]:
                if s not in picks:
                    picks.append(s)
                if len(picks) >= 6:
                    break
            if len(picks) >= 6:
                break
    return picks[:6]


# ==================== 历史频率Top6 ====================
def _bucket_freq(cursor, bucket):
    """同盘口档位下已完场比赛的比分频率(键归一为'1-1'格式)
    返回: (Counter, 总场数, 样本来源说明); 样本<FREQ_MIN_HISTORY回退全体
    """
    group = _BUCKET_GROUPS.get(bucket)
    if not group:
        return None, 0, ''
    ph = ','.join(['%s'] * len(group))
    cursor.execute(f"""
        SELECT score, COUNT(*) FROM `match`
        WHERE status = '完' AND score IS NOT NULL AND score <> ''
          AND page_handicap IN ({ph})
        GROUP BY score ORDER BY COUNT(*) DESC
    """, group)
    rows = cursor.fetchall()
    total = sum(c for _, c in rows)
    source = f"{bucket}档{total}场"
    if total < FREQ_MIN_HISTORY:   # 样本不足回退全体
        cursor.execute("""
            SELECT score, COUNT(*) FROM `match`
            WHERE status = '完' AND score IS NOT NULL AND score <> ''
            GROUP BY score ORDER BY COUNT(*) DESC
        """)
        rows = cursor.fetchall()
        total = sum(c for _, c in rows)
        source = f"全体{total}场({bucket}档样本不足{FREQ_MIN_HISTORY})"
    if total == 0:
        return None, 0, ''
    return Counter({s.replace(':', '-'): c for s, c in rows}), total, source


def _pool_pick6(cnt, total):
    """8比分池选6: 池内按历史频率取最高的6个(被裁掉的2个随频率动态漂移)
    返回: (比分列表, 覆盖概率)"""
    picks = sorted(POOL8_SCORES, key=lambda s: -cnt.get(s, 0))[:6]
    cover = sum(cnt.get(s, 0) for s in picks) / total
    return picks, cover


def _mixed_top6(cnt, total, cursor, match_id, prediction):
    """混合选分: 方向内(方向优先级前两位, 如胜平->平局+主胜)频率Top DIR_TOP_N个
    + 档位其余比分中频率最高FILL_N个补齐; 无预测方向时返回无条件Top6
    保证至少DIR_TOP_N个比分与预测方向一致, 覆盖率损失压到最小
    返回: (比分列表, 覆盖概率, 模式说明)"""
    if not prediction:
        top = cnt.most_common(6)
        return [s for s, _ in top], sum(c for _, c in top) / total, ''
    dirs, _ = _determine_direction(cursor, match_id, prediction)
    dir_set = set(dirs[:2])
    dir_pool = [s for d in dir_set for s in SCORE_POOLS[d]]
    picks = sorted(dir_pool, key=lambda s: -cnt.get(s, 0))[:DIR_TOP_N]
    picked = set(picks)
    for s, _c in cnt.most_common():
        if s in picked:
            continue
        picks.append(s)
        picked.add(s)
        if len(picks) >= DIR_TOP_N + FILL_N:
            break
    cover = sum(cnt.get(s, 0) for s in picks) / total
    return picks[:6], cover, f" (方向{'+'.join(dirs[:2])})"


def _freq_top_scores(cursor, bucket):
    """同盘口档位无条件历史频率Top6(供get_freq_coverage等复用)
    返回: (比分列表['1-0',...], 覆盖概率, 样本量, 样本来源说明)
    """
    cnt, total, source = _bucket_freq(cursor, bucket)
    if not cnt:
        return [], 0.0, 0, ''
    top = cnt.most_common(6)
    cover = sum(c for _, c in top) / total
    return [s for s, _ in top], cover, total, source


def get_freq_coverage(cursor, match_id):
    """该比赛即时主线盘口档位对应的历史频率Top6覆盖概率
    返回: (档位, 覆盖概率, 样本量, 来源说明)
          无即时盘口/档位不在映射内时返回 (None, None, 0, '')
    """
    main_hc, _flat = _main_current_handicap(cursor, match_id)
    bucket = _NUM_TO_BUCKET.get(main_hc)
    if not bucket:
        return None, None, 0, ''
    _scores, cover, n, source = _freq_top_scores(cursor, bucket)
    return bucket, cover, n, source


# ==================== 对外接口 ====================
def predict_six_scores(cursor, match_id, prediction=None):
    """预测6个比分
    参数:
        cursor: 数据库游标
        match_id: 比赛编号
        prediction: filter_matches.predict_match 的预测结果('胜平'/'平负'), 可为None
    返回:
        (6个比分列表['1-0',...], 推导理由字符串)
    """
    if ENABLE_FREQ_POOL:
        main_hc, _flat = _main_current_handicap(cursor, match_id)
        bucket = _NUM_TO_BUCKET.get(main_hc)
        if bucket:
            cnt, total, source = _bucket_freq(cursor, bucket)
            if cnt:
                if POOL_MODE == 'pool8' and bucket in POOL8_BUCKETS:
                    scores, cover = _pool_pick6(cnt, total)
                    dropped = format_scores([s for s in POOL8_SCORES
                                             if s not in scores])
                    reason = (f"8比分池选6 ({source}, 覆盖概率{cover:.0%}, "
                              f"裁掉{dropped}) [即时主线{main_hc:+g} -> {bucket}]")
                    return scores, reason
                scores, cover, mode = _mixed_top6(cnt, total, cursor, match_id,
                                                  prediction)
                reason = (f"历史频率Top6{mode} ({source}, 覆盖概率{cover:.0%}) "
                          f"[即时主线{main_hc:+g} -> {bucket}]")
                return scores, reason

    dirs, dir_reasons = _determine_direction(cursor, match_id, prediction)
    small, goal_reasons = _determine_goals(cursor, match_id)
    scores = _pick_scores(dirs, small)
    reason = "方向: " + " > ".join(dirs) + " (" + "; ".join(dir_reasons) + ")" + \
             " | 进球: " + "; ".join(goal_reasons)
    return scores, reason


def format_scores(scores, sep='、'):
    """'1-0' -> '1:0' 展示格式"""
    return sep.join(s.replace('-', ':') for s in scores)


if __name__ == '__main__':
    # 独立调试: python predict_scores.py <match_id>
    import sys
    import pymysql
    from config import DB_CONFIG

    mid = sys.argv[1] if len(sys.argv) > 1 else None
    if not mid:
        print("用法: python predict_scores.py <match_id>")
        sys.exit(1)

    conn = pymysql.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT home_team, away_team, prediction FROM `match` WHERE match_id = %s", (mid,))
    row = cur.fetchone()
    if not row:
        print(f"未找到比赛 {mid}")
    else:
        ht, at, pred = row
        sc, rs = predict_six_scores(cur, mid, pred)
        print(f"{ht} vs {at}")
        print(f"预测6个比分: {format_scores(sc)}")
        print(rs)
    cur.close()
    conn.close()
