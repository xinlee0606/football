# features.py - 机器学习特征工程模块（Step 1）
# 功能: 构建"一场比赛一行"的特征宽表, 只使用开赛前已知信息(防泄漏)
#
# 特征分组:
#   1. 亚盘(15家机构初盘+终盘): 盘口均值/分歧度/均势盘占比/盘口升降/水位漂移/让球覆盖概率
#   2. 大小球(初盘+终盘): 线均值/分歧度/线变化/大球隐含概率/期望总进球
#   3. 近10场战绩(每队): 场均积分/得失球/小球率/让球赢率/射正率/控球率, 及双队差值
#   4. 对战往绩: 场次/场均进球/池内比分占比/当前主队胜率
#   5. 联赛积分: 总排名差/总积分差/近6场积分差
#   6. 联赛先验: 该联赛历史池内命中率(按时间累积, 无未来函数)
#
# 用法:
#   from features import build_features, load_cache, save_cache
#   df = build_features(conn)     # 返回 pandas.DataFrame(含已完场标签与未完场样本)
#   save_cache(df)                # 缓存到 features_cache.csv, 训练/回测优先读缓存
import os
import re
from collections import Counter, defaultdict

import pandas as pd

from config import DB_CONFIG

# ==================== 常量 ====================
POOL8 = ['1-0', '2-0', '2-1', '3-1', '3-2', '0-0', '1-1', '2-2']
POOL8_SET = set(POOL8)

# 目标盘口范围（训练+预测均只使用这些盘口的比赛）
TARGET_HANDICAPS = ['受让半球', '半球', '受让平/半', '平/半', '平手']
TARGET_HANDICAP_NUMS = {-0.5, 0.5, -0.25, 0.25, 0.0}

# 大小球线最低阈值（初盘和终盘均需 >= 2.25，即 2/2.5）
MIN_OU_LINE = 2.25


def is_target_handicap(ah_cur_mean, ah_init_mean=None):
    """判断盘口均值是否在目标盘口范围内"""
    v = ah_cur_mean
    if v is None:
        v = ah_init_mean
    if v is None:
        return False
    return any(abs(v - t) < 0.15 for t in TARGET_HANDICAP_NUMS)


def is_valid_ou_line(ou_line_init_mean, ou_line_cur_mean=None):
    """判断大小球线是否满足最低阈值（初盘和终盘均 >= 2.25）
    一方缺失时用另一方替代，双方均缺失返回False"""
    init_v = ou_line_init_mean
    cur_v = ou_line_cur_mean
    if init_v is None:
        init_v = cur_v
    if cur_v is None:
        cur_v = init_v
    if init_v is None:
        return False
    return init_v >= MIN_OU_LINE and cur_v >= MIN_OU_LINE


def is_cup_league(league):
    """判断联赛名称是否为杯赛（含'杯'字即为杯赛）"""
    if not league:
        return False
    return '杯' in str(league)

# 元信息列(非特征列)
META_COLS = ['match_id', 'single_match', 'league', 'home_team', 'away_team',
             'match_time', 'score', 'in_pool', 'sp']

# 联赛先验平滑参数: prior = (池内命中 + W*基准) / (场数 + W)
LEAGUE_PRIOR_W = 20
LEAGUE_PRIOR_BASE = 0.55

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'features_cache.csv')

# 比分格式兼容: 数据库存'1-0'(横杠,完场页抓取), 兼容'1:0'(冒号)写法
_SCORE_PAT = re.compile(r'^\d+[:\-]\d+$')

# 盘口字符串 -> 数值映射（正数=主队让球，负数=主队受让）
_HANDICAP_MAP = {
    '受让球半': -1.5, '受球半': -1.5,
    '受让一/球半': -1.25, '受一/球半': -1.25,
    '受让一球': -1.0, '受一球': -1.0,
    '受让半/一': -0.75, '受半/一': -0.75,
    '受让半球': -0.5, '受半球': -0.5,
    '受让平/半': -0.25, '受平/半': -0.25,
    '受让平手/半球': -0.25, '受平手/半球': -0.25,
    '受让平手': 0.0, '受平手': 0.0,
    '平手': 0.0, '平/半': 0.25, '平手/半球': 0.25,
    '半球': 0.5, '半/一': 0.75, '一球': 1.0,
    '一/球半': 1.25, '球半': 1.5, '球半/两球': 1.75, '两球': 2.0,
}


# ==================== 基础工具 ====================
def handicap_to_num(h):
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


def to_float(val):
    """安全转浮点数（兼容'2/2.5'双线写法取均值, '58%'去百分号）"""
    if val is None:
        return None
    s = str(val).replace('%', '').strip()
    if not s:
        return None
    if '/' in s:
        vals = []
        for p in s.split('/'):
            try:
                vals.append(float(p.strip()))
            except (ValueError, TypeError):
                pass
        return sum(vals) / len(vals) if vals else None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def parse_score(score):
    """'1-0'/'1:0' -> (1, 0), 解析失败返回 (None, None)"""
    try:
        a, b = str(score).replace(':', '-').split('-')
        return int(a), int(b)
    except (ValueError, TypeError, AttributeError):
        return None, None


def norm_score(s):
    """'1:1' -> '1-1'"""
    return (s or '').replace(':', '-').strip()


def is_finished(score):
    """比分字段是否为有效完场比分"""
    return bool(score) and bool(_SCORE_PAT.match(str(score).strip()))


def team_similarity(name1, name2):
    """球队名称相似度（与filter_matches.py一致: 清理(中)标记 + 子串 + 字符重叠）"""
    if not name1 or not name2:
        return 0.0
    n1 = name1.replace('(中)', '').replace('（中）', '').strip()
    n2 = name2.replace('(中)', '').replace('（中）', '').strip()
    if n1 == n2:
        return 1.0
    if n1 in n2 or n2 in n1:
        return 0.9
    set1, set2 = set(n1), set(n2)
    union = set1 | set2
    if not union:
        return 0.0
    return len(set1 & set2) / len(union)


def team_side(team_name, home_team, away_team):
    """判断team_name在某场历史比赛中是主场还是客场, 返回 'home'/'away'/None"""
    sim_h = team_similarity(team_name, home_team)
    sim_a = team_similarity(team_name, away_team)
    if sim_h >= sim_a and sim_h >= 0.5:
        return 'home'
    if sim_a >= 0.5:
        return 'away'
    return None


def _mean(vals):
    return sum(vals) / len(vals) if vals else None


def _std(vals):
    if len(vals) < 2:
        return 0.0 if vals else None
    m = _mean(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


def _share_mode(vals):
    """众数占比(共识度)"""
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][1] / len(vals)


# ==================== 亚盘特征 ====================
def asian_features(rows):
    """rows: 该场比赛asian_odds表全部行(dict列表)
    返回特征dict; 无数据返回全NaN dict"""
    keys = ['asian_n', 'ah_init_mean', 'ah_init_std', 'ah_init_flat_share',
            'ah_cur_mean', 'ah_cur_std', 'ah_cur_flat_share', 'ah_move',
            'ah_move_up_share', 'cover_prob_init', 'cover_prob_cur',
            'aw_init_diff', 'aw_cur_diff', 'aw_drift']
    feats = {k: None for k in keys}

    ih, ch, moves = [], [], []
    ho_i, ao_i, ho_c, ao_c = [], [], [], []
    for r in rows:
        v_i = handicap_to_num(r['initial_handicap'])
        v_c = handicap_to_num(r['current_handicap'])
        if v_c is None:
            v_c = v_i
        if v_i is not None:
            ih.append(v_i)
        if v_c is not None:
            ch.append(v_c)
        if v_i is not None and v_c is not None:
            moves.append(v_c - v_i)
        oi, ai_ = to_float(r['initial_home_odds']), to_float(r['initial_away_odds'])
        oc, ac = to_float(r['current_home_odds']), to_float(r['current_away_odds'])
        if oi and ai_:
            ho_i.append(oi)
            ao_i.append(ai_)
        if oc and ac:
            ho_c.append(oc)
            ao_c.append(ac)

    feats['asian_n'] = len(rows)
    if ih:
        feats['ah_init_mean'] = _mean(ih)
        feats['ah_init_std'] = _std(ih)
        feats['ah_init_flat_share'] = sum(1 for v in ih if abs(v) <= 0.25) / len(ih)
    if ch:
        feats['ah_cur_mean'] = _mean(ch)
        feats['ah_cur_std'] = _std(ch)
        feats['ah_cur_flat_share'] = sum(1 for v in ch if abs(v) <= 0.25) / len(ch)
    if ih and ch:
        feats['ah_move'] = _mean(ch) - _mean(ih)
    if moves:
        feats['ah_move_up_share'] = sum(1 for m in moves if m > 0) / len(moves)
    # 让球覆盖隐含概率: P(主队赢盘) = 客水/(主水+客水) (去水归一)
    if ho_i and ao_i:
        feats['cover_prob_init'] = _mean([a / (h + a) for h, a in zip(ho_i, ao_i)])
    if ho_c and ao_c:
        feats['cover_prob_cur'] = _mean([a / (h + a) for h, a in zip(ho_c, ao_c)])
    if ho_i and ao_i:
        feats['aw_init_diff'] = _mean(ho_i) - _mean(ao_i)
    if ho_c and ao_c:
        feats['aw_cur_diff'] = _mean(ho_c) - _mean(ao_c)
    if feats['aw_init_diff'] is not None and feats['aw_cur_diff'] is not None:
        feats['aw_drift'] = feats['aw_cur_diff'] - feats['aw_init_diff']
    return feats


# ==================== 大小球特征 ====================
def ou_features(rows):
    """rows: 该场比赛over_under_odds表全部行(dict列表)"""
    keys = ['ou_n', 'ou_line_init_mean', 'ou_line_init_std', 'ou_line_cur_mean',
            'ou_line_move', 'over_prob_init', 'over_prob_cur', 'exp_total']
    feats = {k: None for k in keys}

    li, lc, lo_i, uo_i, lo_c, uo_c = [], [], [], [], [], []
    for r in rows:
        v_i = to_float(r['initial_line'])
        v_c = to_float(r['current_line'])
        if v_c is None:
            v_c = v_i
        if v_i is not None:
            li.append(v_i)
        if v_c is not None:
            lc.append(v_c)
        oi, ui = to_float(r['initial_over_odds']), to_float(r['initial_under_odds'])
        oc, uc = to_float(r['current_over_odds']), to_float(r['current_under_odds'])
        if oi and ui:
            lo_i.append(oi)
            uo_i.append(ui)
        if oc and uc:
            lo_c.append(oc)
            uo_c.append(uc)

    feats['ou_n'] = len(rows)
    if li:
        feats['ou_line_init_mean'] = _mean(li)
        feats['ou_line_init_std'] = _std(li)
    if lc:
        feats['ou_line_cur_mean'] = _mean(lc)
    if li and lc:
        feats['ou_line_move'] = _mean(lc) - _mean(li)
    # 大球隐含概率: P(大球) = 小水/(大水+小水) (去水归一)
    if lo_i and uo_i:
        feats['over_prob_init'] = _mean([u / (o + u) for o, u in zip(lo_i, uo_i)])
    if lo_c and uo_c:
        feats['over_prob_cur'] = _mean([u / (o + u) for o, u in zip(lo_c, uo_c)])
    feats['exp_total'] = feats['ou_line_cur_mean'] if feats['ou_line_cur_mean'] is not None \
        else feats['ou_line_init_mean']
    return feats


# ==================== 欧赔特征 ====================
def euro_features(rows):
    """rows: 该场比赛european_odds表全部行(dict列表)
    返回特征dict; 无数据返回全NaN dict
    隐含概率计算: P_win = (1/win_odds) / (1/win_odds + 1/draw_odds + 1/loss_odds) 去水归一"""
    keys = ['euro_n',
            'euro_win_prob_init', 'euro_draw_prob_init', 'euro_loss_prob_init',
            'euro_win_prob_cur', 'euro_draw_prob_cur', 'euro_loss_prob_cur',
            'euro_win_std_init', 'euro_win_std_cur',
            'euro_return_mean', 'euro_win_move']
    feats = {k: None for k in keys}

    pw_i, pd_i, pl_i = [], [], []  # 初盘隐含概率
    pw_c, pd_c, pl_c = [], [], []  # 即时隐含概率
    returns = []  # 返还率

    for r in rows:
        iw = to_float(r['initial_win_odds'])
        id_ = to_float(r['initial_draw_odds'])
        il = to_float(r['initial_loss_odds'])
        cw = to_float(r['current_win_odds'])
        cd = to_float(r['current_draw_odds'])
        cl = to_float(r['current_loss_odds'])

        # 初盘隐含概率
        if iw and id_ and il and iw > 1 and id_ > 1 and il > 1:
            inv_w, inv_d, inv_l = 1/iw, 1/id_, 1/il
            total = inv_w + inv_d + inv_l
            pw_i.append(inv_w / total)
            pd_i.append(inv_d / total)
            pl_i.append(inv_l / total)
            returns.append(1 / total)

        # 即时隐含概率
        if cw and cd and cl and cw > 1 and cd > 1 and cl > 1:
            inv_w, inv_d, inv_l = 1/cw, 1/cd, 1/cl
            total = inv_w + inv_d + inv_l
            pw_c.append(inv_w / total)
            pd_c.append(inv_d / total)
            pl_c.append(inv_l / total)

    n = len(rows)
    feats['euro_n'] = n
    if n == 0:
        return feats

    if pw_i:
        feats['euro_win_prob_init'] = _mean(pw_i)
        feats['euro_draw_prob_init'] = _mean(pd_i)
        feats['euro_loss_prob_init'] = _mean(pl_i)
        feats['euro_win_std_init'] = _std(pw_i)
    if pw_c:
        feats['euro_win_prob_cur'] = _mean(pw_c)
        feats['euro_draw_prob_cur'] = _mean(pd_c)
        feats['euro_loss_prob_cur'] = _mean(pl_c)
        feats['euro_win_std_cur'] = _std(pw_c)
    if returns:
        feats['euro_return_mean'] = _mean(returns)
    # 主胜概率变动(初→即)
    if pw_i and pw_c:
        feats['euro_win_move'] = _mean(pw_c) - _mean(pw_i)

    return feats


# ==================== 期望进球衍生特征 ====================
def goal_expectation(ah_feats, ou_feats):
    """由盘口+大小球线推导双队期望进球(粗略先验, 模型可自行修正)
    主队让球越多(ah_cur_mean越大)主队期望进球越高"""
    total = ou_feats.get('exp_total')
    hc = ah_feats.get('ah_cur_mean')
    if hc is None:
        hc = ah_feats.get('ah_init_mean')
    out = {'home_adv': None, 'exp_home': None, 'exp_away': None}
    if hc is not None:
        out['home_adv'] = hc      # 正数=主队让球=主队实力占优
    if total is not None:
        adv = (out['home_adv'] or 0.0) * 0.55
        out['exp_home'] = total / 2 + adv
        out['exp_away'] = total / 2 - adv
    return out


# ==================== 近10场战绩特征（含主客场拆分+加权+连续主客场） ====================
def _weighted_mean(vals, weights):
    """加权平均，vals和weights等长"""
    if not vals:
        return None
    w_sum = sum(weights)
    if w_sum == 0:
        return None
    return sum(v * w for v, w in zip(vals, weights)) / w_sum


def _consecutive_count(sides, current_side):
    """计算包含本场在内的连续主/客场数
    sides: 历史比赛的主客场序列（新在前），如 ['home','away','away','home',...]
    current_side: 本场的主客场
    返回: 连续计数（含本场）
    """
    count = 1  # 本场
    for s in sides:
        if s == current_side:
            count += 1
        else:
            break
    return count


def _recent_team_features(rows, team_name):
    """rows: 该队近10场(dict列表, 新在前), 返回该队视角的特征dict
    
    新增特性:
    1. 主客场拆分: 分别计算主场/客场战绩
    2. 指数衰减加权: 越近的比赛权重越高 (decay=0.85)
    3. 连续主客场: 包含本场的连续主场/客场数
    """
    # 原有特征（保持向后兼容）
    prefix_keys = ['rf_n', 'rf_ppg', 'rf_ppg5', 'rf_gf', 'rf_ga', 'rf_total',
                   'rf_small_rate', 'rf_under_rate', 'rf_hc_win_rate',
                   'rf_sot_rate', 'rf_poss', 'rf_home_share']
    # 主客场拆分特征
    split_keys = ['rf_home_n', 'rf_home_ppg', 'rf_home_gf', 'rf_home_ga',
                  'rf_away_n', 'rf_away_ppg', 'rf_away_gf', 'rf_away_ga']
    # 加权特征
    weighted_keys = ['rf_wppg', 'rf_wppg5', 'rf_wgf', 'rf_wga',
                     'rf_whome_ppg', 'rf_waway_ppg']
    # 连续主客场
    streak_keys = ['rf_consec_home', 'rf_consec_away']
    # 7项技术指标
    tech_keys = ['rf_shots_mean', 'rf_sot_mean', 'rf_corners_mean',
                 'rf_danger_mean', 'rf_poss_mean', 'rf_fouls_mean', 'rf_yellow_mean']
    
    all_keys = prefix_keys + split_keys + weighted_keys + streak_keys + tech_keys
    feats = {k: None for k in all_keys}

    # 解析所有比赛（去重 + 过滤无效比分）
    # 注意: rows 中的 match_id 是当前比赛的 ID，不是历史比赛的 ID
    # 用 (home_team, away_team, match_date) 作为历史比赛的唯一标识
    seen_matches = set()
    use = []
    for r in rows:
        hist_key = (r.get('home_team', ''), r.get('away_team', ''), r.get('match_date', ''))
        if hist_key in seen_matches:
            continue
        seen_matches.add(hist_key)
        
        side = team_side(team_name, r['home_team'], r['away_team'])
        if side is None:
            continue
        h, a = parse_score(r['score'])
        if h is None:
            continue
        gf, ga = (h, a) if side == 'home' else (a, h)
        use.append({'side': side, 'gf': gf, 'ga': ga, 'row': r})
    
    if not use:
        return feats

    # ---- 基础统计（简单平均，保持向后兼容） ----
    feats['rf_n'] = len(use)
    pts = [3 if u['gf'] > u['ga'] else (1 if u['gf'] == u['ga'] else 0) for u in use]
    feats['rf_ppg'] = _mean(pts)
    feats['rf_ppg5'] = _mean(pts[:5])
    feats['rf_gf'] = _mean([u['gf'] for u in use])
    feats['rf_ga'] = _mean([u['ga'] for u in use])
    totals = [u['gf'] + u['ga'] for u in use]
    feats['rf_total'] = _mean(totals)
    feats['rf_small_rate'] = sum(1 for t in totals if t <= 2) / len(totals)

    under = [u['row']['over_under_result'] for u in use if u['row']['over_under_result']]
    if under:
        feats['rf_under_rate'] = sum(1 for x in under if x == '小') / len(under)
    hc = [u['row']['handicap_result'] for u in use
          if u['row']['handicap_result'] in ('赢', '输')]
    if hc:
        feats['rf_hc_win_rate'] = sum(1 for x in hc if x == '赢') / len(hc)
    sot = []
    for u in use:
        shots = u['row']['shots']
        on_t = u['row']['shots_on_target']
        if shots and on_t is not None:
            sot.append(on_t / shots)
    if sot:
        feats['rf_sot_rate'] = _mean(sot)
    poss = [to_float(u['row']['possession']) for u in use]
    poss = [p for p in poss if p is not None]
    if poss:
        feats['rf_poss'] = _mean(poss)
    feats['rf_home_share'] = sum(1 for u in use if u['side'] == 'home') / len(use)

    # ---- 1. 主客场拆分特征 ----
    home_use = [u for u in use if u['side'] == 'home']
    away_use = [u for u in use if u['side'] == 'away']
    
    feats['rf_home_n'] = len(home_use)
    feats['rf_away_n'] = len(away_use)
    
    if home_use:
        home_pts = [3 if u['gf'] > u['ga'] else (1 if u['gf'] == u['ga'] else 0) for u in home_use]
        feats['rf_home_ppg'] = _mean(home_pts)
        feats['rf_home_gf'] = _mean([u['gf'] for u in home_use])
        feats['rf_home_ga'] = _mean([u['ga'] for u in home_use])
    
    if away_use:
        away_pts = [3 if u['gf'] > u['ga'] else (1 if u['gf'] == u['ga'] else 0) for u in away_use]
        feats['rf_away_ppg'] = _mean(away_pts)
        feats['rf_away_gf'] = _mean([u['gf'] for u in away_use])
        feats['rf_away_ga'] = _mean([u['ga'] for u in away_use])

    # ---- 2. 指数衰减加权特征 ----
    # 权重: 最近一场权重最高, decay=0.85
    # w[i] = 0.85^i, i=0 为最近一场
    decay = 0.85
    all_weights = [decay ** i for i in range(len(use))]
    
    # 加权 PPG（全部比赛）
    feats['rf_wppg'] = _weighted_mean(pts, all_weights)
    # 加权 PPG（近5场）
    feats['rf_wppg5'] = _weighted_mean(pts[:5], all_weights[:5])
    # 加权得失球
    feats['rf_wgf'] = _weighted_mean([u['gf'] for u in use], all_weights)
    feats['rf_wga'] = _weighted_mean([u['ga'] for u in use], all_weights)
    
    # 主场加权 PPG
    if home_use:
        home_indices = [i for i, u in enumerate(use) if u['side'] == 'home']
        home_w = [all_weights[i] for i in home_indices]
        home_pts_list = [3 if u['gf'] > u['ga'] else (1 if u['gf'] == u['ga'] else 0) for u in home_use]
        feats['rf_whome_ppg'] = _weighted_mean(home_pts_list, home_w)
    
    # 客场加权 PPG
    if away_use:
        away_indices = [i for i, u in enumerate(use) if u['side'] == 'away']
        away_w = [all_weights[i] for i in away_indices]
        away_pts_list = [3 if u['gf'] > u['ga'] else (1 if u['gf'] == u['ga'] else 0) for u in away_use]
        feats['rf_waway_ppg'] = _weighted_mean(away_pts_list, away_w)

    # ---- 3. 连续主客场特征 ----
    # sides 序列: 历史比赛的主客场（新在前），不含本场
    sides_sequence = [u['side'] for u in use]
    # 当前比赛的主客场由调用方通过 team_type 确定，这里用最近一场同侧的连续数
    # 注意：这里的"连续"是指历史比赛中最近的连续同侧场次数
    # 实际使用时，调用方会根据当前 team_type 选择 rf_consec_home 或 rf_consec_away
    
    # 计算最近连续主场数（从最近一场开始数）
    consec_home = 0
    for s in sides_sequence:
        if s == 'home':
            consec_home += 1
        else:
            break
    feats['rf_consec_home'] = consec_home
    
    # 计算最近连续客场数
    consec_away = 0
    for s in sides_sequence:
        if s == 'away':
            consec_away += 1
        else:
            break
    feats['rf_consec_away'] = consec_away

    # ---- 4. 7项技术指标特征(联赛均值填充缺失值) ----
    # 从rows中提取技术指标
    shots_list, corners_list, danger_list = [], [], []
    fouls_list, yellow_list = [], []
    for u in use:
        row = u['row']
        if row.get('shots') is not None:
            shots_list.append(float(row['shots']))
        if row.get('corners') is not None:
            corners_list.append(float(row['corners']))
        if row.get('dangerous_attacks') is not None:
            danger_list.append(float(row['dangerous_attacks']))
        if row.get('fouls') is not None:
            fouls_list.append(float(row['fouls']))
        if row.get('yellow_cards') is not None:
            yellow_list.append(float(row['yellow_cards']))
    
    # 联赛均值从全局变量获取(在build_features中计算)
    league_avgs = globals().get('_league_tech_avgs', {})
    
    # 计算均值,缺失时用联赛均值填充
    feats['rf_shots_mean'] = _mean(shots_list) if shots_list else league_avgs.get('shots')
    feats['rf_sot_mean'] = feats.get('rf_sot_rate')  # 已有射正率
    feats['rf_corners_mean'] = _mean(corners_list) if corners_list else league_avgs.get('corners')
    feats['rf_danger_mean'] = _mean(danger_list) if danger_list else league_avgs.get('dangerous_attacks')
    feats['rf_fouls_mean'] = _mean(fouls_list) if fouls_list else league_avgs.get('fouls')
    feats['rf_yellow_mean'] = _mean(yellow_list) if yellow_list else league_avgs.get('yellow_cards')
    # 控球率已有
    feats['rf_poss_mean'] = feats.get('rf_poss')

    return feats


# ==================== 对战往绩特征 ====================
def h2h_features(rows, home_team, away_team):
    """rows: head_to_head表该场比赛的往绩行(dict列表)"""
    keys = ['h2h_n', 'h2h_goal_avg', 'h2h_pool_share', 'h2h_home_win_rate',
            'h2h_half_goal_avg', 'h2h_corners_avg']
    feats = {k: None for k in keys}
    if not rows:
        return feats

    feats['h2h_n'] = len(rows)
    totals, pool_hit, home_wins, decided = [], 0, 0, 0
    half_goals_list, corners_list = [], []
    for r in rows:
        h, a = parse_score(r['score'])
        if h is None:
            continue
        totals.append(h + a)
        if norm_score(r['score']) in POOL8_SET:
            pool_hit += 1
        side = team_side(home_team, r['home_team'], r['away_team'])
        if side is None:
            continue
        decided += 1
        win = (h > a) if side == 'home' else (a > h)
        if win:
            home_wins += 1
        # 半场进球
        if r.get('half_score'):
            hh, ha = parse_score(r['half_score'])
            if hh is not None and ha is not None:
                half_goals_list.append(hh + ha)
        # 角球(主队视角)
        if r.get('corners'):
            try:
                c_str = str(r['corners']).replace(' ', '')
                if '-' in c_str:
                    parts = c_str.split('-')
                    c_home, c_away = int(parts[0]), int(parts[1])
                    # 根据主客场视角转换
                    if side == 'home':
                        corners_list.append(c_home + c_away)
                    else:
                        corners_list.append(c_away + c_home)
            except (ValueError, IndexError):
                pass
    if totals:
        feats['h2h_goal_avg'] = _mean(totals)
        feats['h2h_pool_share'] = pool_hit / len(totals)
    if decided:
        feats['h2h_home_win_rate'] = home_wins / decided
    if half_goals_list:
        feats['h2h_half_goal_avg'] = _mean(half_goals_list)
    if corners_list:
        feats['h2h_corners_avg'] = _mean(corners_list)
    return feats


# ==================== 数据加载 ====================
def load_raw(cursor):
    """一次性加载建特征所需的全部原始数据(按match_id分组), 避免逐场查库"""
    data = {}

    cursor.execute("""
        SELECT match_id, single_match, league, home_team, away_team,
               match_time, score, status
        FROM `match` ORDER BY match_id ASC
    """)
    data['matches'] = cursor.fetchall()

    cursor.execute("""
        SELECT match_id, initial_handicap, initial_home_odds, initial_away_odds,
               current_handicap, current_home_odds, current_away_odds
        FROM asian_odds
    """)
    asian = defaultdict(list)
    for mid, ih, iho, iao, ch, cho, cao in cursor.fetchall():
        asian[mid].append({'initial_handicap': ih, 'initial_home_odds': iho,
                           'initial_away_odds': iao, 'current_handicap': ch,
                           'current_home_odds': cho, 'current_away_odds': cao})
    data['asian'] = asian

    cursor.execute("""
        SELECT match_id, initial_line, initial_over_odds, initial_under_odds,
               current_line, current_over_odds, current_under_odds
        FROM over_under_odds
    """)
    ou = defaultdict(list)
    for mid, il, ioo, iuo, cl, coo, cuo in cursor.fetchall():
        ou[mid].append({'initial_line': il, 'initial_over_odds': ioo,
                        'initial_under_odds': iuo, 'current_line': cl,
                        'current_over_odds': coo, 'current_under_odds': cuo})
    data['ou'] = ou

    cursor.execute("""
        SELECT match_id, initial_win_odds, initial_draw_odds, initial_loss_odds,
               current_win_odds, current_draw_odds, current_loss_odds
        FROM european_odds
    """)
    euro = defaultdict(list)
    for mid, iw, id_, il, cw, cd, cl in cursor.fetchall():
        euro[mid].append({'initial_win_odds': iw, 'initial_draw_odds': id_,
                          'initial_loss_odds': il, 'current_win_odds': cw,
                          'current_draw_odds': cd, 'current_loss_odds': cl})
    data['euro'] = euro

    cursor.execute("""
        SELECT match_id, team_type, team_name, home_team, away_team, score,
               handicap_result, over_under_result, shots, shots_on_target, possession,
               corners, dangerous_attacks, fouls, yellow_cards,
               match_date
        FROM recent_form
        WHERE is_current_match = 0
        ORDER BY match_id, match_date DESC
    """)
    recent = defaultdict(lambda: {'home': [], 'away': []})
    for mid, tt, tname, ht, at, sc, hcr, our, shots, sot, poss, corners, danger, fouls, yellow, mdate in cursor.fetchall():
        if tt in ('home', 'away'):
            recent[mid][tt].append({'team_name': tname, 'home_team': ht,
                                    'away_team': at, 'score': sc,
                                    'handicap_result': hcr, 'over_under_result': our,
                                    'shots': shots, 'shots_on_target': sot,
                                    'possession': poss, 'corners': corners,
                                    'dangerous_attacks': danger, 'fouls': fouls,
                                    'yellow_cards': yellow, 'match_date': mdate})
    data['recent'] = recent

    cursor.execute("""
        SELECT match_id, home_team, away_team, score, half_score, corners FROM head_to_head
    """)
    h2h = defaultdict(list)
    for mid, ht, at, sc, hs, co in cursor.fetchall():
        h2h[mid].append({'home_team': ht, 'away_team': at, 'score': sc,
                         'half_score': hs, 'corners': co})
    data['h2h'] = h2h

    cursor.execute("""
        SELECT match_id, team_type, scope, points, ranking FROM league_standings
    """)
    standings = defaultdict(dict)
    for mid, tt, scope, points, ranking in cursor.fetchall():
        standings[mid][(tt, scope)] = {'points': points, 'ranking': ranking}
    data['standings'] = standings

    cursor.execute("""
        SELECT match_id, sp_value FROM result_sp
        WHERE match_id IS NOT NULL AND match_id <> ''
    """)
    data['sp'] = {mid: float(sp) if sp is not None else None
                  for mid, sp in cursor.fetchall()}
    return data


# ==================== 宽表构建 ====================
def _standings_features(st_map):
    """st_map: {(team_type, scope): {points, ranking}}"""
    feats = {'rank_diff': None, 'points_diff': None, 'recent6_points_diff': None}
    hr = st_map.get(('home', 'total'), {}).get('ranking')
    ar = st_map.get(('away', 'total'), {}).get('ranking')
    if hr is not None and ar is not None:
        feats['rank_diff'] = hr - ar
    hp = st_map.get(('home', 'total'), {}).get('points')
    ap = st_map.get(('away', 'total'), {}).get('points')
    if hp is not None and ap is not None:
        feats['points_diff'] = hp - ap
    hp6 = st_map.get(('home', 'recent6'), {}).get('points')
    ap6 = st_map.get(('away', 'recent6'), {}).get('points')
    if hp6 is not None and ap6 is not None:
        feats['recent6_points_diff'] = hp6 - ap6
    return feats


def build_features(conn_or_cursor):
    """构建全量特征宽表(含已完场标签in_pool与未完场样本), 返回DataFrame
    按match_id升序(match_id前8位为日期, 天然时间序)"""
    cursor = conn_or_cursor
    own_cursor = False
    if hasattr(conn_or_cursor, 'cursor'):
        cursor = conn_or_cursor.cursor()
        own_cursor = True
    
    # 计算联赛技术指标均值(用于缺失值填充)
    cursor.execute("""
        SELECT league, 
               AVG(shots) as avg_shots, AVG(corners) as avg_corners,
               AVG(dangerous_attacks) as avg_danger, AVG(fouls) as avg_fouls,
               AVG(yellow_cards) as avg_yellow
        FROM recent_form
        WHERE is_current_match = 0 AND shots IS NOT NULL
        GROUP BY league
    """)
    global _league_tech_avgs
    _league_tech_avgs = {}
    for league, avg_s, avg_c, avg_d, avg_f, avg_y in cursor.fetchall():
        _league_tech_avgs[league] = {
            'shots': avg_s, 'corners': avg_c, 'dangerous_attacks': avg_d,
            'fouls': avg_f, 'yellow_cards': avg_y
        }
    
    try:
        raw = load_raw(cursor)
    finally:
        if own_cursor:
            cursor.close()

    rows = []
    league_cum = defaultdict(lambda: [0, 0])   # 联赛 -> [池内命中数, 完场数]
    for mid, single, league, ht, at, mt, score, _status in raw['matches']:
        ah = asian_features(raw['asian'].get(mid, []))
        ou = ou_features(raw['ou'].get(mid, []))
        euro = euro_features(raw['euro'].get(mid, []))
        goal = goal_expectation(ah, ou)

        rf_h = _recent_team_features(raw['recent'].get(mid, {}).get('home', []), ht)
        rf_a = _recent_team_features(raw['recent'].get(mid, {}).get('away', []), at)
        h2h = h2h_features(raw['h2h'].get(mid, []), ht, at)
        st = _standings_features(raw['standings'].get(mid, {}))

        # 联赛先验(只用该场之前的完场数据, 无未来函数)
        hits, n = league_cum.get(league, [0, 0])
        league_prior = (hits + LEAGUE_PRIOR_W * LEAGUE_PRIOR_BASE) / (n + LEAGUE_PRIOR_W)

        row = {'match_id': mid, 'single_match': single, 'league': league,
               'home_team': ht, 'away_team': at,
               'match_time': mt, 'score': norm_score(score) if is_finished(score) else None,
               'in_pool': None, 'sp': raw['sp'].get(mid)}
        row.update(ah)
        row.update(ou)
        row.update(euro)
        row.update(goal)
        for k, v in rf_h.items():
            row['h_' + k] = v
        for k, v in rf_a.items():
            row['a_' + k] = v
        # 双队差值/组合特征
        if rf_h['rf_gf'] is not None and rf_a['rf_gf'] is not None:
            row['gf_diff'] = rf_h['rf_gf'] - rf_a['rf_gf']
            row['ga_diff'] = rf_h['rf_ga'] - rf_a['rf_ga']
            row['ppg_diff'] = rf_h['rf_ppg'] - rf_a['rf_ppg']
        if rf_h['rf_small_rate'] is not None and rf_a['rf_small_rate'] is not None:
            row['small_rate_avg'] = (rf_h['rf_small_rate'] + rf_a['rf_small_rate']) / 2
        if rf_h['rf_total'] is not None and rf_a['rf_total'] is not None:
            row['recent_total_goals'] = (rf_h['rf_total'] + rf_a['rf_total']) / 2
        
        # 新增交叉特征：主客场拆分差值
        if rf_h.get('rf_home_ppg') is not None and rf_a.get('rf_away_ppg') is not None:
            # 主队主场 PPG vs 客队客场 PPG（核心对比）
            row['home_ppg_vs_away_ppg'] = rf_h['rf_home_ppg'] - rf_a['rf_away_ppg']
        if rf_h.get('rf_away_ppg') is not None and rf_a.get('rf_home_ppg') is not None:
            # 主队客场 PPG vs 客队主场 PPG
            row['away_ppg_vs_home_ppg'] = rf_h['rf_away_ppg'] - rf_a['rf_home_ppg']
        # 主场优势量化
        if rf_h.get('rf_home_ppg') is not None and rf_h.get('rf_away_ppg') is not None:
            row['home_advantage'] = rf_h['rf_home_ppg'] - rf_h['rf_away_ppg']
        if rf_a.get('rf_home_ppg') is not None and rf_a.get('rf_away_ppg') is not None:
            row['away_disadvantage'] = rf_a['rf_away_ppg'] - rf_a['rf_home_ppg']
        
        # 新增交叉特征：加权 PPG 差值
        if rf_h.get('rf_wppg') is not None and rf_a.get('rf_wppg') is not None:
            row['wppg_diff'] = rf_h['rf_wppg'] - rf_a['rf_wppg']
        if rf_h.get('rf_wppg5') is not None and rf_a.get('rf_wppg5') is not None:
            row['wppg5_diff'] = rf_h['rf_wppg5'] - rf_a['rf_wppg5']
        if rf_h.get('rf_wgf') is not None and rf_a.get('rf_wgf') is not None:
            row['wgf_diff'] = rf_h['rf_wgf'] - rf_a['rf_wgf']
        if rf_h.get('rf_wga') is not None and rf_a.get('rf_wga') is not None:
            row['wga_diff'] = rf_h['rf_wga'] - rf_a['rf_wga']
        
        # 新增交叉特征：状态趋势（加权 PPG - 简单 PPG，正值=近期状态上升）
        if rf_h.get('rf_wppg') is not None and rf_h.get('rf_ppg') is not None:
            row['h_form_trend'] = rf_h['rf_wppg'] - rf_h['rf_ppg']
        if rf_a.get('rf_wppg') is not None and rf_a.get('rf_ppg') is not None:
            row['a_form_trend'] = rf_a['rf_wppg'] - rf_a['rf_ppg']
        row.update(h2h)
        row.update(st)
        row['league_pool_prior'] = league_prior
        rows.append(row)

        # 本场完场后纳入联赛累积(供后续场次使用)
        if is_finished(score):
            league_cum[league][1] += 1
            if norm_score(score) in POOL8_SET:
                row['in_pool'] = 1
                league_cum[league][0] += 1
            else:
                row['in_pool'] = 0

    df = pd.DataFrame(rows)
    if not df.empty:
        df['in_pool'] = df['in_pool'].astype('float64')
        df['sp'] = df['sp'].astype('float64')
    return df


# 训练/回测流程额外添加的标签与辅助列, 绝不能当特征(会造成数据泄漏)
_NON_FEATURE_COLS = set(META_COLS) | {'y', 'home_goals', 'away_goals'}


def feature_columns(df):
    """返回特征列名(排除元信息列与标签/辅助列)"""
    return [c for c in df.columns if c not in _NON_FEATURE_COLS]


# ==================== 缓存 ====================
def save_cache(df, path=CACHE_FILE):
    df.to_csv(path, index=False, encoding='utf-8')
    return path


def load_cache(path=CACHE_FILE):
    """读取缓存宽表; 不存在返回None"""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, dtype={'match_id': str})
    return df


def load_or_build(conn=None, rebuild=False, verbose=True):
    """优先读缓存; rebuild=True或缓存不存在时重新构建并写缓存"""
    if not rebuild:
        df = load_cache()
        if df is not None:
            if verbose:
                print(f"✓ 已加载特征缓存: {CACHE_FILE} ({len(df)}场)")
            return df
    import pymysql
    own = conn is None
    if own:
        conn = pymysql.connect(**DB_CONFIG)
    if verbose:
        print("构建特征宽表(全量查库, 请稍候)...")
    df = build_features(conn)
    if own:
        conn.close()
    save_cache(df)
    if verbose:
        print(f"✓ 特征宽表构建完成: {len(df)}场, 已缓存到 {CACHE_FILE}")
    return df
