# features_upside.py - 机器学习特征工程模块（Step 1）
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
#   from features_upside import build_features_upside, load_cache, save_cache
#   df = build_features_upside(conn)     # 返回 pandas.DataFrame(含已完场标签与未完场样本)
#   save_cache(df)                # 缓存到 features_upside_cache.csv, 训练/回测优先读缓存
import os
import re
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count

import pandas as pd

from config import DB_CONFIG

# ==================== 常量 ====================
OVER = ['1-0', '2-0', '2-1', '3-1', '3-2', '0-0', '1-1', '2-2']
OVER_SET = set(OVER)

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
                          'features_upside_cache.csv')

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
def asian_features_upside(rows):
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
def ou_features_upside(rows):
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
def euro_features_upside(rows):
    """rows: 该场比赛european_odds表数据
    支持两种格式:
      1. 旧格式: 逐行dict列表 [{initial_win_odds, ...}, ...]
      2. 新格式(SQL预聚合): [{avg_iw, std_iw, avg_id, ..., count}] 单元素列表
    返回特征dict; 无数据返回全None dict
    隐含概率计算: P_win = (1/win_odds) / (1/win_odds + 1/draw_odds + 1/loss_odds) 去水归一"""
    keys = ['euro_n',
            'euro_win_prob_init', 'euro_draw_prob_init', 'euro_loss_prob_init',
            'euro_win_prob_cur', 'euro_draw_prob_cur', 'euro_loss_prob_cur',
            'euro_win_std_init', 'euro_win_std_cur',
            'euro_return_mean', 'euro_win_move']
    feats = {k: None for k in keys}

    if not rows:
        return feats

    # 检测是否为SQL预聚合格式
    if 'avg_iw' in rows[0]:
        agg = rows[0]
        feats['euro_n'] = agg.get('count', 1)
        iw = to_float(agg.get('avg_iw'))
        id_ = to_float(agg.get('avg_id'))
        il = to_float(agg.get('avg_il'))
        cw = to_float(agg.get('avg_cw'))
        cd = to_float(agg.get('avg_cd'))
        cl = to_float(agg.get('avg_cl'))
        if iw and id_ and il and iw > 1 and id_ > 1 and il > 1:
            inv_w, inv_d, inv_l = 1/iw, 1/id_, 1/il
            total = inv_w + inv_d + inv_l
            feats['euro_win_prob_init'] = inv_w / total
            feats['euro_draw_prob_init'] = inv_d / total
            feats['euro_loss_prob_init'] = inv_l / total
            feats['euro_return_mean'] = 1 / total
        if cw and cd and cl and cw > 1 and cd > 1 and cl > 1:
            inv_w, inv_d, inv_l = 1/cw, 1/cd, 1/cl
            total = inv_w + inv_d + inv_l
            feats['euro_win_prob_cur'] = inv_w / total
            feats['euro_draw_prob_cur'] = inv_d / total
            feats['euro_loss_prob_cur'] = inv_l / total
        feats['euro_win_std_init'] = to_float(agg.get('std_iw'))
        feats['euro_win_std_cur'] = to_float(agg.get('std_cw'))
        if feats['euro_win_prob_init'] is not None and feats['euro_win_prob_cur'] is not None:
            feats['euro_win_move'] = feats['euro_win_prob_cur'] - feats['euro_win_prob_init']
        return feats

    # 旧格式: 逐行处理
    pw_i, pd_i, pl_i = [], [], []
    pw_c, pd_c, pl_c = [], [], []
    returns = []

    for r in rows:
        iw = to_float(r['initial_win_odds'])
        id_ = to_float(r['initial_draw_odds'])
        il = to_float(r['initial_loss_odds'])
        cw = to_float(r['current_win_odds'])
        cd = to_float(r['current_draw_odds'])
        cl = to_float(r['current_loss_odds'])

        if iw and id_ and il and iw > 1 and id_ > 1 and il > 1:
            inv_w, inv_d, inv_l = 1/iw, 1/id_, 1/il
            total = inv_w + inv_d + inv_l
            pw_i.append(inv_w / total)
            pd_i.append(inv_d / total)
            pl_i.append(inv_l / total)
            returns.append(1 / total)

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


def _recent_team_features_upside(rows, team_name):
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
                  'rf_away_n', 'rf_away_ppg', 'rf_away_gf', 'rf_away_ga',
                  'rf_home_hc_win_rate', 'rf_away_hc_win_rate']
    # 加权特征
    weighted_keys = ['rf_wppg', 'rf_wppg5', 'rf_wgf', 'rf_wga',
                     'rf_whome_ppg', 'rf_waway_ppg']
    # 连续主客场
    streak_keys = ['rf_consec_home', 'rf_consec_away']
    # 技术统计特征
    tech_keys = ['rf_shots_avg', 'rf_sot_avg', 'rf_corners_avg',
                 'rf_da_avg', 'rf_attacks_avg', 'rf_fouls_avg',
                 'rf_da_rate', 'rf_pass_rate']
    
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

    # ---- 技术统计特征 ----
    # 场均射门
    shots_list = [u['row']['shots'] for u in use if u['row'].get('shots') is not None]
    if shots_list:
        feats['rf_shots_avg'] = _mean(shots_list)
    # 场均射正
    sot_list = [u['row']['shots_on_target'] for u in use if u['row'].get('shots_on_target') is not None]
    if sot_list:
        feats['rf_sot_avg'] = _mean(sot_list)
    # 场均角球
    corners_list = [u['row']['corners'] for u in use if u['row'].get('corners') is not None]
    if corners_list:
        feats['rf_corners_avg'] = _mean(corners_list)
    # 场均危险进攻
    da_list = [u['row']['dangerous_attacks'] for u in use if u['row'].get('dangerous_attacks') is not None]
    if da_list:
        feats['rf_da_avg'] = _mean(da_list)
    # 场均进攻
    attacks_list = [u['row']['attacks'] for u in use if u['row'].get('attacks') is not None]
    if attacks_list:
        feats['rf_attacks_avg'] = _mean(attacks_list)
    # 场均犯规
    fouls_list = [u['row']['fouls'] for u in use if u['row'].get('fouls') is not None]
    if fouls_list:
        feats['rf_fouls_avg'] = _mean(fouls_list)
    # 危险进攻率 (dangerous_attacks / attacks)
    da_rates = []
    for u in use:
        da = u['row'].get('dangerous_attacks')
        att = u['row'].get('attacks')
        if da is not None and att is not None and att > 0:
            da_rates.append(da / att)
    if da_rates:
        feats['rf_da_rate'] = _mean(da_rates)
    # 传球成功率
    pass_rates = [to_float(u['row']['pass_success_rate']) for u in use if u['row'].get('pass_success_rate') is not None]
    pass_rates = [p for p in pass_rates if p is not None]
    if pass_rates:
        feats['rf_pass_rate'] = _mean(pass_rates)

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
        # P2: 主场赢盘率
        home_hc = [u['row']['handicap_result'] for u in home_use 
                   if u['row'].get('handicap_result') in ('赢', '输')]
        if home_hc:
            feats['rf_home_hc_win_rate'] = sum(1 for x in home_hc if x == '赢') / len(home_hc)
    
    if away_use:
        away_pts = [3 if u['gf'] > u['ga'] else (1 if u['gf'] == u['ga'] else 0) for u in away_use]
        feats['rf_away_ppg'] = _mean(away_pts)
        feats['rf_away_gf'] = _mean([u['gf'] for u in away_use])
        feats['rf_away_ga'] = _mean([u['ga'] for u in away_use])
        # P2: 客场赢盘率
        away_hc = [u['row']['handicap_result'] for u in away_use 
                   if u['row'].get('handicap_result') in ('赢', '输')]
        if away_hc:
            feats['rf_away_hc_win_rate'] = sum(1 for x in away_hc if x == '赢') / len(away_hc)

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

    return feats


# ==================== 对战往绩特征 ====================
def h2h_features_upside(rows, home_team, away_team):
    """rows: head_to_head表该场比赛的往绩行(dict列表)
    P1: 新增初始盘口特征"""
    keys = ['h2h_n', 'h2h_goal_avg', 'h2h_pool_share', 'h2h_home_win_rate',
            'h2h_half_goal_avg', 'h2h_corners_avg']
    # P1: 初始盘口特征
    p1_keys = ['h2h_initial_handicap_avg', 'h2h_home_odds_avg', 'h2h_away_odds_avg',
               'h2h_handicap_win_rate', 'h2h_over_rate']
    feats = {k: None for k in keys + p1_keys}
    if not rows:
        return feats

    feats['h2h_n'] = len(rows)
    totals, pool_hit, home_wins, decided = [], 0, 0, 0
    half_goals_list, corners_list = [], []
    # P1: 初始盘口统计
    ihc_list, iho_list, iao_list = [], [], []
    hcp_wins, over_count, hcp_decided = 0, 0, 0
    
    for r in rows:
        h, a = parse_score(r['score'])
        if h is None:
            continue
        totals.append(h + a)
        if norm_score(r['score']) in OVER_SET:
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
        
        # P1: 初始盘口数据
        ihc = r.get('initial_handicap')
        if ihc:
            ihc_num = handicap_to_num(ihc)
            if ihc_num is not None:
                ihc_list.append(ihc_num)
        iho = r.get('initial_home_odds')
        if iho:
            try:
                iho_list.append(float(iho))
            except (ValueError, TypeError):
                pass
        iao = r.get('initial_away_odds')
        if iao:
            try:
                iao_list.append(float(iao))
            except (ValueError, TypeError):
                pass
        # 赢盘率
        hcr = r.get('handicap_result')
        if hcr in ('赢', '输'):
            hcp_decided += 1
            if hcr == '赢':
                hcp_wins += 1
        # 大球率
        our = r.get('over_under_result')
        if our == '大':
            over_count += 1
    
    if totals:
        feats['h2h_goal_avg'] = _mean(totals)
        feats['h2h_pool_share'] = pool_hit / len(totals)
    if decided:
        feats['h2h_home_win_rate'] = home_wins / decided
    if half_goals_list:
        feats['h2h_half_goal_avg'] = _mean(half_goals_list)
    if corners_list:
        feats['h2h_corners_avg'] = _mean(corners_list)
    
    # P1: 初始盘口特征
    if ihc_list:
        feats['h2h_initial_handicap_avg'] = _mean(ihc_list)
    if iho_list:
        feats['h2h_home_odds_avg'] = _mean(iho_list)
    if iao_list:
        feats['h2h_away_odds_avg'] = _mean(iao_list)
    if hcp_decided > 0:
        feats['h2h_handicap_win_rate'] = hcp_wins / hcp_decided
    if len(rows) > 0:
        feats['h2h_over_rate'] = over_count / len(rows)
    
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
        SELECT match_id,
               AVG(initial_win_odds) as avg_iw, STDDEV_POP(initial_win_odds) as std_iw,
               AVG(initial_draw_odds) as avg_id, STDDEV_POP(initial_draw_odds) as std_id,
               AVG(initial_loss_odds) as avg_il, STDDEV_POP(initial_loss_odds) as std_il,
               AVG(current_win_odds) as avg_cw, STDDEV_POP(current_win_odds) as std_cw,
               AVG(current_draw_odds) as avg_cd, STDDEV_POP(current_draw_odds) as std_cd,
               AVG(current_loss_odds) as avg_cl, STDDEV_POP(current_loss_odds) as std_cl,
               COUNT(*) as cnt
        FROM european_odds
        WHERE match_id IS NOT NULL AND match_id <> ''
        GROUP BY match_id
    """)
    euro = {}
    for row in cursor.fetchall():
        mid = row[0]
        euro[mid] = [{
            'initial_win_odds': row[1], 'initial_win_odds_std': row[2],
            'initial_draw_odds': row[3], 'initial_draw_odds_std': row[4],
            'initial_loss_odds': row[5], 'initial_loss_odds_std': row[6],
            'current_win_odds': row[7], 'current_win_odds_std': row[8],
            'current_draw_odds': row[9], 'current_draw_odds_std': row[10],
            'current_loss_odds': row[11], 'current_loss_odds_std': row[12],
            'count': row[13]
        }]
    data['euro'] = euro

    cursor.execute("""
        SELECT match_id, team_type, team_name, home_team, away_team, score,
               handicap_result, over_under_result, shots, shots_on_target, possession,
               corners, dangerous_attacks, attacks, fouls, pass_success_rate,
               match_date
        FROM recent_form
        WHERE is_current_match = 0
        ORDER BY match_id, match_date DESC
    """)
    recent = defaultdict(lambda: {'home': [], 'away': []})
    for mid, tt, tname, ht, at, sc, hcr, our, shots, sot, poss, corners, da, attacks, fouls, psr, mdate in cursor.fetchall():
        if tt in ('home', 'away'):
            recent[mid][tt].append({'team_name': tname, 'home_team': ht,
                                    'away_team': at, 'score': sc,
                                    'handicap_result': hcr, 'over_under_result': our,
                                    'shots': shots, 'shots_on_target': sot,
                                    'possession': poss, 'corners': corners,
                                    'dangerous_attacks': da, 'attacks': attacks,
                                    'fouls': fouls, 'pass_success_rate': psr,
                                    'match_date': mdate})
    data['recent'] = recent

    cursor.execute("""
        SELECT match_id, home_team, away_team, score, half_score, corners,
               initial_handicap, initial_home_odds, initial_away_odds,
               handicap_result, over_under_result
        FROM head_to_head
    """)
    h2h = defaultdict(list)
    for mid, ht, at, sc, hs, co, ihcp, ihodds, iaodds, hcr, our in cursor.fetchall():
        h2h[mid].append({'home_team': ht, 'away_team': at, 'score': sc,
                         'half_score': hs, 'corners': co,
                         'initial_handicap': ihcp, 'initial_home_odds': ihodds,
                         'initial_away_odds': iaodds, 'handicap_result': hcr,
                         'over_under_result': our})
    data['h2h'] = h2h

    cursor.execute("""
        SELECT match_id, team_type, scope, played, won, drawn, lost,
               goals_for, goals_against, goal_diff, points, ranking, win_rate
        FROM league_standings
    """)
    standings = defaultdict(dict)
    for mid, tt, scope, played, won, drawn, lost, gf, ga, gd, points, ranking, win_rate in cursor.fetchall():
        standings[mid][(tt, scope)] = {
            'played': played, 'won': won, 'drawn': drawn, 'lost': lost,
            'goals_for': gf, 'goals_against': ga, 'goal_diff': gd,
            'points': points, 'ranking': ranking, 'win_rate': win_rate
        }
    data['standings'] = standings

    cursor.execute("""
        SELECT match_id, sp_value FROM result_sp
        WHERE match_id IS NOT NULL AND match_id <> ''
    """)
    data['sp'] = {mid: float(sp) if sp is not None else None
                  for mid, sp in cursor.fetchall()}
    return data


# ==================== 宽表构建 ====================
def _standings_features_upside(st_map):
    """st_map: {(team_type, scope): {played, won, drawn, lost, goals_for, goals_against, goal_diff, points, ranking, win_rate}}
    返回主客队积分榜详细特征 + 差值特征"""
    # 原有特征（保持向后兼容）
    feats = {'rank_diff': None, 'points_diff': None, 'recent6_points_diff': None}
    
    # P0: 新增详细特征
    ls_keys = ['ls_played', 'ls_won', 'ls_drawn', 'ls_lost',
               'ls_goals_for', 'ls_goals_against', 'ls_goal_diff',
               'ls_gf_avg', 'ls_ga_avg', 'ls_ppg', 'ls_win_rate']
    for k in ls_keys:
        feats['h_' + k] = None
        feats['a_' + k] = None
    # 差值特征
    diff_keys = ['ls_gf_diff', 'ls_ga_diff', 'ls_goal_diff_diff',
                 'ls_win_rate_diff', 'ls_ppg_diff', 'ls_home_advantage']
    for k in diff_keys:
        feats[k] = None
    
    # 主队 total 数据
    h_total = st_map.get(('home', 'total'), {})
    a_total = st_map.get(('away', 'total'), {})
    h_home = st_map.get(('home', 'home'), {})
    a_away = st_map.get(('away', 'away'), {})
    h_recent6 = st_map.get(('home', 'recent6'), {})
    a_recent6 = st_map.get(('away', 'recent6'), {})
    
    # 原有差值特征
    hr = h_total.get('ranking')
    ar = a_total.get('ranking')
    if hr is not None and ar is not None:
        feats['rank_diff'] = hr - ar
    hp = h_total.get('points')
    ap = a_total.get('points')
    if hp is not None and ap is not None:
        feats['points_diff'] = hp - ap
    hp6 = h_recent6.get('points')
    ap6 = a_recent6.get('points')
    if hp6 is not None and ap6 is not None:
        feats['recent6_points_diff'] = hp6 - ap6
    
    # P0: 主队详细特征
    if h_total:
        played = h_total.get('played')
        if played and played > 0:
            feats['h_ls_played'] = played
            feats['h_ls_won'] = h_total.get('won')
            feats['h_ls_drawn'] = h_total.get('drawn')
            feats['h_ls_lost'] = h_total.get('lost')
            feats['h_ls_goals_for'] = h_total.get('goals_for')
            feats['h_ls_goals_against'] = h_total.get('goals_against')
            feats['h_ls_goal_diff'] = h_total.get('goal_diff')
            feats['h_ls_gf_avg'] = h_total.get('goals_for') / played
            feats['h_ls_ga_avg'] = h_total.get('goals_against') / played
            feats['h_ls_ppg'] = h_total.get('points') / played
            wr = h_total.get('win_rate')
            if wr:
                try:
                    feats['h_ls_win_rate'] = float(wr.replace('%', '')) / 100
                except (ValueError, AttributeError):
                    pass
    
    # P0: 客队详细特征
    if a_total:
        played = a_total.get('played')
        if played and played > 0:
            feats['a_ls_played'] = played
            feats['a_ls_won'] = a_total.get('won')
            feats['a_ls_drawn'] = a_total.get('drawn')
            feats['a_ls_lost'] = a_total.get('lost')
            feats['a_ls_goals_for'] = a_total.get('goals_for')
            feats['a_ls_goals_against'] = a_total.get('goals_against')
            feats['a_ls_goal_diff'] = a_total.get('goal_diff')
            feats['a_ls_gf_avg'] = a_total.get('goals_for') / played
            feats['a_ls_ga_avg'] = a_total.get('goals_against') / played
            feats['a_ls_ppg'] = a_total.get('points') / played
            wr = a_total.get('win_rate')
            if wr:
                try:
                    feats['a_ls_win_rate'] = float(wr.replace('%', '')) / 100
                except (ValueError, AttributeError):
                    pass
    
    # P0: 差值特征
    if feats['h_ls_gf_avg'] is not None and feats['a_ls_gf_avg'] is not None:
        feats['ls_gf_diff'] = feats['h_ls_gf_avg'] - feats['a_ls_gf_avg']
    if feats['h_ls_ga_avg'] is not None and feats['a_ls_ga_avg'] is not None:
        feats['ls_ga_diff'] = feats['h_ls_ga_avg'] - feats['a_ls_ga_avg']
    if feats['h_ls_goal_diff'] is not None and feats['a_ls_goal_diff'] is not None:
        feats['ls_goal_diff_diff'] = feats['h_ls_goal_diff'] - feats['a_ls_goal_diff']
    if feats['h_ls_win_rate'] is not None and feats['a_ls_win_rate'] is not None:
        feats['ls_win_rate_diff'] = feats['h_ls_win_rate'] - feats['a_ls_win_rate']
    if feats['h_ls_ppg'] is not None and feats['a_ls_ppg'] is not None:
        feats['ls_ppg_diff'] = feats['h_ls_ppg'] - feats['a_ls_ppg']
    
    # 主场优势：主队主场场均进球 - 主队客场场均进球
    if h_home and h_total:
        h_home_played = h_home.get('played')
        h_away_played = h_total.get('played') - h_home_played if h_total.get('played') else None
        if h_home_played and h_home_played > 0 and h_away_played and h_away_played > 0:
            h_home_gf = h_home.get('goals_for', 0)
            h_total_gf = h_total.get('goals_for', 0)
            h_away_gf = h_total_gf - h_home_gf
            h_home_gf_avg = h_home_gf / h_home_played
            h_away_gf_avg = h_away_gf / h_away_played
            feats['ls_home_advantage'] = h_home_gf_avg - h_away_gf_avg
    
    return feats


def build_features_upside(conn_or_cursor):
    """构建全量特征宽表(含已完场标签in_pool与未完场样本), 返回DataFrame
    按match_id升序(match_id前8位为日期, 天然时间序)"""
    cursor = conn_or_cursor
    own_cursor = False
    if hasattr(conn_or_cursor, 'cursor'):
        cursor = conn_or_cursor.cursor()
        own_cursor = True
    try:
        raw = load_raw(cursor)
    finally:
        if own_cursor:
            cursor.close()

    rows = []
    league_cum = defaultdict(lambda: [0, 0])   # 联赛 -> [池内命中数, 完场数]
    total = len(raw['matches'])
    print(f"⏳ 构建特征... (共{total}场)")
    for i, (mid, single, league, ht, at, mt, score, _status) in enumerate(raw['matches']):
        ah = asian_features_upside(raw['asian'].get(mid, []))
        ou = ou_features_upside(raw['ou'].get(mid, []))
        euro = euro_features_upside(raw['euro'].get(mid, []))
        goal = goal_expectation(ah, ou)

        rf_h = _recent_team_features_upside(raw['recent'].get(mid, {}).get('home', []), ht)
        rf_a = _recent_team_features_upside(raw['recent'].get(mid, {}).get('away', []), at)
        h2h = h2h_features_upside(raw['h2h'].get(mid, []), ht, at)
        st = _standings_features_upside(raw['standings'].get(mid, {}))

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
        
        # 技术统计差值特征
        if rf_h.get('rf_shots_avg') is not None and rf_a.get('rf_shots_avg') is not None:
            row['shots_diff'] = rf_h['rf_shots_avg'] - rf_a['rf_shots_avg']
        if rf_h.get('rf_sot_rate') is not None and rf_a.get('rf_sot_rate') is not None:
            row['sot_rate_diff'] = rf_h['rf_sot_rate'] - rf_a['rf_sot_rate']
        if rf_h.get('rf_poss') is not None and rf_a.get('rf_poss') is not None:
            row['poss_diff'] = rf_h['rf_poss'] - rf_a['rf_poss']
        if rf_h.get('rf_da_avg') is not None and rf_a.get('rf_da_avg') is not None:
            row['da_diff'] = rf_h['rf_da_avg'] - rf_a['rf_da_avg']
        if rf_h.get('rf_corners_avg') is not None and rf_a.get('rf_corners_avg') is not None:
            row['corners_diff'] = rf_h['rf_corners_avg'] - rf_a['rf_corners_avg']
        if rf_h.get('rf_da_rate') is not None and rf_a.get('rf_da_rate') is not None:
            row['da_rate_diff'] = rf_h['rf_da_rate'] - rf_a['rf_da_rate']
        # P2: 赢盘率主客场拆分差值
        if rf_h.get('rf_home_hc_win_rate') is not None and rf_a.get('rf_away_hc_win_rate') is not None:
            row['hc_win_rate_diff'] = rf_h['rf_home_hc_win_rate'] - rf_a['rf_away_hc_win_rate']
        row.update(h2h)
        row.update(st)
        row['league_pool_prior'] = league_prior
        # P2: 是否杯赛
        row['is_cup'] = 1 if is_cup_league(league) else 0
        rows.append(row)

        # 进度显示(每1000场)
        if (i + 1) % 1000 == 0:
            print(f"  已处理 {i + 1}/{total} 场")

        # 本场完场后纳入联赛累积(供后续场次使用)
        if is_finished(score):
            league_cum[league][1] += 1
            if norm_score(score) in OVER_SET:
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
    df = build_features_upside(conn)
    if own:
        conn.close()
    save_cache(df)
    if verbose:
        print(f"✓ 特征宽表构建完成: {len(df)}场, 已缓存到 {CACHE_FILE}")
    return df
