# features_ml.py - ML模型(POOL8)专属特征工程模块
# 功能: 构建"一场比赛一行"的特征宽表, 专用于 POOL8 八比分池预测
#       完全独立于 features_1x.py, 使用独立的缓存文件
#
# 用法:
#   from features_ml import build_features_ml, load_cache_ml
#   df = build_features_ml(conn)     # 返回含 in_pool 标签的特征表
#   save_cache_ml(df)                # 缓存到 features_ml_cache.csv
import os
import re
from collections import Counter, defaultdict

import pandas as pd

from config import DB_CONFIG

# ==================== 常量 ====================
# ML模型专用 POOL8：双方进球(BTTS) + 总进球≤5（主胜3种+平局2种+客胜3种，对称覆盖）
POOL8_ML = ['2-1', '3-1', '3-2', '1-1', '2-2', '1-2', '1-3', '2-3']
POOL8_ML_SET = set(POOL8_ML)

# 目标盘口范围（训练用宽口径）
TARGET_HANDICAPS_ML = ['受让半球', '半球', '受让平/半', '平/半', '平手']
TARGET_HANDICAP_NUMS_ML = {-0.5, 0.5, -0.25, 0.25, 0.0}

# 大小球线最低阈值（训练口径: 初盘和终盘均需 >= 2.25，即 2/2.5）
MIN_OU_LINE_ML = 2.25

# 预测推荐环节大小球线最低阈值（窄口径, 仅 predict_ml 预测/补录推荐环节使用: 只推 >= 2.75，即 2.5/3 及以上）
MIN_OU_LINE_PREDICT_ML = 2.75


def is_target_handicap_ml(ah_cur_mean, ah_init_mean=None):
    """判断盘口均值是否在目标盘口范围内"""
    v = ah_cur_mean
    if v is None:
        v = ah_init_mean
    if v is None:
        return False
    return any(abs(v - t) < 0.15 for t in TARGET_HANDICAP_NUMS_ML)


def is_valid_ou_line_ml(ou_line_init_mean, ou_line_cur_mean=None):
    """判断大小球线是否满足最低阈值（训练口径: 初盘和终盘均 >= 2.25）
    一方缺失时用另一方替代，双方均缺失返回False"""
    init_v = ou_line_init_mean
    cur_v = ou_line_cur_mean
    if init_v is None:
        init_v = cur_v
    if cur_v is None:
        cur_v = init_v
    if init_v is None:
        return False
    return init_v >= MIN_OU_LINE_ML and cur_v >= MIN_OU_LINE_ML


def is_valid_ou_line_predict_ml(ou_line_init_mean, ou_line_cur_mean=None):
    """预测推荐环节大小球线过滤（窄口径: 初盘和终盘均 >= 2.75, 严于训练口径）
    一方缺失时用另一方替代，双方均缺失返回False"""
    init_v = ou_line_init_mean
    cur_v = ou_line_cur_mean
    if init_v is None:
        init_v = cur_v
    if cur_v is None:
        cur_v = init_v
    if init_v is None:
        return False
    return init_v >= MIN_OU_LINE_PREDICT_ML and cur_v >= MIN_OU_LINE_PREDICT_ML


def is_cup_league_ml(league):
    """判断联赛名称是否为杯赛（含'杯'字即为杯赛）"""
    if not league:
        return False
    return '杯' in str(league)


# 元信息列(非特征列)
META_COLS_ML = ['match_id', 'single_match', 'league', 'home_team', 'away_team',
                'match_time', 'score', 'in_pool', 'sp']

# 联赛先验平滑参数: prior = (池内命中 + W*基准) / (场数 + W)
LEAGUE_PRIOR_W_ML = 20
LEAGUE_PRIOR_BASE_ML = 0.55

CACHE_FILE_ML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'features_ml_cache.csv')

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

    n = len(rows)
    if n == 0:
        return feats

    if ih:
        feats['ah_init_mean'] = _mean(ih)
        feats['ah_init_std'] = _std(ih)
        feats['ah_init_flat_share'] = _share_mode(ih)
    if ch:
        feats['ah_cur_mean'] = _mean(ch)
        feats['ah_cur_std'] = _std(ch)
        feats['ah_cur_flat_share'] = _share_mode(ch)
    if moves:
        feats['ah_move'] = _mean(moves)
        feats['ah_move_up_share'] = sum(1 for m in moves if m > 0) / len(moves)

    if ho_i and ao_i:
        feats['aw_init_diff'] = _mean(ho_i) - _mean(ao_i)
    if ho_c and ao_c:
        feats['aw_cur_diff'] = _mean(ho_c) - _mean(ao_c)
        if ho_i and ao_i:
            feats['aw_drift'] = (_mean(ho_c) - _mean(ao_c)) - (_mean(ho_i) - _mean(ao_i))

    # 覆盖概率估算(基于赔率倒数归一化)
    if ho_i and ao_i:
        hi_inv = [1/h for h in ho_i]
        ai_inv = [1/a for a in ao_i]
        total_i = sum(hi_inv) + sum(ai_inv)
        if total_i:
            feats['cover_prob_init'] = sum(hi_inv) / total_i
    if ho_c and ao_c:
        hc_inv = [1/h for h in ho_c]
        ac_inv = [1/a for a in ao_c]
        total_c = sum(hc_inv) + sum(ac_inv)
        if total_c:
            feats['cover_prob_cur'] = sum(hc_inv) / total_c

    feats['asian_n'] = n
    return feats


# ==================== 大小球特征 ====================
def over_under_features(rows):
    """rows: 该场比赛over_under_odds表全部行(dict列表)
    返回特征dict; 无数据返回全NaN dict"""
    keys = ['ou_n', 'ou_line_init_mean', 'ou_line_cur_mean', 'ou_line_move',
            'ou_line_std', 'ou_line_flat_share', 'oh_init_mean', 'oh_cur_mean',
            'oh_init_std', 'oh_cur_std', 'oh_drift', 'oh_init_diff', 'oh_cur_diff']
    feats = {k: None for k in keys}

    li, lc, moves = [], [], []
    ohi, ohc = [], []
    for r in rows:
        l_i = to_float(r['initial_line'])
        l_c = to_float(r['current_line'])
        if l_c is None:
            l_c = l_i
        if l_i is not None:
            li.append(l_i)
        if l_c is not None:
            lc.append(l_c)
        if l_i is not None and l_c is not None:
            moves.append(l_c - l_i)
        ohi_val = to_float(r['initial_over_odds'])
        ohc_val = to_float(r['current_over_odds'])
        if ohi_val:
            ohi.append(ohi_val)
        if ohc_val:
            ohc.append(ohc_val)

    n = len(rows)
    if n == 0:
        return feats

    if li:
        feats['ou_line_init_mean'] = _mean(li)
        feats['ou_line_std'] = _std(li)
        feats['ou_line_flat_share'] = _share_mode(li)
    if lc:
        feats['ou_line_cur_mean'] = _mean(lc)
    if moves:
        feats['ou_line_move'] = _mean(moves)
    if ohi:
        feats['oh_init_mean'] = _mean(ohi)
        feats['oh_init_std'] = _std(ohi)
    if ohc:
        feats['oh_cur_mean'] = _mean(ohc)
        feats['oh_cur_std'] = _std(ohc)
        if ohi:
            feats['oh_drift'] = _mean(ohc) - _mean(ohi)
            feats['oh_init_diff'] = _mean(ohi) - 2.0
            feats['oh_cur_diff'] = _mean(ohc) - 2.0

    feats['ou_n'] = n
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


# ==================== 快速版特征函数(使用预加载数据) ====================
def _h2h_features_fast(home_team, away_team, h2h_data, max_h2h=10):
    """快速版历史交锋特征(使用预加载数据)"""
    keys = ['h2h_n', 'h2h_home_win_rate', 'h2h_draw_rate', 'h2h_away_win_rate',
            'h2h_avg_goals', 'h2h_home_goals_mean', 'h2h_away_goals_mean',
            'h2h_btts_rate', 'h2h_over_25_rate',
            'h2h_half_goal_avg', 'h2h_corners_avg']
    feats = {k: None for k in keys}
    
    h2h_rows = h2h_data[:max_h2h]
    n = len(h2h_rows)
    if n == 0:
        return feats
    
    home_wins, draws, away_wins = 0, 0, 0
    goals_list, home_goals_list, away_goals_list = [], [], []
    btts_count, over_25_count = 0, 0
    half_goals_list, corners_list = [], []
    
    for match in h2h_rows:
        hg, ag = parse_score(match['score'])
        if hg is None:
            continue
        
        # 判断主客场
        if home_team == match['home_team']:
            h_goals, a_goals = hg, ag
            side = 'home'
        else:
            h_goals, a_goals = ag, hg
            side = 'away'
        
        goals_list.append(hg + ag)
        home_goals_list.append(h_goals)
        away_goals_list.append(a_goals)
        
        if h_goals > a_goals:
            home_wins += 1
        elif h_goals == a_goals:
            draws += 1
        else:
            away_wins += 1
        
        if h_goals > 0 and a_goals > 0:
            btts_count += 1
        if hg + ag > 2.5:
            over_25_count += 1
        
        # 半场进球
        if match.get('half_score'):
            hh, ha = parse_score(match['half_score'])
            if hh is not None and ha is not None:
                half_goals_list.append(hh + ha)
        # 角球(总角球)
        if match.get('corners'):
            try:
                c_str = str(match['corners']).replace(' ', '')
                if '-' in c_str:
                    parts = c_str.split('-')
                    corners_list.append(int(parts[0]) + int(parts[1]))
            except (ValueError, IndexError):
                pass
    
    if goals_list:
        feats['h2h_n'] = n
        feats['h2h_home_win_rate'] = home_wins / n
        feats['h2h_draw_rate'] = draws / n
        feats['h2h_away_win_rate'] = away_wins / n
        feats['h2h_avg_goals'] = _mean(goals_list)
        feats['h2h_home_goals_mean'] = _mean(home_goals_list)
        feats['h2h_away_goals_mean'] = _mean(away_goals_list)
        feats['h2h_btts_rate'] = btts_count / n
        feats['h2h_over_25_rate'] = over_25_count / n
    if half_goals_list:
        feats['h2h_half_goal_avg'] = _mean(half_goals_list)
    if corners_list:
        feats['h2h_corners_avg'] = _mean(corners_list)
    
    return feats


def _recent_form_features_fast(team_name, team_data, is_home=True):
    """快速版近期战绩特征(使用预加载数据)"""
    keys = ['recent_n', 'recent_win_rate', 'recent_draw_rate', 'recent_loss_rate',
            'recent_goals_for_mean', 'recent_goals_against_mean',
            'recent_clean_sheet_rate', 'recent_btts_rate',
            'recent_over_25_rate', 'recent_unbeaten_rate']
    feats = {k: None for k in keys}
    
    n = len(team_data)
    if n == 0:
        return feats
    
    wins, draws, losses = 0, 0, 0
    goals_for, goals_against = [], []
    clean_sheets, btts_count, over_25_count = 0, 0, 0
    
    for match in team_data:
        hg, ag = parse_score(match['score'])
        if hg is None:
            continue
        
        if is_home:
            gf, ga = hg, ag
        else:
            gf, ga = ag, hg
        
        goals_for.append(gf)
        goals_against.append(ga)
        
        if gf > ga:
            wins += 1
        elif gf == ga:
            draws += 1
        else:
            losses += 1
        
        if ga == 0:
            clean_sheets += 1
        if gf > 0 and ga > 0:
            btts_count += 1
        if gf + ga > 2.5:
            over_25_count += 1
    
    if goals_for:
        feats['recent_n'] = n
        feats['recent_win_rate'] = wins / n
        feats['recent_draw_rate'] = draws / n
        feats['recent_loss_rate'] = losses / n
        feats['recent_goals_for_mean'] = _mean(goals_for)
        feats['recent_goals_against_mean'] = _mean(goals_against)
        feats['recent_clean_sheet_rate'] = clean_sheets / n
        feats['recent_btts_rate'] = btts_count / n
        feats['recent_over_25_rate'] = over_25_count / n
        feats['recent_unbeaten_rate'] = (wins + draws) / n
    
    return feats


def _league_stats_features_fast(league, league_data):
    """快速版联赛统计特征(使用预加载数据)"""
    keys = ['league_n', 'league_home_win_rate', 'league_draw_rate', 'league_away_win_rate',
            'league_avg_goals', 'league_btts_rate', 'league_over_25_rate',
            'league_home_unbeaten_rate']
    feats = {k: None for k in keys}
    
    n = len(league_data)
    if n == 0:
        return feats
    
    home_wins, draws, away_wins = 0, 0, 0
    goals_list, btts_count, over_25_count, home_unbeaten = [], 0, 0, 0
    
    for match in league_data:
        hg, ag = parse_score(match['score'])
        if hg is None:
            continue
        
        goals_list.append(hg + ag)
        if hg > ag:
            home_wins += 1
            home_unbeaten += 1
        elif hg == ag:
            draws += 1
            home_unbeaten += 1
        else:
            away_wins += 1
        
        if hg > 0 and ag > 0:
            btts_count += 1
        if hg + ag > 2.5:
            over_25_count += 1
    
    if goals_list:
        feats['league_n'] = n
        feats['league_home_win_rate'] = home_wins / n
        feats['league_draw_rate'] = draws / n
        feats['league_away_win_rate'] = away_wins / n
        feats['league_avg_goals'] = _mean(goals_list)
        feats['league_btts_rate'] = btts_count / n
        feats['league_over_25_rate'] = over_25_count / n
        feats['league_home_unbeaten_rate'] = home_unbeaten / n
    
    return feats


# ==================== 历史交锋特征 ====================
def h2h_features(home_team, away_team, history_df, max_h2h=10):
    """从history_df中提取两队历史交锋特征
    history_df: 包含所有历史比赛的DataFrame(需有home_team, away_team, score, half_score, corners列)
    返回特征dict"""
    keys = ['h2h_n', 'h2h_home_win_rate', 'h2h_draw_rate', 'h2h_away_win_rate',
            'h2h_avg_goals', 'h2h_home_goals_mean', 'h2h_away_goals_mean',
            'h2h_btts_rate', 'h2h_over_25_rate',
            'h2h_half_goal_avg', 'h2h_corners_avg']
    feats = {k: None for k in keys}

    # 筛选两队交锋记录
    h2h_mask = ((history_df['home_team'].str.contains(home_team, regex=False)) &
                (history_df['away_team'].str.contains(away_team, regex=False))) | \
               ((history_df['home_team'].str.contains(away_team, regex=False)) &
                (history_df['away_team'].str.contains(home_team, regex=False)))
    h2h_rows = history_df[h2h_mask].head(max_h2h)

    n = len(h2h_rows)
    if n == 0:
        return feats

    home_wins, draws, away_wins = 0, 0, 0
    goals_list, home_goals_list, away_goals_list = [], [], []
    btts_count, over_25_count = 0, 0
    half_goals_list, corners_list = [], []

    for _, row in h2h_rows.iterrows():
        hg, ag = parse_score(row['score'])
        if hg is None:
            continue

        # 判断主客场
        if home_team in row['home_team']:
            h_goals, a_goals = hg, ag
        else:
            h_goals, a_goals = ag, hg

        goals_list.append(hg + ag)
        home_goals_list.append(h_goals)
        away_goals_list.append(a_goals)

        if h_goals > a_goals:
            home_wins += 1
        elif h_goals == a_goals:
            draws += 1
        else:
            away_wins += 1

        if h_goals > 0 and a_goals > 0:
            btts_count += 1
        if hg + ag > 2.5:
            over_25_count += 1

        # 半场进球
        if row.get('half_score') and pd.notna(row.get('half_score')):
            hh, ha = parse_score(str(row['half_score']))
            if hh is not None and ha is not None:
                half_goals_list.append(hh + ha)
        # 角球(总角球)
        if row.get('corners') and pd.notna(row.get('corners')):
            try:
                c_str = str(row['corners']).replace(' ', '')
                if '-' in c_str:
                    parts = c_str.split('-')
                    corners_list.append(int(parts[0]) + int(parts[1]))
            except (ValueError, IndexError):
                pass

    if goals_list:
        feats['h2h_n'] = n
        feats['h2h_home_win_rate'] = home_wins / n
        feats['h2h_draw_rate'] = draws / n
        feats['h2h_away_win_rate'] = away_wins / n
        feats['h2h_avg_goals'] = _mean(goals_list)
        feats['h2h_home_goals_mean'] = _mean(home_goals_list)
        feats['h2h_away_goals_mean'] = _mean(away_goals_list)
        feats['h2h_btts_rate'] = btts_count / n
        feats['h2h_over_25_rate'] = over_25_count / n
    if half_goals_list:
        feats['h2h_half_goal_avg'] = _mean(half_goals_list)
    if corners_list:
        feats['h2h_corners_avg'] = _mean(corners_list)

    return feats


# ==================== 近期战绩特征 ====================
def recent_form_features(team_name, history_df, is_home=True, max_recent=10):
    """提取球队近期战绩特征
    team_name: 球队名称
    history_df: 历史比赛DataFrame
    is_home: 是否为主场视角
    max_recent: 最多取最近N场
    返回特征dict"""
    keys = ['recent_n', 'recent_win_rate', 'recent_draw_rate', 'recent_loss_rate',
            'recent_goals_for_mean', 'recent_goals_against_mean',
            'recent_clean_sheet_rate', 'recent_btts_rate',
            'recent_over_25_rate', 'recent_unbeaten_rate']
    feats = {k: None for k in keys}

    # 筛选该队比赛
    if is_home:
        mask = history_df['home_team'].str.contains(team_name, regex=False)
        side = 'home'
    else:
        mask = history_df['away_team'].str.contains(team_name, regex=False)
        side = 'away'

    team_rows = history_df[mask].sort_values('match_id', ascending=False).head(max_recent)
    n = len(team_rows)
    if n == 0:
        return feats

    wins, draws, losses = 0, 0, 0
    goals_for, goals_against = [], []
    clean_sheets, btts_count, over_25_count = 0, 0, 0

    for _, row in team_rows.iterrows():
        hg, ag = parse_score(row['score'])
        if hg is None:
            continue

        if side == 'home':
            gf, ga = hg, ag
        else:
            gf, ga = ag, hg

        goals_for.append(gf)
        goals_against.append(ga)

        if gf > ga:
            wins += 1
        elif gf == ga:
            draws += 1
        else:
            losses += 1

        if ga == 0:
            clean_sheets += 1
        if gf > 0 and ga > 0:
            btts_count += 1
        if gf + ga > 2.5:
            over_25_count += 1

    if goals_for:
        feats['recent_n'] = n
        feats['recent_win_rate'] = wins / n
        feats['recent_draw_rate'] = draws / n
        feats['recent_loss_rate'] = losses / n
        feats['recent_goals_for_mean'] = _mean(goals_for)
        feats['recent_goals_against_mean'] = _mean(goals_against)
        feats['recent_clean_sheet_rate'] = clean_sheets / n
        feats['recent_btts_rate'] = btts_count / n
        feats['recent_over_25_rate'] = over_25_count / n
        feats['recent_unbeaten_rate'] = (wins + draws) / n

    return feats


# ==================== 联赛统计特征 ====================
def league_stats_features(league, history_df):
    """提取联赛级别统计特征
    league: 联赛名称
    history_df: 历史比赛DataFrame
    返回特征dict"""
    keys = ['league_n', 'league_home_win_rate', 'league_draw_rate', 'league_away_win_rate',
            'league_avg_goals', 'league_btts_rate', 'league_over_25_rate',
            'league_home_unbeaten_rate']
    feats = {k: None for k in keys}

    league_rows = history_df[history_df['league'] == league]
    n = len(league_rows)
    if n == 0:
        return feats

    home_wins, draws, away_wins = 0, 0, 0
    goals_list, btts_count, over_25_count, home_unbeaten = [], 0, 0, 0

    for _, row in league_rows.iterrows():
        hg, ag = parse_score(row['score'])
        if hg is None:
            continue

        goals_list.append(hg + ag)
        if hg > ag:
            home_wins += 1
            home_unbeaten += 1
        elif hg == ag:
            draws += 1
            home_unbeaten += 1
        else:
            away_wins += 1

        if hg > 0 and ag > 0:
            btts_count += 1
        if hg + ag > 2.5:
            over_25_count += 1

    if goals_list:
        feats['league_n'] = n
        feats['league_home_win_rate'] = home_wins / n
        feats['league_draw_rate'] = draws / n
        feats['league_away_win_rate'] = away_wins / n
        feats['league_avg_goals'] = _mean(goals_list)
        feats['league_btts_rate'] = btts_count / n
        feats['league_over_25_rate'] = over_25_count / n
        feats['league_home_unbeaten_rate'] = home_unbeaten / n

    return feats


# ==================== 核心构建函数 ====================
def build_features_ml(conn):
    """构建ML模型专属特征表(一场比赛一行)
    采用批量查询优化: 一次性加载所有盘口数据, 避免逐场查库
    conn: MySQL连接对象
    返回: DataFrame, 包含所有特征列 + META_COLS_ML"""
    cursor = conn.cursor()

    # 1. 批量加载所有比赛(含未开赛)
    cursor.execute("""
        SELECT match_id, single_match, league, home_team, away_team, 
               match_time, score
        FROM `match` 
        ORDER BY match_id
    """)
    matches = cursor.fetchall()
    cols = ['match_id', 'single_match', 'league', 'home_team', 'away_team',
            'match_time', 'score']
    df = pd.DataFrame(matches, columns=cols)

    if len(df) == 0:
        return pd.DataFrame()

    # 2. 计算标签: in_pool(完场=0/1, 未完场=None)
    def _calc_in_pool(s):
        if not s or not is_finished(s):
            return None
        return 1 if norm_score(s) in POOL8_ML_SET else 0
    df['in_pool'] = df['score'].apply(_calc_in_pool)
    
    # 3. 批量加载SP值
    cursor.execute("""
        SELECT match_id, sp_value FROM result_sp
        WHERE match_id IS NOT NULL AND match_id <> ''
    """)
    sp_map = {mid: float(sp) if sp is not None else None
              for mid, sp in cursor.fetchall()}
    df['sp'] = df['match_id'].map(sp_map)

    # 4. 批量加载亚盘数据(按match_id分组)
    cursor.execute("""
        SELECT match_id, initial_handicap, current_handicap,
               initial_home_odds, initial_away_odds,
               current_home_odds, current_away_odds
        FROM asian_odds
    """)
    asian_map = defaultdict(list)
    for mid, ih, ch, iho, iao, cho, cao in cursor.fetchall():
        asian_map[mid].append({
            'initial_handicap': ih, 'current_handicap': ch,
            'initial_home_odds': iho, 'initial_away_odds': iao,
            'current_home_odds': cho, 'current_away_odds': cao
        })

    # 5. 批量加载大小球数据(按match_id分组)
    cursor.execute("""
        SELECT match_id, initial_line, current_line,
               initial_over_odds, initial_under_odds,
               current_over_odds, current_under_odds
        FROM over_under_odds
    """)
    ou_map = defaultdict(list)
    for mid, il, cl, ioo, iuo, coo, cuo in cursor.fetchall():
        ou_map[mid].append({
            'initial_line': il, 'current_line': cl,
            'initial_over_odds': ioo, 'initial_under_odds': iuo,
            'current_over_odds': coo, 'current_under_odds': cuo
        })

    # 5b. 批量加载欧赔数据(按match_id分组)
    cursor.execute("""
        SELECT match_id, initial_win_odds, initial_draw_odds, initial_loss_odds,
               current_win_odds, current_draw_odds, current_loss_odds
        FROM european_odds
    """)
    euro_map = defaultdict(list)
    for mid, iw, id_, il, cw, cd, cl in cursor.fetchall():
        euro_map[mid].append({
            'initial_win_odds': iw, 'initial_draw_odds': id_, 'initial_loss_odds': il,
            'current_win_odds': cw, 'current_draw_odds': cd, 'current_loss_odds': cl
        })

    # 6. 加载历史比赛(用于近期战绩/联赛统计)
    cursor.execute("""
        SELECT match_id, league, home_team, away_team, score
        FROM `match` 
        WHERE score IS NOT NULL AND score != ''
        ORDER BY match_id
    """)
    hist_matches = cursor.fetchall()
    history_df = pd.DataFrame(hist_matches, columns=['match_id', 'league', 'home_team', 'away_team', 'score'])

    # 6b. 加载历史交锋数据(含半场比分、角球)
    cursor.execute("""
        SELECT home_team, away_team, score, half_score, corners
        FROM head_to_head
        ORDER BY match_id
    """)
    h2h_records = cursor.fetchall()
    h2h_df = pd.DataFrame(h2h_records, columns=['home_team', 'away_team', 'score', 'half_score', 'corners'])

    # 7. 预建索引加速查找
    print("⏳ 构建索引...")
    # 按球队分组(主场/客场)
    home_team_matches = defaultdict(list)
    away_team_matches = defaultdict(list)
    # 按联赛分组
    league_matches = defaultdict(list)
    # 按球队对分组(历史交锋) - 使用head_to_head表
    h2h_matches = defaultdict(list)
    
    for _, row in history_df.iterrows():
        home = row['home_team']
        away = row['away_team']
        league = row['league']
        match_data = {
            'match_id': row['match_id'],
            'score': row['score'],
            'home_team': home,
            'away_team': away
        }
        home_team_matches[home].append(match_data)
        away_team_matches[away].append(match_data)
        league_matches[league].append(match_data)
    
    # 历史交锋索引(从head_to_head表)
    for _, row in h2h_df.iterrows():
        home = row['home_team']
        away = row['away_team']
        team_pair = tuple(sorted([home, away]))
        h2h_matches[team_pair].append({
            'home_team': home,
            'away_team': away,
            'score': row['score'],
            'half_score': row['half_score'],
            'corners': row['corners']
        })
    
    print(f"✓ 索引构建完成")

    # 8. 逐场构建特征(从内存中读取盘口数据, 不再查库)
    print(f"⏳ 构建特征... (共{len(df)}场)")
    all_rows = []
    for idx, row in df.iterrows():
        mid = row['match_id']
        home = row['home_team']
        away = row['away_team']
        league = row['league']

        # 从内存中获取盘口数据
        af = asian_features(asian_map.get(mid, []))
        ouf = over_under_features(ou_map.get(mid, []))
        ef = euro_features(euro_map.get(mid, []))

        # 历史交锋(使用预建索引)
        team_pair = tuple(sorted([home, away]))
        h2h_data = h2h_matches.get(team_pair, [])
        h2hf = _h2h_features_fast(home, away, h2h_data)
        
        # 近期战绩(使用预建索引)
        home_recent_data = home_team_matches.get(home, [])[-10:]
        home_recent = _recent_form_features_fast(home, home_recent_data, is_home=True)
        away_recent_data = away_team_matches.get(away, [])[-10:]
        away_recent = _recent_form_features_fast(away, away_recent_data, is_home=False)
        
        # 联赛统计(使用预建索引)
        league_data = league_matches.get(league, [])
        league_stats = _league_stats_features_fast(league, league_data)

        # 合并所有特征
        feat_row = {**af, **ouf, **ef, **h2hf, **home_recent, **away_recent, **league_stats}
        meta_row = row.to_dict()
        all_rows.append({**meta_row, **feat_row})

        # 进度显示(每1000场)
        if (idx + 1) % 1000 == 0:
            print(f"  已处理 {idx + 1}/{len(df)} 场")

    result_df = pd.DataFrame(all_rows)
    print(f"✓ 特征构建完成: {len(result_df)}场")
    return result_df


def feature_columns_ml():
    """返回ML模型使用的特征列名列表(不含元信息列)"""
    # 从示例构建中提取所有特征键
    sample_asian = asian_features([])
    sample_ou = over_under_features([])
    sample_euro = euro_features([])
    # 构造空DataFrame但包含必要的列，避免KeyError
    empty_hist_df = pd.DataFrame(columns=['match_id', 'home_team', 'away_team', 'score', 'league'])
    sample_h2h = h2h_features('', '', empty_hist_df)
    sample_recent = recent_form_features('', empty_hist_df, is_home=True)
    sample_league = league_stats_features('', empty_hist_df)

    all_keys = list(sample_asian.keys()) + list(sample_ou.keys()) + list(sample_euro.keys()) + \
               list(sample_h2h.keys()) + \
               list(sample_recent.keys()) + list(sample_league.keys())
    return all_keys


def load_cache_ml():
    """加载ML模型特征缓存, 不存在返回None"""
    if os.path.exists(CACHE_FILE_ML):
        return pd.read_csv(CACHE_FILE_ML)
    return None


def save_cache_ml(df):
    """保存ML模型特征缓存"""
    df.to_csv(CACHE_FILE_ML, index=False)
    print(f"✓ ML特征缓存已保存: {CACHE_FILE_ML} ({len(df)}条)")


def load_or_build_ml(conn=None, rebuild=False):
    """加载或构建ML模型特征表(带缓存)
    conn=None时自动创建数据库连接
    rebuild=True时强制重建缓存"""
    import pymysql
    own = conn is None
    
    if rebuild:
        print("⏳ 强制重建ML特征表...")
        if own:
            conn = pymysql.connect(**DB_CONFIG)
        df = build_features_ml(conn)
        if len(df) > 0:
            save_cache_ml(df)
        if own:
            conn.close()
        return df
    
    cached = load_cache_ml()
    if cached is not None:
        print(f"✓ 加载ML特征缓存: {len(cached)}条")
        return cached

    print("⏳ 构建ML特征表...")
    if own:
        conn = pymysql.connect(**DB_CONFIG)
    df = build_features_ml(conn)
    if len(df) > 0:
        save_cache_ml(df)
    if own:
        conn.close()
    return df


if __name__ == '__main__':
    import pymysql
    conn = pymysql.connect(**DB_CONFIG)
    try:
        df = load_or_build_ml(conn)
        print(f"\nML特征表形状: {df.shape}")
        print(f"列名: {list(df.columns)[:20]}...")
        print(f"\nPOOL8_ML命中率: {df['in_pool'].mean():.2%}")
        print(f"样本数: {len(df)}")
    finally:
        conn.close()
