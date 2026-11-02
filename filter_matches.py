# filter_matches.py - 比赛筛选脚本
# 用法: python filter_matches.py
#       python filter_matches.py --review          # 回顾昨天筛选结果(对照实际比分+SP, 邮件发送)
#       python filter_matches.py --review 20260907 # 回顾指定日期
# 功能: 从数据库中筛选符合多个条件的比赛，每个条件可通过开关自由添加或组合
#       筛选通过场次追加写入filter_results.jsonl, 供次日--review回顾
import argparse
import json
import os
import pymysql
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.header import Header
from config import DB_CONFIG, EMAIL_CONFIG, ENABLE_EMAIL
from predict_scores import predict_six_scores, format_scores, get_freq_coverage
from backtest import update_backtest_result

# ====================================================================
# 筛选条件开关（True=启用, False=关闭，可自由组合）
# ====================================================================
ENABLE_COND1 = True    # 条件1：亚盘机构数（初盘+即时盘为目标盘口的机构数）
ENABLE_COND2 = False   # 条件2：主队主场积分 vs 客队客场积分 分差（回测LOO为负贡献, 已禁用仅参考输出）
ENABLE_COND3 = True    # 条件3：两队近6场积分分差
ENABLE_COND4 = False   # 条件4：近N场对赛往绩胜场优势（回测LOO为负贡献, 已禁用仅参考输出）
ENABLE_COND5 = False   # 条件5：主队近期技术统计优势（多窗口时间衰减+指标加权，评估主队是否明显优于客队，已禁用仅参考输出）
ENABLE_COND6 = True    # 条件6：近N场目标盘口下正常比分概率
ENABLE_COND7 = False    # 条件7：两队近6场平均射正率之和
ENABLE_COND8 = False    # 条件8：联赛黑名单（运行前自动回测生成，命中率低的联赛直接排除）
ENABLE_COND9 = False    # 条件9：历史频率Top6覆盖概率（回测: ≥50%档命中率56.9% vs 40-50%档47.4%）
ENABLE_COND10 = True   # 条件10：大小球机构数（初盘+终盘进球数线均≤目标线的机构数）
ENABLE_COND11 = False   # 条件11：两队近N场均总进球≤上限（排除大球比赛, 六比分池偏小球）


# ====================================================================
# 比赛编号筛选（自动使用当天日期，支持*通配符；留空''表示不限制）
# ====================================================================
# MATCH_ID_PATTERN = datetime.now().strftime('%Y%m%d') + '*'   # 自动筛选当天的比赛
MATCH_ID_PATTERN = '20260912*'

# ====================================================================
# 各条件参数（可自由调整阈值）
# ====================================================================
# 条件1：亚盘初盘和即时盘口都为目标盘口的机构数需 > COND1_MIN_COMPANIES
COND1_TARGET_HANDICAPS = ['半球' , '平手/半球', '平手']   # asian_odds表中的盘口格式
COND1_MIN_COMPANIES = 12                       # 机构数阈值（超过此值）

# 条件2：主队主场积分 与 客队客场积分 分差 <= COND2_MAX_DIFF
COND2_MAX_DIFF = 6

# 条件3：两队近6场积分分差 <= COND3_MAX_DIFF
COND3_MAX_DIFF = 9

# 条件4：近COND4_RECENT_N场对赛往绩，胜场优势 <= COND4_MAX_WIN_DIFF
COND4_RECENT_N = 6
COND4_MAX_WIN_DIFF = 2

# 条件5：主队近期技术统计优势（多窗口时间衰减+指标加权，评估主队是否明显优于客队→上盘信号）
# 时间窗口权重（递增，近3场最重要；总权重=10）
COND5_WINDOW_WEIGHTS = {'recent3': 5, 'recent6': 3, 'recent10': 2}
# 各窗口最小有效场次数（不足则跳过该窗口）
COND5_WINDOW_MIN_MATCHES = {'recent3': 2, 'recent6': 3, 'recent10': 5}
# 进攻技术指标及权重（按预测力排序）
COND5_STATS = ['shots_on_target', 'dangerous_attacks', 'shots', 'corners', 'possession']
COND5_STATS_WEIGHTS = {
    'shots_on_target': 3.0,    # 射正，最直接的质量指标
    'dangerous_attacks': 2.5,  # 危险进攻，直接威胁
    'shots': 1.5,              # 射门，尝试次数
    'corners': 1.5,            # 角球，持续施压
    'possession': 1.0,         # 控球，控制力但非直接进攻
}
COND5_STATS_CN = {             # 中文名（用于输出）
    'shots_on_target': '射正',
    'dangerous_attacks': '危险进攻',
    'shots': '射门',
    'corners': '角球',
    'possession': '控球率',
}
COND5_MIN_ADVANTAGE = 0.10     # 主队加权优势分需≥此值（即主队平均强10%以上）
COND5_MIN_VALID_STATS = 2      # 至少2项指标有效才纳入评估

# 条件6：近COND6_RECENT_N场，目标盘口下正常比分概率 >= COND6_MIN_PROBABILITY（两队合并）
COND6_RECENT_N = 10
COND6_TARGET_HANDICAPS = ['半球','平/半', '平手']       # recent_form表中的盘口格式
COND6_NORMAL_SCORES = ['1-0', '2-0' , '2-1' , '3-1' , '3-2', '0-0', '1-1', '2-2']
COND6_MIN_PROBABILITY = 0.50   # 70%

# 条件7：主队和客队近COND7_RECENT_N场平均射正率（射正/射门）同时 > COND7_MIN_RATE
# 子条件：主客队近COND7_RECENT_N场平均危险进攻次数同时 > COND7_MIN_DANGEROUS_ATTACKS
COND7_RECENT_N = 6
COND7_MIN_RATE = 0.30   # 30%
COND7_MIN_DANGEROUS_ATTACKS = 30   # 平均危险进攻次数

# 条件8：联赛黑名单（自动回测生成: 每次运行时调用backtest.py全量回测,
#        结果写入backtest_result.json并自动加载, 无需手动维护）
ENABLE_AUTO_BACKTEST = True    # 运行前自动跑全量回测刷新黑名单(约几分钟)
COND8_LEAGUE_BLACKLIST = [     # 内置列表(自动回测失败时兜底使用)
    '德甲', '英超', '法甲', '荷甲', '瑞士超', '瑞典甲', '奥乙',
    '瑞士甲', '捷甲', '爱甲', '法乙', '苏冠', '巴西乙', '巴西杯', '阿甲'
]

# 条件9：盘口档位历史频率Top6覆盖概率 >= COND9_MIN_COVERAGE
# （覆盖概率=该档位历史Top6比分场次占比, 即单场命中经验概率）
COND9_MIN_COVERAGE = 0.50

# 条件10：大小球初盘和终盘进球数线均 <= COND10_MAX_LINE 的机构数需 > COND10_MIN_COMPANIES
# 盘口线写法: '2.5'、'2/2.5'(=2.25)、'2.5/3'(=2.75)等, 双侧取平均换算为数值比较
# 回测: 线≤2.25的通过组命中率68.5% > 线≤2.5的64.3%（2.5线市场定价高效, 挡不住大球）
COND10_MAX_LINE = 3       # 2/2.5 即 2.25（小于等于此线）
COND10_MIN_COMPANIES = 12     # 机构数阈值（超过此值）

# 条件11：两队近COND11_RECENT_N场均总进球的平均 <= COND11_MAX_AVG_GOALS
# （六比分预测池偏小球, 需从基本面排除大球比赛; 回测: 叠加后组合命中率68.1%且按月稳定）
COND11_RECENT_N = 6
COND11_MAX_AVG_GOALS = 2.5

# ====================================================================
# 预测模块配置（盘口50% + 基本面50%）
# ====================================================================
ENABLE_PREDICT = True   # 是否启用预测（对所有未开始比赛预测并写入match表）
ENABLE_SCORE_PREDICT = True   # 是否启用六比分预测（在筛选结果中附上预测的6个比分）

# ====================================================================
# 回顾配置(--review对照前一天筛选结果与实际比分/SP)
# ====================================================================
FILTER_RESULT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'filter_results.jsonl')
REVIEW_TITLE = '常规版筛选'
REVIEW_POOL = COND6_NORMAL_SCORES   # 回顾命中口径: 实际比分落在该比分池内

# 盘口因素权重（合计50%）
PREDICT_WEIGHT_ODDS = 0.50

# 基本面因素权重（合计50%）
PREDICT_WEIGHT_STANDINGS = 0.15    # 积分排名
PREDICT_WEIGHT_RECENT = 0.15       # 近期战绩
PREDICT_WEIGHT_H2H = 0.10          # 对赛往绩
PREDICT_WEIGHT_TECH = 0.05         # 技术统计
PREDICT_WEIGHT_OU = 0.05           # 大小球


# ====================================================================
# 工具函数
# ====================================================================
def get_connection():
    return pymysql.connect(**DB_CONFIG)


def team_name_similarity(name1, name2):
    """计算两个球队名称的相似度（用于对赛往绩中匹配球队）"""
    if not name1 or not name2:
        return 0.0
    # 清理(中)标记
    n1 = name1.replace('(中)', '').replace('（中）', '').strip()
    n2 = name2.replace('(中)', '').replace('（中）', '').strip()
    if n1 == n2:
        return 1.0
    # 子串包含
    if n1 in n2 or n2 in n1:
        return 0.9
    # 字符重叠度
    set1, set2 = set(n1), set(n2)
    union = set1 | set2
    if not union:
        return 0.0
    return len(set1 & set2) / len(union)


def match_team_position(target_team, h2h_home, h2h_away):
    """判断target_team在某场对赛往绩中是主场还是客场
    返回: 'home' / 'away' / None
    """
    sim_home = team_name_similarity(target_team, h2h_home)
    sim_away = team_name_similarity(target_team, h2h_away)
    if sim_home >= sim_away and sim_home >= 0.5:
        return 'home'
    if sim_away >= 0.5:
        return 'away'
    return None


def to_float(val):
    """安全转换为浮点数（处理控球率'58%'这类字符串）"""
    if val is None:
        return None
    try:
        return float(str(val).replace('%', '').strip())
    except (ValueError, TypeError):
        return None


def _review_norm_score(s):
    """比分归一为'x-y'格式, 无效(未开赛/'-'/改期等)返回None"""
    s = str(s or '').replace(':', '-').strip()
    parts = s.split('-')
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return s
    return None


def _review_finished(status):
    """match表status是否为已完场"""
    return bool(status) and any(k in str(status) for k in ('完', '已结束', 'Finished'))


# ====================================================================
# 各条件判断函数（返回 (是否通过, 详情) ）
# ====================================================================
# 条件1盘口归一化: 各种写法变体 -> 标准写法(与COND1_TARGET_HANDICAPS对应)
_COND1_ALIASES = {
    '平手': '平手', '受平手': '平手',
    '平手/半球': '平手/半球', '平/半': '平手/半球',
    '受让平手/半球': '受让平手/半球', '受平手/半球': '受让平手/半球',
    '受让平/半': '受让平手/半球', '受平/半': '受让平手/半球',
    '半球': '半球', '受半球': '受让半球', '受让半球': '受让半球',
}


def _cond1_norm(h):
    """盘口字符串归一化, 无法识别返回None"""
    if h is None:
        return None
    return _COND1_ALIASES.get(str(h).strip())


def check_cond1(cursor, match_id):
    """条件1：亚盘初盘和即时盘口都为目标盘口的机构数 > COND1_MIN_COMPANIES
    盘口先经_COND1_ALIASES归一化, 兼容'受让平手/半球'/'受平手/半球'/'受让平/半'等变体写法"""
    cursor.execute("""
        SELECT initial_handicap, current_handicap FROM asian_odds WHERE match_id = %s
    """, (match_id,))
    targets = set(COND1_TARGET_HANDICAPS)
    count = sum(1 for init_h, curr_h in cursor.fetchall()
                if _cond1_norm(init_h) in targets and _cond1_norm(curr_h) in targets)
    return count > COND1_MIN_COMPANIES, f"{count}家机构(需>{COND1_MIN_COMPANIES})"


def check_cond2(cursor, match_id):
    """条件2：主队主场积分 与 客队客场积分 分差 <= COND2_MAX_DIFF"""
    cursor.execute("""
        SELECT team_type, scope, points FROM league_standings
        WHERE match_id = %s AND scope IN ('home', 'away')
    """, (match_id,))
    home_points = away_points = None
    for team_type, scope, points in cursor.fetchall():
        if team_type == 'home' and scope == 'home':
            home_points = points
        elif team_type == 'away' and scope == 'away':
            away_points = points
    if home_points is None or away_points is None:
        return False, "缺少积分数据"
    diff = abs(home_points - away_points)
    return diff <= COND2_MAX_DIFF, f"主主场{home_points}分/客客场{away_points}分, 差{diff}(需≤{COND2_MAX_DIFF})"


def check_cond3(cursor, match_id):
    """条件3：两队近6场积分分差 <= COND3_MAX_DIFF"""
    cursor.execute("""
        SELECT team_type, points FROM league_standings
        WHERE match_id = %s AND scope = 'recent6'
    """, (match_id,))
    home_points = away_points = None
    for team_type, points in cursor.fetchall():
        if team_type == 'home':
            home_points = points
        elif team_type == 'away':
            away_points = points
    if home_points is None or away_points is None:
        return False, "缺少近6场积分数据"
    diff = abs(home_points - away_points)
    return diff <= COND3_MAX_DIFF, f"主近6场{home_points}分/客近6场{away_points}分, 差{diff}(需≤{COND3_MAX_DIFF})"


def check_cond4(cursor, match_id, home_team, away_team):
    """条件4：近COND4_RECENT_N场对赛往绩，胜场优势 <= COND4_MAX_WIN_DIFF"""
    cursor.execute("""
        SELECT home_team, away_team, result FROM head_to_head
        WHERE match_id = %s
        ORDER BY match_date DESC
        LIMIT %s
    """, (match_id, COND4_RECENT_N))
    rows = cursor.fetchall()
    if not rows:
        return False, "无对赛往绩数据"
    home_wins = away_wins = 0
    for h2h_home, h2h_away, result in rows:
        if not result:
            continue
        pos = match_team_position(home_team, h2h_home, h2h_away)
        if pos == 'home':
            if result == '胜':
                home_wins += 1
            elif result == '负':
                away_wins += 1
        elif pos == 'away':
            if result == '负':
                home_wins += 1
            elif result == '胜':
                away_wins += 1
    diff = abs(home_wins - away_wins)
    return diff <= COND4_MAX_WIN_DIFF, f"近{len(rows)}场: 主胜{home_wins}/客胜{away_wins}, 优势{diff}(需≤{COND4_MAX_WIN_DIFF})"


def check_cond5(cursor, match_id):
    """条件5：主队近期技术统计优势（多窗口时间衰减+指标加权）
    评估主队是否明显优于客队（上盘信号）。
    数据不足时返回(True, '数据不足')不过滤。"""
    # 1. 获取主客队近10场技术统计（一次性取最大窗口，子集切片）
    rows_by_team = {}
    for team_type in ['home', 'away']:
        cursor.execute(
            f"SELECT {', '.join(COND5_STATS)} FROM recent_form "
            f"WHERE match_id = %s AND team_type = %s AND is_current_match = 0 "
            f"ORDER BY match_date DESC LIMIT 10",
            (match_id, team_type))
        rows = cursor.fetchall()
        if not rows:
            return True, f"缺少{team_type}近期战绩数据"
        # 预计算浮点值矩阵: rows_float[j][i] = float value or None
        rows_float = []
        for row in rows:
            rows_float.append([to_float(row[i]) for i in range(len(COND5_STATS))])
        rows_by_team[team_type] = rows_float

    # 2. 窗口配置: (name, limit, weight, min_valid_matches)
    windows = [
        ('recent3', 3, 5, 2),
        ('recent6', 6, 3, 3),
        ('recent10', 10, 2, 5),
    ]

    # 3. 对每队每指标，计算各窗口均值（仅当有效场次≥min_valid_matches）
    # avg[team_type][stat_idx][window_name] = mean_value or None
    avg = {}
    for team_type in ['home', 'away']:
        avg[team_type] = {}
        rf = rows_by_team[team_type]
        for i in range(len(COND5_STATS)):
            avg[team_type][i] = {}
            for w_name, w_limit, w_weight, w_min in windows:
                values = [rf[j][i] for j in range(min(w_limit, len(rf))) if rf[j][i] is not None]
                if len(values) >= w_min:
                    avg[team_type][i][w_name] = sum(values) / len(values)

    # 4. 对每指标，计算三窗口加权均值 + 相对优势
    stat_results = []  # [(stat_name, advantage, stat_weight)]
    for i, stat in enumerate(COND5_STATS):
        stat_weight = COND5_STATS_WEIGHTS[stat]
        weighted_h = weighted_a = weight_sum = 0.0
        for w_name, w_limit, w_weight, w_min in windows:
            h_val = avg['home'][i].get(w_name)
            a_val = avg['away'][i].get(w_name)
            if h_val is not None and a_val is not None:
                weighted_h += h_val * w_weight
                weighted_a += a_val * w_weight
                weight_sum += w_weight
        if weight_sum == 0:
            continue  # 该指标无有效窗口数据
        h_w = weighted_h / weight_sum
        a_w = weighted_a / weight_sum
        advantage = (h_w - a_w) / max(h_w, a_w, 1.0)
        stat_results.append((stat, advantage, stat_weight))

    if len(stat_results) < COND5_MIN_VALID_STATS:
        return True, f"数据不足(有效指标{len(stat_results)}<{COND5_MIN_VALID_STATS})"

    # 5. 综合优势分（指标加权平均）
    total_sw = sum(w for _, _, w in stat_results)
    overall = sum(adv * w for _, adv, w in stat_results) / total_sw

    ok = overall >= COND5_MIN_ADVANTAGE
    flag = "\u2713" if ok else "\u2717"

    # 6. 构建详情
    parts = [f"{COND5_STATS_CN[s]}{adv:+.2f}(w{w})" for s, adv, w in stat_results]
    detail = f"加权优势分{overall:+.2f}{flag} [{' '.join(parts)}]"
    return ok, detail


def check_cond6(cursor, match_id):
    """条件6：近COND6_RECENT_N场目标盘口下，正常比分概率 >= COND6_MIN_PROBABILITY（两队合并）"""
    all_rows = []
    for team_type in ['home', 'away']:
        cursor.execute("""
            SELECT score, initial_handicap FROM recent_form
            WHERE match_id = %s AND team_type = %s AND is_current_match = 0
            ORDER BY match_date DESC
            LIMIT %s
        """, (match_id, team_type, COND6_RECENT_N))
        all_rows.extend(cursor.fetchall())

    filtered = [(score, hc) for score, hc in all_rows if hc in COND6_TARGET_HANDICAPS]
    if not filtered:
        return False, "无目标盘口的近期战绩"
    normal_count = sum(1 for score, hc in filtered if score in COND6_NORMAL_SCORES)
    total = len(filtered)
    prob = normal_count / total
    return prob >= COND6_MIN_PROBABILITY, f"目标盘口{total}场, 正常比分{normal_count}场, 概率{prob*100:.0f}%(需≥{COND6_MIN_PROBABILITY*100:.0f}%)"


def check_cond7(cursor, match_id):
    """条件7：主队和客队近COND7_RECENT_N场平均射正率同时 > COND7_MIN_RATE，
    且平均危险进攻次数同时 > COND7_MIN_DANGEROUS_ATTACKS；附加输出黄牌和犯规数"""
    rates = {}
    da_avgs = {}
    yel_avgs = {}
    foul_avgs = {}
    for team_type in ['home', 'away']:
        cursor.execute("""
            SELECT shots_on_target, shots, dangerous_attacks, yellow_cards, fouls FROM recent_form
            WHERE match_id = %s AND team_type = %s AND is_current_match = 0
            ORDER BY match_date DESC
            LIMIT %s
        """, (match_id, team_type, COND7_RECENT_N))
        rows = cursor.fetchall()
        if not rows:
            return False, f"缺少{team_type}近期战绩数据"
        match_rates = []
        da_values = []
        yel_values = []
        foul_values = []
        for sot, sh, da, yel, foul in rows:
            sot_v = to_float(sot)
            sh_v = to_float(sh)
            da_v = to_float(da)
            yel_v = to_float(yel)
            foul_v = to_float(foul)
            if sot_v is not None and sh_v is not None and sh_v > 0:
                match_rates.append(sot_v / sh_v)
            if da_v is not None:
                da_values.append(da_v)
            if yel_v is not None:
                yel_values.append(yel_v)
            if foul_v is not None:
                foul_values.append(foul_v)
        if not match_rates:
            return False, f"{team_type}无有效射门数据"
        rates[team_type] = sum(match_rates) / len(match_rates)
        da_avgs[team_type] = sum(da_values) / len(da_values) if da_values else 0.0
        yel_avgs[team_type] = sum(yel_values) / len(yel_values) if yel_values else 0.0
        foul_avgs[team_type] = sum(foul_values) / len(foul_values) if foul_values else 0.0

    # 子条件1：射正率
    home_rate_pass = rates['home'] > COND7_MIN_RATE
    away_rate_pass = rates['away'] > COND7_MIN_RATE
    rate_pass = home_rate_pass and away_rate_pass
    h_rflag = "✓" if home_rate_pass else "✗"
    a_rflag = "✓" if away_rate_pass else "✗"

    # 子条件2：危险进攻
    home_da_pass = da_avgs['home'] > COND7_MIN_DANGEROUS_ATTACKS
    away_da_pass = da_avgs['away'] > COND7_MIN_DANGEROUS_ATTACKS
    da_pass = home_da_pass and away_da_pass
    h_dflag = "✓" if home_da_pass else "✗"
    a_dflag = "✓" if away_da_pass else "✗"

    both_pass = rate_pass and da_pass
    detail = (f"射正率: 主{rates['home']*100:.1f}%{h_rflag}/客{rates['away']*100:.1f}%{a_rflag}(需各>{COND7_MIN_RATE*100:.0f}%) | "
              f"危险进攻: 主{da_avgs['home']:.0f}{h_dflag}/客{da_avgs['away']:.0f}{a_dflag}(需各>{COND7_MIN_DANGEROUS_ATTACKS}) | "
              f"黄牌: 主{yel_avgs['home']:.1f}/客{yel_avgs['away']:.1f} | "
              f"犯规: 主{foul_avgs['home']:.1f}/客{foul_avgs['away']:.1f}")
    return both_pass, detail


def check_cond8(cursor, match_id, league):
    """条件8：联赛不在黑名单内（黑名单由自动回测生成，见backtest_result.json）"""
    if not COND8_LEAGUE_BLACKLIST:
        return True, "黑名单为空"
    if league in COND8_LEAGUE_BLACKLIST:
        return False, f"{league}在黑名单内(回测命中率偏低)"
    return True, f"{league}不在黑名单"


def _ou_line_to_num(line):
    """大小球盘口线转数值: '2.5'->2.5, '2.5/3'->2.75, '3/3.5'->3.25; 无法识别返回None"""
    if line is None:
        return None
    s = str(line).strip()
    if not s:
        return None
    try:
        if '/' in s:
            a, b = s.split('/', 1)
            return (float(a.strip()) + float(b.strip())) / 2
        return float(s)
    except (ValueError, TypeError):
        return None


def check_cond10(cursor, match_id):
    """条件10：大小球初盘和终盘进球数线均 <= COND10_MAX_LINE 的机构数 > COND10_MIN_COMPANIES"""
    cursor.execute("""
        SELECT initial_line, current_line FROM over_under_odds WHERE match_id = %s
    """, (match_id,))
    rows = cursor.fetchall()
    if not rows:
        return False, "无大小球数据"
    count = 0
    for init_line, curr_line in rows:
        init_v = _ou_line_to_num(init_line)
        curr_v = _ou_line_to_num(curr_line)
        # 终盘缺失('-'表示无变化)时以初盘代替
        if curr_v is None:
            curr_v = init_v
        if init_v is not None and curr_v is not None \
                and init_v <= COND10_MAX_LINE + 1e-9 and curr_v <= COND10_MAX_LINE + 1e-9:
            count += 1
    return count > COND10_MIN_COMPANIES, f"{count}家机构(需>{COND10_MIN_COMPANIES})"


def _score_total_goals(score):
    """比分字符串 -> 总进球数, 无法解析返回None"""
    try:
        a, b = str(score).replace(':', '-').split('-')
        return int(a) + int(b)
    except (ValueError, TypeError, AttributeError):
        return None


def check_cond11(cursor, match_id):
    """条件11：两队近COND11_RECENT_N场均总进球的平均 <= COND11_MAX_AVG_GOALS
    六比分预测池偏小球, 从基本面排除大球倾向的比赛"""
    avgs = {}
    for team_type in ['home', 'away']:
        cursor.execute("""
            SELECT score FROM recent_form
            WHERE match_id = %s AND team_type = %s AND is_current_match = 0
              AND score IS NOT NULL AND score <> ''
            ORDER BY match_date DESC
            LIMIT %s
        """, (match_id, team_type, COND11_RECENT_N))
        totals = []
        for (s,) in cursor.fetchall():
            t = _score_total_goals(s)
            if t is not None:
                totals.append(t)
        if not totals:
            return False, f"缺{team_type}近期比分数据"
        avgs[team_type] = sum(totals) / len(totals)
    avg = (avgs['home'] + avgs['away']) / 2
    ok = avg <= COND11_MAX_AVG_GOALS
    return ok, f"场均总进球: 主{avgs['home']:.1f}/客{avgs['away']:.1f}, 均值{avg:.2f}(需≤{COND11_MAX_AVG_GOALS})"


def check_cond9(cursor, match_id):
    """条件9：盘口档位历史频率Top6覆盖概率 >= COND9_MIN_COVERAGE"""
    bucket, cover, n, source = get_freq_coverage(cursor, match_id)
    if bucket is None or cover is None:
        return False, "无即时盘口或档位无法映射, 无法计算覆盖概率"
    ok = cover >= COND9_MIN_COVERAGE
    return ok, f"{source}, 覆盖概率{cover:.0%}(需≥{COND9_MIN_COVERAGE:.0%})"


# ====================================================================
# 预测模块
# ====================================================================
# 盘口字符串 → 数值映射（正数=主队让球，负数=客队让球）
_HANDICAP_MAP = {
    '受球半': -1.5, '受一/球半': -1.25, '受一球': -1.0, '受半/一': -0.75,
    '受半球': -0.5, '受平/半': -0.25, '受平手': -0.0,
    '平手': 0.0, '平/半': 0.25, '平手/半球': 0.25,
    '半球': 0.5, '半/一': 0.75, '一球': 1.0,
    '一/球半': 1.25, '球半': 1.5, '球半/两': 1.75, '两球': 2.0,
}


def _handicap_to_num(h):
    """盘口字符串转数值，无法识别返回None"""
    if h is None:
        return None
    h = str(h).strip()
    if h in _HANDICAP_MAP:
        return _HANDICAP_MAP[h]
    # 尝试直接转浮点（如 "0.5", "-0.25"）
    try:
        return float(h)
    except (ValueError, TypeError):
        return None


def _predict_odds(cursor, match_id):
    """亚盘盘口+水位评分，返回 -1~1（正=主强，负=客强）"""
    cursor.execute("""
        SELECT initial_handicap, initial_home_odds, initial_away_odds,
               current_handicap, current_home_odds, current_away_odds
        FROM asian_odds WHERE match_id = %s
    """, (match_id,))
    rows = cursor.fetchall()
    if not rows:
        return 0.0, "无亚盘数据"

    scores = []
    for init_h, init_ho, init_ao, curr_h, curr_ho, curr_ao in rows:
        init_num = _handicap_to_num(init_h)
        curr_num = _handicap_to_num(curr_h)
        if init_num is None:
            continue
        # 盘口方向分（-1~1）：盘口数值越大主队越强，归一化到[-1,1]
        dir_score = max(-1.0, min(1.0, init_num / 1.5))
        # 盘口变化分：升盘=主队更强，降盘=客队更强
        move_score = 0.0
        if curr_num is not None:
            diff = curr_num - init_num
            move_score = max(-1.0, min(1.0, diff / 0.5))
        # 水位分：主队水位越低主队越强
        water_score = 0.0
        init_ho_v = to_float(init_ho)
        init_ao_v = to_float(init_ao)
        if init_ho_v is not None and init_ao_v is not None and (init_ho_v + init_ao_v) > 0:
            water_score = (init_ao_v - init_ho_v) / (init_ho_v + init_ao_v)
        # 综合：盘口方向50% + 盘口变化25% + 水位25%
        s = dir_score * 0.50 + move_score * 0.25 + water_score * 0.25
        scores.append(s)

    if not scores:
        return 0.0, "亚盘数据无效"
    avg = sum(scores) / len(scores)
    return avg, f"盘口评分{avg:+.2f}({len(scores)}家)"


def _predict_standings(cursor, match_id):
    """积分排名评分，返回 -1~1"""
    cursor.execute("""
        SELECT team_type, scope, points FROM league_standings
        WHERE match_id = %s AND scope IN ('home', 'away')
    """, (match_id,))
    home_pts = away_pts = None
    for tt, scope, pts in cursor.fetchall():
        if tt == 'home' and scope == 'home':
            home_pts = to_float(pts)
        elif tt == 'away' and scope == 'away':
            away_pts = to_float(pts)
    if home_pts is None or away_pts is None:
        return 0.0, "无积分数据"
    total = home_pts + away_pts
    if total == 0:
        return 0.0, f"主{home_pts:.0f}分/客{away_pts:.0f}分"
    score = (home_pts - away_pts) / total
    score = max(-1.0, min(1.0, score * 2))  # 放大到[-1,1]
    return score, f"积分主{home_pts:.0f}/客{away_pts:.0f}"


def _predict_recent(cursor, match_id):
    """近期战绩评分，返回 -1~1"""
    results = {}
    for tt in ['home', 'away']:
        cursor.execute("""
            SELECT result FROM recent_form
            WHERE match_id = %s AND team_type = %s AND is_current_match = 0
            ORDER BY match_date DESC LIMIT 6
        """, (match_id, tt))
        rows = cursor.fetchall()
        if not rows:
            return 0.0, "无近期战绩"
        w = sum(1 for r, in rows if r == 'W')
        d = sum(1 for r, in rows if r == 'D')
        l = sum(1 for r, in rows if r == 'L')
        total = w + d + l
        # 胜率-负率，范围[-1,1]
        results[tt] = (w - l) / total if total > 0 else 0.0
    score = (results['home'] - results['away']) / 2
    return score, f"近6场主{results['home']:+.2f}/客{results['away']:+.2f}"


def _predict_h2h(cursor, match_id, home_team, away_team):
    """对赛往绩评分，返回 -1~1"""
    cursor.execute("""
        SELECT home_team, away_team, result FROM head_to_head
        WHERE match_id = %s ORDER BY match_date DESC LIMIT 6
    """, (match_id,))
    rows = cursor.fetchall()
    if not rows:
        return 0.0, "无对赛往绩"
    home_wins = away_wins = 0
    for h2h_h, h2h_a, result in rows:
        pos = match_team_position(home_team, h2h_h, h2h_a)
        if pos == 'home':
            if result == '胜': home_wins += 1
            elif result == '负': away_wins += 1
        elif pos == 'away':
            if result == '负': home_wins += 1
            elif result == '胜': away_wins += 1
    total = home_wins + away_wins
    if total == 0:
        return 0.0, f"交锋主胜{home_wins}/客胜{away_wins}"
    score = (home_wins - away_wins) / total
    return score, f"交锋主胜{home_wins}/客胜{away_wins}"


def _predict_tech(cursor, match_id):
    """技术统计评分，返回 -1~1"""
    stats = {}
    for tt in ['home', 'away']:
        cursor.execute("""
            SELECT shots_on_target, dangerous_attacks, possession
            FROM recent_form
            WHERE match_id = %s AND team_type = %s AND is_current_match = 0
            ORDER BY match_date DESC LIMIT 6
        """, (match_id, tt))
        rows = cursor.fetchall()
        if not rows:
            return 0.0, "无技术统计"
        sot_vals = [to_float(r[0]) for r in rows if to_float(r[0]) is not None]
        da_vals = [to_float(r[1]) for r in rows if to_float(r[1]) is not None]
        pos_vals = [to_float(r[2]) for r in rows if to_float(r[2]) is not None]
        stats[tt] = {
            'sot': sum(sot_vals) / len(sot_vals) if sot_vals else 0,
            'da': sum(da_vals) / len(da_vals) if da_vals else 0,
            'pos': sum(pos_vals) / len(pos_vals) if pos_vals else 0,
        }
    # 射正和危险进攻：差值归一化
    sot_diff = stats['home']['sot'] - stats['away']['sot']
    da_diff = stats['home']['da'] - stats['away']['da']
    pos_diff = stats['home']['pos'] - stats['away']['pos']
    # 归一化到[-1,1]
    sot_score = max(-1, min(1, sot_diff / 5)) if sot_diff != 0 else 0
    da_score = max(-1, min(1, da_diff / 15)) if da_diff != 0 else 0
    pos_score = max(-1, min(1, pos_diff / 20)) if pos_diff != 0 else 0
    score = sot_score * 0.4 + da_score * 0.4 + pos_score * 0.2
    return score, f"技术统计评分{score:+.2f}"


def _predict_ou(cursor, match_id):
    """大小球评分，返回 -1~1（大球盘口高通常意味强队进攻强）"""
    cursor.execute("""
        SELECT initial_line, current_line, initial_over_odds, initial_under_odds
        FROM over_under_odds WHERE match_id = %s
    """, (match_id,))
    rows = cursor.fetchall()
    if not rows:
        return 0.0, "无大小球数据"
    scores = []
    for init_line, curr_line, init_oo, init_uo in rows:
        init_l = to_float(init_line)
        curr_l = to_float(curr_line)
        if init_l is None:
            continue
        # 盘口线越高，进球预期越多，对强队有利（但无法区分主客，给一个微弱信号）
        # 水位：大球水位低说明预期进球多
        oo = to_float(init_oo)
        uo = to_float(init_uo)
        if oo is not None and uo is not None and (oo + uo) > 0:
            water = (uo - oo) / (oo + uo)  # 正=大球水位低
            scores.append(water * 0.5)  # 大小球只给微弱信号
    if not scores:
        return 0.0, "大小球数据无效"
    avg = sum(scores) / len(scores)
    return avg, f"大小球评分{avg:+.2f}"


def predict_match(cursor, match_id, home_team, away_team):
    """综合预测，返回 (预测结果, 各因子详情)"""
    factors = []

    odds_score, odds_detail = _predict_odds(cursor, match_id)
    factors.append(('亚盘', odds_score, odds_detail, PREDICT_WEIGHT_ODDS))

    std_score, std_detail = _predict_standings(cursor, match_id)
    factors.append(('积分', std_score, std_detail, PREDICT_WEIGHT_STANDINGS))

    rec_score, rec_detail = _predict_recent(cursor, match_id)
    factors.append(('近绩', rec_score, rec_detail, PREDICT_WEIGHT_RECENT))

    h2h_score, h2h_detail = _predict_h2h(cursor, match_id, home_team, away_team)
    factors.append(('往绩', h2h_score, h2h_detail, PREDICT_WEIGHT_H2H))

    tech_score, tech_detail = _predict_tech(cursor, match_id)
    factors.append(('技术', tech_score, tech_detail, PREDICT_WEIGHT_TECH))

    ou_score, ou_detail = _predict_ou(cursor, match_id)
    factors.append(('大小球', ou_score, ou_detail, PREDICT_WEIGHT_OU))

    # 加权总分
    total_score = sum(s * w for _, s, _, w in factors)
    total_score = max(-1.0, min(1.0, total_score))

    prediction = '胜平' if total_score >= 0 else '平负'

    # 构建详情
    details = []
    for name, score, detail, weight in factors:
        flag = '主' if score > 0.05 else ('客' if score < -0.05 else '平')
        details.append(f"{name}({weight*100:.0f}%):{detail}[{flag}]")

    return prediction, total_score, details



def load_league_blacklist(conn):
    """运行前自动全量回测并加载联赛黑名单; 失败时回退内置列表"""
    try:
        t0 = datetime.now()
        blacklist = update_backtest_result(conn=conn)
        cost = (datetime.now() - t0).total_seconds()
        print(f"✓ 自动回测完成({cost:.0f}秒), 联赛黑名单({len(blacklist)}个): "
              f"{'、'.join(blacklist) if blacklist else '无'}")
        return blacklist
    except Exception as e:
        print(f"⚠ 自动回测失败: {e}, 回退使用内置黑名单({len(COND8_LEAGUE_BLACKLIST)}个)")
        return list(COND8_LEAGUE_BLACKLIST)


# ====================================================================
# 筛选回顾(--review对照筛选结果与实际比分/SP)
# ====================================================================
def append_filter_log(entries):
    """筛选通过记录追加写入jsonl(回顾时按match_id去重保留最后一次)"""
    with open(FILTER_RESULT_FILE, 'a', encoding='utf-8') as fp:
        for e in entries:
            fp.write(json.dumps(e, ensure_ascii=False) + '\n')


def review(target_date, send_mail=True):
    """回看指定日期筛选通过场次: 对照match表实际比分与result_sp开奖SP
    命中口径: 实际比分∈REVIEW_POOL; send_mail=True时结果邮件发送(受ENABLE_EMAIL控制)"""
    lines = []

    def out(s=''):
        lines.append(s)
        print(s)

    if not os.path.exists(FILTER_RESULT_FILE):
        print(f"✗ 未找到筛选记录文件: {FILTER_RESULT_FILE} (记录自上线当天开始积累, 无历史数据)")
        return
    with open(FILTER_RESULT_FILE, encoding='utf-8') as fp:
        logs = [json.loads(l) for l in fp if l.strip()]
    # 按match_id去重(同一场多次筛选保留最后一次), 只取目标日期
    latest = {}
    for l in logs:
        mid = str(l.get('match_id', ''))
        if mid.startswith(target_date):
            latest[mid] = l
    if not latest:
        print(f"日期{target_date}无筛选记录({os.path.basename(FILTER_RESULT_FILE)})")
        return

    ids = list(latest)
    conn = get_connection()
    cursor = conn.cursor()
    fmt = ','.join(['%s'] * len(ids))
    cursor.execute(f"SELECT match_id, score, status FROM `match` WHERE match_id IN ({fmt})", ids)
    info = {r[0]: {'score': r[1], 'status': r[2]} for r in cursor.fetchall()}
    # 开奖SP值(赛果比分的SP, 比分玩法)
    cursor.execute(f"SELECT match_id, sp_value FROM result_sp WHERE CONVERT(match_id USING utf8mb4) COLLATE utf8mb4_general_ci IN ({fmt})", ids)
    sp_map = {r[0]: (float(r[1]) if r[1] is not None else None) for r in cursor.fetchall()}
    cursor.close()
    conn.close()

    out('=' * 78)
    out(f"{REVIEW_TITLE}回顾 {target_date}  (来源: {os.path.basename(FILTER_RESULT_FILE)})")
    out(f"命中口径: 实际比分∈{len(REVIEW_POOL)}比分池 {'/'.join(REVIEW_POOL)}")
    out('=' * 78)
    finished = hit = 0
    for mid in sorted(latest):
        e = latest[mid]
        mi = info.get(mid, {})
        sp = sp_map.get(mid)
        sp_str = f"SP{sp:.2f}" if sp is not None else "SP-"
        single = f"[{e.get('single_match') or '-'}]"
        vs = f"{e.get('home_team', '?')} vs {e.get('away_team', '?')}"
        sc = mi.get('score')
        norm = _review_norm_score(sc)
        if _review_finished(mi.get('status')) and norm:
            finished += 1
            ok = norm in REVIEW_POOL
            hit += int(ok)
            flag = '✓命中' if ok else '✗未中'
            sc_str = norm
        else:
            flag = '未完场'
            sc_str = str(sc) if sc else '-'
        out(f"{mid} {single} {e.get('league', '')} {vs}  实际[{sc_str} {sp_str}]  -> {flag}")
    out('-' * 78)
    rate = f" ({hit / finished * 100:.1f}%)" if finished else ''
    out(f"共{len(latest)}场通过筛选, 已完场{finished}场: 命中{hit}/{finished}{rate}")
    if send_mail:
        send_email('\n'.join(lines),
                   f"{REVIEW_TITLE}回顾 {target_date} 通过{len(latest)}场 "
                   f"完场{finished} 命中{hit}{rate}")


# ====================================================================
# 主流程
# ====================================================================
def main():
    ap = argparse.ArgumentParser(description='比赛筛选')
    ap.add_argument('--review', nargs='?', const='', default=None,
                    help='回顾模式: 对照筛选结果与实际比分/SP, 不带值默认昨天, 可指定YYYYMMDD')
    ap.add_argument('--skip-email', action='store_true', help='回顾模式跳过邮件发送')
    args = ap.parse_args()
    if args.review is not None:
        target = args.review or (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
        review(target, send_mail=not args.skip_email)
        return

    global COND8_LEAGUE_BLACKLIST
    conn = get_connection()
    cursor = conn.cursor()

    # 自动回测刷新联赛黑名单(条件8数据源)
    if ENABLE_COND8 and ENABLE_AUTO_BACKTEST:
        print("=" * 70)
        COND8_LEAGUE_BLACKLIST = load_league_blacklist(conn)

    # 启用的条件列表
    enabled = []
    if ENABLE_COND1: enabled.append(('条件1-亚盘机构数', check_cond1))
    if ENABLE_COND2: enabled.append(('条件2-主客积分差', check_cond2))
    if ENABLE_COND3: enabled.append(('条件3-近6场积分差', check_cond3))
    if ENABLE_COND4: enabled.append(('条件4-对赛往绩胜场', check_cond4))
    if ENABLE_COND5: enabled.append(('条件5-技术统计', check_cond5))
    if ENABLE_COND6: enabled.append(('条件6-正常比分概率', check_cond6))
    if ENABLE_COND7: enabled.append(('条件7-射正率+危险进攻', check_cond7))
    if ENABLE_COND8: enabled.append(('条件8-联赛黑名单', check_cond8))
    if ENABLE_COND9: enabled.append(('条件9-覆盖概率', check_cond9))
    if ENABLE_COND10: enabled.append(('条件10-大小球机构数', check_cond10))
    if ENABLE_COND11: enabled.append(('条件11-场均总进球', check_cond11))

    # 已禁用但仅作参考输出的条件(不参与筛选, 只在通过场次下打印供人工参考)
    reference = []
    if not ENABLE_COND2: reference.append(('条件2-主客积分差', check_cond2))
    if not ENABLE_COND4: reference.append(('条件4-对赛往绩胜场', check_cond4))
    if not ENABLE_COND5: reference.append(('条件5-技术统计', check_cond5))

    print("=" * 70)
    print("比赛筛选")
    print(f"启用条件: {len(enabled)}个 -> {', '.join(name for name, _ in enabled)}")
    if MATCH_ID_PATTERN:
        print(f"比赛编号筛选: match_id LIKE '{MATCH_ID_PATTERN}'")
    print("=" * 70)

    # 获取所有比赛（可按match_id模式筛选，*通配符转换为SQL的%）
    if MATCH_ID_PATTERN:
        like_pattern = MATCH_ID_PATTERN.replace('*', '%')
        cursor.execute("""
            SELECT match_id, single_match, league, home_team, away_team, match_time
            FROM `match` WHERE match_id LIKE %s ORDER BY match_id
        """, (like_pattern,))
    else:
        cursor.execute("""
            SELECT match_id, single_match, league, home_team, away_team, match_time
            FROM `match` ORDER BY match_id
        """)
    matches = cursor.fetchall()
    print(f"待筛选比赛: {len(matches)} 场\n")

    passed = []
    # 统计每个条件淘汰了多少场
    eliminated = {name: 0 for name, _ in enabled}

    def eval_cond(name, func, match_id, league, home_team, away_team):
        """统一分发条件函数参数(条件4需球队名, 条件8需联赛名)"""
        if func is check_cond4:
            return func(cursor, match_id, home_team, away_team)
        if func is check_cond8:
            return func(cursor, match_id, league)
        return func(cursor, match_id)

    for match_id, single_match, league, home_team, away_team, match_time in matches:
        match_ok = True
        match_results = {}
        for name, func in enabled:
            ok, detail = eval_cond(name, func, match_id, league, home_team, away_team)
            match_results[name] = (ok, detail)
            if not ok:
                match_ok = False
                eliminated[name] += 1
        if match_ok:
            passed.append((match_id, single_match, league, home_team, away_team, match_time, match_results))

    # 输出通过筛选的比赛（report_lines用于邮件发送）
    report_lines = []

    def out(line=''):
        print(line)
        report_lines.append(line)

    out("=" * 70)
    out(f"筛选结果: {len(passed)} 场通过 (共{len(matches)}场)")
    out("=" * 70)
    for match_id, single_match, league, home_team, away_team, match_time, results in passed:
        single = f" [{single_match}]" if single_match else ""
        out(f"\n★ [{match_id}]{single} {league}  {home_team} vs {away_team}  ({match_time})")
        for name, (ok, detail) in results.items():
            out(f"    {name}: {detail}")
        # 已禁用条件的参考输出(不影响筛选结果)
        for name, func in reference:
            try:
                ok, detail = eval_cond(name, func, match_id, league, home_team, away_team)
                out(f"    [参考] {name}: {'✓' if ok else '✗'} {detail}")
            except Exception:
                out(f"    [参考] {name}: 数据缺失")
        # 预测结果
        prediction = None
        if ENABLE_PREDICT:
            prediction, score, details = predict_match(cursor, match_id, home_team, away_team)
            out(f"    预测: {prediction}  (综合评分: {score:+.3f})")
        # 六比分预测（定方向 -> 定进球数 -> 锁定6个比分）
        if ENABLE_SCORE_PREDICT:
            scores6, score_reason = predict_six_scores(cursor, match_id, prediction)
            out(f"    预测6个比分: {format_scores(scores6)}")
            out(f"    {score_reason}")

    # 留存筛选结果供次日--review对照实际比分/SP
    entries = [{
        'match_id': m[0], 'single_match': m[1], 'league': m[2],
        'home_team': m[3], 'away_team': m[4], 'match_time': m[5],
        'filtered_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    } for m in passed]
    if entries:
        append_filter_log(entries)
        print(f"\n✓ 已追加{len(entries)}条筛选记录到{os.path.basename(FILTER_RESULT_FILE)}"
              f"(供--review回顾)")

    # 输出各条件淘汰统计
    print("\n" + "=" * 70)
    print("各条件淘汰统计:")
    for name, _ in enabled:
        print(f"    {name}: 淘汰 {eliminated[name]} 场")
    print("=" * 70)

    # ====================================================================
    # 预测写入数据库：对所有未开始比赛预测并写入match表
    # ====================================================================
    if ENABLE_PREDICT:
        # 确保match表存在prediction列（旧库兼容，不存在则自动补建）
        cursor.execute("""
            SELECT COUNT(*) FROM information_schema.columns
            WHERE table_schema = DATABASE() AND table_name = 'match' AND column_name = 'prediction'
        """)
        if cursor.fetchone()[0] == 0:
            cursor.execute("ALTER TABLE `match` ADD COLUMN `prediction` VARCHAR(20) DEFAULT NULL COMMENT '预测结果: 胜平/平负'")
            conn.commit()
        cursor.execute("""
            SELECT match_id, league, home_team, away_team, match_time, status
            FROM `match` ORDER BY match_id
        """)
        all_matches = cursor.fetchall()
        predict_count = 0
        for mid, league, ht, at, mt, status in all_matches:
            if status and any(k in str(status) for k in ['完', '已结束', 'Finished']):
                continue
            prediction, score, details = predict_match(cursor, mid, ht, at)
            cursor.execute("UPDATE `match` SET prediction = %s WHERE match_id = %s",
                           (prediction, mid))
            predict_count += 1
        conn.commit()
        print(f"\n共预测 {predict_count} 场未开始比赛，已写入match表")

    # 邮件通知（受config.py中ENABLE_EMAIL开关控制）
    if ENABLE_EMAIL:
        send_email('\n'.join(report_lines))

    cursor.close()
    conn.close()


def send_email(content, subject=None):
    """将筛选结果通过邮件发送(subject缺省为'比赛筛选结果 日期')"""
    try:
        msg = MIMEText(content, 'plain', 'utf-8')
        today = datetime.now().strftime('%Y-%m-%d')
        msg['Subject'] = Header(subject or f'比赛筛选结果 {today}', 'utf-8')
        msg['From'] = EMAIL_CONFIG['sender']
        msg['To'] = EMAIL_CONFIG['receiver']

        server = smtplib.SMTP_SSL(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.login(EMAIL_CONFIG['sender'], EMAIL_CONFIG['password'])
        server.sendmail(EMAIL_CONFIG['sender'], [EMAIL_CONFIG['receiver']], msg.as_string())
        server.quit()
        print("\n✓ 邮件发送成功")
    except Exception as e:
        print(f"\n✗ 邮件发送失败: {e}")


if __name__ == '__main__':
    main()
