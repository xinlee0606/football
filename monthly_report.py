# monthly_report.py - 每月预测命中率分析报告
# 功能:
#   1. 分析上月 UNDER/OVER 模型预测记录(置信度筛选)
#   2. 多维度命中率分析: 联赛、比赛类型、盘口、大小球线、置信度区间
#   3. 盈亏分析(基于SP值)
#   4. 生成报告并发送邮件
#
# 用法:
#   python monthly_report.py                    # 分析上月(默认)
#   python monthly_report.py --month 202610     # 分析指定月份
#   python monthly_report.py --local            # 使用本地文件(不连服务器)
import argparse
import json
import os
import smtplib
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from email.header import Header
from email.mime.text import MIMEText

import pymysql

from config import DB_CONFIG, EMAIL_CONFIG, ENABLE_EMAIL

# ==================== 配置 ====================
# 预测记录文件路径(服务器)
SERVER_PRED_DIR = '/coding/live'
# 预测记录文件路径(本地)
LOCAL_PRED_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型置信度阈值
OVER_CONF_THRESHOLD = 0.55
UNDER_CONF_THRESHOLD = 0.90

# 模型比分池
OVER_POOL = ['2-1', '1-2', '2-2', '3-1', '3-0', '0-3', '3-2', '1-3', '2-3']
UNDER_POOL = ['1-1', '1-0', '2-1', '0-1', '1-2', '0-0', '2-0', '0-2', '3-0']

# OVER 模型文件名
OVER_PRED_FILE = 'ml_predictions.jsonl'
UNDER_PRED_FILE = 'predictions_under.jsonl'


# ==================== 工具函数 ====================
def parse_score(score):
    """解析比分字符串 -> (home, away) 整数元组"""
    if not score:
        return None, None
    import re
    m = re.match(r'^(\d+)[:\-](\d+)$', str(score).strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def is_finished(score):
    """判断比分是否已完场"""
    if not score:
        return False
    import re
    return bool(re.match(r'^\d+[:\-]\d+$', str(score).strip()))


def norm_score(score):
    """标准化比分格式为 X-Y"""
    if not score:
        return score
    h, a = parse_score(score)
    if h is not None:
        return f"{h}-{a}"
    return score


def get_month_range(year_month=None):
    """获取月份的起止日期
    year_month: 'YYYYMM' 格式字符串, 默认上月
    返回: (start_date, end_date, label) 如 ('20261001', '20261031', '2026年10月')
    """
    if year_month:
        year = int(year_month[:4])
        month = int(year_month[4:6])
    else:
        today = datetime.now()
        first_of_this_month = today.replace(day=1)
        last_month_end = first_of_this_month - timedelta(days=1)
        year = last_month_end.year
        month = last_month_end.month

    start_date = f"{year}{month:02d}01"
    # 计算月末
    if month == 12:
        end_date = f"{year}1231"
    else:
        next_month = datetime(year, month + 1, 1)
        end_date = (next_month - timedelta(days=1)).strftime('%Y%m%d')

    label = f"{year}年{month}月"
    return start_date, end_date, label


def load_predictions(pred_dir, filename, start_date, end_date):
    """加载指定月份的预测记录
    返回: list of dict
    """
    filepath = os.path.join(pred_dir, filename)
    if not os.path.exists(filepath):
        return []

    records = []
    with open(filepath, encoding='utf-8') as fp:
        for line in fp:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # 按日期过滤
            date = rec.get('date', '')
            if date < start_date or date > end_date:
                continue
            records.append(rec)

    # 去重: 同一 match_id 保留最后一条(predicted_at 最大)
    latest = {}
    for rec in records:
        mid = rec.get('match_id', '')
        existing = latest.get(mid)
        if existing is None or rec.get('predicted_at', '') > existing.get('predicted_at', ''):
            latest[mid] = rec

    return list(latest.values())


def handicap_str_to_num(h):
    """盘口中文字符串 -> 数值(正=主队让球, 负=受让)"""
    if h is None:
        return None
    h = str(h).strip()
    _MAP = {
        '受让球半': -1.5, '受球半': -1.5,
        '受让一/球半': -1.25, '受一/球半': -1.25, '受让一球/球半': -1.25,
        '受让一球': -1.0, '受一球': -1.0,
        '受让半/一': -0.75, '受半/一': -0.75, '受让半球/一球': -0.75,
        '受让半球': -0.5, '受半球': -0.5,
        '受让平/半': -0.25, '受平/半': -0.25, '受让平手/半球': -0.25,
        '受让平手': 0.0, '受平手': 0.0,
        '平手': 0.0,
        '平/半': 0.25, '平手/半球': 0.25,
        '半球': 0.5, '半/一': 0.75, '半球/一球': 0.75,
        '一球': 1.0, '一/球半': 1.25, '一球/球半': 1.25,
        '球半': 1.5, '球半/两球': 1.75,
        '两球': 2.0, '两球/两球半': 2.25, '两球半': 2.5,
        '两球半/三球': 2.75, '三球': 3.0,
    }
    return _MAP.get(h)


def ou_line_str_to_num(line):
    """大小球线字符串 -> 数值(如 '2.5/3' -> 2.75, '2.5' -> 2.5)"""
    if line is None:
        return None
    line = str(line).strip()
    if '/' in line:
        parts = line.split('/')
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except (ValueError, IndexError):
            return None
    try:
        return float(line)
    except ValueError:
        return None


def query_match_data(conn, match_ids):
    """从数据库查询比赛详情(比分、盘口、大小球线等)
    返回: dict of match_id -> {score, league, home_team, away_team, single_match,
                              avg_handicap, avg_ou_line, sp_value}
    """
    if not match_ids:
        return {}

    cur = conn.cursor()
    result = {}

    # 分批查询(避免SQL过长)
    batch_size = 500
    for i in range(0, len(match_ids), batch_size):
        batch = match_ids[i:i + batch_size]
        fmt = ','.join(['%s'] * len(batch))

        # 比赛基本信息
        cur.execute(f"""
            SELECT match_id, league, home_team, away_team, score, single_match
            FROM `match` WHERE match_id IN ({fmt})
        """, batch)
        for mid, lg, ht, at, sc, sm in cur.fetchall():
            result.setdefault(mid, {})
            result[mid].update({
                'match_id': mid, 'league': lg, 'home_team': ht,
                'away_team': at, 'score': sc, 'single_match': sm
            })

        # 亚盘(终盘) - 取众数(最常见盘口)
        cur.execute(f"""
            SELECT match_id, current_handicap
            FROM asian_odds 
            WHERE match_id IN ({fmt}) AND current_handicap IS NOT NULL AND current_handicap != ''
        """, batch)
        from collections import Counter
        hc_by_match = defaultdict(list)
        for mid, hc in cur.fetchall():
            hc_by_match[mid].append(hc)
        for mid, hc_list in hc_by_match.items():
            result.setdefault(mid, {})
            # 取出现最多的盘口
            most_common = Counter(hc_list).most_common(1)
            if most_common:
                hc_str = most_common[0][0]
                result[mid]['handicap_str'] = hc_str
                result[mid]['avg_handicap'] = handicap_str_to_num(hc_str)

        # 大小球(终盘) - 取众数
        cur.execute(f"""
            SELECT match_id, current_line
            FROM over_under_odds
            WHERE match_id IN ({fmt}) AND current_line IS NOT NULL AND current_line != ''
        """, batch)
        ou_by_match = defaultdict(list)
        for mid, ol in cur.fetchall():
            ou_by_match[mid].append(ol)
        for mid, ol_list in ou_by_match.items():
            result.setdefault(mid, {})
            most_common = Counter(ol_list).most_common(1)
            if most_common:
                ou_str = most_common[0][0]
                result[mid]['ou_line_str'] = ou_str
                result[mid]['avg_ou_line'] = ou_line_str_to_num(ou_str)

        # SP值
        cur.execute(f"""
            SELECT match_id, sp_value FROM result_sp
            WHERE CONVERT(match_id USING utf8mb4) COLLATE utf8mb4_general_ci IN ({fmt})
        """, batch)
        for mid, sp in cur.fetchall():
            result.setdefault(mid, {})
            result[mid]['sp_value'] = float(sp) if sp is not None else None

    cur.close()
    return result


# ==================== 分析引擎 ====================
def analyze_model(records, match_data, pool, pool_name, conf_threshold):
    """分析单个模型的命中情况
    返回: 分析结果字典
    """
    # 筛选: picked=true 且 p_pool >= threshold
    picked = [r for r in records
              if r.get('picked') and r.get('p_pool', 0) >= conf_threshold]

    if not picked:
        return {'total': 0, 'finished': 0, 'hit': 0, 'rate': 0}

    # 逐场判断命中
    details = []
    for rec in picked:
        mid = rec.get('match_id', '')
        md = match_data.get(mid, {})
        score = md.get('score')
        if not is_finished(score):
            continue

        h, a = parse_score(score)
        if h is None:
            continue

        norm = f"{h}-{a}"
        hit_pool = norm in pool

        # 解析 top_picks
        picks_str = rec.get('top_picks', '')
        picks = [s.split(':')[0] for s in picks_str.split(';') if s]
        hit_top = norm in picks

        details.append({
            'match_id': mid,
            'league': rec.get('league', md.get('league', '?')),
            'date': rec.get('date', ''),
            'p_pool': rec.get('p_pool', 0),
            'score': norm,
            'hit_pool': hit_pool,
            'hit_top': hit_top,
            'sp': md.get('sp_value'),
            'handicap': md.get('avg_handicap'),
            'ou_line': md.get('avg_ou_line'),
            'single_match': md.get('single_match', ''),
            'home_team': md.get('home_team', ''),
            'away_team': md.get('away_team', ''),
        })

    if not details:
        return {'total': len(picked), 'finished': 0, 'hit': 0, 'rate': 0,
                'details': [], 'by_league': {}, 'by_handicap': {},
                'by_ou_line': {}, 'by_conf': {}, 'profit': 0}

    # 总体统计
    n_finished = len(details)
    n_hit_pool = sum(1 for d in details if d['hit_pool'])
    n_hit_top = sum(1 for d in details if d['hit_top'])
    rate_pool = n_hit_pool / n_finished if n_finished else 0
    rate_top = n_hit_top / n_finished if n_finished else 0

    # 按联赛统计
    by_league = defaultdict(lambda: {'total': 0, 'hit_pool': 0, 'hit_top': 0})
    for d in details:
        lg = d['league']
        by_league[lg]['total'] += 1
        by_league[lg]['hit_pool'] += int(d['hit_pool'])
        by_league[lg]['hit_top'] += int(d['hit_top'])

    # 按盘口统计
    by_handicap = defaultdict(lambda: {'total': 0, 'hit_pool': 0, 'hit_top': 0})
    for d in details:
        hc = d['handicap']
        if hc is None:
            bucket = '未知'
        elif hc <= -1.5:
            bucket = '受让1.5+'
        elif hc <= -0.75:
            bucket = '受让0.75~1.5'
        elif hc <= -0.25:
            bucket = '受让0.25~0.75'
        elif hc < 0.25:
            bucket = '平手'
        elif hc < 0.75:
            bucket = '让0.25'
        elif hc < 1.25:
            bucket = '让0.5~0.75'
        elif hc < 1.75:
            bucket = '让1~1.5'
        else:
            bucket = '让1.75+'
        by_handicap[bucket]['total'] += 1
        by_handicap[bucket]['hit_pool'] += int(d['hit_pool'])
        by_handicap[bucket]['hit_top'] += int(d['hit_top'])

    # 按大小球线统计
    by_ou_line = defaultdict(lambda: {'total': 0, 'hit_pool': 0, 'hit_top': 0})
    for d in details:
        ou = d['ou_line']
        if ou is None:
            bucket = '未知'
        elif ou < 2.0:
            bucket = '<2.0'
        elif ou < 2.5:
            bucket = '2.0~2.25'
        elif ou < 3.0:
            bucket = '2.5~2.75'
        elif ou < 3.5:
            bucket = '3.0~3.25'
        else:
            bucket = '3.5+'
        by_ou_line[bucket]['total'] += 1
        by_ou_line[bucket]['hit_pool'] += int(d['hit_pool'])
        by_ou_line[bucket]['hit_top'] += int(d['hit_top'])

    # 按置信度区间统计
    by_conf = defaultdict(lambda: {'total': 0, 'hit_pool': 0, 'hit_top': 0})
    for d in details:
        p = d['p_pool']
        if p < 0.6:
            bucket = '55%-60%'
        elif p < 0.7:
            bucket = '60%-70%'
        elif p < 0.8:
            bucket = '70%-80%'
        elif p < 0.9:
            bucket = '80%-90%'
        else:
            bucket = '90%+'
        by_conf[bucket]['total'] += 1
        by_conf[bucket]['hit_pool'] += int(d['hit_pool'])
        by_conf[bucket]['hit_top'] += int(d['hit_top'])

    # 按比赛类型统计(单场/串关 + 杯赛/联赛)
    by_match_type = defaultdict(lambda: {'total': 0, 'hit_pool': 0, 'hit_top': 0})
    for d in details:
        sm = d.get('single_match', '')
        lg = d.get('league', '')
        # 单场/串关
        if sm and ('单场' in str(sm)):
            match_type = '单场'
        else:
            match_type = '串关'
        by_match_type[match_type]['total'] += 1
        by_match_type[match_type]['hit_pool'] += int(d['hit_pool'])
        by_match_type[match_type]['hit_top'] += int(d['hit_top'])
        # 杯赛/联赛(简单规则: 含'杯'/'甲'/'乙'等关键字)
        if any(k in lg for k in ['杯', '锦标', '决赛']):
            league_type = '杯赛'
        else:
            league_type = '联赛'
        by_match_type[league_type]['total'] += 1
        by_match_type[league_type]['hit_pool'] += int(d['hit_pool'])
        by_match_type[league_type]['hit_top'] += int(d['hit_top'])

    # 盈亏分析(投注命中时: SP*2*0.65 - 注数, 未中: -注数)
    # 这里按9注计算(OVER/UNDER都是9比分池)
    n_picks = 9
    profit = 0.0
    profit_count = 0
    for d in details:
        if d['sp'] is not None:
            profit_count += 1
            if d['hit_top']:
                profit += d['sp'] * 2 * 0.65 - n_picks
            else:
                profit -= n_picks

    return {
        'total': len(picked),
        'finished': n_finished,
        'hit_pool': n_hit_pool,
        'hit_top': n_hit_top,
        'rate_pool': rate_pool,
        'rate_top': rate_top,
        'details': details,
        'by_league': dict(by_league),
        'by_handicap': dict(by_handicap),
        'by_ou_line': dict(by_ou_line),
        'by_conf': dict(by_conf),
        'by_match_type': dict(by_match_type),
        'profit': profit,
        'profit_count': profit_count,
    }


# ==================== 报告生成 ====================
def format_report(over_result, under_result, month_label):
    """格式化月度分析报告"""
    lines = []

    def out(s=''):
        lines.append(s)

    out("=" * 70)
    out(f"  北单比分模型 {month_label} 预测命中率月度报告")
    out(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out("=" * 70)
    out()

    # ---------- OVER 模型 ----------
    out("━" * 70)
    out("【一、OVER 模型】置信度阈值 ≥ 55%")
    out("━" * 70)
    if over_result.get('finished', 0) == 0:
        out("  本月无OVER模型推荐记录或均未完场")
    else:
        r = over_result
        out(f"  推荐场次: {r['total']}  已完场: {r['finished']}")
        out(f"  池内命中: {r['hit_pool']}/{r['finished']} ({r['rate_pool']*100:.1f}%)")
        out(f"  投注命中: {r['hit_top']}/{r['finished']} ({r['rate_top']*100:.1f}%)")
        if r.get('profit_count', 0) > 0:
            out(f"  盈亏统计: {r['profit']:+.1f} 元 / {r['profit_count']}场 (9注/场)")
            roi = r['profit'] / (r['profit_count'] * 9) * 100
            out(f"  ROI: {roi:+.1f}%")
        out()

        # 按置信度区间
        out("  ▸ 置信度区间分析:")
        conf_order = ['55%-60%', '60%-70%', '70%-80%', '80%-90%', '90%+']
        for bucket in conf_order:
            d = r['by_conf'].get(bucket)
            if d and d['total'] > 0:
                rate = d['hit_pool'] / d['total'] * 100
                out(f"    {bucket:>10}: {d['hit_pool']}/{d['total']} ({rate:.1f}%)  投注{d['hit_top']}中")
        out()

        # 按联赛(Top 10)
        out("  ▸ 联赛分析(按推荐场次降序, Top 10):")
        lg_sorted = sorted(r['by_league'].items(), key=lambda x: -x[1]['total'])[:10]
        for lg, d in lg_sorted:
            rate = d['hit_pool'] / d['total'] * 100 if d['total'] else 0
            out(f"    {lg:<12}: {d['hit_pool']}/{d['total']} ({rate:.1f}%)  投注{d['hit_top']}中")
        out()

        # 按比赛类型
        out("  ▸ 比赛类型分析:")
        mt_order = ['单场', '串关', '联赛', '杯赛']
        for mt in mt_order:
            d = r['by_match_type'].get(mt)
            if d and d['total'] > 0:
                rate = d['hit_pool'] / d['total'] * 100
                out(f"    {mt:>6}: {d['hit_pool']}/{d['total']} ({rate:.1f}%)  投注{d['hit_top']}中")
        out()

        # 按盘口
        out("  ▸ 盘口分析:")
        hc_order = ['受让1.5+', '受让0.75~1.5', '受让0.25~0.75', '平手',
                    '让0.25', '让0.5~0.75', '让1~1.5', '让1.75+', '未知']
        for bucket in hc_order:
            d = r['by_handicap'].get(bucket)
            if d and d['total'] > 0:
                rate = d['hit_pool'] / d['total'] * 100
                out(f"    {bucket:>14}: {d['hit_pool']}/{d['total']} ({rate:.1f}%)  投注{d['hit_top']}中")
        out()

        # 按大小球线
        out("  ▸ 大小球线分析:")
        ou_order = ['<2.0', '2.0~2.25', '2.5~2.75', '3.0~3.25', '3.5+', '未知']
        for bucket in ou_order:
            d = r['by_ou_line'].get(bucket)
            if d and d['total'] > 0:
                rate = d['hit_pool'] / d['total'] * 100
                out(f"    {bucket:>10}: {d['hit_pool']}/{d['total']} ({rate:.1f}%)  投注{d['hit_top']}中")

    out()

    # ---------- UNDER 模型 ----------
    out("━" * 70)
    out("【二、UNDER 模型】置信度阈值 ≥ 90%")
    out("━" * 70)
    if under_result.get('finished', 0) == 0:
        out("  本月无UNDER模型推荐记录或均未完场")
    else:
        r = under_result
        out(f"  推荐场次: {r['total']}  已完场: {r['finished']}")
        out(f"  池内命中: {r['hit_pool']}/{r['finished']} ({r['rate_pool']*100:.1f}%)")
        out(f"  投注命中: {r['hit_top']}/{r['finished']} ({r['rate_top']*100:.1f}%)")
        if r.get('profit_count', 0) > 0:
            out(f"  盈亏统计: {r['profit']:+.1f} 元 / {r['profit_count']}场 (9注/场)")
            roi = r['profit'] / (r['profit_count'] * 9) * 100
            out(f"  ROI: {roi:+.1f}%")
        out()

        # 按置信度区间
        out("  ▸ 置信度区间分析:")
        conf_order = ['55%-60%', '60%-70%', '70%-80%', '80%-90%', '90%+']
        for bucket in conf_order:
            d = r['by_conf'].get(bucket)
            if d and d['total'] > 0:
                rate = d['hit_pool'] / d['total'] * 100
                out(f"    {bucket:>10}: {d['hit_pool']}/{d['total']} ({rate:.1f}%)  投注{d['hit_top']}中")
        out()

        # 按联赛(Top 10)
        out("  ▸ 联赛分析(按推荐场次降序, Top 10):")
        lg_sorted = sorted(r['by_league'].items(), key=lambda x: -x[1]['total'])[:10]
        for lg, d in lg_sorted:
            rate = d['hit_pool'] / d['total'] * 100 if d['total'] else 0
            out(f"    {lg:<12}: {d['hit_pool']}/{d['total']} ({rate:.1f}%)  投注{d['hit_top']}中")
        out()

        # 按比赛类型
        out("  ▸ 比赛类型分析:")
        mt_order = ['单场', '串关', '联赛', '杯赛']
        for mt in mt_order:
            d = r['by_match_type'].get(mt)
            if d and d['total'] > 0:
                rate = d['hit_pool'] / d['total'] * 100
                out(f"    {mt:>6}: {d['hit_pool']}/{d['total']} ({rate:.1f}%)  投注{d['hit_top']}中")
        out()

        # 按盘口
        out("  ▸ 盘口分析:")
        hc_order = ['受让1.5+', '受让0.75~1.5', '受让0.25~0.75', '平手',
                    '让0.25', '让0.5~0.75', '让1~1.5', '让1.75+', '未知']
        for bucket in hc_order:
            d = r['by_handicap'].get(bucket)
            if d and d['total'] > 0:
                rate = d['hit_pool'] / d['total'] * 100
                out(f"    {bucket:>14}: {d['hit_pool']}/{d['total']} ({rate:.1f}%)  投注{d['hit_top']}中")
        out()

        # 按大小球线
        out("  ▸ 大小球线分析:")
        ou_order = ['<2.0', '2.0~2.25', '2.5~2.75', '3.0~3.25', '3.5+', '未知']
        for bucket in ou_order:
            d = r['by_ou_line'].get(bucket)
            if d and d['total'] > 0:
                rate = d['hit_pool'] / d['total'] * 100
                out(f"    {bucket:>10}: {d['hit_pool']}/{d['total']} ({rate:.1f}%)  投注{d['hit_top']}中")

    out()

    # ---------- 综合总结 ----------
    out("━" * 70)
    out("【三、综合总结】")
    out("━" * 70)

    total_profit = 0.0
    total_matches = 0

    if over_result.get('finished', 0) > 0:
        r = over_result
        out(f"  OVER模型: 池内命中率 {r['rate_pool']*100:.1f}%, "
            f"投注命中率 {r['rate_top']*100:.1f}%")
        if r.get('profit_count', 0) > 0:
            out(f"            盈亏 {r['profit']:+.1f} 元")
            total_profit += r['profit']
            total_matches += r['profit_count']

    if under_result.get('finished', 0) > 0:
        r = under_result
        out(f"  UNDER模型: 池内命中率 {r['rate_pool']*100:.1f}%, "
            f"投注命中率 {r['rate_top']*100:.1f}%")
        if r.get('profit_count', 0) > 0:
            out(f"            盈亏 {r['profit']:+.1f} 元")
            total_profit += r['profit']
            total_matches += r['profit_count']

    if total_matches > 0:
        out(f"\n  合计盈亏: {total_profit:+.1f} 元 / {total_matches}场")
        roi = total_profit / (total_matches * 9) * 100
        out(f"  综合ROI: {roi:+.1f}%")

    out()
    out("=" * 70)
    out("  报告结束")
    out("=" * 70)

    return '\n'.join(lines)


# ==================== 邮件发送 ====================
def send_email(subject, content):
    """发送报告邮件"""
    if not ENABLE_EMAIL:
        print(f"(邮件开关ENABLE_EMAIL=False, 跳过发送: {subject})")
        return False
    try:
        print(f"正在发送邮件: {subject} -> {EMAIL_CONFIG['receiver']}")
        msg = MIMEText(content, 'plain', 'utf-8')
        msg['Subject'] = Header(subject, 'utf-8')
        msg['From'] = EMAIL_CONFIG['sender']
        msg['To'] = EMAIL_CONFIG['receiver']
        server = smtplib.SMTP_SSL(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.login(EMAIL_CONFIG['sender'], EMAIL_CONFIG['password'])
        server.sendmail(EMAIL_CONFIG['sender'], [EMAIL_CONFIG['receiver']], msg.as_string())
        server.quit()
        print(f"✓ 邮件发送成功: {subject}")
        return True
    except Exception as e:
        print(f"✗ 邮件发送失败: {e}")
        return False


# ==================== 主流程 ====================
def main():
    ap = argparse.ArgumentParser(description='月度预测命中率分析报告')
    ap.add_argument('--month', default=None, metavar='YYYYMM',
                    help='分析指定月份(默认上月)')
    ap.add_argument('--local', action='store_true',
                    help='使用本地文件(不连服务器SSH)')
    ap.add_argument('--skip-email', action='store_true',
                    help='跳过邮件发送')
    ap.add_argument('--save', action='store_true',
                    help='保存报告到文件')
    args = ap.parse_args()

    # 确定分析月份
    start_date, end_date, month_label = get_month_range(args.month)
    print(f"分析月份: {month_label} ({start_date} ~ {end_date})")

    # 确定预测文件目录
    if args.local:
        pred_dir = LOCAL_PRED_DIR
    else:
        # 尝试从服务器获取(通过SSH下载)
        pred_dir = LOCAL_PRED_DIR
        try:
            import paramiko
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect('8.130.188.90', username='root', password='Xin7231013@@@')
            sftp = ssh.open_sftp()

            for fname in [OVER_PRED_FILE, UNDER_PRED_FILE]:
                remote_path = f"{SERVER_PRED_DIR}/{fname}"
                local_path = os.path.join(pred_dir, fname)
                try:
                    sftp.get(remote_path, local_path)
                    print(f"  ✓ 下载 {fname}")
                except FileNotFoundError:
                    print(f"  ⚠ {fname} 服务器上不存在")

            sftp.close()
            ssh.close()
        except ImportError:
            print("⚠ paramiko未安装, 使用本地文件")
        except Exception as e:
            print(f"⚠ SSH连接失败({e}), 使用本地文件")

    # 加载预测记录
    print(f"\n加载预测记录...")
    over_records = load_predictions(pred_dir, OVER_PRED_FILE, start_date, end_date)
    under_records = load_predictions(pred_dir, UNDER_PRED_FILE, start_date, end_date)
    print(f"  OVER: {len(over_records)} 条")
    print(f"  UNDER: {len(under_records)} 条")

    if not over_records and not under_records:
        print("本月无预测记录, 跳过分析")
        if not args.skip_email:
            send_email(f'月度报告 {month_label} 空仓',
                       f'{month_label}无OVER/UNDER模型预测记录')
        return

    # 收集所有 match_id
    all_ids = set()
    for rec in over_records + under_records:
        all_ids.add(rec.get('match_id', ''))
    all_ids = list(all_ids)

    # 查询数据库获取比赛详情
    print(f"\n查询数据库({len(all_ids)}场比赛)...")
    conn = pymysql.connect(**DB_CONFIG)
    match_data = query_match_data(conn, all_ids)
    conn.close()
    print(f"  ✓ 获取到 {len(match_data)} 场比赛数据")

    # 分析 OVER 模型
    print(f"\n分析 OVER 模型(置信度 ≥ {OVER_CONF_THRESHOLD*100:.0f}%)...")
    over_result = analyze_model(over_records, match_data, OVER_POOL,
                                'OVER', OVER_CONF_THRESHOLD)
    print(f"  推荐 {over_result.get('total', 0)} 场, "
          f"完场 {over_result.get('finished', 0)} 场, "
          f"池内命中 {over_result.get('hit_pool', 0)} 场")

    # 分析 UNDER 模型
    print(f"\n分析 UNDER 模型(置信度 ≥ {UNDER_CONF_THRESHOLD*100:.0f}%)...")
    under_result = analyze_model(under_records, match_data, UNDER_POOL,
                                 'UNDER', UNDER_CONF_THRESHOLD)
    print(f"  推荐 {under_result.get('total', 0)} 场, "
          f"完场 {under_result.get('finished', 0)} 场, "
          f"池内命中 {under_result.get('hit_pool', 0)} 场")

    # 生成报告
    report = format_report(over_result, under_result, month_label)
    print("\n" + report)

    # 保存报告到文件
    if args.save:
        report_file = os.path.join(LOCAL_PRED_DIR,
                                   f'monthly_report_{start_date[:6]}.txt')
        with open(report_file, 'w', encoding='utf-8') as fp:
            fp.write(report)
        print(f"\n✓ 报告已保存: {report_file}")

    # 发送邮件
    if not args.skip_email:
        # 邮件标题包含关键数据
        parts = []
        if over_result.get('finished', 0) > 0:
            parts.append(f"OVER {over_result['rate_pool']*100:.0f}%")
        if under_result.get('finished', 0) > 0:
            parts.append(f"UNDER {under_result['rate_pool']*100:.0f}%")
        summary = ' / '.join(parts) if parts else '空仓'
        subject = f"月度报告 {month_label} {summary}"
        send_email(subject, report)

    # 清理下载的临时文件
    if not args.local:
        for fname in [OVER_PRED_FILE]:
            fpath = os.path.join(pred_dir, fname)
            if os.path.exists(fpath) and fname == OVER_PRED_FILE:
                # OVER 文件本地可能本来就没有，清理下载的
                pass


if __name__ == '__main__':
    main()
