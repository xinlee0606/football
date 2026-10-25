# tech_stats_compare.py - 技术统计对比模块
# 功能: 从 recent_form 表提取两队技术统计数据，生成对比报告和盘口推荐
#
# 用法:
#   from tech_stats_compare import get_tech_stats_comparison, format_tech_stats_report
#   report, recommendation = get_tech_stats_comparison(conn, home_team, away_team, match_id)
#   print(format_tech_stats_report(report, recommendation))
import pymysql
from config import DB_CONFIG

# 技术指标配置
TECH_METRICS = [
    ('shots', '射门', 0.20),
    ('shots_on_target', '射正', 0.20),
    ('dangerous_attacks', '危险进攻', 0.20),
    ('attacks', '进攻', 0.15),
    ('possession', '控球率%', 0.10),
    ('pass_success_rate', '传球成功率%', 0.10),
    ('corners', '角球', 0.05),
]

# 时间窗口权重
WINDOW_WEIGHTS = {
    3: 0.50,
    6: 0.30,
    10: 0.20,
}


def _safe_float(val):
    """安全转浮点数（兼容varchar存储的数字）"""
    if val is None:
        return None
    try:
        s = str(val).strip().replace('%', '')
        return float(s) if s else None
    except (ValueError, TypeError):
        return None


def _get_team_tech_stats(cursor, team_name, match_id, max_matches=10):
    """获取球队近期技术统计（按match_id倒序，排除当前比赛）
    返回: [{shots, shots_on_target, ...}, ...] 最多max_matches条"""
    cursor.execute("""
        SELECT shots, shots_on_target, possession, dangerous_attacks, attacks,
               corners, pass_success_rate
        FROM recent_form
        WHERE team_name = %s
          AND match_id < %s
          AND is_current_match = 0
          AND shots IS NOT NULL
        ORDER BY match_id DESC
        LIMIT %s
    """, (team_name, match_id, max_matches))
    
    rows = cursor.fetchall()
    stats = []
    for r in rows:
        stats.append({
            'shots': _safe_float(r[0]),
            'shots_on_target': _safe_float(r[1]),
            'possession': _safe_float(r[2]),
            'dangerous_attacks': _safe_float(r[3]),
            'attacks': _safe_float(r[4]),
            'corners': _safe_float(r[5]),
            'pass_success_rate': _safe_float(r[6]),
        })
    return stats


def _calc_window_avg(stats, window):
    """计算指定窗口的均值（取前window条有数据的记录）"""
    if not stats:
        return {}
    
    result = {}
    for key in ['shots', 'shots_on_target', 'possession', 'dangerous_attacks',
                'attacks', 'corners', 'pass_success_rate']:
        vals = [s[key] for s in stats[:window] if s[key] is not None]
        result[key] = sum(vals) / len(vals) if vals else None
    return result


def _calc_score(avgs):
    """计算综合评分（加权归一化）"""
    if not avgs:
        return 0.0
    
    score = 0.0
    total_weight = 0.0
    
    for key, _, weight in TECH_METRICS:
        val = avgs.get(key)
        if val is not None:
            score += val * weight
            total_weight += weight
    
    return score / total_weight if total_weight > 0 else 0.0


def get_tech_stats_comparison(conn, home_team, away_team, match_id):
    """获取两队技术统计对比
    返回: (report_dict, recommendation_str)
    report_dict: {
        'home': {3: {...}, 6: {...}, 10: {...}},
        'away': {3: {...}, 6: {...}, 10: {...}},
        'home_score': float,
        'away_score': float,
    }
    recommendation_str: '上盘(胜平)' / '下盘(平负)' / '数据不足'
    """
    cursor = conn.cursor()
    
    # 获取两队技术统计
    home_stats = _get_team_tech_stats(cursor, home_team, match_id, max_matches=10)
    away_stats = _get_team_tech_stats(cursor, away_team, match_id, max_matches=10)
    
    cursor.close()
    
    # 计算各窗口均值
    report = {
        'home': {
            3: _calc_window_avg(home_stats, 3),
            6: _calc_window_avg(home_stats, 6),
            10: _calc_window_avg(home_stats, 10),
        },
        'away': {
            3: _calc_window_avg(away_stats, 3),
            6: _calc_window_avg(away_stats, 6),
            10: _calc_window_avg(away_stats, 10),
        },
    }
    
    # 计算加权综合评分
    home_score = 0.0
    away_score = 0.0
    has_data = False
    
    for window, weight in WINDOW_WEIGHTS.items():
        home_avg = report['home'][window]
        away_avg = report['away'][window]
        
        home_s = _calc_score(home_avg)
        away_s = _calc_score(away_avg)
        
        if home_s > 0 or away_s > 0:
            has_data = True
            home_score += home_s * weight
            away_score += away_s * weight
    
    report['home_score'] = home_score
    report['away_score'] = away_score
    report['has_data'] = has_data
    
    # 生成推荐
    if not has_data:
        recommendation = '数据不足'
    elif home_score > away_score:
        recommendation = '上盘(胜平)'
    else:
        recommendation = '下盘(平负)'
    
    report['recommendation'] = recommendation
    
    return report, recommendation


def format_tech_stats_report(report, recommendation, indent='    '):
    """格式化技术统计对比报告
    返回: 字符串列表（每行一个字符串）"""
    
    if not report.get('has_data'):
        return [f'{indent}技术统计: 数据不足']
    
    lines = []
    lines.append(f'{indent}─── 技术统计对比 ───')
    
    # 表头
    header = f'{indent}{"指标":<10} {"主队(近3)":>10} {"客队(近3)":>10} | {"主队(近6)":>10} {"客队(近6)":>10} | {"主队(近10)":>10} {"客队(近10)":>10}'
    lines.append(header)
    
    # 数据行
    for key, label, _ in TECH_METRICS:
        h3 = report['home'][3].get(key)
        a3 = report['away'][3].get(key)
        h6 = report['home'][6].get(key)
        a6 = report['away'][6].get(key)
        h10 = report['home'][10].get(key)
        a10 = report['away'][10].get(key)
        
        def fmt(v):
            return f'{v:.1f}' if v is not None else 'N/A'
        
        row = f'{indent}{label:<10} {fmt(h3):>10} {fmt(a3):>10} | {fmt(h6):>10} {fmt(a6):>10} | {fmt(h10):>10} {fmt(a10):>10}'
        lines.append(row)
    
    # 综合评分和推荐
    score_diff = report["home_score"] - report["away_score"]
    diff_sign = '+' if score_diff > 0 else ''
    lines.append(f'{indent}综合评分: 主队 {report["home_score"]:.1f} vs 客队 {report["away_score"]:.1f} (分差 {diff_sign}{score_diff:.1f}) → 推荐{recommendation}')
    
    return lines


# 测试
if __name__ == '__main__':
    conn = pymysql.connect(**DB_CONFIG)
    try:
        # 测试用例
        report, rec = get_tech_stats_comparison(conn, '阿森纳', '切尔西', '2026102001')
        lines = format_tech_stats_report(report, rec)
        for line in lines:
            print(line)
    finally:
        conn.close()
