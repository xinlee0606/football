# predict_over.py - 混合模型每日预测脚本（Step 4）
# 功能:
#   1. 对当日未开赛比赛生成特征, 加载最新模型预测池内概率与11比分概率分布
#   2. 投注纪律: P_pool >= 置信度阈值 + 排除联赛黑名单, 不限推荐场次数量
#   3. 预测方向(胜平/平负)写入match.prediction, 沿用现有格式不破坏下游
#   4. 推荐记录追加到 ml_predictions.jsonl, 供持续学习监控
#   5. 预测结果邮件发送(受config.py的ENABLE_EMAIL开关控制, --skip-email可跳过)
#   6. 监控: 滚动近MONITOR_N场已完场推荐的池内命中率, 跌破基线一定比例邮件告警
#
# 用法:
#   python -u predict_over.py                 # 预测当天比赛
#   python -u predict_over.py --date 20260818 # 指定日期
#   python -u predict_over.py --all-unfinished # 预测全部未开赛比赛(不限日期)
#   python -u predict_over.py --review          # 回看昨天预测与实际比分对照
#   python -u predict_over.py --review 20260816  # 回看指定日期预测结果
#   python -u predict_over.py --backfill 20260906 --min-prob 0.75  # 补录指定日期预测记录
import argparse
import json
import os
import smtplib
import subprocess
from datetime import datetime, timedelta
from email.header import Header
from email.mime.text import MIMEText

import numpy as np
import pymysql

from config import DB_CONFIG, EMAIL_CONFIG, ENABLE_EMAIL
from features_over import OVER, OVER_SET, is_target_handicap_over, is_valid_ou_line_over, is_cup_league_over, is_finished, load_or_build_over, parse_score
from incremental_features import load_features_incremental
from train_over import (MODEL_DIR, TOP_PICKS, dc_matrix, load_model,
                         matrix_pool_scores, predict_rows)
from tech_stats_compare import get_tech_stats_comparison, format_tech_stats_report

PRED_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'ml_predictions.jsonl')
BASELINE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'baseline_report.json')
BLACKLIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'backtest_result.json')

# ==================== 投注纪律配置 ====================
# 推荐方式: 不限场次数量, 只按置信度筛选(P_pool >= 阈值且不在联赛黑名单)
# 阈值优先级: 命令行--min-prob > 模型训练时按EV选出的阈值 > 兜底值
MIN_POOL_PROB_FALLBACK = 0.60   # 模型未提供阈值时的兜底阈值
MONITOR_N = 100             # 监控滚动窗口场次
MONITOR_DROP_RATIO = 0.85   # 命中率跌破 基线*该比例 触发告警


# ==================== 辅助 ====================
def scrape_euro_odds_for_date(target_date):
    """爬取指定日期比赛的最新欧赔数据"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scrape_euro_odds.py')
    if not os.path.exists(script_path):
        print(f"  ⚠ 未找到欧赔爬虫脚本: {script_path}")
        return False
    try:
        # 使用虚拟环境中的Python解释器
        python_exe = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv', 'bin', 'python')
        result = subprocess.run(
            [python_exe, '-u', script_path, '--date', target_date],
            cwd=os.path.dirname(script_path),
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        if result.returncode == 0:
            print("  ✓ 欧赔数据爬取完成")
            return True
        else:
            print(f"   欧赔爬虫失败: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print("   欧赔爬虫超时(>5分钟)")
        return False
    except Exception as e:
        print(f"  ✗ 欧赔爬虫异常: {e}")
        return False


def load_blacklist():
    """加载联赛黑名单(backtest.py产出), 失败返回空列表"""
    try:
        with open(BLACKLIST_FILE, encoding='utf-8') as fp:
            return json.load(fp).get('league_blacklist', [])
    except (OSError, json.JSONDecodeError):
        return []


def load_baseline():
    try:
        with open(BASELINE_FILE, encoding='utf-8') as fp:
            return json.load(fp).get('baseline_pool_rate')
    except (OSError, json.JSONDecodeError):
        return None


def direction_from_matrix(lh, la, rho):
    """由比分矩阵推导方向, 返回 '胜平'/'平负' (兼容match.prediction现有格式)"""
    m = dc_matrix(lh, la, rho)
    p_home = sum(m[x, y] for x in range(m.shape[0]) for y in range(m.shape[1]) if x > y)
    p_away = sum(m[x, y] for x in range(m.shape[0]) for y in range(m.shape[1]) if x < y)
    return '胜平' if p_home >= p_away else '平负'


def fmt_meta(r, col, default='-'):
    """安全取元信息列(兼容旧缓存缺列/NaN), 如single_match='单场99', match_time='02:30'"""
    if col not in r.index:
        return default
    v = r[col]
    if v is None or (isinstance(v, float) and v != v):
        return default
    s = str(v).strip()
    return s if s else default


def send_email(subject, content):
    """邮件通知(受config.py中ENABLE_EMAIL开关控制)"""
    if not ENABLE_EMAIL:
        print(f"  (邮件开关ENABLE_EMAIL=False, 跳过发送: {subject})")
        return False
    try:
        print(f"  正在发送邮件: {subject} -> {EMAIL_CONFIG['receiver']}")
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


# ==================== 持续学习监控 ====================
def append_prediction_log(entries):
    with open(PRED_LOG_FILE, 'a', encoding='utf-8') as fp:
        for e in entries:
            fp.write(json.dumps(e, ensure_ascii=False) + '\n')


def replace_prediction_log(entries, target_date=None, keep_backfill=True):
    """覆盖式写入预测记录: 先删除同日旧记录再写入新记录,
    保证jsonl每个日期只保留最后一次预测的候选集, --review回顾与最新预测邮件场次一致.
    target_date=None时按本次entries的match_id集合覆盖(--all-unfinished跨日期场景);
    keep_backfill=True时保留补录记录(常规预测只覆盖常规记录),
    补录调用传keep_backfill=False覆盖同日全部旧记录(补录含已完场, 集合更全)."""
    old = []
    if os.path.exists(PRED_LOG_FILE):
        with open(PRED_LOG_FILE, encoding='utf-8') as fp:
            old = [json.loads(line) for line in fp if line.strip()]
    if target_date is not None:
        def _drop(l):
            return str(l.get('match_id', '')).startswith(str(target_date)) \
                and not (keep_backfill and l.get('backfill'))
    else:
        ids = {str(e['match_id']) for e in entries}
        def _drop(l):
            return str(l.get('match_id', '')) in ids \
                and not (keep_backfill and l.get('backfill'))
    kept = [l for l in old if not _drop(l)]
    with open(PRED_LOG_FILE, 'w', encoding='utf-8') as fp:
        for l in kept + list(entries):
            fp.write(json.dumps(l, ensure_ascii=False) + '\n')


def monitor(conn):
    """核对历史推荐与完场比分, 滚动命中率跌破基线比例则告警"""
    if not os.path.exists(PRED_LOG_FILE):
        return
    with open(PRED_LOG_FILE, encoding='utf-8') as fp:
        logs = [json.loads(line) for line in fp if line.strip()]
    picked = [l for l in logs if l.get('picked')]
    if not picked:
        return
    ids = [l['match_id'] for l in picked]
    fmt = ','.join(['%s'] * len(ids))
    cur = conn.cursor()
    cur.execute(f"SELECT match_id, score FROM `match` WHERE match_id IN ({fmt})", ids)
    scores = {mid: sc for mid, sc in cur.fetchall()}
    cur.close()

    judged = []
    for l in picked:
        sc = scores.get(l['match_id'])
        if not is_finished(sc):  # 兼容'1-0'(横杠)与'1:0'(冒号)两种格式
            continue
        h, a = parse_score(sc)
        if h is None:
            continue
        judged.append({'match_id': l['match_id'],
                       'hit': f"{h}-{a}" in OVER})
    if len(judged) < 20:
        return
    recent = judged[-MONITOR_N:]
    hit_rate = sum(j['hit'] for j in recent) / len(recent)
    baseline = load_baseline() or 0.55
    print(f"\n持续学习监控: 近{len(recent)}场已完场推荐池内命中率 "
          f"{hit_rate * 100:.1f}% (基线{baseline * 100:.1f}%)")
    if hit_rate < baseline * MONITOR_DROP_RATIO:
        print(f"⚠ 命中率跌破基线的{MONITOR_DROP_RATIO:.0%}, 触发告警")
        send_email('比分模型命中率告警',
                   f"近{len(recent)}场推荐池内命中率 {hit_rate * 100:.1f}%, "
                   f"低于基线{baseline * 100:.1f}%的{MONITOR_DROP_RATIO:.0%}。\n"
                   f"建议: 运行 python -u backtest_over.py 复核, "
                   f"并提前执行重训 python -u train_over.py")


# ==================== 预测回顾 ====================
def review(target_date, send_mail=True):
    """回看指定日期预测: 读取ml_predictions.jsonl, 与实际比分对照命中情况
    命中口径: 池内=实际比分落在OVER(11比分池); 投注命中=实际比分落在所选11个投注比分
    send_mail=True时回顾结果一并邮件发送(受config.py的ENABLE_EMAIL开关控制)
    返回: (lines列表, stats字典) 或 (None, None)（无数据时）"""
    lines = []

    def out(s=''):
        lines.append(s)
        print(s)

    if not os.path.exists(PRED_LOG_FILE):
        print(f" 未找到预测记录文件: {PRED_LOG_FILE}")
        return None, None
    with open(PRED_LOG_FILE, encoding='utf-8') as fp:
        logs = [json.loads(line) for line in fp if line.strip()]

    # 按match_id去重(同一场多次预测保留最后一次), 只取目标日期
    latest = {}
    for l in logs:
        mid = str(l.get('match_id', ''))
        if mid.startswith(target_date):
            latest[mid] = l
    if not latest:
        print(f"日期{target_date}无预测记录(ml_predictions.jsonl)")
        return None, None

    # 查实际比分与对阵
    ids = list(latest.keys())
    conn = pymysql.connect(**DB_CONFIG)
    cur = conn.cursor()
    fmt = ','.join(['%s'] * len(ids))
    cur.execute(f"SELECT match_id, single_match, home_team, away_team, score "
                f"FROM `match` WHERE match_id IN ({fmt})", ids)
    info = {r[0]: {'single': r[1], 'home': r[2], 'away': r[3], 'score': r[4]}
            for r in cur.fetchall()}
    # 开奖SP值(赛果比分的SP, 比分玩法), 用于计算真实盈亏
    cur.execute(f"SELECT match_id, sp_value FROM result_sp "
                f"WHERE CONVERT(match_id USING utf8mb4) COLLATE utf8mb4_general_ci IN ({fmt})", ids)
    sp_map = {r[0]: (float(r[1]) if r[1] is not None else None)
              for r in cur.fetchall()}
    cur.close()
    # conn 保持开放，供技术统计对比使用

    out("=" * 78)
    out(f"预测回顾 {target_date}  (来源: ml_predictions.jsonl, 按池内概率由高到低)")
    out("命中口径: 池内=实际比分∈11比分池  ✓中=实际比分∈所选11个投注比分")
    out("=" * 78)

    # 按池内概率(置信度)由高到低排序
    def _conf(e):
        v = e.get('p_pool')
        return v if isinstance(v, (int, float)) else -1.0
    ordered = sorted(latest.items(), key=lambda kv: _conf(kv[1]), reverse=True)

    n_picked = pf_total = pf_pool = pf_top = pf_sp_n = 0
    pf_pl = 0.0
    for mid, e in ordered:
        picked = bool(e.get('picked'))
        p_pool = e.get('p_pool')
        picks = [s.split(':')[0] for s in str(e.get('top_picks', '')).split(';') if s]
        mi = info.get(mid, {})
        single = str(mi.get('single') or '').strip()
        single_part = f"[{single}]" if single else "[-]"
        vs = f"{mi.get('home', '?')} vs {mi.get('away', '?')}"
        sc = mi.get('score')
        flag = '★推荐' if picked else '  -   '
        pstr = f"{p_pool * 100:.1f}%" if isinstance(p_pool, (int, float)) else '-'
        top8 = '/'.join(picks[:8]) if picks else '-'
        sp = sp_map.get(mid)
        sp_str = f"SP{sp:.2f}" if sp is not None else "SP-"
        if is_finished(sc):
            h, a = parse_score(sc)
            norm = f"{h}-{a}"
            in_pool = norm in OVER_SET
            in_top = norm in picks
            hit = '✓中' if in_top else ('池内' if in_pool else '未中')
            sc_str = norm
            if picked:
                pf_total += 1
                pf_pool += int(in_pool)
                pf_top += int(in_top)
                if sp is not None:
                    pf_sp_n += 1
                    pl = (2 * sp * 0.65 - 18) if in_top else -18
                    pf_pl += pl
                    hit = f"{hit}{pl:+.1f}"
        else:
            hit = '未完场'
            sc_str = str(sc) if sc else '-'
        if picked:
            n_picked += 1
        out(f"{mid}  {single_part}  {flag}  池内{pstr:>7}  {vs}  "
            f"实际[{sc_str} {sp_str}]  投注:{top8}  -> {hit}")
        
        # 推荐场次显示技术统计对比
        if picked:
            try:
                tech_report, tech_rec = get_tech_stats_comparison(
                    conn, mi.get('home', ''), mi.get('away', ''), mid)
                tech_lines = format_tech_stats_report(tech_report, tech_rec, indent='  ')
                for tl in tech_lines:
                    out(tl)
            except Exception:
                pass  # 回顾模式静默处理技术统计错误

    out("-" * 78)
    out("【准确率统计】")
    out(f"共 {len(latest)} 场预测, 其中推荐 {n_picked} 场")
    if pf_total:
        rate_pool = pf_pool / pf_total * 100
        rate_top = pf_top / pf_total * 100
        out(f"推荐且已完场 {pf_total} 场:")
        out(f"  池内命中率: {pf_pool}/{pf_total} ({rate_pool:.1f}%)")
        out(f"  投注命中率(Top8): {pf_top}/{pf_total} ({rate_top:.1f}%)")
        if pf_sp_n:
            roi = pf_pl / (pf_sp_n * 18) * 100
            out(f"  实际盈亏: {pf_pl:+.1f} 元 / {pf_sp_n}场 (ROI {roi:+.1f}%)")
    else:
        out("推荐场次均未完场, 暂无命中统计")
    out("-" * 78)

    stats = {
        'total': len(latest),
        'n_picked': n_picked,
        'pf_total': pf_total,
        'pf_pool': pf_pool,
        'pf_top': pf_top,
        'rate_pool': pf_pool / pf_total * 100 if pf_total else 0,
        'rate_top': pf_top / pf_total * 100 if pf_total else 0,
        'pf_pl': pf_pl,
        'pf_sp_n': pf_sp_n,
    }

    # 回顾结果邮件发送(受ENABLE_EMAIL开关控制)
    if send_mail:
        if pf_total:
            tail = (f"推荐{n_picked}场 投注命中{pf_top}/{pf_total} "
                    f"({pf_top / pf_total * 100:.0f}%)")
        else:
            tail = f"推荐{n_picked}场 未完场暂无命中"
        send_email(f"OVER预测回顾 {target_date} {tail}", '\n'.join(lines))

    conn.close()
    return lines, stats


# ==================== 预测记录补录 ====================
def backfill(target_date, min_prob=None):
    """补录指定日期预测记录到ml_predictions.jsonl(含已完场比赛).
    用赛前特征预测(特征不含比分/标签, 等价回测口径, 无泄漏); 覆盖同日旧记录后写入jsonl,
    不写match.prediction、不发邮件、不跑监控, 供--review回顾历史日期使用."""
    model = load_model()
    if model is None:
        print(f"✗ 未找到模型文件({MODEL_DIR}), 请先运行: python -u train_over.py")
        return
    meta = model.get('meta', {})
    threshold = min_prob if min_prob is not None \
        else meta.get('threshold', MIN_POOL_PROB_FALLBACK)
    df = load_or_build_over(rebuild=True)
    sub = df[df['match_id'].str.startswith(target_date)].copy()
    if sub.empty:
        print(f"日期{target_date}无比赛记录(特征宽表)")
        return
    # 盘口筛选
    hc_mask = sub.apply(
        lambda r: is_target_handicap_over(r.get('ah_cur_mean'), r.get('ah_init_mean')),
        axis=1)
    sub = sub[hc_mask]
    # 大小球线筛选(推荐窄口径: 初盘和终盘均 >= 2.75)
    ou_mask = sub.apply(
        lambda r: is_valid_ou_line_over(r.get('ou_line_init_mean'), r.get('ou_line_cur_mean')),
        axis=1)
    sub = sub[ou_mask]
    # 排除杯赛
    sub = sub[~sub['league'].apply(is_cup_league_over)]
    if sub.empty:
        print(f"日期{target_date}无满足过滤条件的比赛")
        return
    pred = predict_rows(model, sub)
    blacklist = set(load_blacklist())
    picked_ids = set(pred[(pred['p_pool'] >= threshold)
                          & (~pred['league'].isin(blacklist))]['match_id'])
    entries = []
    for _, r in pred.iterrows():
        entries.append({
            'match_id': r['match_id'], 'date': target_date, 'league': r['league'],
            'p_pool': round(float(r['p_pool']), 4),
            'picked': r['match_id'] in picked_ids,
            'top_picks': r['top_picks'],
            'predicted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'backfill': True,
        })
    replace_prediction_log(entries, target_date=target_date, keep_backfill=False)
    print(f"✓ 已补录{len(entries)}场 {target_date} 预测记录到 ml_predictions.jsonl "
          f"(阈值{threshold:.2f}, 推荐{len(picked_ids)}场, 标记backfill=true)")
    print("  提示: 补录为事后用赛前特征预测(回测口径), 仅用于--review回顾; "
          "补录覆盖同日全部旧记录, review按match_id去重保留最后一次")


# ==================== 主流程 ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=datetime.now().strftime('%Y%m%d'),
                    help='预测指定日期YYYYMMDD的比赛(默认当天)')
    ap.add_argument('--all-unfinished', action='store_true',
                    help='预测全部未开赛比赛(不限日期)')
    ap.add_argument('--skip-monitor', action='store_true', help='跳过历史推荐监控')
    ap.add_argument('--min-prob', type=float, default=None,
                    help='手动指定置信度阈值(覆盖模型EV阈值, 如 0.65 只推高置信场次)')
    ap.add_argument('--skip-email', action='store_true', help='跳过预测结果邮件发送')
    ap.add_argument('--review', nargs='?', const='', default=None,
                    help='回看模式: 对照ml_predictions.jsonl与实际比分, 不带值默认昨天, 可指定YYYYMMDD')
    ap.add_argument('--backfill', default=None, metavar='YYYYMMDD',
                    help='补录模式: 对指定日期全部比赛(含已完场)用赛前特征预测并追加到ml_predictions.jsonl, 供--review回顾')
    args = ap.parse_args()

    # 回看模式: 不做预测, 直接对照历史推荐与实际比分
    if args.review is not None:
        target = args.review or (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
        review(target, send_mail=not args.skip_email)
        return

    # 补录模式: 只追加jsonl, 不写库/不发邮件/不跑监控
    if args.backfill:
        backfill(args.backfill, min_prob=args.min_prob)
        return

    model = load_model()
    if model is None:
        print(f"✗ 未找到模型文件({MODEL_DIR}), 请先运行: python -u train_over.py")
        return
    meta = model.get('meta', {})
    threshold = args.min_prob if args.min_prob is not None \
        else meta.get('threshold', MIN_POOL_PROB_FALLBACK)
    
    # 自动回顾前一天（在预测前）
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    print("\n" + "=" * 70)
    print("【第一部分：回顾前一天预测结果】")
    print("=" * 70)
    review_lines, review_stats = review(yesterday, send_mail=False)
    
    # 预测今天
    print("\n" + "=" * 70)
    print("【第二部分：今日预测】")
    print("=" * 70 + "\n")
    
    print("=" * 70)
    print("OVER比分预测 (LightGBM筛选门 + Dixon-Coles比分矩阵)")
    print(f"模型版本: {meta.get('tag')}  训练时间: {meta.get('trained_at')}")
    print(f"池内概率阈值: {threshold:.2f}  推荐方式: 全部高置信场次(不限数量)")
    print("=" * 70)

    # 特征前爬取最新欧赔数据（保证使用最新盘口）
    print("\n⏳ 正在爬取最新欧赔数据...")
    scrape_euro_odds_for_date(args.date)
    
    # 特征(增量更新模式，只更新盘口数据，大幅提升速度)
    print("\n⏳ 加载特征数据（增量模式）...")
    df = load_features_incremental(rebuild_base=False)

    # 当日未开赛样本: 无比分(in_pool为空)
    unfinished = df[df['in_pool'].isna()].copy()
    if not args.all_unfinished:
        unfinished = unfinished[unfinished['match_id'].str.startswith(args.date)]
    if unfinished.empty:
        print(f"日期{args.date}无未开赛比赛")
        return
    # 盘口筛选: 只保留目标盘口范围内的比赛
    handicap_mask = unfinished.apply(
        lambda r: is_target_handicap_over(r.get('ah_cur_mean'), r.get('ah_init_mean')),
        axis=1)
    before_hc = len(unfinished)
    unfinished = unfinished[handicap_mask]
    # 大小球线筛选（推荐窄口径: 初盘和终盘均 >= 2.75; 训练仍用宽口径 MIN_OU_LINE_OVER=2.25）
    ou_mask = unfinished.apply(
        lambda r: is_valid_ou_line_over(r.get('ou_line_init_mean'), r.get('ou_line_cur_mean')),
        axis=1)
    before_ou = len(unfinished)
    unfinished = unfinished[ou_mask]
    # 排除杯赛
    before_cup = len(unfinished)
    unfinished = unfinished[~unfinished['league'].apply(is_cup_league_over)]
    print(f"过滤: {before_hc}场 → 盘口{before_ou}场 → 大小球{before_cup}场 → 排除杯赛{len(unfinished)}场")
    if unfinished.empty:
        print(f"日期{args.date}无满足过滤条件的未开赛比赛")
        return
    print(f"待预测比赛: {len(unfinished)}场\n")

    pred = predict_rows(model, unfinished)
    blacklist = set(load_blacklist())

    # 全部场次概览(按池内概率降序)
    pred = pred.sort_values('p_pool', ascending=False).reset_index(drop=True)
    print(f"{'编号':<12}{'单场':<9}{'联赛':<10}{'对阵':<26}{'时间':<8}"
          f"{'池内概率':>8}{'状态':>6}")
    for _, r in pred.iterrows():
        vs = f"{r['home_team']} vs {r['away_team']}"
        single = fmt_meta(r, 'single_match')
        mtime = fmt_meta(r, 'match_time')
        if r['league'] in blacklist:
            flag = '黑名单'
        elif r['p_pool'] >= threshold:
            flag = '候选'
        else:
            flag = '-'
        print(f"{r['match_id']:<12}{single:<9}{str(r['league']):<10}{vs:<26}"
              f"{mtime:<8}{r['p_pool'] * 100:>7.1f}%{flag:>6}")

    # 投注筛选: 置信度阈值 + 黑名单, 不限场次数量(pred已按p_pool降序)
    picks_df = pred[(pred['p_pool'] >= threshold)
                    & (~pred['league'].isin(blacklist))]

    report = []
    report.append(f"OVER比分预测 {args.date} (阈值{threshold:.2f}, 不限场次)")
    report.append(f"模型: {meta.get('tag')}  训练时间: {meta.get('trained_at')}")
    
    # 提前创建数据库连接（技术统计对比需要）
    conn = pymysql.connect(**DB_CONFIG)
    
    if picks_df.empty:
        report.append("今日无满足条件的比赛, 空仓")
        print("\n★ 今日无满足条件的比赛, 空仓")
    else:
        print(f"\n★ 今日推荐 {len(picks_df)} 场:")
        for _, r in picks_df.iterrows():
            m = dc_matrix(r['lambda_home'], r['lambda_away'], model['rho'])
            pool_all = matrix_pool_scores(m)
            scores_str = '、'.join(f"{s}({p * 100:.1f}%)" for s, p in pool_all)
            # 标题行格式与filter_matches.py一致: ★ [编号] [单场XX] 联赛 主队 vs 客队 (时间)
            single = fmt_meta(r, 'single_match', default='')
            single_part = f" [{single}]" if single else ""
            mtime = fmt_meta(r, 'match_time', default='')
            time_part = f"  ({mtime})" if mtime else ""
            line1 = (f"★ [{r['match_id']}]{single_part} {r['league']}  "
                     f"{r['home_team']} vs {r['away_team']}{time_part}")
            line2 = (f"    池内概率: {r['p_pool'] * 100:.1f}%  "
                     f"(GBM {r['p_lgb'] * 100:.1f}% / DC {r['p_dc'] * 100:.1f}%)  "
                     f"λ 主{r['lambda_home']:.2f}/客{r['lambda_away']:.2f}")
            line3 = f"    11比分概率(降序): {scores_str}"
            print(line1)
            print(line2)
            print(line3)
            report.extend([line1, line2, line3])
            
            # 技术统计对比
            try:
                tech_report, tech_rec = get_tech_stats_comparison(
                    conn, r['home_team'], r['away_team'], r['match_id'])
                tech_lines = format_tech_stats_report(tech_report, tech_rec)
                for tl in tech_lines:
                    print(tl)
                report.extend(tech_lines)
            except Exception as e:
                err_line = f'    技术统计: 获取失败({e})'
                print(err_line)
                report.append(err_line)

    # 邮件发送预测结果(受ENABLE_EMAIL开关控制)
    if not args.skip_email:
        subject = (f"OVER比分预测 {args.date} "
                   + (f"推荐{len(picks_df)}场" if not picks_df.empty else "空仓"))
        
        # 构建邮件内容：回顾 + 预测
        email_parts = []
        
        # 第一部分：回顾（如果有数据）
        if review_lines:
            email_parts.extend(review_lines)
            email_parts.append('')
            email_parts.append('=' * 70)
            email_parts.append('')
        
        # 第二部分：预测
        email_parts.extend(report)
        
        send_email(subject, '\n'.join(email_parts))

    # 写入match.prediction(方向格式兼容现有下游) + 追加推荐日志
    cur = conn.cursor()
    entries = []
    picked_ids = set(picks_df['match_id'])
    for _, r in pred.iterrows():
        direction = direction_from_matrix(r['lambda_home'], r['lambda_away'],
                                          model['rho'])
        cur.execute("UPDATE `match` SET prediction = %s WHERE match_id = %s",
                    (direction, r['match_id']))
        entries.append({
            'match_id': r['match_id'], 'date': args.date, 'league': r['league'],
            'p_pool': round(float(r['p_pool']), 4),
            'picked': r['match_id'] in picked_ids,
            'top_picks': r['top_picks'],
            'predicted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        })
    conn.commit()
    replace_prediction_log(entries,
                           target_date=None if args.all_unfinished else args.date)
    print(f"\n已写入{len(entries)}场预测方向到match表, "
          f"推荐记录写入 ml_predictions.jsonl (覆盖同日旧记录)")

    # 持续学习监控
    if not args.skip_monitor:
        monitor(conn)
    cur.close()
    conn.close()


if __name__ == '__main__':
    main()
