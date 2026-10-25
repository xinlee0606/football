# predict_upside.py - UPSIDE模型每日预测脚本
# 功能:
#   1. 对当日未开赛比赛生成特征, 加载最新UPSIDE模型预测两层概率:
#      - Layer 1: LightGBM 预测主队不败概率 P_UPSIDE
#      - Layer 2: 三阶段混合(同UNDER): LightGBM池内筛选门 + 泊松λ +
#        Dixon-Coles矩阵池概率 → α融合 → Isotonic校准 → P_pool
#   2. 投注纪律: P_UPSIDE >= UPSIDE阈值 或 P_pool >= 池内阈值 → 推荐, 不限场次
#   3. 推荐记录写入 predictions_upside.jsonl (同日旧记录被覆盖), 供持续学习监控
#   4. 预测结果邮件发送(受config.py的ENABLE_EMAIL开关控制, --skip-email可跳过)
#   5. 监控: 滚动近MONITOR_N场已完场推荐的池内命中率, 跌破基线一定比例邮件告警
#
# 与UNDER模型差异:
#   - 两层推荐: UPSIDE(主队不败双选) + 9池比分(同UNDER池)
#   - 比分池: POOL_UPSIDE (9种) vs UNDER (9种, 不同比分组合)
#   - 不做大小球线过滤
#   - 不写match.prediction
#
# 用法:
#   python -u predict_upside.py                 # 预测当天比赛
#   python -u predict_upside.py --date 20260818 # 指定日期
#   python -u predict_upside.py --all-unfinished # 预测全部未开赛比赛(不限日期)
#   python -u predict_upside.py --review          # 回看昨天预测与实际比分对照
#   python -u predict_upside.py --review 20260816  # 回看指定日期预测结果
#   python -u predict_upside.py --backfill 20260906 --min-prob 0.75  # 补录指定日期预测记录
import argparse
import json
import os
import smtplib
import subprocess
from datetime import datetime, timedelta
from email.header import Header
from email.mime.text import MIMEText

import joblib
import numpy as np
import pymysql

from config import DB_CONFIG, EMAIL_CONFIG, ENABLE_EMAIL
from features_under import is_predict_handicap_under
from features_upside import is_cup_league, parse_score, is_finished
from features_upside_ext import (POOL_UPSIDE, POOL_UPSIDE_SET, build_features_upside_ext,
                         feature_columns_upside)
from train_upside import (MODEL_DIR, dc_matrix, matrix_pool_prob,
                      matrix_top_scores_pool)
from tech_stats_compare import get_tech_stats_comparison, format_tech_stats_report

PRED_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'predictions_upside.jsonl')
BLACKLIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'backtest_upside_result.json')

# ==================== 投注纪律配置 ====================
THRESHOLD_UPSIDE = 0.75             # UPSIDE 推荐阈值
THRESHOLD_9POOL_FALLBACK = 0.65     # 9 池推荐阈值回退值(优先用 meta 的 threshold_pool)
MONITOR_N = 100
MONITOR_DROP_RATIO = 0.85


# ==================== 辅助 ====================
def scrape_euro_odds_for_date(target_date):
    """爬取指定日期比赛的最新欧赔数据"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scrape_euro_odds.py')
    if not os.path.exists(script_path):
        print(f"   未找到欧赔爬虫脚本: {script_path}")
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
        print(f"   欧赔爬虫异常: {e}")
        return False


def load_blacklist():
    """加载联赛黑名单(backtest_upside.py产出), 失败返回空列表"""
    try:
        with open(BLACKLIST_FILE, encoding='utf-8') as fp:
            return json.load(fp).get('league_blacklist', [])
    except (OSError, json.JSONDecodeError):
        return []


def recommend_strategy(p_upside, p_underool, thr_pool):
    """动态选场策略"""
    if p_upside >= THRESHOLD_UPSIDE:
        return 'UPSIDE（双选胜平）'
    elif p_underool >= thr_pool:
        return '9 池比分'
    else:
        return '空仓'


def fmt_meta(r, col, default='-'):
    """安全取元信息列"""
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
        if not is_finished(sc):
            continue
        h, a = parse_score(sc)
        if h is None:
            continue
        judged.append({'match_id': l['match_id'],
                       'hit': f"{h}-{a}" in POOL_UPSIDE_SET})
    if len(judged) < 20:
        return
    recent = judged[-MONITOR_N:]
    hit_rate = sum(j['hit'] for j in recent) / len(recent)
    baseline = 0.55
    print(f"\n持续学习监控: 近{len(recent)}场已完场推荐池内命中率 "
          f"{hit_rate * 100:.1f}% (基线{baseline * 100:.1f}%)")
    if hit_rate < baseline * MONITOR_DROP_RATIO:
        print(f"⚠ 命中率跌破基线的{MONITOR_DROP_RATIO:.0%}, 触发告警")
        send_email('UPSIDE比分模型命中率告警',
                   f"近{len(recent)}场推荐池内命中率 {hit_rate * 100:.1f}%, "
                   f"低于基线{baseline * 100:.1f}%的{MONITOR_DROP_RATIO:.0%}。\n"
                   f"建议: 运行 python -u backtest_upside.py 复核, "
                   f"并提前执行重训 python -u train_upside.py")


# ==================== 预测回顾 ====================
def review(target_date, send_mail=True):
    """回看指定日期预测: 读取predictions_upside.jsonl, 与实际比分对照命中情况
    命中口径: UPSIDE=主队不败(主胜或平局); 9池=实际比分∈POOL_UPSIDE(9池)
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

    latest = {}
    for l in logs:
        mid = str(l.get('match_id', ''))
        if mid.startswith(target_date):
            latest[mid] = l
    if not latest:
        print(f"日期{target_date}无预测记录(predictions_upside.jsonl)")
        return None, None

    ids = list(latest.keys())
    conn = pymysql.connect(**DB_CONFIG)
    cur = conn.cursor()
    fmt = ','.join(['%s'] * len(ids))
    cur.execute(f"SELECT match_id, single_match, home_team, away_team, score "
                f"FROM `match` WHERE match_id IN ({fmt})", ids)
    info = {r[0]: {'single': r[1], 'home': r[2], 'away': r[3], 'score': r[4]}
            for r in cur.fetchall()}
    cur.execute(f"SELECT match_id, sp_value FROM result_sp "
                f"WHERE CONVERT(match_id USING utf8mb4) COLLATE utf8mb4_general_ci IN ({fmt})", ids)
    sp_map = {r[0]: (float(r[1]) if r[1] is not None else None)
              for r in cur.fetchall()}
    cur.close()
    # conn 保持开放，供技术统计对比使用

    out("=" * 78)
    out(f"UPSIDE预测回顾 {target_date}  (来源: predictions_upside.jsonl, 按池内概率由高到低)")
    out("命中口径: UPSIDE=主队不败(主胜或平局)  9池=实际比分∈9池")
    out("=" * 78)

    def _conf(e):
        v = e.get('p_pool')
        return v if isinstance(v, (int, float)) else -1.0
    ordered = sorted(latest.items(), key=lambda kv: _conf(kv[1]), reverse=True)

    n_picked = 0
    pf_total = pf_upside = pf_pool = pf_sp_n = 0
    pf_pl = 0.0
    for mid, e in ordered:
        picked = bool(e.get('picked'))
        rec = e.get('recommendation', '-')
        p_pool = e.get('p_pool', 0)
        picks = [s.split(':')[0] for s in str(e.get('top_picks', '')).split(';') if s]
        mi = info.get(mid, {})
        single = str(mi.get('single') or '').strip()
        single_part = f"[{single}]" if single else "[-]"
        vs = f"{mi.get('home', '?')} vs {mi.get('away', '?')}"
        sc = mi.get('score')
        flag = '★推荐' if picked else '  -   '
        pstr = f"{p_pool * 100:.1f}%" if isinstance(p_pool, (int, float)) else '-'
        top9 = '/'.join(picks[:9]) if picks else '-'
        sp = sp_map.get(mid)
        sp_str = f"SP{sp:.2f}" if sp is not None else "SP-"
        if is_finished(sc):
            h, a = parse_score(sc)
            norm = f"{h}-{a}"
            is_upside_hit = h >= a          # 主队不败
            in_pool = norm in POOL_UPSIDE_SET
            # 按推荐类型判定命中
            if rec == 'UPSIDE（双选胜平）':
                hit = '✓中' if is_upside_hit else '未中'
            elif rec == '9 池比分':
                hit = '✓中' if in_pool else '未中'
            else:
                hit = '参考'
            sc_str = norm
            if picked:
                pf_total += 1
                pf_upside += int(is_upside_hit)
                pf_pool += int(in_pool)
                if rec == '9 池比分' and sp is not None:
                    pf_sp_n += 1
                    pl = (2 * sp * 0.65 - 18) if in_pool else -18
                    pf_pl += pl
                    hit = f"{hit}{pl:+.1f}"
        else:
            hit = '未完场'
            sc_str = str(sc) if sc else '-'
        if picked:
            n_picked += 1
        out(f"{mid}  {single_part}  {flag}  池内{pstr:>7}  {vs}  "
            f"实际[{sc_str} {sp_str}]  投注:{top9}  -> {hit}")
        
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
        rate_upside = pf_upside / pf_total * 100
        rate_pool = pf_pool / pf_total * 100
        out(f"推荐且已完场 {pf_total} 场:")
        out(f"  UPSIDE命中率(主队不败): {pf_upside}/{pf_total} ({rate_upside:.1f}%)")
        out(f"  9池命中率(Top9): {pf_pool}/{pf_total} ({rate_pool:.1f}%)")
        if pf_sp_n:
            roi = pf_pl / (pf_sp_n * 18) * 100
            out(f"  实际盈亏(9池): {pf_pl:+.1f} 元 / {pf_sp_n}场 (ROI {roi:+.1f}%)")
    else:
        out("推荐场次均未完场, 暂无命中统计")
    out("-" * 78)

    stats = {
        'total': len(latest),
        'n_picked': n_picked,
        'pf_total': pf_total,
        'pf_upside': pf_upside,
        'pf_pool': pf_pool,
        'rate_upside': pf_upside / pf_total * 100 if pf_total else 0,
        'rate_pool': pf_pool / pf_total * 100 if pf_total else 0,
        'pf_pl': pf_pl,
        'pf_sp_n': pf_sp_n,
    }

    if send_mail:
        if pf_total:
            tail = (f"UPSIDE推荐{n_picked}场 9池命中{pf_pool}/{pf_total} "
                    f"({pf_pool / pf_total * 100:.0f}%)")
        else:
            tail = f"UPSIDE推荐{n_picked}场 未完场暂无命中"
        send_email(f"UPSIDE预测回顾 {target_date} {tail}", '\n'.join(lines))

    conn.close()
    return lines, stats


# ==================== 预测记录补录 ====================
def backfill(target_date, min_prob=None):
    """补录指定日期预测记录到predictions_upside.jsonl(含已完场比赛).
    用赛前特征预测(特征不含比分/标签, 等价回测口径, 无泄漏); 覆盖同日旧记录后写入jsonl,
    不写match.prediction、不发邮件、不跑监控, 供--review回顾历史日期使用."""
    model, meta = load_model()
    if model is None:
        print(f"✗ 未找到UPSIDE模型文件({MODEL_DIR}), 请先运行: python -u train_upside.py")
        return
    gh, ga, rho = load_goal_models(meta)
    pool_clf, calib = load_pool_model(meta)
    alpha = meta.get('alpha', 0.5)
    thr_pool = min_prob if min_prob is not None \
        else meta.get('threshold_pool', THRESHOLD_9POOL_FALLBACK)

    df = build_features_upside_ext(rebuild=True)
    df['match_id'] = df['match_id'].astype(str)
    sub = df[df['match_id'].str.startswith(target_date)].copy()
    if sub.empty:
        print(f"日期{target_date}无比赛记录(特征宽表)")
        return
    # 盘口筛选(基于特征均值)
    hc_mask = sub.apply(
        lambda r: is_predict_handicap_under(r.get('ah_cur_mean'), r.get('ah_init_mean')),
        axis=1)
    sub = sub[hc_mask]
    # 排除杯赛
    sub = sub[~sub['league'].apply(is_cup_league)]
    # 黑名单
    blacklist = set(load_blacklist())
    sub = sub[~sub['league'].isin(blacklist)]
    if sub.empty:
        print(f"日期{target_date}无满足过滤条件的比赛")
        return

    feats = meta.get('features_upside', feature_columns_upside(sub))
    sub = sub.copy()
    sub['p_upside'] = model.predict_proba(sub[feats])[:, 1]
    sub = attach_pool_probs(sub, feats, pool_clf, calib, alpha, gh, ga, rho)

    entries = []
    picked_ids = set()
    for _, r in sub.iterrows():
        rec = recommend_strategy(r['p_upside'], r['p_underool'], thr_pool)
        if rec != '空仓':
            picked_ids.add(r['match_id'])
        entries.append({
            'match_id': r['match_id'], 'date': target_date, 'league': r['league'],
            'home_team': r.get('home_team', ''), 'away_team': r.get('away_team', ''),
            'p_pool': round(float(r['p_underool']), 4),
            'picked': r['match_id'] in picked_ids,
            'recommendation': rec,
            'top_picks': r.get('top_picks', ''),
            'single_match': r.get('single_match', ''),
            'predicted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'backfill': True,
        })
    replace_prediction_log(entries, target_date=target_date, keep_backfill=False)
    print(f"✓ 已补录{len(entries)}场 {target_date} 预测记录到 predictions_upside.jsonl "
          f"(阈值{thr_pool:.2f}, 推荐{len(picked_ids)}场, 标记backfill=true)")
    print("  提示: 补录为事后用赛前特征预测(回测口径), 仅用于--review回顾; "
          "补录覆盖同日全部旧记录, review按match_id去重保留最后一次")


# ==================== UNDER池三阶段混合(同UNDER) ====================
def load_goal_models(meta):
    """加载泊松进球模型+rho; 旧模型无该产物时返回(None,None,None)降级"""
    rho = meta.get('rho')
    gh_path = os.path.join(MODEL_DIR, 'lgb_upside_goals_home.joblib')
    ga_path = os.path.join(MODEL_DIR, 'lgb_upside_goals_away.joblib')
    if rho is None or not (os.path.exists(gh_path) and os.path.exists(ga_path)):
        return None, None, None
    return joblib.load(gh_path), joblib.load(ga_path), rho


def load_pool_model(meta):
    """加载UNDER池池内筛选门+Isotonic校准器; 旧模型无该产物时返回(None,None)降级"""
    pool_path = os.path.join(MODEL_DIR, 'lgb_upside_pool.joblib')
    calib_path = os.path.join(MODEL_DIR, 'calib_upside.joblib')
    if not (os.path.exists(pool_path) and os.path.exists(calib_path)):
        return None, None
    return joblib.load(pool_path), joblib.load(calib_path)


def attach_pool_probs(df, feats, pool_clf, calib, alpha, gh, ga, rho):
    """UNDER池三阶段混合: 池内门P_lgb + 泊松λ + DC矩阵池概率P_dc → α融合 →
    Isotonic校准 → P_underool; 同时输出UNDER池内Top具体比分top_picks.
    缺进球模型时降级为不出具体比分; 缺门/校准器时P_underool用DC原值"""
    d = df.copy()
    X = d[feats]
    if gh is None or ga is None or rho is None:
        d['p_underool'] = 0.0
        d['top_picks'] = ''
        return d
    lh = np.clip(gh.predict(X), 0.1, 4.5)
    la = np.clip(ga.predict(X), 0.1, 4.5)
    d['lambda_home'] = lh
    d['lambda_away'] = la
    p_dc = np.array([matrix_pool_prob(dc_matrix(float(h), float(a), rho))
                     for h, a in zip(lh, la)])
    if pool_clf is not None and calib is not None:
        p_lgb = pool_clf.predict_proba(X)[:, 1]
        fused = alpha * p_lgb + (1 - alpha) * p_dc
        d['p_underool'] = calib.predict(fused)
    else:
        d['p_underool'] = p_dc  # 降级: DC原值池概率
    picks = []
    for h, a in zip(lh, la):
        m = dc_matrix(float(h), float(a), rho)
        top = matrix_top_scores_pool(m, POOL_UPSIDE, 9)
        picks.append(';'.join(f"{s}:{p:.3f}" for s, p in top))
    d['top_picks'] = picks
    return d


def load_model():
    """加载 UPSIDE 模型和元信息"""
    model_path = os.path.join(MODEL_DIR, 'lgb_upside.txt')
    meta_path = os.path.join(MODEL_DIR, 'meta_upside.json')

    if not os.path.exists(model_path):
        print(f"✗ 未找到模型文件: {model_path}")
        print("  请先运行: python -u train_upside.py")
        return None, None

    model = joblib.load(model_path)
    with open(meta_path, encoding='utf-8') as fp:
        meta = json.load(fp)

    return model, meta


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
                    help='回看模式: 对照predictions_upside.jsonl与实际比分, 不带值默认昨天, 可指定YYYYMMDD')
    ap.add_argument('--backfill', default=None, metavar='YYYYMMDD',
                    help='补录模式: 对指定日期全部比赛(含已完场)用赛前特征预测并追加到predictions_upside.jsonl')
    args = ap.parse_args()

    # 回看模式
    if args.review is not None:
        target = args.review or (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
        review(target, send_mail=not args.skip_email)
        return

    # 补录模式
    if args.backfill:
        backfill(args.backfill, min_prob=args.min_prob)
        return

    model, meta = load_model()
    if model is None:
        return
    gh, ga, rho = load_goal_models(meta)
    pool_clf, calib = load_pool_model(meta)
    alpha = meta.get('alpha', 0.5)
    threshold = args.min_prob if args.min_prob is not None \
        else meta.get('threshold_pool', THRESHOLD_9POOL_FALLBACK)

    # 自动回顾前一天
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    print("\n" + "=" * 70)
    print("【第一部分：回顾前一天预测结果】")
    print("=" * 70)
    review_lines, review_stats = review(yesterday, send_mail=False)

    # 预测今天
    print("\n" + "=" * 70)
    print("【第二部分：今日UPSIDE预测】")
    print("=" * 70 + "\n")

    print("=" * 70)
    print("UPSIDE比分预测 (两层模型: UPSIDE + 9池)")
    print(f"模型版本: {meta.get('tag')}  训练时间: {meta.get('trained_at')}")
    print(f"UPSIDE阈值: {THRESHOLD_UPSIDE}  9池阈值: {threshold:.2f}")
    print(f"推荐方式: 全部高置信场次(不限数量)")
    print(f"UNDER池三阶段: 池内门+泊松+DC, α={alpha:.1f}"
          + ("" if pool_clf is not None else " (降级: 缺门/校准器产物)"))
    print("=" * 70)

    # 特征(每日重建缓存)
    print("⏳ 正在爬取最新欧赔数据...")
    scrape_euro_odds_for_date(args.date)
    df = build_features_upside_ext(rebuild=True)
    df['match_id'] = df['match_id'].astype(str)

    # 当日未开赛样本
    unfinished = df[df['home_unbeaten'].isna()].copy()
    if not args.all_unfinished:
        unfinished = unfinished[unfinished['match_id'].str.startswith(args.date)]
    if unfinished.empty:
        print(f"日期{args.date}无未开赛比赛")
        return
    # 盘口筛选(基于特征均值)
    handicap_mask = unfinished.apply(
        lambda r: is_predict_handicap_under(r.get('ah_cur_mean'), r.get('ah_init_mean')),
        axis=1)
    before_hc = len(unfinished)
    unfinished = unfinished[handicap_mask]
    # 排除杯赛
    before_cup = len(unfinished)
    unfinished = unfinished[~unfinished['league'].apply(is_cup_league)]
    # 黑名单
    blacklist = set(load_blacklist())
    before_bl = len(unfinished)
    unfinished = unfinished[~unfinished['league'].isin(blacklist)]
    print(f"过滤: {before_hc}场 → 盘口筛选 → 排除杯赛 → 黑名单 → {len(unfinished)}场")
    if unfinished.empty:
        print(f"日期{args.date}无满足过滤条件的未开赛比赛")
        return
    print(f"待预测比赛: {len(unfinished)}场\n")

    # 特征列
    feats = meta.get('features_upside', feature_columns_upside(unfinished))

    # Layer 1: P_UPSIDE
    unfinished = unfinished.copy()
    unfinished['p_upside'] = model.predict_proba(unfinished[feats])[:, 1]

    # Layer 2: 9池三阶段混合
    unfinished = attach_pool_probs(unfinished, feats, pool_clf, calib,
                                   alpha, gh, ga, rho)
    # 推荐策略
    unfinished['recommendation'] = unfinished.apply(
        lambda r: recommend_strategy(r['p_upside'], r['p_underool'], threshold), axis=1
    )

    # 按池内概率降序
    unfinished = unfinished.sort_values('p_underool', ascending=False).reset_index(drop=True)

    # 全部场次概览
    print(f"{'编号':<12}{'单场':<9}{'联赛':<10}{'对阵':<26}{'时间':<8}"
          f"{'池内概率':>8}{'状态':>6}")
    for _, r in unfinished.iterrows():
        vs = f"{r['home_team']} vs {r['away_team']}"
        single = fmt_meta(r, 'single_match')
        mtime = fmt_meta(r, 'match_time')
        if r['league'] in blacklist:
            flag = '黑名单'
        elif r['recommendation'] != '空仓':
            flag = '候选'
        else:
            flag = '-'
        print(f"{r['match_id']:<12}{single:<9}{str(r['league']):<10}{vs:<26}"
              f"{mtime:<8}{r['p_underool'] * 100:>7.1f}%{flag:>6}")

    # 投注筛选
    picks_df = unfinished[unfinished['recommendation'] != '空仓']

    report = []
    report.append(f"UPSIDE比分预测 {args.date} (UPSIDE>{THRESHOLD_UPSIDE} 或 9池>{threshold:.2f}, 不限场次)")
    report.append(f"模型: {meta.get('tag')}  训练时间: {meta.get('trained_at')}")
    
    # 提前创建数据库连接（技术统计对比需要）
    conn = pymysql.connect(**DB_CONFIG)
    
    if picks_df.empty:
        report.append("今日无满足条件的比赛, 空仓")
        print("\n★ 今日无满足条件的比赛, 空仓")
    else:
        print(f"\n★ 今日推荐 {len(picks_df)} 场:")
        for _, r in picks_df.iterrows():
            rec = r['recommendation']
            m = dc_matrix(float(r['lambda_home']), float(r['lambda_away']), rho)
            pool_all = matrix_top_scores_pool(m, POOL_UPSIDE, 9)
            scores_str = '、'.join(f"{s}({p * 100:.1f}%)" for s, p in pool_all)
            single = fmt_meta(r, 'single_match', default='')
            single_part = f" [{single}]" if single else ""
            mtime = fmt_meta(r, 'match_time', default='')
            time_part = f"  ({mtime})" if mtime else ""
            line1 = (f"★ [{r['match_id']}]{single_part} {r['league']}  "
                     f"{r['home_team']} vs {r['away_team']}{time_part}")
            line2 = (f"    池内概率: {r['p_underool'] * 100:.1f}%  "
                     f"(P_UPSIDE {r['p_upside'] * 100:.1f}%)  "
                     f"λ 主{r['lambda_home']:.2f}/客{r['lambda_away']:.2f}  "
                     f"→ {rec}")
            line3 = f"    9比分概率(降序): {scores_str}"
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

    # 邮件发送
    if not args.skip_email:
        subject = (f"UPSIDE比分预测 {args.date} "
                   + (f"推荐{len(picks_df)}场" if not picks_df.empty else "空仓"))
        email_parts = []
        if review_lines:
            email_parts.extend(review_lines)
            email_parts.append('')
            email_parts.append('=' * 70)
            email_parts.append('')
        email_parts.extend(report)
        send_email(subject, '\n'.join(email_parts))

    # 追加推荐日志(不写match.prediction)
    cur = conn.cursor()
    entries = []
    picked_ids = set(picks_df['match_id'])
    for _, r in unfinished.iterrows():
        sm = r.get('single_match', '')
        if isinstance(sm, float) and sm != sm:
            sm = ''
        entries.append({
            'match_id': r['match_id'], 'date': args.date, 'league': r['league'],
            'home_team': r.get('home_team', ''), 'away_team': r.get('away_team', ''),
            'p_pool': round(float(r['p_underool']), 4),
            'picked': r['match_id'] in picked_ids,
            'recommendation': r['recommendation'],
            'top_picks': r.get('top_picks', ''),
            'single_match': sm,
            'predicted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        })
    replace_prediction_log(entries,
                           target_date=None if args.all_unfinished else args.date)
    print(f"\n已写入{len(entries)}场推荐记录到 predictions_upside.jsonl "
          f"(覆盖同日旧记录, 不写match.prediction)")

    # 持续学习监控
    if not args.skip_monitor:
        monitor(conn)
    cur.close()
    conn.close()


if __name__ == '__main__':
    main()
