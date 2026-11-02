# predict_1x.py - 两层模型每日预测脚本（1X + 8 池）
# 功能:
#   1. Layer 1: LightGBM 预测主队不败概率 P_1X
#   2. Layer 2: P_8pool = P_1X × 0.865（固定条件概率）
#   3. 动态选场:
#      - P_1X > 0.75 → 推荐 1X（双选胜平）
#      - P_8pool > 0.65 → 推荐 8 池比分
#      - 否则 → 空仓
#   4. 预测结果写入 match.prediction（兼容现有格式）
#   5. 推荐记录追加到 predictions_1x.jsonl
#   6. 邮件发送（受 config.py 的 ENABLE_EMAIL 控制）
#   7. --review 回顾前一天预测结果（对照实际比分）
#   8. --backfill 补录指定日期预测记录（供--review回顾历史日期）
#
# 用法:
#   python -u predict_1x.py                 # 预测当天比赛
#   python -u predict_1x.py --date 20260912 # 指定日期
#   python -u predict_1x.py --all-unfinished # 预测全部未开赛比赛
#   python -u predict_1x.py --review        # 回顾昨天预测结果
#   python -u predict_1x.py --review 20260912 # 回顾指定日期
#   python -u predict_1x.py --backfill 20260911 # 补录指定日期预测记录
import argparse
import json
import os
import smtplib
from datetime import datetime, timedelta
from email.header import Header
from email.mime.text import MIMEText

import joblib
import numpy as np
import pymysql

from config import DB_CONFIG, EMAIL_CONFIG, ENABLE_EMAIL
from features import is_cup_league
from features_1x import build_features_1x, feature_columns_1x, label_in_pool_8
from train_1x import (MODEL_DIR, TOP_SCORES_8POOL, dc_matrix,
                       matrix_top_scores_pool)

PRED_LOG_FILE = 'predictions_1x.jsonl'

# ==================== 配置 ====================
P_8POOL_GIVEN_1X = 0.865    # Layer 2 固定条件概率
THRESHOLD_1X = 0.75         # 1X 推荐阈值
THRESHOLD_8POOL = 0.65      # 8 池推荐阈值

# 8 池比分列表
POOL_8 = ['1-0', '2-0', '2-1', '3-1', '3-2', '0-0', '1-1', '2-2']

# ==================== 亚盘盘口过滤配置 ====================
# 只推荐盘口为这些值的比赛（否则即使命中SP值也很低）
# '受让平手/半球' 条件概率高达 90%，命中1X时比分极可能在8池内
HANDICAP_TARGETS = {'半球', '平手/半球', '平手', '受让平手/半球'}
HANDICAP_MIN_COMPANIES = 12  # 至少需要多少家机构的初盘+即时盘都在目标盘口内

# 盘口归一化别名（兼容各种写法）
_HANDICAP_ALIASES = {
    '平手': '平手', '受平手': '平手',
    '平手/半球': '平手/半球', '平/半': '平手/半球',
    '受让平手/半球': '受让平手/半球', '受平手/半球': '受让平手/半球',
    '受让平/半': '受让平手/半球', '受平/半': '受让平手/半球',
    '半球': '半球', '受半球': '受让半球', '受让半球': '受让半球',
}


def _norm_handicap(h):
    """盘口归一化，无法识别返回None"""
    if h is None:
        return None
    return _HANDICAP_ALIASES.get(str(h).strip())


def filter_by_handicap(match_ids, conn):
    """过滤掉亚盘盘口不在目标集合的比赛
    返回: 满足条件的 match_id 集合
    """
    if not match_ids:
        return set()
    
    cursor = conn.cursor()
    placeholders = ','.join(['%s'] * len(match_ids))
    cursor.execute(
        f"SELECT match_id, initial_handicap, current_handicap FROM asian_odds WHERE match_id IN ({placeholders})",
        match_ids
    )
    
    # 统计每场比赛符合条件的机构数
    match_counts = {}
    for mid, init_h, curr_h in cursor.fetchall():
        mid_str = str(mid)
        init_norm = _norm_handicap(init_h)
        curr_norm = _norm_handicap(curr_h)
        if init_norm in HANDICAP_TARGETS and curr_norm in HANDICAP_TARGETS:
            match_counts[mid_str] = match_counts.get(mid_str, 0) + 1
    
    cursor.close()
    
    # 返回机构数达标的比赛
    return {mid for mid, count in match_counts.items() if count >= HANDICAP_MIN_COMPANIES}


def load_model():
    """加载 1X 模型和元信息"""
    model_path = os.path.join(MODEL_DIR, 'lgb_1x.txt')
    meta_path = os.path.join(MODEL_DIR, 'meta_1x.json')
    
    if not os.path.exists(model_path):
        print(f"✗ 未找到模型文件: {model_path}")
        print("  请先运行: python -u train_1x.py")
        return None, None
    
    import joblib
    model = joblib.load(model_path)
    
    with open(meta_path, encoding='utf-8') as fp:
        meta = json.load(fp)
    
    return model, meta


def send_email(subject, content):
    """邮件通知"""
    if not ENABLE_EMAIL:
        print(f"  (邮件开关 ENABLE_EMAIL=False, 跳过)")
        return False
    try:
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


def recommend_strategy(p_1x, p_8pool):
    """动态选场策略"""
    if p_1x >= THRESHOLD_1X:
        return '1X（双选胜平）'
    elif p_8pool >= THRESHOLD_8POOL:
        return '8 池比分'
    else:
        return '空仓'


def _norm_score(s):
    """比分归一为'x-y'格式，无效返回None"""
    s = str(s or '').replace(':', '-').strip()
    parts = s.split('-')
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return s
    return None


def _is_finished(status):
    """match表status是否为已完场"""
    return bool(status) and any(k in str(status) for k in ('完', '已结束', 'Finished'))


def review(target_date, send_mail=True):
    """回看指定日期预测结果：对照 predictions_1x.jsonl 与实际比分
    命中口径:
    - 1X命中：实际比分主队不败（主胜或平局）
    - 8池命中：实际比分落在8池内
    返回：(lines列表，统计信息字典) 或 None（无数据时）
    """
    lines = []
    
    def out(s=''):
        lines.append(s)
        print(s)
    
    if not os.path.exists(PRED_LOG_FILE):
        print(f"✗ 未找到预测记录文件：{PRED_LOG_FILE}")
        return None, None
    
    with open(PRED_LOG_FILE, encoding='utf-8') as fp:
        logs = [json.loads(l) for l in fp if l.strip()]
    
    # 按 match_id 去重（同一场多次预测保留最后一次），只取目标日期
    latest = {}
    for l in logs:
        mid = str(l.get('match_id', ''))
        if mid.startswith(target_date):
            latest[mid] = l
    
    if not latest:
        print(f"日期 {target_date} 无预测记录 ({os.path.basename(PRED_LOG_FILE)})")
        return None, None
    
    # 查询实际比分、状态、单场ID和SP值
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    ids = list(latest.keys())
    placeholders = ','.join(['%s'] * len(ids))
    cursor.execute(
        f"SELECT match_id, score, status, single_match FROM `match` WHERE match_id IN ({placeholders})",
        ids
    )
    info = {r[0]: {'score': r[1], 'status': r[2], 'single_match': r[3]} for r in cursor.fetchall()}
        
    # 查询开奖SP值
    cursor.execute(
        f"SELECT match_id, sp_value FROM result_sp WHERE CONVERT(match_id USING utf8mb4) COLLATE utf8mb4_general_ci IN ({placeholders})",
        ids
    )
    sp_map = {r[0]: (float(r[1]) if r[1] is not None else None) for r in cursor.fetchall()}
    cursor.close()
    conn.close()
    
    out('=' * 78)
    out(f"1X+8池模型回顾 {target_date}  (来源：{os.path.basename(PRED_LOG_FILE)})")
    out(f"命中口径：1X=主队不败（主胜或平局），8池={len(POOL_8)}种比分 {'/'.join(POOL_8)}")
    out('=' * 78)
    
    finished = hit_1x = hit_8pool = 0
    
    for mid in sorted(latest.keys()):
        e = latest[mid]
        mi = info.get(mid, {})
        # 优先使用 match 表的 single_match，其次使用 jsonl 记录的
        single_val = mi.get('single_match') or e.get('single_match', '')
        single = f"[{single_val}]" if single_val else "[-]"
        vs = f"{e.get('home_team', '?')} vs {e.get('away_team', '?')}"
        rec = e.get('recommendation', '-')
        p_1x = e.get('p_1x', 0)
        p_8pool = e.get('p_8pool', 0)
        
        sc = mi.get('score')
        norm = _norm_score(sc)
        sp = sp_map.get(mid)
        sp_str = f"SP{sp:.2f}" if sp is not None else "SP-"
                
        if _is_finished(mi.get('status')) and norm:
            finished += 1
            parts = norm.split('-')
            home_goals, away_goals = int(parts[0]), int(parts[1])
            is_1x = home_goals >= away_goals  # 主队不败
            in_pool8 = norm in POOL_8
                    
            if is_1x:
                hit_1x += 1
            if in_pool8:
                hit_8pool += 1
                    
            flag_1x = '✓' if is_1x else '✗'
            flag_8pool = '✓' if in_pool8 else '✗'
            sc_str = f"{norm} {sp_str}"
        else:
            flag_1x = '-'
            flag_8pool = '-'
            sc_str = f"{str(sc) if sc else '-'} {sp_str}"
                
        out(f"{mid} {single} {e.get('league', '')} {vs}")
        out(f"  预测：{rec} (P_1X={p_1x*100:.1f}%, P_8pool={p_8pool*100:.1f}%)")
        out(f"  实际：{sc_str}  -> 1X:{flag_1x}  8池:{flag_8pool}")
        out('')
    
    out('-' * 78)
    rate_1x = f" ({hit_1x / finished * 100:.1f}%)" if finished else ''
    rate_8pool = f" ({hit_8pool / finished * 100:.1f}%)" if finished else ''
    out(f"【准确率统计】")
    out(f"共 {len(latest)} 场预测，已完场 {finished} 场")
    out(f"  1X 命中率：{hit_1x}/{finished}{rate_1x}")
    out(f"  8池命中率：{hit_8pool}/{finished}{rate_8pool}")
    out('-' * 78)
    
    stats = {
        'total': len(latest),
        'finished': finished,
        'hit_1x': hit_1x,
        'hit_8pool': hit_8pool,
        'rate_1x': rate_1x,
        'rate_8pool': rate_8pool,
    }
    
    if send_mail:
        subject = f"1X+8池模型回顾 {target_date} 预测{len(latest)}场 完场{finished} " \
                  f"1X命中{hit_1x}{rate_1x} 8池命中{hit_8pool}{rate_8pool}"
        send_email(subject, '\n'.join(lines))
    
    return lines, stats


# ==================== Dixon-Coles 8池具体比分(参考POOL9三阶段) ====================
def load_goal_models(meta):
    """加载泊松进球模型+rho; 旧模型无该产物时返回(None,None,None)降级为不出具体比分"""
    rho = meta.get('rho')
    gh_path = os.path.join(MODEL_DIR, 'lgb_1x_goals_home.joblib')
    ga_path = os.path.join(MODEL_DIR, 'lgb_1x_goals_away.joblib')
    if rho is None or not (os.path.exists(gh_path) and os.path.exists(ga_path)):
        return None, None, None
    return joblib.load(gh_path), joblib.load(ga_path), rho


def attach_dc_scores(df, feats, gh, ga, rho, n=TOP_SCORES_8POOL):
    """用泊松λ+Dixon-Coles矩阵在8池内选Top具体比分, 返回新增top_picks列的df副本"""
    d = df.copy()
    if gh is None or ga is None or rho is None:
        d['top_picks'] = ''
        return d
    X = d[feats]
    lh = np.clip(gh.predict(X), 0.1, 4.5)
    la = np.clip(ga.predict(X), 0.1, 4.5)
    d['lambda_home'] = lh
    d['lambda_away'] = la
    picks = []
    for h, a in zip(lh, la):
        m = dc_matrix(float(h), float(a), rho)
        top = matrix_top_scores_pool(m, POOL_8, n)
        picks.append(';'.join(f"{s}:{p:.3f}" for s, p in top))
    d['top_picks'] = picks
    return d


# ==================== 预测记录补录 ====================
def backfill(target_date):
    """补录指定日期预测记录到 predictions_1x.jsonl（含已完场比赛）.
    用赛前特征预测（特征不含比分/标签，等价回测口径，无泄漏）; 只追加 jsonl,
    不写 match.prediction、不发邮件，供--review回顾历史日期使用."""
    model, meta = load_model()
    if model is None:
        return
    gh, ga, rho = load_goal_models(meta)
    
    # 重建缓存获取目标日期数据
    df = build_features_1x(rebuild=True)
    sub = df[df['match_id'].astype(str).str.startswith(target_date)].copy()
    
    if sub.empty:
        print(f"日期{target_date}无比赛记录（特征宽表）")
        return
    
    # 筛选未开赛比赛（补录时也包含已完场的，用赛前特征预测）
    # 注意：补录时使用全部比赛（含已完场），因为我们要回测历史预测效果
    print(f"日期{target_date}共 {len(sub)} 场比赛")
    
    # 亚盘盘口过滤
    conn = pymysql.connect(**DB_CONFIG)
    valid_ids = filter_by_handicap(list(sub['match_id'].astype(str)), conn)
    conn.close()
    
    sub = sub[sub['match_id'].astype(str).isin(valid_ids)]
    print(f"盘口过滤后: {len(sub)} 场")
    
    # 排除杯赛（与 ML 模型保持一致）
    sub = sub[~sub['league'].apply(is_cup_league)]
    print(f"排除杯赛后: {len(sub)} 场")
    
    if sub.empty:
        print(f"日期{target_date}无符合盘口条件的比赛")
        return
    
    # 特征列
    feats = meta.get('features', feature_columns_1x(sub))
    
    # Layer 1: 预测 P_1X
    p_1x = model.predict_proba(sub[feats])[:, 1]
    
    # Layer 2: P_8pool = P_1X × 0.865
    p_8pool = p_1x * P_8POOL_GIVEN_1X
    
    # 添加预测结果
    sub = sub.copy()
    sub['p_1x'] = p_1x
    sub['p_8pool'] = p_8pool
    sub['recommendation'] = sub.apply(
        lambda r: recommend_strategy(r['p_1x'], r['p_8pool']), axis=1
    )
    
    # Dixon-Coles 8池具体比分(参考POOL9三阶段)
    sub = attach_dc_scores(sub, feats, gh, ga, rho)
    
    # 生成补录记录
    entries = []
    for _, r in sub.iterrows():
        match_date = str(r['match_id'])[:8]
        entries.append({
            'match_id': r['match_id'],
            'date': match_date,
            'league': r['league'],
            'home_team': r['home_team'],
            'away_team': r['away_team'],
            'single_match': r.get('single_match', ''),  # 单场 ID
            'p_1x': round(float(r['p_1x']), 4),
            'p_8pool': round(float(r['p_8pool']), 4),
            'recommendation': r['recommendation'],
            'top_picks': r.get('top_picks', ''),
            'predicted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'backfill': True,
        })
    
    # 追加到 jsonl
    with open(PRED_LOG_FILE, 'a', encoding='utf-8') as fp:
        for e in entries:
            fp.write(json.dumps(e, ensure_ascii=False) + '\n')
    
    n_1x = sum(1 for e in entries if e['recommendation'] == '1X（双选胜平）')
    n_8pool = sum(1 for e in entries if e['recommendation'] == '8 池比分')
    n_empty = sum(1 for e in entries if e['recommendation'] == '空仓')
    
    print(f"✓ 已补录{len(entries)}场 {target_date} 预测记录到 {PRED_LOG_FILE}")
    print(f"  推荐：1X={n_1x}, 8池={n_8pool}, 空仓={n_empty}")
    print(f"  提示: 补录为事后用赛前特征预测（回测口径），仅用于--review回顾")
    print(f"        review按match_id去重保留最后一次，补录记录会覆盖同日旧记录")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=datetime.now().strftime('%Y%m%d'),
                    help='预测指定日期 YYYYMMDD 的比赛（默认当天）')
    ap.add_argument('--all-unfinished', action='store_true',
                    help='预测全部未开赛比赛（不限日期）')
    ap.add_argument('--skip-email', action='store_true',
                    help='跳过邮件发送')
    ap.add_argument('--review', nargs='?', const='', default=None,
                    help='回顾模式：对照预测结果与实际比分，不带值默认昨天，可指定 YYYYMMDD')
    ap.add_argument('--backfill', default=None, metavar='YYYYMMDD',
                    help='补录模式: 对指定日期全部比赛(含已完场)用赛前特征预测并追加到predictions_1x.jsonl, 供--review回顾')
    args = ap.parse_args()
    
    report_lines = []  # 提前初始化，避免后续无数据分支 UnboundLocalError
    
    # 补录模式: 只追加 jsonl, 不写库/不发邮件
    if args.backfill:
        backfill(args.backfill)
        return
    
    # 回顾模式
    if args.review is not None:
        target = args.review or (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
        review(target, send_mail=not args.skip_email)
        return
    
    # 加载模型
    model, meta = load_model()
    if model is None:
        return
    gh, ga, rho = load_goal_models(meta)
    n_1x = n_8pool = n_empty = 0
    
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
    print("1X + 8 池两层模型预测")
    print("=" * 70)
    print(f"模型版本: {meta.get('tag')}")
    print(f"训练时间: {meta.get('trained_at')}")
    print(f"Layer 1 阈值: P_1X > {THRESHOLD_1X} → 1X")
    print(f"Layer 2 阈值: P_8pool > {THRESHOLD_8POOL} → 8 池")
    print(f"P_8pool|1X = {P_8POOL_GIVEN_1X:.3f}（固定值）")
    print("=" * 70)
    
    # 构建特征（如果目标日期不在缓存中，自动重建）
    df = build_features_1x(rebuild=False)
    
    # 检查目标日期是否在缓存中
    if not args.all_unfinished:
        has_date = df['match_id'].astype(str).str.startswith(args.date).any()
        if not has_date:
            print(f"\n缓存中无 {args.date} 数据，自动重建缓存...")
            df = build_features_1x(rebuild=True)
            has_date = df['match_id'].astype(str).str.startswith(args.date).any()
            if not has_date:
                print(f"\n日期 {args.date} 无比赛数据（数据库中也无）")
                report_lines.append(f"日期 {args.date} 无比赛数据")
                # 邮件发送
                if not args.skip_email:
                    subject = f"1X+8池预测 {args.date} (无数据)"
                    email_parts = []
                    if review_lines:
                        email_parts.extend(review_lines)
                        email_parts.append('')
                        email_parts.append('=' * 70)
                        email_parts.append('')
                    email_parts.append(f"1X + 8 池两层模型预测 {args.date}")
                    email_parts.append(f"模型：{meta.get('tag')}")
                    email_parts.append(f"日期 {args.date} 无比赛数据")
                    send_email(subject, '\n'.join(email_parts))
                return
    
    # 筛选未开赛比赛
    unfinished = df[df['home_unbeaten'].isna()].copy()
    if not args.all_unfinished:
        unfinished = unfinished[unfinished['match_id'].astype(str).str.startswith(args.date)]
    
    if unfinished.empty:
        print(f"\n日期 {args.date} 无未开赛比赛")
        report_lines.append(f"日期 {args.date} 无未开赛比赛")
    else:
    
        # 亚盘盘口过滤：只保留目标盘口的比赛
        conn_filter = pymysql.connect(**DB_CONFIG)
        valid_ids = filter_by_handicap(list(unfinished['match_id'].astype(str)), conn_filter)
        conn_filter.close()
            
        n_before = len(unfinished)
        unfinished = unfinished[unfinished['match_id'].astype(str).isin(valid_ids)]
        n_filtered = n_before - len(unfinished)
        
        # 排除杯赛（与 ML 模型保持一致）
        n_before_cup = len(unfinished)
        unfinished = unfinished[~unfinished['league'].apply(is_cup_league)]
        n_cup = n_before_cup - len(unfinished)
            
        if unfinished.empty:
            print(f"\n日期 {args.date} 无符合盘口条件的比赛（过滤掉 {n_filtered} 场，排除杯赛 {n_cup} 场）")
            report_lines.append(f"日期 {args.date} 无符合盘口条件的比赛（过滤掉 {n_filtered} 场，排除杯赛 {n_cup} 场）")
        else:
            print(f"\n待预测比赛：{len(unfinished)} 场（盘口过滤掉 {n_filtered} 场，排除杯赛 {n_cup} 场）\n")
                
            # 特征列
            feats = meta.get('features', feature_columns_1x(unfinished))
                
            # Layer 1: 预测 P_1X
            p_1x = model.predict_proba(unfinished[feats])[:, 1]
                
            # Layer 2: P_8pool = P_1X × 0.865
            p_8pool = p_1x * P_8POOL_GIVEN_1X
                
            # 添加预测结果
            unfinished = unfinished.copy()
            unfinished['p_1x'] = p_1x
            unfinished['p_8pool'] = p_8pool
            unfinished['recommendation'] = unfinished.apply(
                lambda r: recommend_strategy(r['p_1x'], r['p_8pool']), axis=1
            )
                
            # Dixon-Coles 8池具体比分(参考POOL9三阶段)
            unfinished = attach_dc_scores(unfinished, feats, gh, ga, rho)
                
            # 按 P_1X 降序排序
            unfinished = unfinished.sort_values('p_1x', ascending=False).reset_index(drop=True)
                
            # 输出概览
            print(f"{'编号':<18}{'联赛':<10}{'对阵':<26}{'P_1X':>7}{'P_8pool':>8}{'推荐':>12}")
            print("-" * 96)
                
            entries = []
                
            for _, r in unfinished.iterrows():
                vs = f"{r['home_team']} vs {r['away_team']}"
                rec = r['recommendation']
                flag = "★" if rec != '空仓' else " "
                single = f"[{r['single_match']}]" if r.get('single_match') else "[-]"
                    
                line = (f"{flag} {r['match_id']:<10}{single:<8}{str(r['league']):<10}{vs:<26}"
                        f"{r['p_1x']*100:>6.1f}%{r['p_8pool']*100:>7.1f}%{rec:>12}")
                print(line)
                report_lines.append(line)
                    
                # 8池推荐额外输出Dixon-Coles具体比分
                if rec == '8 池比分' and r.get('top_picks'):
                    scores_line = f"      8池具体比分: {r['top_picks'].replace(';', '  ')}"
                    print(scores_line)
                    report_lines.append(scores_line)
                    
                # 记录推荐（date 使用 match_id 中的日期，确保回顾时能正确匹配）
                match_date = str(r['match_id'])[:8]  # 从 match_id 提取日期
                entries.append({
                    'match_id': r['match_id'],
                    'date': match_date,  # 使用比赛日期而非预测执行日期
                    'league': r['league'],
                    'home_team': r['home_team'],
                    'away_team': r['away_team'],
                    'p_1x': round(float(r['p_1x']), 4),
                    'p_8pool': round(float(r['p_8pool']), 4),
                    'recommendation': rec,
                    'top_picks': r.get('top_picks', ''),
                    'predicted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                })
                
            # 统计
            n_1x = sum(1 for e in entries if e['recommendation'] == '1X（双选胜平）')
            n_8pool = sum(1 for e in entries if e['recommendation'] == '8 池比分')
            n_empty = sum(1 for e in entries if e['recommendation'] == '空仓')
                
            print("\n" + "=" * 70)
            print(f"推荐统计:")
            print(f"  1X（双选胜平）: {n_1x} 场")
            print(f"  8 池比分：       {n_8pool} 场")
            print(f"  空仓：           {n_empty} 场")
            print("=" * 70)
                
            # 写入 match.prediction（方向格式兼容现有下游）
            conn = pymysql.connect(**DB_CONFIG)
            cur = conn.cursor()
                
            for _, r in unfinished.iterrows():
                # 根据推荐类型写 prediction
                if r['recommendation'] == '1X（双选胜平）':
                    pred_dir = '胜平'  # 兼容现有格式
                elif r['recommendation'] == '8 池比分':
                    # 8池已用Dixon-Coles输出具体比分(top_picks), 方向仍写"胜平"兼容下游
                    pred_dir = '胜平'
                else:
                    pred_dir = None
                    
                if pred_dir:
                    cur.execute("UPDATE `match` SET prediction = %s WHERE match_id = %s",
                               (pred_dir, r['match_id']))
                
            conn.commit()
            cur.close()
                
            # 追加推荐日志
            with open(PRED_LOG_FILE, 'a', encoding='utf-8') as fp:
                for e in entries:
                    fp.write(json.dumps(e, ensure_ascii=False) + '\n')
                
            print(f"\n✓ 已写入 {len(entries)} 场预测到 match 表")
            print(f"✓ 推荐记录已追加到 {PRED_LOG_FILE}")
            conn.close()
    
    # 邮件发送（合并回顾 + 预测）
    if not args.skip_email:
        subject = f"1X+8池预测 {args.date} (1X:{n_1x} 8池:{n_8pool} 空仓:{n_empty})"
            
        # 构建邮件内容：回顾 + 预测
        email_parts = []
            
        # 第一部分：回顾（如果有数据）
        if review_lines:
            email_parts.extend(review_lines)
            email_parts.append('')
            email_parts.append('=' * 70)
            email_parts.append('')
            
        # 第二部分：预测
        report_header = [
            f"1X + 8 池两层模型预测 {args.date}",
            f"模型：{meta.get('tag')}",
            f"推荐：1X={n_1x}, 8池={n_8pool}, 空仓={n_empty}",
            "-" * 70,
        ]
        email_parts.extend(report_header)
        email_parts.extend(report_lines)
            
        send_email(subject, '\n'.join(email_parts))


if __name__ == '__main__':
    main()
# predict_1x.py - 两层模型每日预测脚本（1X + 8 池）
# 功能:
#   1. Layer 1: LightGBM 预测主队不败概率 P_1X
#   2. Layer 2: P_8pool = P_1X × 0.865（固定条件概率）
#   3. 动态选场:
#      - P_1X > 0.75 → 推荐 1X（双选胜平）
#      - P_8pool > 0.65 → 推荐 8 池比分
#      - 否则 → 空仓
#   4. 预测结果写入 match.prediction（兼容现有格式）
#   5. 推荐记录追加到 predictions_1x.jsonl
#   6.