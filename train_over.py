# train_over.py - OVER模型训练脚本（Step 2）
# 架构: 混合模型
#   1. LightGBM二分类(筛选门): 输出池内概率 P_lgb
#   2. LightGBM泊松回归 x2: 预测主队/客队期望进球 λ_home/λ_away
#   3. Dixon-Coles比分矩阵: 用λ+全局rho生成比分概率矩阵, 池内求和得 P_dc
#   4. 融合+校准: P_pool = α·P_lgb + (1-α)·P_dc, Isotonic校准
#   5. 投注阈值: 在留出集上以期望收益EV(关联result_sp开奖SP)选择
#
# 用法:
#   python -u train_over.py                 # 全量完场数据训练(70/15/15时间切分)
#   python -u train_over.py --end 20260601  # 只用指定日期之前的数据训练(回测用)
# 产物(models/目录, 带日期版本):
#   lgb_pool.txt       池内筛选LightGBM
#   lgb_goals_home.txt / lgb_goals_away.txt  泊松进球模型
#   calib.joblib       Isotonic校准器
#   meta.json          特征列/α/rho/阈值/基线等元信息
import argparse
import inspect
import json
import math
import os
from datetime import datetime

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from config import DB_CONFIG
from features_over import (META_COLS_OVER, OVER, OVER_SET, TARGET_HANDICAPS_OVER,
                         feature_columns_over, is_target_handicap_over,
                         is_valid_ou_line_over, is_cup_league_over, parse_score,
                         load_or_build_over)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

# ==================== 训练超参 ====================
TRAIN_RATIO = 0.70          # 训练集占比(按时间顺序)
VALID_RATIO = 0.15          # 留出集占比(选α/rho/阈值/早停)
MAX_GOALS = 8               # 比分矩阵截断进球数
TOP_PICKS = 11              # 推荐比分个数(每场下注注数, 与OVER池大小一致)

LGB_CLF_PARAMS = dict(
    objective='binary', n_estimators=600, learning_rate=0.05,
    num_leaves=31, max_depth=-1, min_child_samples=60,
    subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1,
    verbosity=-1,
)
LGB_POISSON_PARAMS = dict(
    objective='poisson', n_estimators=400, learning_rate=0.05,
    num_leaves=31, min_child_samples=80,
    subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1,
    verbosity=-1,
)
RHO_GRID = [round(x * 0.02, 2) for x in range(-15, 16)]   # -0.30~0.30


def _eval_kwargs(X, y):
    """兼容LightGBM新旧版本: 新版弃用eval_set改用eval_X/eval_y"""
    params = inspect.signature(lgb.LGBMClassifier.fit).parameters
    if 'eval_X' in params:
        return {'eval_X': X, 'eval_y': y}
    return {'eval_set': [(X, y)]}


# ==================== Dixon-Coles ====================
def _poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * lam ** k / math.factorial(k)


def dc_tau(x, y, lh, la, rho):
    """Dixon-Coles低比分相关修正因子"""
    if x == 0 and y == 0:
        return 1.0 - lh * la * rho
    if x == 0 and y == 1:
        return 1.0 + lh * rho
    if x == 1 and y == 0:
        return 1.0 + la * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def dc_matrix(lh, la, rho, max_goals=MAX_GOALS):
    """生成(主0..max_goals x 客0..max_goals)比分概率矩阵(归一化)"""
    m = np.zeros((max_goals + 1, max_goals + 1))
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            p = _poisson_pmf(x, lh) * _poisson_pmf(y, la) * dc_tau(x, y, lh, la, rho)
            m[x, y] = max(p, 0.0)
    s = m.sum()
    return m / s if s > 0 else m


def matrix_pool_prob(m):
    """矩阵池内(11比分)概率和"""
    total = 0.0
    for sc in OVER:
        h, a = parse_score(sc)
        if h < m.shape[0] and a < m.shape[1]:
            total += m[h, a]
    return total


def matrix_top_scores(m, n=TOP_PICKS):
    """矩阵中概率最高的n个池内比分, 返回 [(比分, 概率)...] 按概率降序"""
    cand = matrix_pool_scores(m)
    return cand[:n]


def matrix_pool_scores(m):
    """全部池内比分及概率, 返回 [(比分, 概率)...] 按概率降序"""
    cand = []
    for sc in OVER:
        h, a = parse_score(sc)
        if h < m.shape[0] and a < m.shape[1]:
            cand.append((sc, float(m[h, a])))
    cand.sort(key=lambda t: -t[1])
    return cand


def matrix_score_prob(m, score):
    """矩阵中指定比分的概率"""
    h, a = parse_score(score)
    if h is None or h >= m.shape[0] or a >= m.shape[1]:
        return 0.0
    return float(m[h, a])


def fit_rho(lh_arr, la_arr, goals, rho_grid=RHO_GRID):
    """网格搜索rho使训练集实际比分的DC对数似然最大
    goals: [(主队进球, 客队进球)...]"""
    best_rho, best_ll = 0.0, -np.inf
    for rho in rho_grid:
        ll = 0.0
        for lh, la, (x, y) in zip(lh_arr, la_arr, goals):
            x, y = min(x, MAX_GOALS), min(y, MAX_GOALS)
            p = _poisson_pmf(x, lh) * _poisson_pmf(y, la) * dc_tau(x, y, lh, la, rho)
            ll += math.log(max(p, 1e-12))
        if ll > best_ll:
            best_ll, best_rho = ll, rho
    return best_rho


# ==================== 数据准备 ====================
def prepare_data(df, end_date=None, handicap_filter=True):
    """取已完场样本(有in_pool标签), 返回按时间排序的DataFrame
    handicap_filter=True时依次过滤: 目标盘口 → 大小球线 → 排除杯赛"""
    d = df[df['in_pool'].notna()].copy()
    if handicap_filter:
        # 1. 盘口筛选
        mask = d.apply(lambda r: is_target_handicap_over(
            r.get('ah_cur_mean'), r.get('ah_init_mean')), axis=1)
        before = len(d)
        d = d[mask]
        print(f"盘口筛选(目标: {TARGET_HANDICAPS_OVER}): {before} -> {len(d)}场")
        # 2. 大小球线筛选（初盘和终盘均 >= 2.25）
        mask_ou = d.apply(lambda r: is_valid_ou_line_over(
            r.get('ou_line_init_mean'), r.get('ou_line_cur_mean')), axis=1)
        before = len(d)
        d = d[mask_ou]
        print(f"大小球筛选(>= 2.25): {before} -> {len(d)}场")
        # 3. 排除杯赛
        mask_no_cup = ~d['league'].apply(is_cup_league_over)
        before = len(d)
        d = d[mask_no_cup]
        print(f"排除杯赛: {before} -> {len(d)}场")
    if end_date:
        d = d[d['match_id'] <= end_date + '99']
    d = d.sort_values('match_id').reset_index(drop=True)
    d['y'] = d['in_pool'].astype(int)
    d['home_goals'], d['away_goals'] = zip(*d['score'].map(parse_score))
    return d


def split_by_time(d, train_ratio=TRAIN_RATIO, valid_ratio=VALID_RATIO):
    """按时间顺序三段切分: train / valid(调参) / test(仅报告)"""
    n = len(d)
    i_train = int(n * train_ratio)
    i_valid = int(n * (train_ratio + valid_ratio))
    return d.iloc[:i_train], d.iloc[i_train:i_valid], d.iloc[i_valid:]


# ==================== 训练核心 ====================
def train_over(df_train, df_valid, feats, verbose=True):
    """训练全套模型, 返回model字典(含booster/校准器/超参/指标)"""
    # 防泄漏护栏: 标签/辅助列绝不能进入特征(否则训练看似完美但预测无效)
    leak = {'y', 'home_goals', 'away_goals', 'in_pool', 'score', 'sp'} & set(feats)
    if leak:
        raise ValueError(f"特征列表混入了标签/辅助列, 拒绝训练: {sorted(leak)}")
    X_tr, y_tr = df_train[feats], df_train['y']
    X_va, y_va = df_valid[feats], df_valid['y']

    # --- 1. 池内筛选LightGBM ---
    clf = lgb.LGBMClassifier(**LGB_CLF_PARAMS)
    clf.fit(X_tr, y_tr, **_eval_kwargs(X_va, y_va),
            callbacks=[lgb.early_stopping(60, verbose=False),
                       lgb.log_evaluation(0)])
    best_iter = clf.best_iteration_ or LGB_CLF_PARAMS['n_estimators']
    clf_final = lgb.LGBMClassifier(**{**LGB_CLF_PARAMS, 'n_estimators': best_iter})
    clf_final.fit(pd.concat([X_tr, X_va]), pd.concat([y_tr, y_va]))

    # --- 2. 泊松进球模型 ---
    goals_models = {}
    lambdas_va = {}
    for side in ['home', 'away']:
        gm = lgb.LGBMRegressor(**LGB_POISSON_PARAMS)
        gm.fit(df_train[feats], df_train[f'{side}_goals'],
               **_eval_kwargs(df_valid[feats], df_valid[f'{side}_goals']),
               callbacks=[lgb.early_stopping(60, verbose=False),
                          lgb.log_evaluation(0)])
        gi = gm.best_iteration_ or LGB_POISSON_PARAMS['n_estimators']
        gm_final = lgb.LGBMRegressor(**{**LGB_POISSON_PARAMS, 'n_estimators': gi})
        gm_final.fit(pd.concat([df_train[feats], df_valid[feats]]),
                     pd.concat([df_train[f'{side}_goals'], df_valid[f'{side}_goals']]))
        goals_models[side] = gm_final
        # rho拟合用训练集内的λ估计(避免valid信息泄漏到rho)
        lambdas_va[side] = np.clip(gm.predict(df_train[feats]), 0.1, 4.5)

    # --- 3. 全局rho(训练集DC对数似然最大化) ---
    goals_train = list(zip(df_train['home_goals'], df_train['away_goals']))
    rho = fit_rho(lambdas_va['home'], lambdas_va['away'], goals_train)

    # --- 4. 融合α + Isotonic校准(valid集) ---
    p_lgb_va = clf.predict_proba(X_va)[:, 1]
    lh_va = np.clip(goals_models['home'].predict(X_va), 0.1, 4.5)
    la_va = np.clip(goals_models['away'].predict(X_va), 0.1, 4.5)
    p_dc_va = np.array([matrix_pool_prob(dc_matrix(h, a, rho))
                        for h, a in zip(lh_va, la_va)])

    best_alpha, best_brier = 0.5, np.inf
    for alpha in [round(x * 0.1, 1) for x in range(11)]:
        p = alpha * p_lgb_va + (1 - alpha) * p_dc_va
        b = brier_score_loss(y_va, p)
        if b < best_brier:
            best_brier, best_alpha = b, alpha
    p_fused_va = best_alpha * p_lgb_va + (1 - best_alpha) * p_dc_va

    calib = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    calib.fit(p_fused_va, y_va.values)
    p_calib_va = calib.predict(p_fused_va)

    # --- 5. 投注阈值(valid集按EV选择, 无SP则按命中率-覆盖平衡) ---
    sp_va = df_valid['sp'].values
    threshold, ev_curve = choose_threshold(
        p_calib_va, lh_va, la_va, rho, sp_va,
        df_valid['score'].values)

    model = {
        'clf': clf_final, 'goals': goals_models, 'calib': calib,
        'alpha': best_alpha, 'rho': rho, 'threshold': threshold,
        'feats': feats, 'ev_curve': ev_curve,
        'metrics': {
            'valid_auc': round(float(roc_auc_score(y_va, p_calib_va)), 4),
            'valid_brier': round(float(brier_score_loss(y_va, p_calib_va)), 4),
            'valid_pool_rate': round(float(y_va.mean()), 4),
            'valid_n': int(len(y_va)),
        },
    }
    if verbose:
        m = model['metrics']
        print(f"  LightGBM筛选门: best_iter={best_iter}, "
              f"valid AUC={m['valid_auc']}, Brier={m['valid_brier']}")
        print(f"  Dixon-Coles: rho={rho:+.2f}, 融合α={best_alpha:.1f}")
        print(f"  投注阈值: P_pool >= {threshold:.2f}")
    return model


def choose_threshold(p_pool, lh, la, rho, sp_arr, scores,
                     ths=(0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)):
    """在给定概率/矩阵参数下选择投注阈值: 有SP用实现EV, 否则用池内命中率×覆盖率"""
    curve = []
    for th in ths:
        sel = p_pool >= th
        n_sel = int(sel.sum())
        if n_sel == 0:
            curve.append({'threshold': th, 'n': 0, 'pool_hit_rate': None,
                          'ev_per_match': None})
            continue
        realized, pool_hit = [], 0
        for i in np.where(sel)[0]:
            m = dc_matrix(float(lh[i]), float(la[i]), rho)
            picks = [s for s, _ in matrix_top_scores(m)]
            actual = scores[i]
            if actual in OVER_SET:
                pool_hit += 1
            if sp_arr is not None and not np.isnan(sp_arr[i]):
                # 每注1元x8注: 命中=SP-8, 未中=-8
                profit = float(sp_arr[i]) - TOP_PICKS if actual in picks else -TOP_PICKS
                realized.append(profit)
        ev = float(np.mean(realized)) if realized else None
        curve.append({'threshold': th, 'n': n_sel,
                      'pool_hit_rate': round(pool_hit / n_sel, 4),
                      'ev_per_match': round(ev, 3) if ev is not None else None})
    # 选最优阈值
    valid_pts = [c for c in curve if c['n'] >= max(10, len(p_pool) // 50)]
    if not valid_pts:
        return 0.60, curve
    if any(c['ev_per_match'] is not None for c in valid_pts):
        best = max((c for c in valid_pts if c['ev_per_match'] is not None),
                   key=lambda c: c['ev_per_match'])
    else:
        best = max(valid_pts, key=lambda c: c['pool_hit_rate'] * (c['n'] / len(p_pool)) ** 0.5)
    return best['threshold'], curve


# ==================== 预测(回测/每日预测共用) ====================
def _proba_pool(clf, X):
    """兼容sklearn包装器(LGBMClassifier)与原生Booster两种模型"""
    if hasattr(clf, 'predict_proba'):
        return clf.predict_proba(X)[:, 1]
    return clf.predict(X.values if hasattr(X, 'values') else X)


def predict_rows(model, df):
    """对df批量预测, 返回附加列的DataFrame副本:
    p_lgb/p_dc/p_pool(校准后)/lambda_home/lambda_away/top_picks(比分:概率;...)/ev_model"""
    d = df.copy()
    feats = model['feats']
    X = d[feats]   # 保留列名(与训练时一致, 避免sklearn特征名校验警告)
    d['p_lgb'] = _proba_pool(model['clf'], X)
    d['lambda_home'] = np.clip(model['goals']['home'].predict(X), 0.1, 4.5)
    d['lambda_away'] = np.clip(model['goals']['away'].predict(X), 0.1, 4.5)
    d['p_dc'] = [matrix_pool_prob(dc_matrix(h, a, model['rho']))
                 for h, a in zip(d['lambda_home'], d['lambda_away'])]
    fused = model['alpha'] * d['p_lgb'] + (1 - model['alpha']) * d['p_dc']
    d['p_pool'] = model['calib'].predict(fused.values)

    picks_list, ev_list = [], []
    sp_arr = d.get('sp')
    for idx, (h, a) in enumerate(zip(d['lambda_home'], d['lambda_away'])):
        m = dc_matrix(float(h), float(a), model['rho'])
        top = matrix_top_scores(m)
        picks_list.append(';'.join(f"{s}:{p:.3f}" for s, p in top))
        sp = sp_arr.iloc[idx] if sp_arr is not None else None
        if sp is not None and not pd.isna(sp):
            ev_list.append(round(sum(p * sp for _, p in top) - TOP_PICKS, 3))
        else:
            ev_list.append(None)
    d['top_picks'] = picks_list
    d['ev_model'] = ev_list
    return d


# ==================== 模型存取 ====================
def save_model(model, tag=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    tag = tag or datetime.now().strftime('%Y%m%d')
    paths = {
        'clf': os.path.join(MODEL_DIR, f'lgb_pool_{tag}.txt'),
        'goals_home': os.path.join(MODEL_DIR, f'lgb_goals_home_{tag}.txt'),
        'goals_away': os.path.join(MODEL_DIR, f'lgb_goals_away_{tag}.txt'),
        'calib': os.path.join(MODEL_DIR, f'calib_{tag}.joblib'),
        'meta': os.path.join(MODEL_DIR, f'meta_{tag}.json'),
    }
    model['clf'].booster_.save_model(paths['clf'])
    model['goals']['home'].booster_.save_model(paths['goals_home'])
    model['goals']['away'].booster_.save_model(paths['goals_away'])
    joblib.dump(model['calib'], paths['calib'])
    meta = {
        'tag': tag, 'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'features_upside': model['feats'], 'alpha': model['alpha'], 'rho': model['rho'],
        'threshold': model['threshold'], 'metrics': model['metrics'],
        'ev_curve': model['ev_curve'], 'top_picks': TOP_PICKS, 'pool11': OVER,
    }
    with open(paths['meta'], 'w', encoding='utf-8') as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=1)
    # 更新latest指针
    with open(os.path.join(MODEL_DIR, 'latest.json'), 'w', encoding='utf-8') as fp:
        json.dump({'tag': tag}, fp)
    print(f"✓ 模型已保存(tag={tag}): {MODEL_DIR}")
    return paths


def load_model(tag=None):
    """加载模型; tag=None读取latest指向的最新版本"""
    if tag is None:
        latest = os.path.join(MODEL_DIR, 'latest.json')
        if not os.path.exists(latest):
            return None
        with open(latest, encoding='utf-8') as fp:
            tag = json.load(fp)['tag']
    meta_path = os.path.join(MODEL_DIR, f'meta_{tag}.json')
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, encoding='utf-8') as fp:
        meta = json.load(fp)
    model = {
        'clf': lgb.Booster(model_file=os.path.join(MODEL_DIR, f'lgb_pool_{tag}.txt')),
        'goals': {
            'home': lgb.Booster(model_file=os.path.join(MODEL_DIR, f'lgb_goals_home_{tag}.txt')),
            'away': lgb.Booster(model_file=os.path.join(MODEL_DIR, f'lgb_goals_away_{tag}.txt')),
        },
        'calib': joblib.load(os.path.join(MODEL_DIR, f'calib_{tag}.joblib')),
        'alpha': meta['alpha'], 'rho': meta['rho'],
        'threshold': meta['threshold'], 'feats': meta['features_upside'],
        'meta': meta,
    }
    return model


# ==================== 主流程 ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--end', default=None, help='只用该日期(YYYYMMDD)之前的数据训练')
    ap.add_argument('--rebuild', action='store_true', help='强制重建特征缓存')
    args = ap.parse_args()

    df = load_or_build_over(rebuild=args.rebuild)
    d = prepare_data(df, end_date=args.end)
    print(f"完场样本: {len(d)} 场 "
          f"({str(d['match_id'].iloc[0])[:8]} ~ {str(d['match_id'].iloc[-1])[:8]})")
    print(f"池内样本: {int(d['y'].sum())} ({d['y'].mean() * 100:.1f}%)")

    tr, va, te = split_by_time(d)
    feats = feature_columns_over()
    print(f"\n时间切分: 训练{len(tr)} / 调参{len(va)} / 验收{len(te)}")
    print(f"特征数: {len(feats)}\n")

    print("训练混合模型...")
    model = train_over(tr, va, feats)

    # 验收段一次性报告(不参与任何调参)
    if len(te) >= 50:
        pred_te = predict_rows(model, te)
        sel = pred_te[pred_te['p_pool'] >= model['threshold']]
        hit_all = pred_te['y'].mean()
        hit_sel = sel['y'].mean() if len(sel) else 0.0
        top3_hit = sum(
            1 for _, r in pred_te.iterrows()
            if r['score'] in [s.split(':')[0] for s in r['top_picks'].split(';')[:3]]
        )
        print(f"\n验收段({len(te)}场, {str(te['match_id'].iloc[0])[:8]}起, 未参与调参):")
        print(f"  基线池内率: {hit_all * 100:.1f}%")
        print(f"  阈值{model['threshold']:.2f}筛选后: {len(sel)}场, "
              f"池内命中率 {hit_sel * 100:.1f}%")
        print(f"  全样本Top3比分命中率: {top3_hit / len(te) * 100:.1f}%")

    save_model(model)
    print("\nEV阈值曲线(valid):")
    for c in model['ev_curve']:
        ev = f"{c['ev_per_match']:+.2f}" if c['ev_per_match'] is not None else '-'
        hr = f"{c['pool_hit_rate'] * 100:.1f}%" if c['pool_hit_rate'] is not None else '-'
        print(f"  P>={c['threshold']:.2f}: {c['n']:>5}场 池内率{hr:>7} 每场EV {ev}")


if __name__ == '__main__':
    main()
