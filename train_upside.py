# train_upside.py - UPSIDE + 9 池模型训练脚本
# 功能: Layer 1: LightGBM 二分类预测主队不败概率 P_UPSIDE
#       9 池分支: 三阶段混合(同UNDER): LightGBM池内筛选门 + 泊松回归λ +
#                 Dixon-Coles矩阵池概率 → α融合 → Isotonic校准 → P_underool
#       投注阈值: 在验证集上以期望收益EV(关联result_sp开奖SP, 9注/场)选择
#       采用 70/15/15 时间切分, 早停防止过拟合
#
# 用法:
#   python -u train_upside.py                 # 全量数据训练
#   python -u train_upside.py --end 20260601  # 只用指定日期前的数据（回测用）
#
# 产物(models/目录):
#   lgb_upside.txt      LightGBM UPSIDE 模型
#   lgb_upside_pool.joblib  UNDER池池内筛选门 LightGBM 模型
#   calib_upside.joblib     UNDER池概率 Isotonic 校准器
#   lgb_upside_goals_home.joblib / lgb_upside_goals_away.joblib  泊松进球模型
#   meta_upside.json    元信息（特征列、阈值、alpha、rho、基线等）
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
from sklearn.metrics import roc_auc_score, brier_score_loss

from config import DB_CONFIG
from features_upside import is_cup_league, norm_score, parse_score
from features_upside_ext import (POOL_UPSIDE, POOL_UPSIDE_SET, build_features_upside_ext,
                         feature_columns_upside)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ==================== 训练超参 ====================
TRAIN_RATIO = 0.70          # 训练集占比
VALID_RATIO = 0.15          # 验证集占比
THRESHOLD_UPSIDE = 0.75         # UPSIDE 推荐阈值（P_UPSIDE > 此值推荐双选胜平）

LGB_UPSIDE_PARAMS = dict(
    objective='binary',
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=60,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)

# Dixon-Coles 比分矩阵参数(参考UNDER三阶段混合架构, 供UNDER池输出具体比分)
MAX_GOALS = 8               # 比分矩阵截断进球数
TOP_SCORES_POOL = 3         # UNDER池推荐时输出的具体比分个数
BETS_PER_MATCH = 9          # 每场投注注数(UNDER池全部比分各1注)

LGB_POISSON_PARAMS = dict(
    objective='poisson',
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)

LGB_POOL_PARAMS = dict(
    objective='binary',
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=60,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)

RHO_GRID = [round(-0.30 + 0.02 * i, 2) for i in range(31)]


def _eval_kwargs(X, y):
    """兼容 LightGBM 新旧版本"""
    params = inspect.signature(lgb.LGBMClassifier.fit).parameters
    if 'eval_X' in params:
        return {'eval_X': X, 'eval_y': y}
    return {'eval_set': [(X, y)]}


# ==================== Dixon-Coles 比分矩阵(参考UNDER) ====================
def _poisson_pmf(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def dc_tau(x, y, lh, la, rho):
    """Dixon-Coles 低比分相关修正因子"""
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


def fit_rho(lh_arr, la_arr, goals, rho_grid=RHO_GRID):
    """网格搜索rho使训练集实际比分的DC对数似然最大"""
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


def matrix_top_scores_pool(m, pool, n=TOP_SCORES_POOL):
    """矩阵中给定比分池内概率最高的n个比分, 返回[(比分,概率)...]降序"""
    cand = []
    for sc in pool:
        h, a = parse_score(sc)
        if h is not None and h < m.shape[0] and a < m.shape[1]:
            cand.append((sc, float(m[h, a])))
    cand.sort(key=lambda t: -t[1])
    return cand[:n]


def matrix_pool_prob(m, pool_set=POOL_UPSIDE_SET):
    """Dixon-Coles 矩阵中 9 池比分概率之和(未校准池概率)"""
    total = 0.0
    for sc in pool_set:
        h, a = parse_score(sc)
        if h is not None and h < m.shape[0] and a < m.shape[1]:
            total += float(m[h, a])
    return total


def choose_threshold_pool(p_pool, sp_arr, scores, n_total,
                          ths=(0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)):
    """在验证集上选择UNDER池投注阈值: 有SP用实现EV(9注/场: 命中=2*SP*0.65, 未中=-18),
    否则用池内命中率×覆盖率; 返回(阈值, EV曲线)"""
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
            actual = norm_score(str(scores[i]))
            if actual in POOL_UPSIDE_SET:
                pool_hit += 1
            if sp_arr is not None and not pd.isna(sp_arr[i]):
                profit = 2 * float(sp_arr[i]) * 0.65 - 18 \
                    if actual in POOL_UPSIDE_SET else -18
                realized.append(profit)
        ev = float(np.mean(realized)) if realized else None
        curve.append({'threshold': th, 'n': n_sel,
                      'pool_hit_rate': round(pool_hit / n_sel, 4),
                      'ev_per_match': round(ev, 3) if ev is not None else None})
    # 选最优阈值(样本量护栏: 至少10场且不低于总量2%)
    valid_pts = [c for c in curve if c['n'] >= max(10, n_total // 50)]
    if not valid_pts:
        return 0.60, curve
    if any(c['ev_per_match'] is not None for c in valid_pts):
        best = max((c for c in valid_pts if c['ev_per_match'] is not None),
                   key=lambda c: c['ev_per_match'])
    else:
        best = max(valid_pts,
                   key=lambda c: c['pool_hit_rate'] * (c['n'] / n_total) ** 0.5)
    return best['threshold'], curve


# ==================== 数据准备 ====================
def prepare_data(df, end_date=None):
    """取已完场样本（有 home_unbeaten 标签），按时间排序，排除杯赛"""
    d = df[df['home_unbeaten'].notna()].copy()
    # 排除杯赛（与OVER模型保持一致）
    d = d[~d['league'].apply(is_cup_league)]
    if end_date:
        d = d[d['match_id'] <= end_date + '99']
    d = d.sort_values('match_id').reset_index(drop=True)
    # 进球标签(供泊松回归/Dixon-Coles使用)
    d['home_goals'], d['away_goals'] = zip(*d['score'].map(parse_score))
    d = d[d['home_goals'].notna()].reset_index(drop=True)
    return d


def split_by_time(d, train_ratio=TRAIN_RATIO, valid_ratio=VALID_RATIO):
    """按时间顺序三段切分: train / valid / test"""
    n = len(d)
    i_train = int(n * train_ratio)
    i_valid = int(n * (train_ratio + valid_ratio))
    return d.iloc[:i_train], d.iloc[i_train:i_valid], d.iloc[i_valid:]


# ==================== 训练核心 ====================
def train_upside_model(df_train, df_valid, feats, verbose=True):
    """训练 UPSIDE 二分类模型 + UNDER池三阶段混合分支(同UNDER)"""
    # 防泄漏检查
    leak = {'home_unbeaten', 'in_pool_9', 'in_pool_8', 'in_pool', 'score', 'sp'} & set(feats)
    if leak:
        raise ValueError(f"特征列表混入标签列: {sorted(leak)}")
    
    X_tr, y_tr = df_train[feats], df_train['home_unbeaten']
    X_va, y_va = df_valid[feats], df_valid['home_unbeaten']
    
    # 训练
    clf = lgb.LGBMClassifier(**LGB_UPSIDE_PARAMS)
    clf.fit(
        X_tr, y_tr,
        **_eval_kwargs(X_va, y_va),
        callbacks=[
            lgb.early_stopping(60, verbose=False),
            lgb.log_evaluation(50 if verbose else 0)
        ]
    )
    
    best_iter = clf.best_iteration_ or LGB_UPSIDE_PARAMS['n_estimators']
    
    # 用 train+valid 重训最终模型
    X_all = pd.concat([X_tr, X_va])
    y_all = pd.concat([y_tr, y_va])
    clf_final = lgb.LGBMClassifier(**{**LGB_UPSIDE_PARAMS, 'n_estimators': best_iter})
    clf_final.fit(X_all, y_all)
    
    # --- 泊松进球模型 + 全局rho(参考UNDER三阶段, 供8池输出具体比分) ---
    goals_models = {}
    lambdas_tr = {}
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
        lambdas_tr[side] = np.clip(gm.predict(df_train[feats]), 0.1, 4.5)
    goals_tr = list(zip(df_train['home_goals'], df_train['away_goals']))
    rho = fit_rho(lambdas_tr['home'], lambdas_tr['away'], goals_tr)
    
    # --- UNDER池三阶段混合(同UNDER): 池内筛选门 + DC池概率 → α融合 → Isotonic校准 ---
    y_pool_tr = df_train['in_pool_9']
    y_pool_va = df_valid['in_pool_9']
    pool_clf = lgb.LGBMClassifier(**LGB_POOL_PARAMS)
    pool_clf.fit(X_tr, y_pool_tr, **_eval_kwargs(X_va, y_pool_va),
                 callbacks=[lgb.early_stopping(60, verbose=False),
                            lgb.log_evaluation(0)])
    pi = pool_clf.best_iteration_ or LGB_POOL_PARAMS['n_estimators']
    pool_clf_final = lgb.LGBMClassifier(**{**LGB_POOL_PARAMS, 'n_estimators': pi})
    pool_clf_final.fit(X_all, pd.concat([y_pool_tr, y_pool_va]))
    
    p_lgb_pool_va = pool_clf.predict_proba(X_va)[:, 1]
    lh_va = np.clip(goals_models['home'].predict(X_va), 0.1, 4.5)
    la_va = np.clip(goals_models['away'].predict(X_va), 0.1, 4.5)
    p_dc_va = np.array([matrix_pool_prob(dc_matrix(h, a, rho))
                        for h, a in zip(lh_va, la_va)])
    
    best_alpha, best_brier = 0.5, np.inf
    for alpha in [round(x * 0.1, 1) for x in range(11)]:
        p = alpha * p_lgb_pool_va + (1 - alpha) * p_dc_va
        b = brier_score_loss(y_pool_va, p)
        if b < best_brier:
            best_brier, best_alpha = b, alpha
    p_fused_va = best_alpha * p_lgb_pool_va + (1 - best_alpha) * p_dc_va
    
    calib = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    calib.fit(p_fused_va, y_pool_va.values)
    p_pool_va = calib.predict(p_fused_va)
    
    # --- UNDER池投注阈值(valid集按EV选择, 9注/场: 命中=2*SP*0.65, 未中=-18) ---
    threshold_pool, ev_curve = choose_threshold_pool(
        p_pool_va, df_valid['sp'].values, df_valid['score'].values, len(y_pool_va))
    
    # 验证集评估
    p_va = clf.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, p_va)
    brier = brier_score_loss(y_va, p_va)
    pool_auc = roc_auc_score(y_pool_va, p_pool_va)
    
    # 基础率对比
    baseline = y_va.mean()
    hit_rate = (p_va >= THRESHOLD_UPSIDE).mean()
    hit_correct = ((p_va >= THRESHOLD_UPSIDE) & (y_va == 1)).sum()
    hit_total = (p_va >= THRESHOLD_UPSIDE).sum()
    model_hit_rate = hit_correct / hit_total if hit_total > 0 else 0
    
    if verbose:
        print("\n" + "=" * 70)
        print("UPSIDE + 9 池模型训练完成")
        print("=" * 70)
        print(f"最佳迭代: {best_iter}")
        print(f"验证集 AUC: {auc:.4f}")
        print(f"验证集 Brier Score: {brier:.4f}")
        print(f"\n基础率（主队不败占比）: {baseline:.1%}")
        print(f"模型推荐率（P_UPSIDE>{THRESHOLD_UPSIDE}）: {hit_rate:.1%}")
        print(f"模型命中率: {model_hit_rate:.1%} ({int(hit_correct)}/{int(hit_total)})")
        print(f"提升幅度: {(model_hit_rate - baseline) / baseline * 100:+.1f}%")
        print(f"\nUNDER池三阶段: 池内门AUC={pool_auc:.4f}, 融合α={best_alpha:.1f}, "
              f"rho={rho:+.2f}")
        print(f"UNDER池投注阈值: P_underool >= {threshold_pool:.2f} (valid集EV选择)")
        for c in ev_curve:
            hr = f"{c['pool_hit_rate']*100:.1f}%" if c['pool_hit_rate'] is not None else '  -  '
            ev = f"{c['ev_per_match']:+.2f}" if c['ev_per_match'] is not None else '  -  '
            print(f"  P>={c['threshold']:.2f}: {c['n']:>5}场 池内率{hr:>7} 每场EV {ev}")
        print("=" * 70)
    
    metrics = {
        'auc': auc,
        'brier': brier,
        'baseline': baseline,
        'threshold': THRESHOLD_UPSIDE,
        'best_iter': best_iter,
        'model_hit_rate': model_hit_rate,
        'rho': rho,
        'pool_auc': round(float(pool_auc), 4),
        'alpha': best_alpha,
        'threshold_pool': threshold_pool,
        'ev_curve': ev_curve,
    }
    return (clf_final, pool_clf_final, calib, goals_models, rho,
            best_alpha, threshold_pool, metrics)


# ==================== 主流程 ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--end', default=None, help='只用指定日期前的数据训练（YYYYMMDD）')
    args = ap.parse_args()
    
    print("=" * 70)
    print("UPSIDE + 9 池模型训练")
    print("=" * 70)
    
    # 构建特征
    df = build_features_upside_ext(rebuild=False)
    df = prepare_data(df, end_date=args.end)
    
    if len(df) == 0:
        print("✗ 无有效样本")
        return
    
    print(f"\n总样本: {len(df)} 场")
    
    # 时间切分
    df_train, df_valid, df_test = split_by_time(df)
    print(f"训练集: {len(df_train)} 场 ({df_train['match_id'].iloc[0]} ~ {df_train['match_id'].iloc[-1]})")
    print(f"验证集: {len(df_valid)} 场 ({df_valid['match_id'].iloc[0]} ~ {df_valid['match_id'].iloc[-1]})")
    print(f"测试集: {len(df_test)} 场 ({df_test['match_id'].iloc[0]} ~ {df_test['match_id'].iloc[-1]})")
    
    # 特征列
    feats = feature_columns_upside(df)
    print(f"\n特征数: {len(feats)}")
    
    # 训练
    (model, pool_clf, calib, goals_models, rho, alpha,
     threshold_pool, metrics) = train_upside_model(df_train, df_valid, feats)
    
    # 保存模型
    model_path = os.path.join(MODEL_DIR, 'lgb_upside.txt')
    joblib.dump(model, model_path)
    print(f"\n✓ 模型已保存: {model_path}")
    
    # 保存UNDER池池内筛选门 + Isotonic校准器
    pool_path = os.path.join(MODEL_DIR, 'lgb_upside_pool.joblib')
    calib_path = os.path.join(MODEL_DIR, 'calib_upside.joblib')
    joblib.dump(pool_clf, pool_path)
    joblib.dump(calib, calib_path)
    print(f"✓ UNDER池筛选门/校准器已保存: {pool_path}, {calib_path}")
    
    # 保存泊松进球模型(UNDER池DC矩阵用)
    gh_path = os.path.join(MODEL_DIR, 'lgb_upside_goals_home.joblib')
    ga_path = os.path.join(MODEL_DIR, 'lgb_upside_goals_away.joblib')
    joblib.dump(goals_models['home'], gh_path)
    joblib.dump(goals_models['away'], ga_path)
    print(f"✓ 泊松进球模型已保存: {gh_path}, {ga_path}")
    
    # 保存元信息
    meta = {
        'tag': f'1x_v2_{datetime.now().strftime("%Y%m%d_%H%M")}',
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'features_upside': feats,
        'n_features_upside': len(feats),
        'metrics': metrics,
        'threshold_upside': THRESHOLD_UPSIDE,
        'threshold_pool': threshold_pool,  # UNDER池投注阈值(valid集EV选择)
        'alpha': alpha,                    # 池内门与DC池概率融合系数
        'rho': rho,                        # Dixon-Coles 相关系数
        'pool': POOL_UPSIDE,                   # 9 比分池
        'top_scores_pool': TOP_SCORES_POOL,
        'bets_per_match': BETS_PER_MATCH,
    }
    meta_path = os.path.join(MODEL_DIR, 'meta_upside.json')
    with open(meta_path, 'w', encoding='utf-8') as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)
    print(f"✓ 元信息已保存: {meta_path}")
    
    # 测试集简单评估（可选）
    if len(df_test) > 0:
        p_test = model.predict_proba(df_test[feats])[:, 1]
        auc_test = roc_auc_score(df_test['home_unbeaten'], p_test)
        print(f"\n测试集 AUC: {auc_test:.4f}")


if __name__ == '__main__':
    main()
