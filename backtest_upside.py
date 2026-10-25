# backtest_upside.py - UPSIDE + 9 池两层模型回测脚本
# 功能:
#   1. 用测试集（后 15% 数据）回测
#   2. 统计指标:
#      - UPSIDE 命中率（P_UPSIDE > 0.75 的场次中，实际主队不败占比）
#      - 9 池命中率（P_underool >= 阈值 的场次中，实际 9 池占比）
#      - 9 池实现盈亏（9注/场: 命中=SP-9, 未中=-9, 关联 result_sp）
#   3. 输出详细报告
#
# 用法:
#   python -u backtest_upside.py                 # 全量回测
#   python -u backtest_upside.py --start 20260601 # 指定起始日期
import argparse
import json
import os

import numpy as np
import pandas as pd
import pymysql
from sklearn.metrics import roc_auc_score, brier_score_loss

from config import DB_CONFIG
from features_upside import is_cup_league, norm_score
from features_upside_ext import POOL_UPSIDE_SET, build_features_upside_ext, feature_columns_upside
from predict_upside import (THRESHOLD_9POOL_FALLBACK, attach_pool_probs,
                        load_goal_models, load_pool_model)
from train_upside import BETS_PER_MATCH, MODEL_DIR, split_by_time, prepare_data

# ==================== 配置 ====================
THRESHOLD_UPSIDE = 0.75


def load_model():
    """加载模型"""
    model_path = os.path.join(MODEL_DIR, 'lgb_upside.txt')
    meta_path = os.path.join(MODEL_DIR, 'meta_upside.json')
    
    if not os.path.exists(model_path):
        print(f"✗ 未找到模型: {model_path}")
        return None, None
    
    import joblib
    model = joblib.load(model_path)
    
    with open(meta_path, encoding='utf-8') as fp:
        meta = json.load(fp)
    
    return model, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default=None, help='回测起始日期 YYYYMMDD（默认用测试集）')
    args = ap.parse_args()
    
    print("=" * 70)
    print("UPSIDE + 9 池两层模型回测")
    print("=" * 70)
    
    # 加载模型
    model, meta = load_model()
    if model is None:
        return
    
    print(f"模型版本: {meta.get('tag')}")
    print(f"训练时间: {meta.get('trained_at')}")
    thr_pool = meta.get('threshold_pool', THRESHOLD_9POOL_FALLBACK)
    print(f"9 池阈值: P_underool >= {thr_pool} (meta EV选择)")
    print("=" * 70)
    
    # 构建特征
    df = build_features_upside_ext(rebuild=False)
    
    # 筛选回测范围
    if args.start:
        df_test = df[df['match_id'] >= args.start].copy()
        df_test = df_test[df_test['home_unbeaten'].notna()].sort_values('match_id').reset_index(drop=True)
    else:
        # 默认用测试集（后 15%，prepare_data 已排除杯赛）
        df_full = prepare_data(df)
        _, _, df_test = split_by_time(df_full)
    
    # 排除杯赛（确保 args.start 路径也排除）
    df_test = df_test[~df_test['league'].apply(is_cup_league)].reset_index(drop=True)
    
    if len(df_test) == 0:
        print("✗ 无有效测试样本")
        return
    
    print(f"\n测试集: {len(df_test)} 场 ({df_test['match_id'].iloc[0]} ~ {df_test['match_id'].iloc[-1]})")
    
    # 特征列
    feats = meta.get('features_upside', feature_columns_upside(df_test))
    
    # 预测: Layer 1 + UNDER池三阶段混合
    df_test = df_test.copy()
    df_test['p_upside'] = model.predict_proba(df_test[feats])[:, 1]
    gh, ga, rho = load_goal_models(meta)
    pool_clf, calib = load_pool_model(meta)
    alpha = meta.get('alpha', 0.5)
    df_test = attach_pool_probs(df_test, feats, pool_clf, calib, alpha, gh, ga, rho)
    
    # ==================== UPSIDE 回测 ====================
    picks_upside = df_test[df_test['p_upside'] >= THRESHOLD_UPSIDE].copy()
    
    if len(picks_upside) > 0:
        hit_upside = (picks_upside['home_unbeaten'] == 1).mean()
        baseline_upside = df_test['home_unbeaten'].mean()
        
        auc_upside = roc_auc_score(df_test['home_unbeaten'], df_test['p_upside'])
        brier_upside = brier_score_loss(df_test['home_unbeaten'], df_test['p_upside'])
        
        print("\n" + "=" * 70)
        print("【Layer 1: UPSIDE（主队不败）回测】")
        print("=" * 70)
        print(f"推荐场次: {len(picks_upside)} 场")
        print(f"基础率（全体测试集）: {baseline_upside:.1%}")
        print(f"模型命中率: {hit_upside:.1%}")
        print(f"提升幅度: {(hit_upside - baseline_upside) / baseline_upside * 100:+.1f}%")
        print(f"AUC: {auc_upside:.4f}")
        print(f"Brier Score: {brier_upside:.4f}")
        
        # 按月统计稳定性
        picks_upside = picks_upside.copy()
        picks_upside['month'] = picks_upside['match_id'].astype(str).str[:6]
        monthly_stats = picks_upside.groupby('month').agg(
            total=('home_unbeaten', 'count'),
            hits=('home_unbeaten', lambda x: (x == 1).sum())
        ).reset_index()
        monthly_stats['hit_rate'] = monthly_stats['hits'] / monthly_stats['total']
        
        print(f"\n月度稳定性:")
        for _, row in monthly_stats.iterrows():
            print(f"  {row['month']}: {row['hit_rate']*100:.1f}% ({int(row['hits'])}/{int(row['total'])})")
    else:
        print("\n【Layer 1: UPSIDE】无推荐场次（阈值过高或模型失效）")
    
    # ==================== 9 池回测 ====================
    picks_pool = df_test[df_test['p_underool'] >= thr_pool].copy()
    
    def in_pool_actual(score):
        s = norm_score(str(score))
        return 1 if s in POOL_UPSIDE_SET else 0
    
    if len(picks_pool) > 0:
        picks_pool['in_pool_actual'] = picks_pool['score'].apply(in_pool_actual)
        
        hit_pool = picks_pool['in_pool_actual'].mean()
        baseline_pool = df_test['score'].apply(in_pool_actual).mean()
        
        print("\n" + "=" * 70)
        print("【9 池比分回测（三阶段混合: 池内门+泊松+DC+校准）】")
        print("=" * 70)
        print(f"推荐场次: {len(picks_pool)} 场")
        print(f"基础率（全体测试集）: {baseline_pool:.1%}")
        print(f"模型命中率: {hit_pool:.1%}")
        print(f"提升幅度: {(hit_pool - baseline_pool) / baseline_pool * 100:+.1f}%")
        
        # 实现盈亏: 9注/场(UNDER池全部比分各1注), 命中=SP-9, 未中=-9
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        ids = [str(m) for m in picks_pool['match_id']]
        placeholders = ','.join(['%s'] * len(ids))
        cursor.execute(
            f"SELECT match_id, sp_value FROM result_sp WHERE CONVERT(match_id USING utf8mb4) COLLATE utf8mb4_general_ci IN ({placeholders})",
            ids)
        sp_map = {r[0]: (float(r[1]) if r[1] is not None else None)
                  for r in cursor.fetchall()}
        cursor.close()
        conn.close()
        
        profits = []
        for _, r in picks_pool.iterrows():
            sp = sp_map.get(str(r['match_id']))
            if sp is None:
                continue
            profits.append(2 * sp * 0.65 - 18 if r['in_pool_actual'] == 1
                           else -18)
        if profits:
            total_profit = sum(profits)
            ev_per_match = total_profit / len(profits)
            roi = total_profit / (len(profits) * 18)
            print(f"\n实现盈亏(有SP的 {len(profits)} 场, {BETS_PER_MATCH}注/场):")
            print(f"  总盈亏: {total_profit:+.1f} 元  每场EV: {ev_per_match:+.2f} 元  ROI: {roi:+.1%}")
        else:
            total_profit, ev_per_match, roi = None, None, None
            print("\n实现盈亏: 无SP数据")
        
        # 与 Layer 1 对比
        if len(picks_upside) > 0:
            overlap = set(picks_upside['match_id']) & set(picks_pool['match_id'])
            print(f"\n重叠场次: {len(overlap)} 场（同时满足 UPSIDE 和 9 池条件）")
    else:
        print("\n【9 池】无推荐场次（阈值过高或模型失效）")
        total_profit, ev_per_match, roi = None, None, None
    
    # ==================== 综合评估 ====================
    print("\n" + "=" * 70)
    print("【综合评估】")
    print("=" * 70)
    
    if len(picks_upside) > 0 and len(picks_pool) > 0:
        # 2 串 1 理论命中率
        theoretical_2x1_upside = hit_upside ** 2
        theoretical_2x1_pool = hit_pool ** 2
        
        print(f"UPSIDE 2串1理论命中率: {theoretical_2x1_upside*100:.1f}%")
        print(f"9 池 2串1理论命中率: {theoretical_2x1_pool*100:.1f}%")
        
        # 期望收益（UPSIDE 假设赔率 1.4; 9 池用实现EV）
        ev_upside = hit_upside * 1.4 - 1
        print(f"\nUPSIDE 期望收益（赔率1.4）: {ev_upside:+.3f} {'✓盈利' if ev_upside > 0 else '✗亏损'}")
        if ev_per_match is not None:
            print(f"9 池实现每场EV（{BETS_PER_MATCH}注/场）: {ev_per_match:+.2f} "
                  f"{'✓盈利' if ev_per_match > 0 else '✗亏损'}")
    
    print("=" * 70)
    
    # 保存回测报告
    report = {
        'model_tag': meta.get('tag'),
        'test_range': f"{df_test['match_id'].iloc[0]} ~ {df_test['match_id'].iloc[-1]}",
        'n_test': len(df_test),
        'layer1_upside': {
            'threshold': THRESHOLD_UPSIDE,
            'n_picks': len(picks_upside),
            'baseline': float(df_test['home_unbeaten'].mean()),
            'hit_rate': float(hit_upside) if len(picks_upside) > 0 else None,
            'auc': float(auc_upside) if len(picks_upside) > 0 else None,
        },
        'layer2_underool': {
            'threshold': thr_pool,
            'n_picks': len(picks_pool),
            'baseline': float(baseline_pool) if len(picks_pool) > 0 else None,
            'hit_rate': float(hit_pool) if len(picks_pool) > 0 else None,
            'profit_total': total_profit,
            'ev_per_match': ev_per_match,
            'roi': roi,
        },
    }
    
    report_path = 'backtest_upside_report.json'
    with open(report_path, 'w', encoding='utf-8') as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 回测报告已保存: {report_path}")


if __name__ == '__main__':
    main()
