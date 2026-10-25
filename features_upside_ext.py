# features_upside_ext.py - UPSIDE（主队不败）模型特征工程
# 功能: 构建"一场比赛一行"的特征宽表, 标签为 home_unbeaten（主胜或平局=1, 客胜=0）
#       完全复用 features_upside.py 的特征构建逻辑, 仅替换标签列
#
# 用法:
#   from features_upside_ext import build_features_upside_ext
#   df = build_features_upside_ext(conn)     # 返回含 home_unbeaten 标签的特征表
import os
import re
from collections import Counter, defaultdict

import pandas as pd

from config import DB_CONFIG
from features_upside import (META_COLS, OVER, OVER_SET, feature_columns,
                      load_or_build, parse_score, team_similarity, team_side,
                      _mean, _std, _share_mode, handicap_to_num, to_float,
                      norm_score, is_finished)

# ==================== 常量 ====================
CACHE_FILE_UPSIDE = 'features_upside_ext_cache.csv'

# 9 比分池(UPSIDE 模型): 原 8 池(主队不败低比分) + 0-1
POOL_UPSIDE = ['1-0', '2-0', '2-1', '3-1', '3-2', '0-0', '1-1', '2-2', '0-1']
POOL_UPSIDE_SET = set(POOL_UPSIDE)

# ==================== 标签构建 ====================
def label_home_unbeaten(score):
    """主队不败标签: 主胜或平局=1, 客胜=0, 无效=None"""
    h, a = parse_score(score)
    if h is None:
        return None
    return 1 if h >= a else 0


def label_in_pool_9(score):
    """9 池标签: 比分在 9 池内=1, 否则=0, 无效=None
    9 池: ['1-0', '2-0', '2-1', '3-1', '3-2', '0-0', '1-1', '2-2', '0-1']"""
    if score is None or (isinstance(score, float) and pd.isna(score)):
        return None
    s = norm_score(str(score))
    if not s or s not in POOL_UPSIDE_SET:
        return 0 if is_finished(score) else None
    return 1


# ==================== 特征构建（复用 features_upside.py）====================
def build_features_upside_ext(conn=None, rebuild=False):
    """构建 UPSIDE 模型特征表
    
    Args:
        conn: 数据库连接（rebuild=True 时需要）
        rebuild: 是否强制重建缓存
        
    Returns:
        DataFrame 含以下列:
        - META_COLS: match_id, league, home_team, away_team, score 等
        - 所有 features_upside.py 的特征列（亚盘、大小球、近10场、往绩、积分等）
        - home_unbeaten: 主队不败标签（1/0/None）
        - in_pool_9: 9 池标签（1/0/None）
    """
    cache_path = CACHE_FILE_UPSIDE
    
    # 优先读缓存
    if not rebuild and os.path.exists(cache_path):
        print(f"[UPSIDE] 读取缓存: {cache_path}")
        df = pd.read_csv(cache_path)
        return df
    
    # 从 features_upside.py 复用特征构建
    print("[UPSIDE] 构建特征表（复用 features_upside.py 逻辑）...")
    df_base = load_or_build(conn=conn, rebuild=rebuild)
    
    # 新增标签列
    df_base['home_unbeaten'] = df_base['score'].apply(label_home_unbeaten)
    df_base['in_pool_9'] = df_base['score'].apply(label_in_pool_9)
    
    # 保存缓存
    df_base.to_csv(cache_path, index=False)
    print(f"[UPSIDE] 缓存已保存: {cache_path}")
    print(f"[UPSIDE] 样本数: {len(df_base)}, 完场: {df_base['home_unbeaten'].notna().sum()}")
    
    # 统计标签分布
    finished = df_base[df_base['home_unbeaten'].notna()]
    if len(finished) > 0:
        unbeaten_rate = finished['home_unbeaten'].mean()
        pool9_rate = finished['in_pool_9'].mean()
        print(f"[UPSIDE] 主队不败率: {unbeaten_rate:.1%} ({int(unbeaten_rate * len(finished))}/{len(finished)})")
        print(f"[UPSIDE] 9 池命中率: {pool9_rate:.1%} ({int(pool9_rate * len(finished))}/{len(finished)})")
    
    return df_base


def feature_columns_upside(df):
    """返回 UPSIDE 模型的特征列名（排除标签和元信息）"""
    base_feats = feature_columns(df)  # 复用 features_upside.py 的特征列表
    exclude = {'home_unbeaten', 'in_pool_9', 'in_pool_8', 'in_pool', 'sp', 'score'}
    return [f for f in base_feats if f not in exclude]


if __name__ == '__main__':
    import pymysql
    import os
    
    conn = pymysql.connect(**DB_CONFIG)
    df = build_features_upside_ext(conn, rebuild=True)
    print("\n特征列示例:")
    print(df.columns.tolist()[:20])
    conn.close()
