import pandas as pd
import sys
sys.path.insert(0, '/coding/live')
from train_over import prepare_data, _add_missing_indicators
from features_over import feature_columns_over, load_or_build_over

print("=== 加载数据 ===")
df = load_or_build_over()
print(f"总样本: {len(df)}")

print("\n=== 准备数据 ===")
d = prepare_data(df)
print(f"完场样本: {len(d)}")

print("\n=== 获取特征列表 ===")
feats = feature_columns_over()
print(f"特征数: {len(feats)}")

print("\n=== 检查 league_goal_strength 在原始数据中的类型 ===")
print(f"dtype: {d['league_goal_strength'].dtype}")
print(f"sample values: {d['league_goal_strength'].head().tolist()}")

print("\n=== 添加缺失指示器 ===")
d_with_ind, feats_with_ind = _add_missing_indicators(d, feats)
print(f"新特征数: {len(feats_with_ind)}")

print("\n=== 检查 league_goal_strength 在添加指示器后的类型 ===")
print(f"dtype: {d_with_ind['league_goal_strength'].dtype}")
print(f"sample values: {d_with_ind['league_goal_strength'].head().tolist()}")

print("\n=== 检查 league_goal_strength_missing 的类型 ===")
if 'league_goal_strength_missing' in d_with_ind.columns:
    print(f"dtype: {d_with_ind['league_goal_strength_missing'].dtype}")
    print(f"sample values: {d_with_ind['league_goal_strength_missing'].head().tolist()}")

print("\n=== 填充中位数 ===")
medians = d_with_ind[feats_with_ind].median().to_dict()
X = d_with_ind[feats_with_ind].fillna(d_with_ind[feats_with_ind].median())

print("\n=== 检查填充后的类型 ===")
print(f"league_goal_strength dtype: {X['league_goal_strength'].dtype}")
print(f"league_goal_strength sample: {X['league_goal_strength'].head().tolist()}")
