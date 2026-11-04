"""
Fix prediction errors:
1. league_goal_strength type issue (None -> NaN)
2. attach_pool_probs duplicate indicator creation
"""
import re

# Fix 1: Ensure league_goal_strength is always numeric in features_over.py
print('=== Fix 1: league_goal_strength type in features_over.py ===')
with open('/coding/live/features_over.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the league_goal_strength assignment to handle None case
old = "            feats['league_goal_strength'] = league_avg / 2.7  # baseline ~2.7 goals/game"
new = "            feats['league_goal_strength'] = float(league_avg) / 2.7 if league_avg is not None else None  # baseline ~2.7 goals/game"
content = content.replace(old, new)

with open('/coding/live/features_over.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('✓ Fixed features_over.py')

# Fix 1b: Same fix in features_under.py
print('\n=== Fix 1b: league_goal_strength type in features_under.py ===')
with open('/coding/live/features_under.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(old, new)

with open('/coding/live/features_under.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('✓ Fixed features_under.py')

# Fix 2: attach_pool_probs in predict_upside.py - skip existing _missing columns
print('\n=== Fix 2: attach_pool_probs indicator dedup in predict_upside.py ===')
with open('/coding/live/predict_upside.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the indicator creation loop and add skip for _missing columns
old_indicator = """        # 添加缺失指示器(为所有特征列创建,与训练一致)
        indicator_cols = []
        for col in available_feats:
            indicator_name = f'{col}_missing'
            X[indicator_name] = X[col].isnull().astype(int)
            indicator_cols.append(indicator_name)
        available_feats = available_feats + indicator_cols"""

new_indicator = """        # 添加缺失指示器(只为原始特征创建,不为已有的指示器创建)
        indicator_cols = []
        for col in available_feats:
            if col.endswith('_missing'):  # 跳过已有的指示器列
                continue
            indicator_name = f'{col}_missing'
            X[indicator_name] = X[col].isnull().astype(int)
            indicator_cols.append(indicator_name)
        available_feats = [f for f in available_feats if not f.endswith('_missing')] + indicator_cols"""

content = content.replace(old_indicator, new_indicator)

with open('/coding/live/predict_upside.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('✓ Fixed predict_upside.py')

# Fix 3: Also fix train_under.py predict_rows to ensure league_goal_strength is converted
print('\n=== Fix 3: Ensure numeric types in train_under.py predict_rows ===')
with open('/coding/live/train_under.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add a line to convert object columns to numeric before creating indicators
old_x = "    X = d[available_feats]   # 保留列名 (与训练时一致，避免 sklearn 特征名校验警告)"
new_x = """    X = d[available_feats]   # 保留列名 (与训练时一致，避免 sklearn 特征名校验警告)
    # 确保所有特征列为数值类型(防止object类型导致LightGBM报错)
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors='coerce')"""
content = content.replace(old_x, new_x)

with open('/coding/live/train_under.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('✓ Fixed train_under.py')

# Fix 3b: Same fix in train_over.py
print('\n=== Fix 3b: Same numeric type fix in train_over.py ===')
with open('/coding/live/train_over.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(old_x, new_x)

with open('/coding/live/train_over.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('✓ Fixed train_over.py')

print('\n=== All fixes applied ===')
