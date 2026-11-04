# 修复 predict_rows 函数，添加数值类型转换

with open('/coding/live/train_over.py', 'r') as f:
    content = f.read()

# 找到 predict_rows 中的填充代码，添加类型转换
old_code = '''        # 更新feats以包含指示器列
        feats = [f for f in feats if f in X.columns] + indicator_cols
        X = X.fillna(fill_medians)
    d['p_lgb'] = _proba_pool(model['clf'], X)'''

new_code = '''        # 更新feats以包含指示器列
        feats = [f for f in feats if f in X.columns] + indicator_cols
        X = X.fillna(fill_medians)
    
    # 确保所有特征列都是数值类型
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)
    
    d['p_lgb'] = _proba_pool(model['clf'], X)'''

content = content.replace(old_code, new_code)

with open('/coding/live/train_over.py', 'w') as f:
    f.write(content)

print("✓ 修复 train_over.py predict_rows - 添加数值类型转换")

# 同样修复 train_under.py
with open('/coding/live/train_under.py', 'r') as f:
    content = f.read()

old_code_under = '''        # 更新feats以包含指示器列
        feats = [f for f in feats if f in X.columns] + indicator_cols
        X = X.fillna(fill_medians)
    d['p_lgb'] = _proba_pool(model['clf'], X)'''

new_code_under = '''        # 更新feats以包含指示器列
        feats = [f for f in feats if f in X.columns] + indicator_cols
        X = X.fillna(fill_medians)
    
    # 确保所有特征列都是数值类型
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)
    
    d['p_lgb'] = _proba_pool(model['clf'], X)'''

content = content.replace(old_code_under, new_code_under)

with open('/coding/live/train_under.py', 'w') as f:
    f.write(content)

print("✓ 修复 train_under.py predict_rows - 添加数值类型转换")
