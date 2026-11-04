# 修复 train_over.py，在训练前强制转换所有特征列为数值类型

with open('/coding/live/train_over.py', 'r') as f:
    content = f.read()

# 找到填充中位数的代码，在其后添加类型转换
old_code = '''    # 新增：在训练集上计算中位数，用于填充缺失值
    medians = df_train[feats].median().to_dict()
    # 用中位数填充训练集和验证集的缺失值
    X_tr = df_train[feats].fillna(df_train[feats].median())
    X_va = df_valid[feats].fillna(df_valid[feats].median())'''

new_code = '''    # 新增：在训练集上计算中位数，用于填充缺失值
    medians = df_train[feats].median().to_dict()
    # 用中位数填充训练集和验证集的缺失值
    X_tr = df_train[feats].fillna(df_train[feats].median())
    X_va = df_valid[feats].fillna(df_valid[feats].median())
    
    # 确保所有特征列都是数值类型
    for col in X_tr.columns:
        X_tr[col] = pd.to_numeric(X_tr[col], errors='coerce')
        X_va[col] = pd.to_numeric(X_va[col], errors='coerce')
    
    # 再次填充（如果有新产生的NaN）
    X_tr = X_tr.fillna(0)
    X_va = X_va.fillna(0)'''

content = content.replace(old_code, new_code)

with open('/coding/live/train_over.py', 'w') as f:
    f.write(content)

print("✓ 修复 train_over.py - 添加数值类型转换")

# 同样修复 train_under.py
with open('/coding/live/train_under.py', 'r') as f:
    content = f.read()

old_code_under = '''    # 新增：在训练集上计算中位数，用于填充缺失值
    medians = df_train[feats].median().to_dict()
    # 用中位数填充训练集和验证集的缺失值
    X_tr = df_train[feats].fillna(df_train[feats].median())
    X_va = df_valid[feats].fillna(df_valid[feats].median())'''

new_code_under = '''    # 新增：在训练集上计算中位数，用于填充缺失值
    medians = df_train[feats].median().to_dict()
    # 用中位数填充训练集和验证集的缺失值
    X_tr = df_train[feats].fillna(df_train[feats].median())
    X_va = df_valid[feats].fillna(df_valid[feats].median())
    
    # 确保所有特征列都是数值类型
    for col in X_tr.columns:
        X_tr[col] = pd.to_numeric(X_tr[col], errors='coerce')
        X_va[col] = pd.to_numeric(X_va[col], errors='coerce')
    
    # 再次填充（如果有新产生的NaN）
    X_tr = X_tr.fillna(0)
    X_va = X_va.fillna(0)'''

content = content.replace(old_code_under, new_code_under)

with open('/coding/live/train_under.py', 'w') as f:
    f.write(content)

print("✓ 修复 train_under.py - 添加数值类型转换")
