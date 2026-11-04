import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("A/B 测试：180特征 vs 90特征")
print("=" * 60)

# 加载缓存数据
print("\n[1] 加载数据...")
df = pd.read_csv('features_under_cache.csv')
print(f"总样本数: {len(df)}")

# 加载当前模型获取特征列表
clf_old = lgb.Booster(model_file='models/lgb_pool_under_20261103.txt')
all_feats = clf_old.feature_name()
orig_feats = [f for f in all_feats if not f.endswith('_missing')]
ind_feats = [f for f in all_feats if f.endswith('_missing')]
print(f"原始特征: {len(orig_feats)}, 指示器: {len(ind_feats)}")

# 筛选有标签的样本
df = df[df['in_pool'].notna()].copy()
df['y'] = df['in_pool'].astype(int)
print(f"有效样本: {len(df)}")

# 按时间排序划分
if 'match_id' in df.columns:
    df = df.sort_values('match_id')
split_idx = int(len(df) * 0.8)
df_train = df.iloc[:split_idx]
df_valid = df.iloc[split_idx:]
print(f"训练集: {len(df_train)}, 验证集: {len(df_valid)}")

# 创建缺失指示器
print("\n[2] 创建缺失指示器...")
for col in orig_feats:
    if col in df_train.columns:
        ind_name = f'{col}_missing'
        df_train[ind_name] = df_train[col].isnull().astype(int)
        df_valid[ind_name] = df_valid[col].isnull().astype(int)

# 模型A：180特征（完整特征）
print("\n[3] 训练模型A（180特征）...")
X_train_a = df_train[all_feats].copy()
y_train = df_train['y']
X_valid_a = df_valid[all_feats].copy()
y_valid = df_valid['y']

# 中位数填充
medians = X_train_a.median()
X_train_a = X_train_a.fillna(medians)
X_valid_a = X_valid_a.fillna(medians)

train_data_a = lgb.Dataset(X_train_a, label=y_train)
params = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5
}
model_a = lgb.train(params, train_data_a, num_boost_round=200)
pred_a = model_a.predict(X_valid_a)
auc_a = roc_auc_score(y_valid, pred_a)
acc_a = accuracy_score(y_valid, (pred_a > 0.5).astype(int))
print(f"模型A AUC: {auc_a:.4f}, 准确率: {acc_a:.4f}")

# 模型B：90特征（仅原始特征）
print("\n[4] 训练模型B（90特征）...")
X_train_b = df_train[orig_feats].copy()
X_valid_b = df_valid[orig_feats].copy()

# 中位数填充
medians_b = X_train_b.median()
X_train_b = X_train_b.fillna(medians_b)
X_valid_b = X_valid_b.fillna(medians_b)

train_data_b = lgb.Dataset(X_train_b, label=y_train)
model_b = lgb.train(params, train_data_b, num_boost_round=200)
pred_b = model_b.predict(X_valid_b)
auc_b = roc_auc_score(y_valid, pred_b)
acc_b = accuracy_score(y_valid, (pred_b > 0.5).astype(int))
print(f"模型B AUC: {auc_b:.4f}, 准确率: {acc_b:.4f}")

# 对比结果
print("\n" + "=" * 60)
print("对比结果")
print("=" * 60)
print(f"模型A (180特征): AUC={auc_a:.4f}, Acc={acc_a:.4f}")
print(f"模型B (90特征):  AUC={auc_b:.4f}, Acc={acc_b:.4f}")
print(f"差异: AUC={auc_a-auc_b:+.4f}, Acc={acc_a-acc_b:+.4f}")

if auc_a > auc_b:
    print(f"\n结论: 保留缺失指示器 (AUC提升 {auc_a-auc_b:.4f})")
elif auc_b > auc_a:
    print(f"\n结论: 删除缺失指示器 (AUC提升 {auc_b-auc_a:.4f})")
else:
    print(f"\n结论: 两者无差异")
