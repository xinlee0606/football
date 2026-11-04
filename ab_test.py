import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("A/B 测试：180特征 vs 90特征")
print("=" * 60)

df = pd.read_csv('features_under_cache.csv')
print("总样本数:", len(df))

clf_old = lgb.Booster(model_file='models/lgb_pool_under_20261103.txt')
all_feats = clf_old.feature_name()
orig_feats = [f for f in all_feats if not f.endswith('_missing')]
ind_feats = [f for f in all_feats if f.endswith('_missing')]
print("原始特征:", len(orig_feats), "指示器:", len(ind_feats))

df = df[df['y'].notna()].copy()
print("有效样本:", len(df))

if 'match_date' in df.columns:
    df = df.sort_values('match_date')
split_idx = int(len(df) * 0.8)
df_train = df.iloc[:split_idx]
df_valid = df.iloc[split_idx:]
print("训练集:", len(df_train), "验证集:", len(df_valid))

for col in orig_feats:
    if col in df_train.columns:
        df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
        df_valid[col] = pd.to_numeric(df_valid[col], errors='coerce')

# 模型A: 180特征
print("\n" + "=" * 60)
print("模型A: 180特征（原始90 + 指示器90）")
print("=" * 60)

df_train_a = df_train.copy()
df_valid_a = df_valid.copy()
for col in orig_feats:
    if col in df_train_a.columns:
        df_train_a[col + '_missing'] = df_train_a[col].isnull().astype(int)
        df_valid_a[col + '_missing'] = df_valid_a[col].isnull().astype(int)

feats_a = orig_feats + [c + '_missing' for c in orig_feats if c in df_train_a.columns]
feats_a = [f for f in feats_a if f in df_train_a.columns]

X_tr_a = df_train_a[feats_a].fillna(df_train_a[feats_a].median())
X_va_a = df_valid_a[feats_a].fillna(df_valid_a[feats_a].median())
y_tr = df_train_a['y']
y_va = df_valid_a['y']

clf_a = lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
clf_a.fit(X_tr_a, y_tr, eval_set=[(X_va_a, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])

pred_a = clf_a.predict_proba(X_va_a)[:, 1]
auc_a = roc_auc_score(y_va, pred_a)

best_acc_a = 0
best_thr_a = 0.5
for thr in np.arange(0.5, 0.95, 0.01):
    acc = accuracy_score(y_va, (pred_a >= thr).astype(int))
    if acc > best_acc_a:
        best_acc_a = acc
        best_thr_a = thr

print("AUC:", round(auc_a, 4))
print("最佳阈值:", round(best_thr_a, 2))
print("最佳准确率:", round(best_acc_a, 4))
print("阈值0.80准确率:", round(accuracy_score(y_va, (pred_a >= 0.80).astype(int)), 4))
print("阈值0.85准确率:", round(accuracy_score(y_va, (pred_a >= 0.85).astype(int)), 4))

# 模型B: 90特征
print("\n" + "=" * 60)
print("模型B: 90特征（仅原始，无指示器）")
print("=" * 60)

feats_b = [f for f in orig_feats if f in df_train.columns]
X_tr_b = df_train[feats_b].fillna(df_train[feats_b].median())
X_va_b = df_valid[feats_b].fillna(df_valid[feats_b].median())

clf_b = lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
clf_b.fit(X_tr_b, y_tr, eval_set=[(X_va_b, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])

pred_b = clf_b.predict_proba(X_va_b)[:, 1]
auc_b = roc_auc_score(y_va, pred_b)

best_acc_b = 0
best_thr_b = 0.5
for thr in np.arange(0.5, 0.95, 0.01):
    acc = accuracy_score(y_va, (pred_b >= thr).astype(int))
    if acc > best_acc_b:
        best_acc_b = acc
        best_thr_b = thr

print("AUC:", round(auc_b, 4))
print("最佳阈值:", round(best_thr_b, 2))
print("最佳准确率:", round(best_acc_b, 4))
print("阈值0.80准确率:", round(accuracy_score(y_va, (pred_b >= 0.80).astype(int)), 4))
print("阈值0.85准确率:", round(accuracy_score(y_va, (pred_b >= 0.85).astype(int)), 4))

# 对比
print("\n" + "=" * 60)
print("对比总结")
print("=" * 60)
print("指标                模型A(180)    模型B(90)     差异")
print("-" * 60)
print("AUC                ", round(auc_a, 4), "     ", round(auc_b, 4), "     ", round(auc_b - auc_a, 4))
print("最佳准确率          ", round(best_acc_a, 4), "     ", round(best_acc_b, 4), "     ", round(best_acc_b - best_acc_a, 4))
acc80_a = accuracy_score(y_va, (pred_a >= 0.80).astype(int))
acc80_b = accuracy_score(y_va, (pred_b >= 0.80).astype(int))
print("阈值0.80准确率      ", round(acc80_a, 4), "     ", round(acc80_b, 4), "     ", round(acc80_b - acc80_a, 4))
acc85_a = accuracy_score(y_va, (pred_a >= 0.85).astype(int))
acc85_b = accuracy_score(y_va, (pred_b >= 0.85).astype(int))
print("阈值0.85准确率      ", round(acc85_a, 4), "     ", round(acc85_b, 4), "     ", round(acc85_b - acc85_a, 4))

n_high_a = int((pred_a >= 0.85).sum())
n_high_b = int((pred_b >= 0.85).sum())
print("\n阈值0.85以上样本数: A=" + str(n_high_a) + ", B=" + str(n_high_b))

if auc_b >= auc_a - 0.005:
    print("\n结论: 90特征模型性能相当，建议删除指示器简化模型")
else:
    print("\n结论: 180特征模型AUC高" + str(round(auc_a - auc_b, 4)) + "，建议保留指示器")
