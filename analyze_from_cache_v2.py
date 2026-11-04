import pandas as pd
import numpy as np

print("=" * 70)
print("从特征缓存分析盘口与大小球线分布")
print("=" * 70)

# 加载 OVER 缓存
print("\n[1] 加载 OVER 特征缓存...")
df = pd.read_csv('/coding/live/features_over_cache.csv')
print(f"总比赛数: {len(df)}")

# 解析比分
def parse_score(score):
    try:
        if pd.isna(score) or score == '':
            return None, None
        parts = str(score).split('-')
        if len(parts) == 2:
            return int(parts[0].strip()), int(parts[1].strip())
    except:
        pass
    return None, None

df[['home_goals', 'away_goals']] = df['score'].apply(lambda x: pd.Series(parse_score(x)))
df = df[df['home_goals'].notna() & df['away_goals'].notna()].copy()
df['total_goals'] = df['home_goals'] + df['away_goals']

print(f"有效比赛数: {len(df)}")

# 分析亚盘盘口分布
print("\n" + "=" * 70)
print("亚盘盘口分布分析")
print("=" * 70)

# 简化的盘口分类
def categorize_handicap(x):
    if pd.isna(x):
        return '未知'
    if x <= -1.25:
        return '受让球半+'
    elif x <= -0.75:
        return '受让一球~受让平/半'
    elif x <= -0.25:
        return '受让半球'
    elif x <= 0.25:
        return '平手'
    elif x <= 0.75:
        return '让平/半~让半球'
    elif x <= 1.25:
        return '让一球'
    else:
        return '让球半+'

df['ah_category'] = df['ah_cur_mean'].apply(categorize_handicap)
ah_dist = df['ah_category'].value_counts().sort_index()
print("\n终盘亚盘分布:")
for bin_name, count in ah_dist.items():
    pct = count / len(df) * 100
    print(f"  {bin_name:25s}: {count:5d}场 ({pct:5.1f}%)")

# 分析大小球线分布
print("\n" + "=" * 70)
print("大小球线分布分析")
print("=" * 70)

def categorize_ou(x):
    if pd.isna(x):
        return '未知'
    if x < 2.0:
        return '<2.0'
    elif x < 2.5:
        return '2.0~2.25'
    elif x < 2.75:
        return '2.5'
    elif x < 3.0:
        return '2.5~2.75'
    elif x < 3.25:
        return '3.0'
    elif x < 3.5:
        return '3.0~3.25'
    else:
        return '3.5+'

df['ou_category'] = df['ou_cur_mean'].apply(categorize_ou)
ou_dist = df['ou_category'].value_counts().sort_index()
print("\n终盘大小球线分布:")
for bin_name, count in ou_dist.items():
    pct = count / len(df) * 100
    print(f"  {bin_name:15s}: {count:5d}场 ({pct:5.1f}%)")

# 分析总进球数分布
print("\n" + "=" * 70)
print("总进球数分布")
print("=" * 70)

goals_dist = df['total_goals'].value_counts().sort_index()
print("\n总进球数分布:")
for goals, count in goals_dist.items():
    pct = count / len(df) * 100
    cum_pct = df[df['total_goals'] <= goals].shape[0] / len(df) * 100
    print(f"  {int(goals)}球: {count:5d}场 ({pct:5.1f}%, 累计{cum_pct:5.1f}%)")

# 分析不同盘口下的进球分布
print("\n" + "=" * 70)
print("不同盘口下的平均进球数")
print("=" * 70)

for ah_cat in ['受让一球~受让平/半', '受让半球', '平手', '让平/半~让半球', '让一球']:
    subset = df[df['ah_category'] == ah_cat]
    if len(subset) > 0:
        avg_goals = subset['total_goals'].mean()
        over25 = (subset['total_goals'] > 2.5).mean() * 100
        over225 = (subset['total_goals'] > 2.25).mean() * 100
        print(f"  {ah_cat:25s}: {len(subset):5d}场, 均{avg_goals:.2f}球, >2.5球{over25:.1f}%, >2.25球{over225:.1f}%")

# 分析不同大小球线下的进球分布
print("\n" + "=" * 70)
print("不同大小球线下的实际进球分布")
print("=" * 70)

for ou_cat in ['2.0~2.25', '2.5', '2.5~2.75', '3.0', '3.0~3.25']:
    subset = df[df['ou_category'] == ou_cat]
    if len(subset) > 0:
        avg_goals = subset['total_goals'].mean()
        print(f"  {ou_cat:15s}: {len(subset):5d}场, 均{avg_goals:.2f}球")

# 分析SP值分布
print("\n" + "=" * 70)
print("SP值分布分析")
print("=" * 70)

df['sp'] = pd.to_numeric(df['sp'], errors='coerce')
sp_valid = df[df['sp'].notna() & (df['sp'] > 0)]
print(f"有SP值的比赛: {len(sp_valid)}场")

if len(sp_valid) > 0:
    print(f"\nSP值统计:")
    print(f"  均值: {sp_valid['sp'].mean():.2f}")
    print(f"  中位数: {sp_valid['sp'].median():.2f}")
    print(f"  最小值: {sp_valid['sp'].min():.2f}")
    print(f"  最大值: {sp_valid['sp'].max():.2f}")
    
    # 分析不同进球数的SP值
    print(f"\n不同进球数的SP值:")
    for goals in range(0, 7):
        subset = sp_valid[sp_valid['total_goals'] == goals]
        if len(subset) > 0:
            avg_sp = subset['sp'].mean()
            median_sp = subset['sp'].median()
            print(f"  {goals}球: {len(subset):5d}场, 平均SP={avg_sp:.2f}, 中位SP={median_sp:.2f}")

# 分析当前模型使用的盘口和大小球线
print("\n" + "=" * 70)
print("当前模型使用的盘口和大小球线")
print("=" * 70)

import sys
sys.path.insert(0, '/coding/live')
from features_over import TARGET_HANDICAPS_OVER, MAX_OU_LINE_OVER
from features_under import TARGET_HANDICAPS_UNDER, MAX_OU_LINE_UNDER

print(f"\nOVER模型:")
print(f"  目标盘口: {TARGET_HANDICAPS_OVER}")
print(f"  大小球线上限: {MAX_OU_LINE_OVER}")

print(f"\nUNDER模型:")
print(f"  目标盘口: {TARGET_HANDICAPS_UNDER}")
print(f"  大小球线上限: {MAX_OU_LINE_UNDER}")

print("\n" + "=" * 70)
print("分析完成")
print("=" * 70)
