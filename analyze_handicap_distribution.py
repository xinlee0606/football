import pandas as pd
import numpy as np
import pymysql
import json

# 连接数据库
conn = pymysql.connect(
    host='rm-bp1x6y77p9z5w4b5h.mysql.rds.aliyuncs.com',
    user='football_user',
    password='Xin7231013@@@',
    database='football',
    charset='utf8mb4'
)

print("=" * 70)
print("数据库比赛数据分析 - 盘口与大小球线分布")
print("=" * 70)

# 查询所有已完赛比赛的数据
query = """
SELECT 
    match_id,
    league,
    home_team,
    away_team,
    score,
    ah_init_mean,
    ah_cur_mean,
    ou_init_mean,
    ou_cur_mean,
    sp
FROM matches 
WHERE score IS NOT NULL 
    AND score != '' 
    AND match_time < NOW()
ORDER BY match_time DESC
"""

print("\n[1] 加载数据...")
df = pd.read_sql(query, conn)
conn.close()

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

print(f"有效比赛数（有比分）: {len(df)}")

# 分析亚盘盘口分布
print("\n" + "=" * 70)
print("亚盘盘口分布分析")
print("=" * 70)

handicap_bins = [
    -float('inf'), -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, float('inf')
]
handicap_labels = [
    '受让球半+', '受让球半', '受让一球', '受让平/半', '受让半球', '受让平手',
    '平手', '让平手', '让平/半', '让半球', '让一球', '让球半', '让球半+'
]

df['ah_cur_bin'] = pd.cut(df['ah_cur_mean'], bins=handicap_bins, labels=handicap_labels)
ah_dist = df['ah_cur_bin'].value_counts().sort_index()
print("\n终盘亚盘分布:")
for bin_name, count in ah_dist.items():
    pct = count / len(df) * 100
    print(f"  {bin_name:12s}: {count:5d}场 ({pct:5.1f}%)")

# 分析大小球线分布
print("\n" + "=" * 70)
print("大小球线分布分析")
print("=" * 70)

ou_bins = [0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0, float('inf')]
ou_labels = ['1.5以下', '1.5-2', '2-2.25', '2.25-2.5', '2.5-2.75', '2.75-3', '3-3.25', '3.25-3.5', '3.5-4', '4以上']

df['ou_cur_bin'] = pd.cut(df['ou_cur_mean'], bins=ou_bins, labels=ou_labels)
ou_dist = df['ou_cur_bin'].value_counts().sort_index()
print("\n终盘大小球线分布:")
for bin_name, count in ou_dist.items():
    pct = count / len(df) * 100
    print(f"  {bin_name:12s}: {count:5d}场 ({pct:5.1f}%)")

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

for ah_label in ['受让平/半', '受让半球', '平手', '让平/半', '让半球']:
    subset = df[df['ah_cur_bin'] == ah_label]
    if len(subset) > 0:
        avg_goals = subset['total_goals'].mean()
        over25 = (subset['total_goals'] > 2.5).mean() * 100
        over225 = (subset['total_goals'] > 2.25).mean() * 100
        print(f"  {ah_label:12s}: {len(subset):5d}场, 均{avg_goals:.2f}球, >2.5球{over25:.1f}%, >2.25球{over225:.1f}%")

# 分析不同大小球线下的进球分布
print("\n" + "=" * 70)
print("不同大小球线下的实际进球分布")
print("=" * 70)

for ou_label in ['2-2.25', '2.25-2.5', '2.5-2.75', '2.75-3', '3-3.25']:
    subset = df[df['ou_cur_bin'] == ou_label]
    if len(subset) > 0:
        avg_goals = subset['total_goals'].mean()
        over_line = (subset['total_goals'] > float(ou_label.split('-')[0])).mean() * 100
        under_line = (subset['total_goals'] < float(ou_label.split('-')[-1])).mean() * 100
        print(f"  {ou_label:12s}: {len(subset):5d}场, 均{avg_goals:.2f}球, 大球{over_line:.1f}%, 小球{under_line:.1f}%")

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
    
    sp_bins = [0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, float('inf')]
    sp_labels = ['<1.5', '1.5-2', '2-2.5', '2.5-3', '3-4', '4-5', '5-10', '10+']
    sp_valid['sp_bin'] = pd.cut(sp_valid['sp'], bins=sp_bins, labels=sp_labels)
    sp_dist = sp_valid['sp_bin'].value_counts().sort_index()
    print(f"\nSP值分布:")
    for bin_name, count in sp_dist.items():
        pct = count / len(sp_valid) * 100
        print(f"  {bin_name:8s}: {count:5d}场 ({pct:5.1f}%)")

# 分析不同进球数的SP值
print("\n" + "=" * 70)
print("不同进球数的SP值分布")
print("=" * 70)

for goals in range(0, 7):
    subset = sp_valid[sp_valid['total_goals'] == goals]
    if len(subset) > 0:
        avg_sp = subset['sp'].mean()
        median_sp = subset['sp'].median()
        print(f"  {goals}球: {len(subset):5d}场, 平均SP={avg_sp:.2f}, 中位SP={median_sp:.2f}")

print("\n" + "=" * 70)
print("分析完成")
print("=" * 70)
