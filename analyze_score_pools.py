import pandas as pd
import numpy as np
from collections import Counter

print("=" * 70)
print("比分池分析 - OVER大球模型 vs UNDER小球模型")
print("=" * 70)

# 加载缓存
print("\n[1] 加载数据...")
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
df['sp'] = pd.to_numeric(df['sp'], errors='coerce')

print(f"有效比赛数: {len(df)}")

# 生成比分的函数
def format_score(h, a):
    return f"{h}-{a}"

# 分析所有比分的分布
print("\n" + "=" * 70)
print("所有比分分布（按出现频率排序）")
print("=" * 70)

score_counts = Counter()
score_sp_sum = {}
score_sp_count = {}

for _, row in df.iterrows():
    h, a = int(row['home_goals']), int(row['away_goals'])
    score = format_score(h, a)
    score_counts[score] += 1
    
    if pd.notna(row['sp']) and row['sp'] > 0:
        if score not in score_sp_sum:
            score_sp_sum[score] = 0
            score_sp_count[score] = 0
        score_sp_sum[score] += row['sp']
        score_sp_count[score] += 1

total_matches = len(df)
print(f"\n{'比分':<10} {'场次':<8} {'命中率':<10} {'平均SP':<10} {'SP*命中率':<12}")
print("-" * 70)

score_stats = []
for score, count in score_counts.most_common(50):
    hit_rate = count / total_matches * 100
    avg_sp = score_sp_sum.get(score, 0) / score_sp_count.get(score, 1) if score_sp_count.get(score, 0) > 0 else 0
    sp_x_hit = avg_sp * hit_rate / 100
    score_stats.append({
        'score': score,
        'count': count,
        'hit_rate': hit_rate,
        'avg_sp': avg_sp,
        'sp_x_hit': sp_x_hit
    })
    print(f"{score:<10} {count:<8} {hit_rate:>6.2f}%    {avg_sp:>8.2f}    {sp_x_hit:>8.2f}")

# 分析OVER模型（3、4、5球）
print("\n" + "=" * 70)
print("OVER 大球模型分析（3、4、5球）")
print("=" * 70)

over_scores = df[df['total_goals'].isin([3, 4, 5])].copy()
print(f"\n3-5球比赛总数: {len(over_scores)}场 ({len(over_scores)/len(df)*100:.1f}%)")

over_score_counts = Counter()
over_score_sp_sum = {}
over_score_sp_count = {}

for _, row in over_scores.iterrows():
    h, a = int(row['home_goals']), int(row['away_goals'])
    score = format_score(h, a)
    over_score_counts[score] += 1
    
    if pd.notna(row['sp']) and row['sp'] > 0:
        if score not in over_score_sp_sum:
            over_score_sp_sum[score] = 0
            over_score_sp_count[score] = 0
        over_score_sp_sum[score] += row['sp']
        over_score_sp_count[score] += 1

total_over = len(over_scores)
print(f"\n{'比分':<10} {'场次':<8} {'命中率':<10} {'平均SP':<10} {'SP*命中率':<12} {'累计命中率':<12}")
print("-" * 80)

over_stats = []
cumulative_hit = 0
for score, count in over_score_counts.most_common(30):
    hit_rate = count / total_over * 100
    cumulative_hit += hit_rate
    avg_sp = over_score_sp_sum.get(score, 0) / over_score_sp_count.get(score, 1) if over_score_sp_count.get(score, 0) > 0 else 0
    sp_x_hit = avg_sp * hit_rate / 100
    over_stats.append({
        'score': score,
        'count': count,
        'hit_rate': hit_rate,
        'cumulative_hit': cumulative_hit,
        'avg_sp': avg_sp,
        'sp_x_hit': sp_x_hit
    })
    print(f"{score:<10} {count:<8} {hit_rate:>6.2f}%    {avg_sp:>8.2f}    {sp_x_hit:>8.2f}    {cumulative_hit:>8.2f}%")

# 选择最优9个比分（综合考虑命中率和SP值）
print("\n" + "=" * 70)
print("OVER 模型最优9个比分选择")
print("=" * 70)

# 按 SP*命中率 排序
over_sorted_by_sp_x_hit = sorted(over_stats, key=lambda x: x['sp_x_hit'], reverse=True)
print("\n按 SP*命中率 排序（前15）:")
for i, stat in enumerate(over_sorted_by_sp_x_hit[:15]):
    print(f"  {i+1}. {stat['score']:<8} 命中率={stat['hit_rate']:.2f}% SP={stat['avg_sp']:.2f} SP*命中率={stat['sp_x_hit']:.2f}")

# 选择前9个
over_top9 = over_sorted_by_sp_x_hit[:9]
over_top9_scores = [s['score'] for s in over_top9]
over_top9_total_hit = sum(s['hit_rate'] for s in over_top9)
over_top9_avg_sp = sum(s['avg_sp'] for s in over_top9) / len(over_top9)

print(f"\n✓ OVER 模型推荐9个比分: {over_top9_scores}")
print(f"  总命中率: {over_top9_total_hit:.2f}%")
print(f"  平均SP: {over_top9_avg_sp:.2f}")

# 分析UNDER模型（0、1、2、3球）
print("\n" + "=" * 70)
print("UNDER 小球模型分析（0、1、2、3球）")
print("=" * 70)

under_scores = df[df['total_goals'].isin([0, 1, 2, 3])].copy()
print(f"\n0-3球比赛总数: {len(under_scores)}场 ({len(under_scores)/len(df)*100:.1f}%)")

under_score_counts = Counter()
under_score_sp_sum = {}
under_score_sp_count = {}

for _, row in under_scores.iterrows():
    h, a = int(row['home_goals']), int(row['away_goals'])
    score = format_score(h, a)
    under_score_counts[score] += 1
    
    if pd.notna(row['sp']) and row['sp'] > 0:
        if score not in under_score_sp_sum:
            under_score_sp_sum[score] = 0
            under_score_sp_count[score] = 0
        under_score_sp_sum[score] += row['sp']
        under_score_sp_count[score] += 1

total_under = len(under_scores)
print(f"\n{'比分':<10} {'场次':<8} {'命中率':<10} {'平均SP':<10} {'SP*命中率':<12} {'累计命中率':<12}")
print("-" * 80)

under_stats = []
cumulative_hit = 0
for score, count in under_score_counts.most_common(30):
    hit_rate = count / total_under * 100
    cumulative_hit += hit_rate
    avg_sp = under_score_sp_sum.get(score, 0) / under_score_sp_count.get(score, 1) if under_score_sp_count.get(score, 0) > 0 else 0
    sp_x_hit = avg_sp * hit_rate / 100
    under_stats.append({
        'score': score,
        'count': count,
        'hit_rate': hit_rate,
        'cumulative_hit': cumulative_hit,
        'avg_sp': avg_sp,
        'sp_x_hit': sp_x_hit
    })
    print(f"{score:<10} {count:<8} {hit_rate:>6.2f}%    {avg_sp:>8.2f}    {sp_x_hit:>8.2f}    {cumulative_hit:>8.2f}%")

# 选择最优9个比分
print("\n" + "=" * 70)
print("UNDER 模型最优9个比分选择")
print("=" * 70)

# 按 SP*命中率 排序
under_sorted_by_sp_x_hit = sorted(under_stats, key=lambda x: x['sp_x_hit'], reverse=True)
print("\n按 SP*命中率 排序（前15）:")
for i, stat in enumerate(under_sorted_by_sp_x_hit[:15]):
    print(f"  {i+1}. {stat['score']:<8} 命中率={stat['hit_rate']:.2f}% SP={stat['avg_sp']:.2f} SP*命中率={stat['sp_x_hit']:.2f}")

# 选择前9个
under_top9 = under_sorted_by_sp_x_hit[:9]
under_top9_scores = [s['score'] for s in under_top9]
under_top9_total_hit = sum(s['hit_rate'] for s in under_top9)
under_top9_avg_sp = sum(s['avg_sp'] for s in under_top9) / len(under_top9)

print(f"\n✓ UNDER 模型推荐9个比分: {under_top9_scores}")
print(f"  总命中率: {under_top9_total_hit:.2f}%")
print(f"  平均SP: {under_top9_avg_sp:.2f}")

# 对比分析
print("\n" + "=" * 70)
print("对比分析")
print("=" * 70)

print(f"\nOVER 模型（3-5球）:")
print(f"  推荐比分: {over_top9_scores}")
print(f"  总命中率: {over_top9_total_hit:.2f}%")
print(f"  平均SP: {over_top9_avg_sp:.2f}")
print(f"  预期EV: {over_top9_avg_sp * over_top9_total_hit / 100 - 1:.2f}")

print(f"\nUNDER 模型（0-3球）:")
print(f"  推荐比分: {under_top9_scores}")
print(f"  总命中率: {under_top9_total_hit:.2f}%")
print(f"  平均SP: {under_top9_avg_sp:.2f}")
print(f"  预期EV: {under_top9_avg_sp * under_top9_total_hit / 100 - 1:.2f}")

print("\n" + "=" * 70)
print("分析完成")
print("=" * 70)
