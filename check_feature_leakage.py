"""
短期改进3: 检查并修复潜在的特征泄露
确保 league_goal_strength 不使用未来数据
"""

# Check if league_goal_strength calculation uses only historical data
print('检查 league_goal_strength 特征是否存在数据泄露...')

files_to_check = [
    '/coding/live/features_over.py',
    '/coding/live/features_under.py'
]

for filepath in files_to_check:
    print(f'\n检查 {filepath}:')
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the league_goal_strength calculation
    import re
    pattern = r"feats\['league_goal_strength'\]\s*=\s*([^\n]+)"
    matches = re.findall(pattern, content)
    
    if matches:
        for match in matches:
            print(f'  当前计算: {match}')
            if 'league_avg' in match and '/ 2.7' in match:
                print(f'  ✓ 使用联赛平均进球数，看起来合理')
                print(f'  ⚠ 但需要确保: league_avg 只使用预测时点之前的数据')
    else:
        print(f'  ⚠ 未找到 league_goal_strength 计算')

print('\n' + '='*80)
print('建议: 检查 _league_stats_features_fast 函数中的数据筛选逻辑')
print('确保只使用 match_time < 当前预测比赛时间 的数据')
print('='*80)
