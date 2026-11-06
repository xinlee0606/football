import re

def update_feature_columns_func(filename, func_name, sample_recent_func, sample_league_func, is_over=True):
    """Update feature_columns function to include h_/a_ prefixed and combo features"""
    print(f'\n=== Updating {func_name} in {filename} ===')
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the function and replace it
    old_pattern = rf'def {func_name}\(\):.*?return all_keys'
    
    if is_over:
        new_func = f'''def {func_name}():
    """返回OVER模型使用的特征列名列表(不含元信息列)"""
    # 从示例构建中提取所有特征键
    sample_asian = asian_features([])
    sample_ou = over_under_features([])
    sample_euro = euro_features([])
    # 构造空DataFrame但包含必要的列，避免KeyError
    empty_hist_df = pd.DataFrame(columns=['match_id', 'home_team', 'away_team', 'score', 'league'])
    sample_h2h = h2h_features('', '', empty_hist_df)
    sample_recent = {sample_recent_func}('', [], is_home=True)
    sample_league = {sample_league_func}('', [])

    # 基础特征(盘口+交锋+联赛)
    base_keys = list(sample_asian.keys()) + list(sample_ou.keys()) + list(sample_euro.keys()) + \\
                list(sample_h2h.keys()) + list(sample_league.keys())
    
    # 近期战绩特征(添加h_/a_前缀区分主客场)
    recent_keys = list(sample_recent.keys())
    h_recent_keys = [f'h_{{k}}' for k in recent_keys]
    a_recent_keys = [f'a_{{k}}' for k in recent_keys]
    
    # 组合特征
    combo_keys = ['attack_eff_diff', 'possession_diff', 'corners_diff', 
                  'fouls_diff', 'yellow_diff', 'form_momentum_h', 'form_momentum_a']
    
    all_keys = base_keys + h_recent_keys + a_recent_keys + combo_keys
    return all_keys'''
    else:
        new_func = f'''def {func_name}():
    """返回UNDER模型使用的特征列名列表(不含元信息列)"""
    # 从示例构建中提取所有特征键
    sample_asian = asian_features_upside([])
    sample_ou = over_under_features_upside([])
    sample_euro = euro_features_upside([])
    # 构造空DataFrame但包含必要的列，避免KeyError
    empty_hist_df = pd.DataFrame(columns=['match_id', 'home_team', 'away_team', 'score', 'league'])
    sample_h2h = h2h_features_upside('', '', empty_hist_df)
    sample_recent = {sample_recent_func}('', [], is_home=True)
    sample_league = {sample_league_func}('', [])

    # 基础特征(盘口+交锋+联赛)
    base_keys = list(sample_asian.keys()) + list(sample_ou.keys()) + list(sample_euro.keys()) + \\
                list(sample_h2h.keys()) + list(sample_league.keys())
    
    # 近期战绩特征(添加h_/a_前缀区分主客场)
    recent_keys = list(sample_recent.keys())
    h_recent_keys = [f'h_{{k}}' for k in recent_keys]
    a_recent_keys = [f'a_{{k}}' for k in recent_keys]
    
    # 组合特征
    combo_keys = ['attack_eff_diff', 'possession_diff', 'corners_diff', 
                  'fouls_diff', 'yellow_diff', 'form_momentum_h', 'form_momentum_a']
    
    all_keys = base_keys + h_recent_keys + a_recent_keys + combo_keys
    return all_keys'''
    
    new_content = re.sub(old_pattern, new_func, content, flags=re.DOTALL)
    
    if new_content != content:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'✓ {func_name} updated')
        return True
    else:
        print(f'⚠ No changes to {func_name}')
        return False

# Update both functions
update_feature_columns_func('/coding/live/features_over.py', 'feature_columns_over', 
                            '_recent_form_features_fast', '_league_stats_features_fast', is_over=True)
update_feature_columns_func('/coding/live/features_under.py', 'feature_columns_under',
                            '_recent_form_features_upside_fast', '_league_stats_features_upside_fast', is_over=False)

print('\n=== Done ===')
