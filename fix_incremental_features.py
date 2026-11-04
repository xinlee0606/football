"""Fix incremental_features.py to add h_/a_ prefixes and combo features"""

filename = '/coding/live/incremental_features.py'
print(f'Fixing {filename}...')

with open(filename, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the merge section
old_merge = """        # 合并所有特征（不含技术统计、排名、初始盘口、页面盘口）
        feat_row = {**af, **ouf, **ef, **h2hf, **home_recent, **away_recent, **league_stats}"""

new_merge = """        # Add h_/a_ prefixes to recent form features
        home_recent_prefixed = {f'h_{k}': v for k, v in home_recent.items()}
        away_recent_prefixed = {f'a_{k}': v for k, v in away_recent.items()}
        
        # Compute combination features
        combo = {}
        h_sot = home_recent.get('rf_sot_mean')
        h_shots = home_recent.get('rf_shots_mean')
        a_sot = away_recent.get('rf_sot_mean')
        a_shots = away_recent.get('rf_shots_mean')
        if h_sot is not None and h_shots and h_shots > 0:
            h_eff = h_sot / h_shots
        else:
            h_eff = None
        if a_sot is not None and a_shots and a_shots > 0:
            a_eff = a_sot / a_shots
        else:
            a_eff = None
        if h_eff is not None and a_eff is not None:
            combo['attack_eff_diff'] = h_eff - a_eff
        
        h_poss = home_recent.get('rf_poss_mean')
        a_poss = away_recent.get('rf_poss_mean')
        if h_poss is not None and a_poss is not None:
            combo['possession_diff'] = h_poss - a_poss
        
        h_corners = home_recent.get('rf_corners_mean')
        a_corners = away_recent.get('rf_corners_mean')
        if h_corners is not None and a_corners is not None:
            combo['corners_diff'] = h_corners - a_corners
        
        h_fouls = home_recent.get('rf_fouls_mean')
        a_fouls = away_recent.get('rf_fouls_mean')
        if h_fouls is not None and a_fouls is not None:
            combo['fouls_diff'] = h_fouls - a_fouls
        
        h_yellow = home_recent.get('rf_yellow_mean')
        a_yellow = away_recent.get('rf_yellow_mean')
        if h_yellow is not None and a_yellow is not None:
            combo['yellow_diff'] = h_yellow - a_yellow
        
        h_win_rate = home_recent.get('recent_win_rate')
        a_win_rate = away_recent.get('recent_win_rate')
        league_h_wr = league_stats.get('league_home_win_rate')
        league_a_wr = league_stats.get('league_away_win_rate')
        if h_win_rate is not None and league_h_wr is not None:
            combo['form_momentum_h'] = h_win_rate - league_h_wr
        if a_win_rate is not None and league_a_wr is not None:
            combo['form_momentum_a'] = a_win_rate - league_a_wr
        
        # 合并所有特征（不含技术统计、排名、初始盘口、页面盘口）
        feat_row = {**af, **ouf, **ef, **h2hf, **home_recent_prefixed, **away_recent_prefixed, **league_stats, **combo}"""

if old_merge in content:
    content = content.replace(old_merge, new_merge)
    print('✓ Replaced merge section with h_/a_ prefixes and combo features')
else:
    print('⚠ Could not find exact merge section, trying flexible match...')
    import re
    pattern = r'        # 合并所有特征.*?feat_row = \{\*\*af, \*\*ouf, \*\*ef, \*\*h2hf, \*\*home_recent, \*\*away_recent, \*\*league_stats\}'
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, new_merge, content, flags=re.DOTALL)
        print('✓ Replaced merge section using regex')
    else:
        print('✗ Could not find merge section')

with open(filename, 'w', encoding='utf-8') as f:
    f.write(content)

print(f'✓ Fixed {filename}')
