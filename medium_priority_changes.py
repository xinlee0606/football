"""
Medium priority feature engineering improvements:
1. League target encoding: add league_goal_strength feature
2. Time decay weights for recent form: exponential decay
"""
import re

def add_league_target_encoding(filename):
    """Add league target encoding features to _league_stats_features_fast"""
    print(f'\n=== Adding league target encoding to {filename} ===')
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the keys list in _league_stats_features_fast and add new features
    # Look for the pattern: keys = ['league_n', ..., 'league_home_unbeaten_rate']
    old_keys_pattern = r"(keys = \['league_n', 'league_home_win_rate', 'league_draw_rate', 'league_away_win_rate',\s+'league_avg_goals', 'league_btts_rate', 'league_over_25_rate',\s+'league_home_unbeaten_rate'\])"
    
    new_keys = "keys = ['league_n', 'league_home_win_rate', 'league_draw_rate', 'league_away_win_rate',\n            'league_avg_goals', 'league_btts_rate', 'league_over_25_rate',\n            'league_home_unbeaten_rate', 'league_goal_strength']"
    
    content = re.sub(old_keys_pattern, new_keys, content)
    
    # Add league_goal_strength computation before "return feats"
    # Find the return statement in _league_stats_features_fast
    old_return = r"(        feats\['league_home_unbeaten_rate'\] = home_unbeaten / n\n\n    return feats)"
    
    new_return = """        feats['league_home_unbeaten_rate'] = home_unbeaten / n
        # League goal strength: how many goals per game vs baseline (2.7)
        league_avg = _mean(goals_list)
        if league_avg is not None:
            feats['league_goal_strength'] = league_avg / 2.7  # baseline ~2.7 goals/game

    return feats"""
    
    content = re.sub(old_return, new_return, content)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'✓ League target encoding added to {filename}')


def add_time_decay_weights(filename, recent_func_name='_recent_form_features_fast'):
    """Add time decay weights to recent form features"""
    print(f'\n=== Adding time decay weights to {filename} ===')
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the _recent_form_features_fast function and modify it
    # The key change: instead of simple counting, use weighted counting with exponential decay
    
    # First, add TIME_DECAY_LAMBDA constant at the top of the file (after imports)
    if 'TIME_DECAY_LAMBDA' not in content:
        # Find the first def or import line to add the constant before it
        insert_pattern = r"(def is_target_handicap)"
        insert_text = """# Time decay parameter for recent form weighting
TIME_DECAY_LAMBDA = 0.08  # Exponential decay rate: w_i = exp(-lambda * (n-1-i))


\\1"""
        content = re.sub(insert_pattern, insert_text, content, count=1)
    
    # Now modify the _recent_form_features_fast function
    # Replace the simple counting loop with weighted counting
    
    # Find the function body
    old_func_start = r'''(def _recent_form_features_fast\(team_name, team_data, is_home=True, league_avgs=None\):
    """快速版近期战绩特征\(使用预加载数据\)"""
    keys = \['recent_n', 'recent_win_rate', 'recent_draw_rate',)'''
    
    # Replace the entire function with a new version that includes time decay
    old_func_pattern = r'def _recent_form_features_fast\(team_name, team_data, is_home=True, league_avgs=None\):.*?return feats'
    
    new_func = '''def _recent_form_features_fast(team_name, team_data, is_home=True, league_avgs=None):
    """快速版近期战绩特征(使用预加载数据, 含时间衰减权重)"""
    keys = ['recent_n', 'recent_win_rate', 'recent_draw_rate',
            'recent_goals_for_mean', 'recent_goals_against_mean',
            'recent_clean_sheet_rate', 'recent_btts_rate',
            'recent_over_25_rate', 'recent_unbeaten_rate',
            'rf_shots_mean', 'rf_sot_mean', 'rf_corners_mean',
            'rf_danger_mean', 'rf_poss_mean', 'rf_fouls_mean', 'rf_yellow_mean']
    feats = {k: None for k in keys}

    n = len(team_data)
    if n == 0:
        return feats

    # Time decay weights: most recent match gets weight 1.0
    # w_i = exp(-lambda * (n-1-i)) where i is the index (0=oldest, n-1=newest)
    weights = [np.exp(-TIME_DECAY_LAMBDA * (n - 1 - i)) for i in range(n)]
    total_weight = sum(weights)

    wins, draws, losses = 0.0, 0.0, 0.0
    goals_for_w, goals_against_w = 0.0, 0.0
    goals_for_count, goals_against_count = 0, 0
    clean_sheets_w, btts_count_w, over_25_count_w = 0.0, 0.0, 0.0

    for idx, match in enumerate(team_data):
        hg, ag = parse_score(match['score'])
        if hg is None:
            continue

        w = weights[idx]

        if is_home:
            gf, ga = hg, ag
        else:
            gf, ga = ag, hg

        goals_for_w += gf * w
        goals_against_w += ga * w
        goals_for_count += 1
        goals_against_count += 1

        if gf > ga:
            wins += w
        elif gf == ga:
            draws += w
        else:
            losses += w

        if ga == 0:
            clean_sheets_w += w
        if gf > 0 and ga > 0:
            btts_count_w += w
        if gf + ga > 2.5:
            over_25_count_w += w

    if goals_for_count > 0:
        feats['recent_n'] = n
        feats['recent_win_rate'] = wins / total_weight
        feats['recent_draw_rate'] = draws / total_weight
        feats['recent_goals_for_mean'] = goals_for_w / total_weight
        feats['recent_goals_against_mean'] = goals_against_w / total_weight
        feats['recent_clean_sheet_rate'] = clean_sheets_w / total_weight
        feats['recent_btts_rate'] = btts_count_w / total_weight
        feats['recent_over_25_rate'] = over_25_count_w / total_weight
        feats['recent_unbeaten_rate'] = (wins + draws) / total_weight

    # 7项技术指标(联赛均值填充缺失值, 使用时间衰减加权)
    if league_avgs is None:
        league_avgs = {}
    
    # Weighted tech stats
    shots_w, shots_count = 0.0, 0
    sot_w, sot_count = 0.0, 0
    corners_w, corners_count = 0.0, 0
    danger_w, danger_count = 0.0, 0
    poss_w, poss_count = 0.0, 0
    fouls_w, fouls_count = 0.0, 0
    yellow_w, yellow_count = 0.0, 0
    
    for idx, m in enumerate(team_data):
        w = weights[idx]
        if m.get('shots') is not None:
            shots_w += m['shots'] * w
            shots_count += 1
        if m.get('shots_on_target') is not None:
            sot_w += m['shots_on_target'] * w
            sot_count += 1
        if m.get('corners') is not None:
            corners_w += m['corners'] * w
            corners_count += 1
        if m.get('dangerous_attacks') is not None:
            danger_w += m['dangerous_attacks'] * w
            danger_count += 1
        poss_val = to_float(m.get('possession'))
        if poss_val is not None:
            poss_w += poss_val * w
            poss_count += 1
        if m.get('fouls') is not None:
            fouls_w += m['fouls'] * w
            fouls_count += 1
        if m.get('yellow_cards') is not None:
            yellow_w += m['yellow_cards'] * w
            yellow_count += 1
    
    feats['rf_shots_mean'] = (shots_w / shots_count) if shots_count > 0 else league_avgs.get('shots')
    feats['rf_sot_mean'] = (sot_w / sot_count) if sot_count > 0 else league_avgs.get('shots_on_target')
    feats['rf_corners_mean'] = (corners_w / corners_count) if corners_count > 0 else league_avgs.get('corners')
    feats['rf_danger_mean'] = (danger_w / danger_count) if danger_count > 0 else league_avgs.get('dangerous_attacks')
    feats['rf_poss_mean'] = (poss_w / poss_count) if poss_count > 0 else league_avgs.get('possession')
    feats['rf_fouls_mean'] = (fouls_w / fouls_count) if fouls_count > 0 else league_avgs.get('fouls')
    feats['rf_yellow_mean'] = (yellow_w / yellow_count) if yellow_count > 0 else league_avgs.get('yellow_cards')

    return feats'''
    
    content = re.sub(old_func_pattern, new_func, content, flags=re.DOTALL)
    
    # Add numpy import if not present
    if 'import numpy as np' not in content:
        content = content.replace('import pandas as pd', 'import pandas as pd\nimport numpy as np')
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'✓ Time decay weights added to {filename}')


# Apply changes to both files
for f in ['/coding/live/features_over.py', '/coding/live/features_under.py']:
    add_league_target_encoding(f)
    add_time_decay_weights(f)

print('\n=== All medium priority changes applied ===')
