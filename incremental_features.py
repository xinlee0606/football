# incremental_features.py - 增量特征更新模块
# 功能: 实现特征的增量更新，避免每次预测都重建全部特征
#       缓存历史数据结构，只重新查询盘口数据
#
# 用法:
#   from incremental_features import load_features_incremental
#   df = load_features_incremental(conn, rebuild_base=False)  # 增量更新（快）
#   df = load_features_incremental(conn, rebuild_base=True)   # 重建基础缓存（慢）
#
# 缓存文件:
#   features_over_base_cache.csv  - 基础缓存（完整特征，每周更新）
#   features_over_cache.csv       - 完整缓存（每日更新）

import os
import pickle
import time
from datetime import datetime
from collections import defaultdict

import pandas as pd
import pymysql

from config import DB_CONFIG
from features_over import (
    build_features_over, load_cache_over, save_cache_over,
    CACHE_FILE_OVER, META_COLS_OVER,
    asian_features_upside, over_under_features_upside, euro_features_upside,
    _h2h_features_upside_fast, _recent_form_features_upside_fast,
    _league_stats_features_upside_fast, _tech_stats_features_fast,
    _league_detailed_features, _h2h_initial_handicap_features, _page_features,
    is_cup_league_over, handicap_to_num, to_float
)

# 基础数据缓存文件（pickle格式，保存中间数据结构）
BASE_DATA_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'features_over_base_data.pkl')

# 基础缓存有效期（秒）：7天
BASE_CACHE_TTL = 7 * 24 * 3600


def is_base_cache_valid():
    """检查基础缓存是否有效（未过期）"""
    if not os.path.exists(BASE_DATA_CACHE):
        return False
    
    file_age = time.time() - os.path.getmtime(BASE_DATA_CACHE)
    return file_age < BASE_CACHE_TTL


def load_base_data_cache():
    """加载基础数据缓存（历史数据结构）"""
    if not os.path.exists(BASE_DATA_CACHE):
        return None
    
    try:
        with open(BASE_DATA_CACHE, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ 加载基础数据缓存: {len(data['matches'])}场比赛")
        return data
    except Exception as e:
        print(f"✗ 加载基础数据缓存失败: {e}")
        return None


def save_base_data_cache(data):
    """保存基础数据缓存（历史数据结构）"""
    try:
        with open(BASE_DATA_CACHE, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ 保存基础数据缓存: {len(data['matches'])}场比赛")
        return True
    except Exception as e:
        print(f"✗ 保存基础数据缓存失败: {e}")
        return False


def load_historical_data(conn):
    """加载历史数据（不经常变化的部分）
    
    Returns:
        dict: 包含所有历史数据结构的字典
    """
    cursor = conn.cursor()
    
    # 1. 加载所有比赛
    cursor.execute("""
        SELECT match_id, single_match, league, home_team, away_team, 
               match_time, score
        FROM `match` 
        ORDER BY match_id
    """)
    matches = cursor.fetchall()
    cols = ['match_id', 'single_match', 'league', 'home_team', 'away_team',
            'match_time', 'score']
    df = pd.DataFrame(matches, columns=cols)
    
    # 2. 计算标签
    from features_over import is_finished, norm_score, OVER_SET
    def _calc_in_pool(s):
        if not s or not is_finished(s):
            return None
        return 1 if norm_score(s) in OVER_SET else 0
    df['in_pool'] = df['score'].apply(_calc_in_pool)
    
    # 3. 加载SP值
    cursor.execute("""
        SELECT match_id, sp_value FROM result_sp
        WHERE match_id IS NOT NULL AND match_id <> ''
    """)
    sp_map = {mid: float(sp) if sp is not None else None
              for mid, sp in cursor.fetchall()}
    df['sp'] = df['match_id'].map(sp_map)
    
    # 4. 加载历史比赛（用于近期战绩/联赛统计）
    cursor.execute("""
        SELECT match_id, league, home_team, away_team, score
        FROM `match` 
        WHERE score IS NOT NULL AND score != ''
        ORDER BY match_id
    """)
    hist_matches = cursor.fetchall()
    history_df = pd.DataFrame(hist_matches, columns=['match_id', 'league', 'home_team', 'away_team', 'score'])
    
    # 5. 加载recent_form技术统计数据
    cursor.execute("""
        SELECT team_name, shots, shots_on_target, possession, corners,
               dangerous_attacks, attacks, fouls, pass_success_rate
        FROM recent_form
        WHERE is_current_match = 0
    """)
    recent_form_records = cursor.fetchall()
    team_tech_data = defaultdict(list)
    for tname, shots, sot, poss, corners, da, attacks, fouls, psr in recent_form_records:
        if tname:
            team_tech_data[tname].append({
                'shots': shots, 'shots_on_target': sot, 'possession': poss,
                'corners': corners, 'dangerous_attacks': da, 'attacks': attacks,
                'fouls': fouls, 'pass_success_rate': psr
            })
    
    # 6. 加载历史交锋数据
    cursor.execute("""
        SELECT home_team, away_team, score, half_score, corners,
               initial_handicap, initial_home_odds, initial_away_odds,
               handicap_result, over_under_result
        FROM head_to_head
        ORDER BY match_id
    """)
    h2h_records = cursor.fetchall()
    h2h_df = pd.DataFrame(h2h_records, columns=['home_team', 'away_team', 'score', 'half_score', 'corners',
                                                  'initial_handicap', 'initial_home_odds', 'initial_away_odds',
                                                  'handicap_result', 'over_under_result'])
    
    # 7. 加载league_standings详细数据
    cursor.execute("""
        SELECT match_id, team_type, scope, played, won, drawn, lost,
               goals_for, goals_against, goal_diff, points, ranking, win_rate
        FROM league_standings
    """)
    standings_map = defaultdict(dict)
    for mid, tt, scope, played, won, drawn, lost, gf, ga, gd, pts, rank, wr in cursor.fetchall():
        standings_map[mid][(tt, scope)] = {
            'played': played, 'won': won, 'drawn': drawn, 'lost': lost,
            'goals_for': gf, 'goals_against': ga, 'goal_diff': gd,
            'points': pts, 'ranking': rank, 'win_rate': wr
        }
    
    # 8. 加载match表页面盘口和排名数据
    cursor.execute("""
        SELECT match_id, page_handicap, page_goals, home_rank, away_rank
        FROM `match`
        WHERE match_id IS NOT NULL AND match_id <> ''
    """)
    page_map = {}
    for mid, ph, pg, hr, ar in cursor.fetchall():
        page_map[mid] = {
            'page_handicap': ph, 'page_goals': pg,
            'home_rank': hr, 'away_rank': ar
        }
    
    # 9. 构建索引
    print("⏳ 构建历史数据索引...")
    home_team_matches = defaultdict(list)
    away_team_matches = defaultdict(list)
    league_matches = defaultdict(list)
    h2h_matches = defaultdict(list)
    
    for _, row in history_df.iterrows():
        home = row['home_team']
        away = row['away_team']
        league = row['league']
        match_data = {
            'match_id': row['match_id'],
            'score': row['score'],
            'home_team': home,
            'away_team': away
        }
        home_team_matches[home].append(match_data)
        away_team_matches[away].append(match_data)
        league_matches[league].append(match_data)
    
    for _, row in h2h_df.iterrows():
        home = row['home_team']
        away = row['away_team']
        team_pair = tuple(sorted([home, away]))
        h2h_matches[team_pair].append({
            'home_team': home,
            'away_team': away,
            'score': row['score'],
            'half_score': row['half_score'],
            'corners': row['corners']
        })
    
    cursor.close()
    
    return {
        'matches': df,
        'history_df': history_df,
        'team_tech_data': dict(team_tech_data),
        'h2h_df': h2h_df,
        'standings_map': dict(standings_map),
        'page_map': page_map,
        'home_team_matches': dict(home_team_matches),
        'away_team_matches': dict(away_team_matches),
        'league_matches': dict(league_matches),
        'h2h_matches': dict(h2h_matches),
    }


def load_latest_odds(conn, match_ids):
    """加载最新盘口数据（每日变化的部分）
    
    Args:
        conn: 数据库连接
        match_ids: 比赛ID列表
    
    Returns:
        dict: 包含asian_map, ou_map, euro_map的字典
    """
    if not match_ids:
        return {'asian_map': {}, 'ou_map': {}, 'euro_map': {}}
    
    cursor = conn.cursor()
    placeholders = ','.join(['%s'] * len(match_ids))
    
    # 亚盘数据
    cursor.execute(f"""
        SELECT match_id, initial_handicap, current_handicap,
               initial_home_odds, initial_away_odds,
               current_home_odds, current_away_odds
        FROM asian_odds
        WHERE match_id IN ({placeholders})
    """, match_ids)
    
    asian_map = defaultdict(list)
    for mid, ih, ch, iho, iao, cho, cao in cursor.fetchall():
        asian_map[mid].append({
            'initial_handicap': ih, 'current_handicap': ch,
            'initial_home_odds': iho, 'initial_away_odds': iao,
            'current_home_odds': cho, 'current_away_odds': cao
        })
    
    # 大小球数据
    cursor.execute(f"""
        SELECT match_id, initial_line, current_line,
               initial_over_odds, initial_under_odds,
               current_over_odds, current_under_odds
        FROM over_under_odds
        WHERE match_id IN ({placeholders})
    """, match_ids)
    
    ou_map = defaultdict(list)
    for mid, il, cl, ioo, iuo, coo, cuo in cursor.fetchall():
        ou_map[mid].append({
            'initial_line': il, 'current_line': cl,
            'initial_over_odds': ioo, 'initial_under_odds': iuo,
            'current_over_odds': coo, 'current_under_odds': cuo
        })
    
    # 欧赔数据（SQL预聚合）
    cursor.execute(f"""
        SELECT match_id,
               AVG(initial_win_odds) as avg_iw, STDDEV_POP(initial_win_odds) as std_iw,
               AVG(initial_draw_odds) as avg_id, STDDEV_POP(initial_draw_odds) as std_id,
               AVG(initial_loss_odds) as avg_il, STDDEV_POP(initial_loss_odds) as std_il,
               AVG(current_win_odds) as avg_cw, STDDEV_POP(current_win_odds) as std_cw,
               AVG(current_draw_odds) as avg_cd, STDDEV_POP(current_draw_odds) as std_cd,
               AVG(current_loss_odds) as avg_cl, STDDEV_POP(current_loss_odds) as std_cl,
               COUNT(*) as cnt
        FROM european_odds
        WHERE match_id IN ({placeholders})
        GROUP BY match_id
    """, match_ids)
    
    euro_map = {}
    for row in cursor.fetchall():
        mid = row[0]
        euro_map[mid] = [{
            'initial_win_odds': row[1], 'initial_win_odds_std': row[2],
            'initial_draw_odds': row[3], 'initial_draw_odds_std': row[4],
            'initial_loss_odds': row[5], 'initial_loss_odds_std': row[6],
            'current_win_odds': row[7], 'current_win_odds_std': row[8],
            'current_draw_odds': row[9], 'current_draw_odds_std': row[10],
            'current_loss_odds': row[11], 'current_loss_odds_std': row[12],
            'count': row[13]
        }]
    
    cursor.close()
    
    return {
        'asian_map': dict(asian_map),
        'ou_map': dict(ou_map),
        'euro_map': euro_map
    }


def build_features_from_cached_data(base_data, odds_data):
    """从缓存的历史数据和最新盘口数据构建特征
    
    Args:
        base_data: 历史数据结构（来自load_historical_data或缓存）
        odds_data: 最新盘口数据（来自load_latest_odds）
    
    Returns:
        DataFrame: 完整特征表
    """
    df = base_data['matches'].copy()
    asian_map = odds_data['asian_map']
    ou_map = odds_data['ou_map']
    euro_map = odds_data['euro_map']
    
    home_team_matches = base_data['home_team_matches']
    away_team_matches = base_data['away_team_matches']
    league_matches = base_data['league_matches']
    h2h_matches = base_data['h2h_matches']
    team_tech_data = base_data['team_tech_data']
    h2h_df = base_data['h2h_df']
    standings_map = base_data['standings_map']
    page_map = base_data['page_map']
    
    print(f"⏳ 构建特征... (共{len(df)}场)")
    all_rows = []
    
    for idx, row in df.iterrows():
        mid = row['match_id']
        home = row['home_team']
        away = row['away_team']
        league = row['league']
        
        # 盘口特征（使用最新数据）
        af = asian_features_upside(asian_map.get(mid, []))
        ouf = over_under_features_upside(ou_map.get(mid, []))
        ef = euro_features_upside(euro_map.get(mid, []))
        
        # 历史特征（使用缓存数据）
        team_pair = tuple(sorted([home, away]))
        h2h_data = h2h_matches.get(team_pair, [])
        h2hf = _h2h_features_upside_fast(home, away, h2h_data)
        
        home_recent_data = home_team_matches.get(home, [])[-10:]
        home_recent = _recent_form_features_upside_fast(home, home_recent_data, is_home=True)
        away_recent_data = away_team_matches.get(away, [])[-10:]
        away_recent = _recent_form_features_upside_fast(away, away_recent_data, is_home=False)
        
        league_data = league_matches.get(league, [])
        league_stats = _league_stats_features_upside_fast(league, league_data)
        
        home_tech_data = team_tech_data.get(home, [])[-10:]
        home_tech = _tech_stats_features_fast(home_tech_data)
        away_tech_data = team_tech_data.get(away, [])[-10:]
        away_tech = _tech_stats_features_fast(away_tech_data)
        
        h_standings = _league_detailed_features(home, standings_map)
        a_standings = _league_detailed_features(away, standings_map)
        
        h2h_p1 = _h2h_initial_handicap_features(home, away, h2h_df)
        p2_feats = _page_features(mid, page_map)
        
        # 合并所有特征
        feat_row = {**af, **ouf, **ef, **h2hf, **home_recent, **away_recent, **league_stats,
                    **{f'h_{k}': v for k, v in home_tech.items()},
                    **{f'a_{k}': v for k, v in away_tech.items()},
                    **{f'h_{k}': v for k, v in h_standings.items()},
                    **{f'a_{k}': v for k, v in a_standings.items()},
                    **h2h_p1, **p2_feats}
        
        # 差值特征
        tech_diff_keys = ['tech_shots_avg', 'tech_sot_avg', 'tech_sot_rate',
                          'tech_poss_avg', 'tech_corners_avg', 'tech_da_avg',
                          'tech_attacks_avg', 'tech_fouls_avg', 'tech_da_rate', 'tech_pass_rate']
        for k in tech_diff_keys:
            h_key = f'h_{k}'
            a_key = f'a_{k}'
            if feat_row.get(h_key) is not None and feat_row.get(a_key) is not None:
                feat_row[f'{k}_diff'] = feat_row[h_key] - feat_row[a_key]
        
        ls_diff_keys = ['ls_gf_avg', 'ls_ga_avg', 'ls_ppg', 'ls_win_rate']
        for k in ls_diff_keys:
            h_key = f'h_{k}'
            a_key = f'a_{k}'
            if feat_row.get(h_key) is not None and feat_row.get(a_key) is not None:
                feat_row[f'{k}_diff'] = feat_row[h_key] - feat_row[a_key]
        
        if feat_row.get('h_ls_gf_avg') is not None and feat_row.get('a_ls_gf_avg') is not None:
            feat_row['ls_home_advantage'] = feat_row['h_ls_gf_avg'] - feat_row['a_ls_gf_avg']
        
        feat_row['is_cup'] = 1 if is_cup_league_over(league) else 0
        
        meta_row = row.to_dict()
        all_rows.append({**meta_row, **feat_row})
        
        if (idx + 1) % 1000 == 0:
            print(f"  已处理 {idx + 1}/{len(df)} 场")
    
    result_df = pd.DataFrame(all_rows)
    print(f"✓ 特征构建完成: {len(result_df)}场")
    return result_df


def load_features_incremental(conn=None, rebuild_base=False):
    """增量加载特征（推荐用于预测）
    
    Args:
        conn: 数据库连接（None时自动创建）
        rebuild_base: 是否重建基础缓存（默认False，使用增量模式）
    
    Returns:
        完整特征DataFrame
    """
    import pymysql
    
    own = conn is None
    if own:
        conn = pymysql.connect(**DB_CONFIG)
    
    try:
        if rebuild_base or not is_base_cache_valid():
            # 重建基础缓存（完整构建）
            print("⏳ 重建基础缓存（加载历史数据）...")
            base_data = load_historical_data(conn)
            save_base_data_cache(base_data)
            
            # 加载最新盘口
            match_ids = base_data['matches']['match_id'].tolist()
            odds_data = load_latest_odds(conn, match_ids)
            
            # 构建特征
            df = build_features_from_cached_data(base_data, odds_data)
            
            # 保存缓存
            if len(df) > 0:
                save_cache_over(df)
            
            return df
        else:
            # 增量更新模式
            print("⏳ 增量更新特征（加载缓存+最新盘口）...")
            base_data = load_base_data_cache()
            
            if base_data is None:
                # 缓存不存在，降级为完整构建
                print("⚠ 基础缓存不存在，降级为完整构建")
                base_data = load_historical_data(conn)
                save_base_data_cache(base_data)
            
            # 加载最新盘口（只查询这部分，很快）
            match_ids = base_data['matches']['match_id'].tolist()
            print("⏳ 查询最新盘口数据...")
            odds_data = load_latest_odds(conn, match_ids)
            
            # 使用缓存的历史数据 + 最新盘口构建特征
            df = build_features_from_cached_data(base_data, odds_data)
            
            # 保存完整缓存
            if len(df) > 0:
                save_cache_over(df)
            
            return df
    
    finally:
        if own:
            conn.close()


if __name__ == '__main__':
    conn = pymysql.connect(**DB_CONFIG)
    try:
        # 测试增量加载
        df = load_features_incremental(conn, rebuild_base=False)
        print(f"\n特征表形状: {df.shape}")
        print(f"样本数: {len(df)}")
    finally:
        conn.close()
