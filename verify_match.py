"""验证1场比赛的完整数据"""
import pymysql
from config import DB_CONFIG

conn = pymysql.connect(**DB_CONFIG)
c = conn.cursor()
mid = '2026072301'

print('=' * 60)
print(f'比赛 {mid} 完整数据验证')
print('=' * 60)

# 1. match表
c.execute("SELECT match_id, league, home_team, away_team, match_time FROM `match` WHERE match_id=%s", (mid,))
r = c.fetchone()
print(f'\n[match] 1场比赛:')
print(f'  {r}')

# 2. league_standings
c.execute("SELECT COUNT(*) FROM league_standings WHERE match_id=%s", (mid,))
cnt = c.fetchone()[0]
print(f'\n[league_standings] 积分排名: {cnt} 条 (期望>=4)')
c.execute("SELECT team_type, scope, team_name, ranking, points FROM league_standings WHERE match_id=%s LIMIT 4", (mid,))
for x in c.fetchall():
    print(f'  {x}')

# 3. head_to_head
c.execute("SELECT COUNT(*) FROM head_to_head WHERE match_id=%s", (mid,))
cnt = c.fetchone()[0]
print(f'\n[head_to_head] 对赛往绩: {cnt} 条 (期望>=3)')
c.execute("SELECT match_date, home_team, score, away_team, corners, initial_home_odds, initial_handicap, initial_away_odds FROM head_to_head WHERE match_id=%s LIMIT 3", (mid,))
for x in c.fetchall():
    print(f'  {x}')

# 4. recent_form
c.execute("SELECT COUNT(*) FROM recent_form WHERE match_id=%s AND is_current_match=0", (mid,))
cnt = c.fetchone()[0]
print(f'\n[recent_form] 近期战绩: {cnt} 条 (期望=20, 主客各10)')
c.execute("SELECT team_type, team_name, COUNT(*) as cnt FROM recent_form WHERE match_id=%s AND is_current_match=0 GROUP BY team_type, team_name", (mid,))
for x in c.fetchall():
    print(f'  {x[0]} | {x[1]} | {x[2]}场')

# 技术统计验证
c.execute("SELECT COUNT(*) FROM recent_form WHERE match_id=%s AND is_current_match=0 AND corners IS NOT NULL", (mid,))
tech_cnt = c.fetchone()[0]
print(f'  含技术统计: {tech_cnt}/{cnt} 条')

c.execute("SELECT team_name, home_team, away_team, score, corners, shots, shots_on_target, attacks, dangerous_attacks FROM recent_form WHERE match_id=%s AND is_current_match=0 AND corners IS NOT NULL LIMIT 3", (mid,))
print(f'  示例(含技术统计):')
for x in c.fetchall():
    print(f'    关注={x[0]} | {x[1]} vs {x[2]} | 比分{x[3]} | 角球={x[4]} 射门={x[5]} 射正={x[6]} 进攻={x[7]} 危险进攻={x[8]}')

# 5. asian_odds
c.execute("SELECT COUNT(*) FROM asian_odds WHERE match_id=%s", (mid,))
cnt = c.fetchone()[0]
print(f'\n[asian_odds] 亚盘: {cnt} 条 (期望>=3)')
c.execute("SELECT company, initial_handicap, initial_home_odds, initial_away_odds, current_handicap, current_home_odds, current_away_odds FROM asian_odds WHERE match_id=%s LIMIT 3", (mid,))
for x in c.fetchall():
    print(f'  {x}')

# 6. over_under_odds
c.execute("SELECT COUNT(*) FROM over_under_odds WHERE match_id=%s", (mid,))
cnt = c.fetchone()[0]
print(f'\n[over_under_odds] 大小球: {cnt} 条 (期望>=3)')
c.execute("SELECT company, initial_line, initial_over_odds, initial_under_odds, current_line, current_over_odds, current_under_odds FROM over_under_odds WHERE match_id=%s LIMIT 3", (mid,))
for x in c.fetchall():
    print(f'  {x}')

# 7. season_stats
c.execute("SELECT COUNT(*) FROM season_stats WHERE match_id=%s", (mid,))
cnt = c.fetchone()[0]
print(f'\n[season_stats] 赛季统计: {cnt} 条')

print('\n' + '=' * 60)
print('验证完成!')
print('=' * 60)

c.close()
conn.close()
