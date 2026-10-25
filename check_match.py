"""验证最新爬取结果"""
import pymysql
from config import DB_CONFIG

MATCH_ID = '2026072301'

conn = pymysql.connect(**DB_CONFIG)
c = conn.cursor()

# 检查近期战绩 team_name 和 odds
c.execute("SELECT id, team_type, team_name, home_team, away_team, match_date, initial_home_odds, initial_handicap, initial_away_odds FROM recent_form WHERE match_id = %s AND is_current_match = 0 ORDER BY id LIMIT 5", (MATCH_ID,))
rows = c.fetchall()
print(f"=== 近期战绩 (主队前5条) ===")
for r in rows:
    print(f"  {r[1]} | tn={r[2]} | h={r[3]} | a={r[4]} | date={r[5]} | odds={r[6]}/{r[7]}/{r[8]}")

# 检查技术统计
c.execute("SELECT id, team_type, team_name, home_team, away_team, corners, shots, shots_on_target, attacks, possession FROM recent_form WHERE match_id = %s AND is_current_match = 1", (MATCH_ID,))
rows = c.fetchall()
print(f"\n=== 技术统计 (is_current_match=1) ===")
for r in rows:
    print(f"  {r[1]} | tn={r[2]} | h={r[3]} | a={r[4]} | corners={r[5]} shots={r[6]} on_target={r[7]} attacks={r[8]} poss={r[9]}")
if not rows:
    print("  ✗ 未找到技术统计数据")

# 检查各表记录数
for table in ['league_standings', 'head_to_head', 'season_stats', 'asian_odds', 'over_under_odds']:
    c.execute(f"SELECT COUNT(*) FROM {table} WHERE match_id = %s", (MATCH_ID,))
    print(f"  {table}: {c.fetchone()[0]} 条")

c.close()
conn.close()
