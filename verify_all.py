"""验证1场比赛的完整数据流程 - 按数据库表关联关系"""
from config import DB_CONFIG
import pymysql

conn = pymysql.connect(**DB_CONFIG)
c = conn.cursor()
mid = 2026072301

print("=" * 60)
print(f"比赛ID: {mid} 完整数据验证")
print("=" * 60)

# [1] match表
c.execute("SELECT match_id, league, home_team, away_team, match_time FROM `match` WHERE match_id=%s", (mid,))
r = c.fetchone()
print(f"\n[1] match表 (1场比赛):")
print(f"    {r}")

# [2] asian_odds表 - 大概16个机构盘口, 测试需要3条数据
c.execute("SELECT COUNT(*) FROM asian_odds WHERE match_id=%s", (mid,))
cnt = c.fetchone()[0]
c.execute("SELECT company, initial_home_odds, initial_handicap, initial_away_odds FROM asian_odds WHERE match_id=%s LIMIT 3", (mid,))
rows = c.fetchall()
status = "✓" if cnt >= 3 else "✗"
print(f"\n[2] asian_odds表: {cnt}条 {status} (需>=3)")
for r in rows:
    print(f"    {r}")

# [3] over_under_odds表 - 大概16个机构大小球, 测试需要3条数据
c.execute("SELECT COUNT(*) FROM over_under_odds WHERE match_id=%s", (mid,))
cnt = c.fetchone()[0]
c.execute("SELECT company, initial_line, initial_over_odds, initial_under_odds FROM over_under_odds WHERE match_id=%s LIMIT 3", (mid,))
rows = c.fetchall()
status = "✓" if cnt >= 3 else "✗"
print(f"\n[3] over_under_odds表: {cnt}条 {status} (需>=3)")
for r in rows:
    print(f"    {r}")

# [4] league_standings表 - 4条积分数据
c.execute("SELECT COUNT(*) FROM league_standings WHERE match_id=%s", (mid,))
cnt = c.fetchone()[0]
c.execute("SELECT team_type, team_name, scope, played, points, ranking FROM league_standings WHERE match_id=%s LIMIT 4", (mid,))
rows = c.fetchall()
status = "✓" if cnt >= 4 else "✗"
print(f"\n[4] league_standings表: {cnt}条 {status} (需>=4)")
for r in rows:
    print(f"    {r}")

# [5] head_to_head表 - 大概10场历史对战, 测试需要3条数据
c.execute("SELECT COUNT(*) FROM head_to_head WHERE match_id=%s", (mid,))
cnt = c.fetchone()[0]
c.execute("SELECT match_date, home_team, score, away_team, corners, initial_home_odds, initial_handicap, initial_away_odds FROM head_to_head WHERE match_id=%s LIMIT 3", (mid,))
rows = c.fetchall()
status = "✓" if cnt >= 3 else "✗"
print(f"\n[5] head_to_head表: {cnt}条 {status} (需>=3)")
for r in rows:
    print(f"    {r}")

# [6] recent_form表 - 主客队各10场历史战绩(20条), 测试主客队各需要3条数据
c.execute("SELECT COUNT(*) FROM recent_form WHERE match_id=%s AND is_current_match=0 AND team_type='home'", (mid,))
hcnt = c.fetchone()[0]
c.execute("SELECT COUNT(*) FROM recent_form WHERE match_id=%s AND is_current_match=0 AND team_type='away'", (mid,))
acnt = c.fetchone()[0]
status = "✓" if hcnt >= 3 and acnt >= 3 else "✗"
print(f"\n[6] recent_form表(近期战绩): 主队{hcnt}条 + 客队{acnt}条 {status} (各需>=3)")

c.execute("SELECT team_type, team_name, match_date, score FROM recent_form WHERE match_id=%s AND is_current_match=0 AND team_type='home' LIMIT 3", (mid,))
rows = c.fetchall()
print(f"    主队样本:")
for r in rows:
    print(f"      {r}")

c.execute("SELECT team_type, team_name, match_date, score FROM recent_form WHERE match_id=%s AND is_current_match=0 AND team_type='away' LIMIT 3", (mid,))
rows = c.fetchall()
print(f"    客队样本:")
for r in rows:
    print(f"      {r}")

# 技术统计 (is_current_match=1)
c.execute("SELECT team_type, team_name, home_team, away_team, corners FROM recent_form WHERE match_id=%s AND is_current_match=1", (mid,))
rows = c.fetchall()
status = "✓" if len(rows) >= 2 else "✗"
print(f"\n[6b] recent_form表(技术统计): {len(rows)}条 {status} (需2条:主+客)")
for r in rows:
    print(f"    {r}")

# [7] season_stats表 (额外)
c.execute("SELECT COUNT(*) FROM season_stats WHERE match_id=%s", (mid,))
cnt = c.fetchone()[0]
print(f"\n[7] season_stats表: {cnt}条")

# 汇总
print("\n" + "=" * 60)
print("验证汇总:")
print(f"  match表:            1条 ✓")
print(f"  asian_odds表:       {cnt}条 (需>=3)")
c.execute("SELECT COUNT(*) FROM asian_odds WHERE match_id=%s", (mid,))
print(f"  asian_odds表:       {c.fetchone()[0]}条")
c.execute("SELECT COUNT(*) FROM over_under_odds WHERE match_id=%s", (mid,))
print(f"  over_under_odds表:  {c.fetchone()[0]}条")
c.execute("SELECT COUNT(*) FROM league_standings WHERE match_id=%s", (mid,))
print(f"  league_standings表: {c.fetchone()[0]}条")
c.execute("SELECT COUNT(*) FROM head_to_head WHERE match_id=%s", (mid,))
print(f"  head_to_head表:     {c.fetchone()[0]}条")
print(f"  recent_form(战绩):  主{hcnt}+客{acnt}条")
print(f"  recent_form(统计):  {len(rows)}条")
print(f"  season_stats表:     {cnt}条")
print("=" * 60)

c.close()
conn.close()
