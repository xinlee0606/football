import pymysql
from config import DB_CONFIG
db = pymysql.connect(**DB_CONFIG)
c = db.cursor()

# 1. recent_form rows
c.execute("SELECT id, team_type, team_name, is_current_match, shots, corners, possession, shots_on_target, attacks, dangerous_attacks, yellow_cards FROM recent_form WHERE match_id = '2024112310' ORDER BY team_type, is_current_match, id")
cols = [d[0] for d in c.description]
rows = c.fetchall()
print(f"recent_form rows: {len(rows)}")
for r in rows:
    d = dict(zip(cols, r))
    print(f"  id={d['id']}, team={d['team_type']}, is_curr={d['is_current_match']}, shots={d['shots']}, corners={d['corners']}, possession={d['possession']}, team_name={d['team_name']}")

# 2. match table
c.execute("SELECT match_id, home_team, away_team, detail_url, analysis_url FROM `match` WHERE match_id = '2024112310'")
row = c.fetchone()
if row:
    print(f"\nmatch: {row[0]} {row[1]} vs {row[2]}")
    print(f"  detail_url: {row[3]}")
    print(f"  analysis_url: {row[4]}")

# 3. season_stats
c.execute("SELECT COUNT(*) FROM season_stats WHERE match_id = '2024112310'")
print(f"\nseason_stats rows: {c.fetchone()[0]}")

db.close()
