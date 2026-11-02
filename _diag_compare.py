import pymysql
from config import DB_CONFIG
db = pymysql.connect(**DB_CONFIG)
c = db.cursor()

cols_show = ['team_type', 'team_name', 'match_date', 'league', 'home_team', 'score', 'away_team',
             'shots', 'corners', 'possession', 'yellow_cards', 'is_current_match']

# 1. Match WITH recent form tech stats (2026101401 - scraped by daily scraper)
print("=== 2026101401 (daily scraper 抓取，有技术统计) ===")
c.execute("SELECT " + ",".join(cols_show) + " FROM recent_form WHERE match_id = '2026101401' ORDER BY team_type, is_current_match, id LIMIT 25")
cols = [d[0] for d in c.description]
rows = c.fetchall()
print(f"Rows: {len(rows)}")
for r in rows[:5]:
    d = dict(zip(cols, r))
    print(f"  {d['team_type']} curr={d['is_current_match']} | {d['match_date']} {d['league']} {d['home_team']} {d['score']} {d['away_team']} | shots={d['shots']} corners={d['corners']} poss={d['possession']}")
print("  ...")

# 2. Match WITHOUT recent form tech stats (2024112310 - history_scraper, 缺技术统计)
print()
print("=== 2024112310 (history_scraper 抓取，缺技术统计) ===")
c.execute("SELECT " + ",".join(cols_show) + " FROM recent_form WHERE match_id = '2024112310' ORDER BY team_type, is_current_match, id LIMIT 25")
rows = c.fetchall()
print(f"Rows: {len(rows)}")
for r in rows[:5]:
    d = dict(zip(cols, r))
    print(f"  {d['team_type']} curr={d['is_current_match']} | {d['match_date']} {d['league']} {d['home_team']} {d['score']} {d['away_team']} | shots={d['shots']} corners={d['corners']} poss={d['possession']}")
print("  ...")

# 3. Global stats
print()
print("=== 全局统计 ===")
c.execute("""SELECT 
    COUNT(DISTINCT match_id) as total_matches,
    COUNT(DISTINCT CASE WHEN shots IS NOT NULL THEN match_id END) as with_tech,
    COUNT(*) as total_rows,
    COUNT(CASE WHEN shots IS NOT NULL THEN 1 END) as rows_with_tech,
    COUNT(CASE WHEN is_current_match = 0 AND shots IS NULL THEN 1 END) as recent_form_missing_tech
    FROM recent_form""")
r = c.fetchone()
print(f"Total matches in recent_form: {r[0]}")
print(f"Matches with ANY tech stats: {r[1]}")
print(f"Total rows: {r[2]}, rows with tech data: {r[3]}")
print(f"Recent form rows missing tech (is_curr=0 AND shots=NULL): {r[4]}")

# 4. Coverage by date
print()
print("=== 按日期统计近期战绩技术统计覆盖率 ===")
c.execute("""SELECT LEFT(m.match_id, 8) as dt, 
    COUNT(DISTINCT m.match_id) as matches,
    COUNT(DISTINCT CASE WHEN rf.shots IS NOT NULL AND rf.is_current_match = 0 THEN m.match_id END) as with_rf_tech
    FROM `match` m
    LEFT JOIN recent_form rf ON m.match_id = rf.match_id
    GROUP BY LEFT(m.match_id, 8) ORDER BY dt""")
for row in c.fetchall():
    pct = f"{row[2]/row[1]*100:.0f}%" if row[1] > 0 else "0%"
    print(f"  {row[0]}: {row[2]}/{row[1]} matches ({pct})")

db.close()
