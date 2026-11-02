
import pymysql
from config import DB_CONFIG

conn = pymysql.connect(**DB_CONFIG)
cur = conn.cursor()

# Check today's matches
cur.execute("SELECT COUNT(*) FROM `match` WHERE match_id LIKE '20261031%'")
count = cur.fetchone()[0]
print(f'今天比赛数：{count}')

if count > 0:
    cur.execute("SELECT match_id, home_team, away_team, match_time FROM `match` WHERE match_id LIKE '20261031%' LIMIT 5")
    for r in cur.fetchall():
        print(r)

# Check match status distribution
cur.execute("SELECT match_status, COUNT(*) FROM `match` WHERE match_id LIKE '20261031%' GROUP BY match_status")
print('\n比赛状态分布:')
for r in cur.fetchall():
    print(f'  {r[0]}: {r[1]}场')

conn.close()
