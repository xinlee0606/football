import pymysql
from config import DB_CONFIG

db = pymysql.connect(**DB_CONFIG)
cursor = db.cursor()

print("=== 查找20261020或20261019的布拉加比赛 ===")
cursor.execute("""
    SELECT match_id, single_match, home_team, away_team, score, match_time
    FROM `match`
    WHERE (home_team LIKE '%布拉加%' OR away_team LIKE '%布拉加%')
    AND match_id LIKE '202610%'
    ORDER BY match_id DESC
""")
rows = cursor.fetchall()
for row in rows:
    print(row)

print("\n=== 查找20261020或20261019的吉尔维森特/吉维森特比赛 ===")
cursor.execute("""
    SELECT match_id, single_match, home_team, away_team, score, match_time
    FROM `match`
    WHERE (home_team LIKE '%吉尔维森特%' OR away_team LIKE '%吉尔维森特%' OR home_team LIKE '%吉维森特%' OR away_team LIKE '%吉维森特%')
    AND match_id LIKE '202610%'
    ORDER BY match_id DESC
""")
rows = cursor.fetchall()
for row in rows:
    print(row)

print("\n=== 查看20261020所有比赛 ===")
cursor.execute("""
    SELECT match_id, single_match, home_team, away_team, score
    FROM `match`
    WHERE match_id LIKE '20261020%'
    ORDER BY single_match
""")
rows = cursor.fetchall()
for row in rows:
    print(row)

db.close()
