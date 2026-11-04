import pymysql
conn = pymysql.connect(host='localhost', user='root', password='Xin7231013@@@', database='football')
cur = conn.cursor()
cur.execute("SELECT match_id, home_team, away_team, match_time FROM matches WHERE match_time > NOW() AND match_time < DATE_ADD(NOW(), INTERVAL 3 DAY) ORDER BY match_time LIMIT 5")
rows = cur.fetchall()
print(f'未来3天比赛: {len(rows)}场')
for r in rows:
    print(f'  {r[0]}: {r[1]} vs {r[2]} ({r[3]})')
conn.close()
