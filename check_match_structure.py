import pymysql
import sys
sys.path.insert(0, '/coding/live')
from config import DB_CONFIG

conn = pymysql.connect(**DB_CONFIG)
cur = conn.cursor()
cur.execute('DESCRIBE `match`')
cols = cur.fetchall()
print('Match table columns:')
for c in cols:
    print(f'  {c[0]}: {c[1]}')
conn.close()
