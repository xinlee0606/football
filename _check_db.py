import pymysql
conn = pymysql.connect(host='rm-0jlt516knd10vpy18ko.mysql.rds.aliyuncs.com', port=3306, user='root', password='Xin7231013@@@', database='match_live')
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM recent_form WHERE is_current_match=0 AND shots IS NULL AND detail_url IS NOT NULL AND detail_url!=''")
total = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(DISTINCT detail_url) FROM recent_form WHERE is_current_match=0 AND shots IS NULL AND detail_url IS NOT NULL AND detail_url!=''")
unique = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM recent_form WHERE is_current_match=0 AND shots IS NOT NULL")
done = cursor.fetchone()[0]
print(f'Need backfill: {total} rows, {unique} unique URLs')
print(f'Already done: {done} rows')
conn.close()
