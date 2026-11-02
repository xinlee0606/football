import pymysql
from config import DB_CONFIG

db = pymysql.connect(**DB_CONFIG)
cursor = db.cursor()

print("=== 总记录数 ===")
cursor.execute("SELECT COUNT(*) FROM recent_form")
total_records = cursor.fetchone()[0]
print(f"recent_form表总记录数: {total_records}")

# 每场比赛有2条记录（主客队各一条）
cursor.execute("SELECT COUNT(DISTINCT match_id) FROM recent_form")
total_matches = cursor.fetchone()[0]
print(f"涉及的比赛场次数: {total_matches}")

cursor.execute("SELECT COUNT(*) FROM `match`")
match_table_total = cursor.fetchone()[0]
print(f"match表总比赛数: {match_table_total}")

print("\n=== 各项技术统计的覆盖情况 ===")
stats = [
    ('shots', '射门'),
    ('shots_on_target', '射正'),
    ('shots_off_target', '射偏'),
    ('shots_blocked', '被封堵射门'),
    ('possession', '控球率'),
    ('half_possession', '半场控球率'),
    ('corners', '角球'),
    ('half_corners', '半场角球'),
    ('fouls', '犯规'),
    ('yellow_cards', '黄牌'),
    ('offsides', '越位'),
    ('passes', '传球'),
    ('pass_success_rate', '传球成功率'),
    ('attacks', '进攻'),
    ('dangerous_attacks', '危险进攻'),
    ('free_kicks', '任意球'),
]

for col, name in stats:
    cursor.execute(f"""
        SELECT COUNT(*) FROM recent_form 
        WHERE {col} IS NOT NULL AND {col} != ''
    """)
    count = cursor.fetchone()[0]
    pct = count / total_records * 100 if total_records > 0 else 0
    print(f"{name}: {count} 条记录 ({pct:.1f}%)")

print("\n=== 按比赛日期统计最近比赛的覆盖情况 ===")
cursor.execute("""
    SELECT 
        SUBSTRING(match_id, 1, 8) as date_prefix,
        COUNT(DISTINCT match_id) as match_count,
        SUM(CASE WHEN shots IS NOT NULL THEN 1 ELSE 0 END) as has_shots,
        SUM(CASE WHEN possession IS NOT NULL THEN 1 ELSE 0 END) as has_possession
    FROM recent_form
    WHERE match_id LIKE '2026%'
    GROUP BY SUBSTRING(match_id, 1, 8)
    ORDER BY date_prefix DESC
    LIMIT 10
""")
rows = cursor.fetchall()
print(f"{'日期':<12} {'比赛数':<8} {'有射门':<10} {'有控球率':<10}")
for row in rows:
    date_prefix = row[0]
    match_count = row[1]
    has_shots = row[2]
    has_possession = row[3]
    # 每场比赛2条记录
    match_count_real = match_count // 2
    has_shots_real = has_shots // 2
    has_possession_real = has_possession // 2
    shots_pct = has_shots_real / match_count_real * 100 if match_count_real > 0 else 0
    possession_pct = has_possession_real / match_count_real * 100 if match_count_real > 0 else 0
    print(f"{date_prefix:<12} {match_count_real:<8} {has_shots_real}({shots_pct:.0f}%){'':<4} {has_possession_real}({possession_pct:.0f}%)")

db.close()
