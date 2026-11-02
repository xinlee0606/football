"""检查数据库中用户提到的特征相关数据可用性 - 通过Python连接"""
import pymysql
import sys

try:
    conn = pymysql.connect(host='rm-0jlt516knd10vpy18ko.mysql.rds.aliyuncs.com', port=3306,
                           user='root', password='Xin7231013@@@', 
                           database='match_live', charset='utf8mb4')
    cursor = conn.cursor()
    
    # 1. 球队身份
    print("\n【1. 球队身份 - 球队数量统计】")
    cursor.execute("""
        SELECT 
            COUNT(DISTINCT home_team) as home_teams,
            COUNT(DISTINCT away_team) as away_teams
        FROM `match` WHERE score IS NOT NULL AND score != ''
    """)
    row = cursor.fetchone()
    print(f"  主队数: {row[0]}, 客队数: {row[1]}")
    
    # 2. 赛前排名
    print("\n【2. 赛前排名 - league_standings覆盖率】")
    cursor.execute("""
        SELECT 
            COUNT(DISTINCT match_id) as matches_with_ranking,
            COUNT(*) as total_records
        FROM league_standings
    """)
    row = cursor.fetchone()
    print(f"  有排名的比赛: {row[0]}, 总记录: {row[1]}")
    
    # 3. 交锋记录
    print("\n【3. 交锋记录 - head_to_head覆盖率】")
    cursor.execute("""
        SELECT 
            COUNT(DISTINCT match_id) as matches_with_h2h,
            COUNT(*) as total_records
        FROM head_to_head
    """)
    row = cursor.fetchone()
    print(f"  有交锋的比赛: {row[0]}, 总记录: {row[1]}")
    
    # 4. 近期状态
    print("\n【4. 近期状态 - recent_form覆盖率】")
    cursor.execute("""
        SELECT 
            COUNT(DISTINCT match_id) as matches_with_recent,
            COUNT(*) as total_records,
            SUM(CASE WHEN is_current_match = 0 THEN 1 ELSE 0 END) as history_records
        FROM recent_form
    """)
    row = cursor.fetchone()
    print(f"  有近绩的比赛: {row[0]}, 总记录: {row[1]}, 历史记录: {row[2]}")
    
    # 5. 盘口信息
    print("\n【5. 盘口信息 - asian_odds机构数量】")
    cursor.execute("""
        SELECT 
            COUNT(DISTINCT match_id) as matches_with_asian,
            COUNT(*) as total_records
        FROM asian_odds
    """)
    row = cursor.fetchone()
    print(f"  有亚盘的比赛: {row[0]}, 总记录: {row[1]}")
    # 计算平均每场机构数
    cursor.execute("""
        SELECT AVG(cnt) FROM (
            SELECT match_id, COUNT(*) as cnt
            FROM asian_odds
            GROUP BY match_id
        ) t
    """)
    avg_cnt = cursor.fetchone()
    print(f"  平均每场机构数: {avg_cnt[0]:.1f}")
    
    # 6. 赢盘记录
    print("\n【6. 赢盘记录 - handicap_result覆盖率】")
    cursor.execute("""
        SELECT 
            SUM(CASE WHEN handicap_result IN ('赢','输') THEN 1 ELSE 0 END) as decided_count,
            SUM(CASE WHEN handicap_result = '赢' THEN 1 ELSE 0 END) as win_count,
            SUM(CASE WHEN handicap_result = '输' THEN 1 ELSE 0 END) as lose_count
        FROM recent_form
        WHERE is_current_match = 0 AND handicap_result IS NOT NULL
    """)
    row = cursor.fetchone()
    print(f"  有结果: {row[0]}, 赢: {row[1]}, 输: {row[2]}")
    
    # 7. 技术统计
    print("\n【7. 技术统计 - 字段覆盖率】")
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN shots IS NOT NULL AND shots != '' THEN 1 ELSE 0 END) as shots_cnt,
            SUM(CASE WHEN shots_on_target IS NOT NULL THEN 1 ELSE 0 END) as sot_cnt,
            SUM(CASE WHEN possession IS NOT NULL AND possession != '' THEN 1 ELSE 0 END) as poss_cnt,
            SUM(CASE WHEN corners IS NOT NULL AND corners != '' THEN 1 ELSE 0 END) as corners_cnt,
            SUM(CASE WHEN dangerous_attacks IS NOT NULL THEN 1 ELSE 0 END) as da_cnt,
            SUM(CASE WHEN fouls IS NOT NULL THEN 1 ELSE 0 END) as fouls_cnt,
            SUM(CASE WHEN yellow_cards IS NOT NULL THEN 1 ELSE 0 END) as yellow_cnt
        FROM recent_form
        WHERE is_current_match = 0
    """)
    row = cursor.fetchone()
    total = row[0]
    print(f"  总记录: {total}")
    print(f"  射门: {row[1]} ({row[1]*100//total if total else 0}%)")
    print(f"  射正: {row[2]} ({row[2]*100//total if total else 0}%)")
    print(f"  控球: {row[3]} ({row[3]*100//total if total else 0}%)")
    print(f"  角球: {row[4]} ({row[4]*100//total if total else 0}%)")
    print(f"  危险进攻: {row[5]} ({row[5]*100//total if total else 0}%)")
    print(f"  犯规: {row[6]} ({row[6]*100//total if total else 0}%)")
    print(f"  黄牌: {row[7]} ({row[7]*100//total if total else 0}%)")
    
    # 8. 主客场效应
    print("\n【8. 主客场效应 - 中立场标记】")
    cursor.execute("""
        SELECT 
            COUNT(*) as total_matches,
            SUM(CASE WHEN home_team LIKE '%(中)%' OR home_team LIKE '%（中）%' THEN 1 ELSE 0 END) as neutral_count
        FROM `match`
        WHERE score IS NOT NULL AND score != ''
    """)
    row = cursor.fetchone()
    print(f"  总比赛: {row[0]}, 中立场: {row[1]}")
    
    # 9. 时间特征
    print("\n【9. 时间特征 - 赛季轮次信息】")
    cursor.execute("""
        SELECT 
            COUNT(*) as total_matches,
            COUNT(DISTINCT league) as leagues,
            MIN(match_time) as earliest,
            MAX(match_time) as latest
        FROM `match`
        WHERE score IS NOT NULL AND score != ''
    """)
    row = cursor.fetchone()
    print(f"  总比赛: {row[0]}, 联赛数: {row[1]}")
    print(f"  时间范围: {row[2]} ~ {row[3]}")
    
    # 10. season_stats
    print("\n【10. season_stats表数据量】")
    cursor.execute("""
        SELECT 
            COUNT(DISTINCT match_id) as matches,
            COUNT(*) as total_records
        FROM season_stats
    """)
    row = cursor.fetchone()
    print(f"  比赛数: {row[0]}, 总记录: {row[1]}")
    
    # 11. 交锋场次分布
    print("\n【11. 交锋记录场次分布】")
    cursor.execute("""
        SELECT 
            CASE 
                WHEN cnt >= 10 THEN '10+'
                WHEN cnt >= 5 THEN '5-9'
                WHEN cnt >= 3 THEN '3-4'
                WHEN cnt >= 1 THEN '1-2'
                ELSE '0'
            END as h2h_games,
            COUNT(*) as match_count
        FROM (
            SELECT match_id, COUNT(*) as cnt
            FROM head_to_head
            GROUP BY match_id
        ) t
        GROUP BY h2h_games
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}场: {row[1]}场比赛")
    
    conn.close()
    print("\n完成。")
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
