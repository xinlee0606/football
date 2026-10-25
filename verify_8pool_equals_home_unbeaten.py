#!/usr/bin/env python3
# verify_8pool_equals_home_unbeaten.py
# 验证：在平手/平半/半球盘口下，8 比分池是否真的等于"主队不败"
import pymysql
from config import DB_CONFIG

def main():
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # 1. 获取所有已完场且有亚盘数据的比赛
    cursor.execute("""
        SELECT m.match_id, m.score
        FROM `match` m
        JOIN asian_odds ao ON m.match_id = ao.match_id
        WHERE m.status LIKE '%完%' 
          AND m.score IS NOT NULL 
          AND TRIM(m.score) <> ''
          AND ao.initial_handicap IN ('半球', '平手/半球', '平手', '受平手', '平/半')
          AND ao.current_handicap IN ('半球', '平手/半球', '平手', '受平手', '平/半')
        GROUP BY m.match_id, m.score
        HAVING COUNT(*) > 12  -- 条件 1：机构数>12
    """)
    
    rows = cursor.fetchall()
    total = len(rows)
    
    # 2. 统计各类结果
    home_win = 0
    draw = 0
    away_win = 0
    in_8pool = 0
    home_unbeaten = 0  # 主胜 + 平局
    
    pool_8 = {'1-0', '2-0', '2-1', '3-1', '3-2', '0-0', '1-1', '2-2'}
    
    for match_id, score in rows:
        # 解析比分
        parts = score.split('-')
        if len(parts) != 2:
            continue
        try:
            h_score, a_score = int(parts[0]), int(parts[1])
        except:
            continue
        
        # 判断结果
        if h_score > a_score:
            home_win += 1
            is_home_unbeaten = True
        elif h_score == a_score:
            draw += 1
            is_home_unbeaten = True
        else:
            away_win += 1
            is_home_unbeaten = False
        
        if is_home_unbeaten:
            home_unbeaten += 1
        
        # 是否在 8 池
        if score in pool_8:
            in_8pool += 1
    
    # 3. 输出统计
    print("=" * 70)
    print(f"样本总数：{total} 场（平手/平半/半球，机构数>12，已完场）")
    print("=" * 70)
    print(f"\n【赛果分布】")
    print(f"  主胜：{home_win} 场 ({home_win/total*100:.1f}%)")
    print(f"  平局：{draw} 场 ({draw/total*100:.1f}%)")
    print(f"  客胜：{away_win} 场 ({away_win/total*100:.1f}%)")
    print(f"  主队不败（胜 + 平）：{home_unbeaten} 场 ({home_unbeaten/total*100:.1f}%)")
    
    print(f"\n【8 池命中率】")
    print(f"  8 池命中：{in_8pool} 场 ({in_8pool/total*100:.1f}%)")
    print(f"  8 池未命中：{total - in_8pool} 场 ({(total-in_8pool)/total*100:.1f}%)")
    
    print(f"\n【关键对比】")
    print(f"  主队不败 vs 8 池命中：")
    print(f"    主队不败：{home_unbeaten/total*100:.1f}%")
    print(f"    8 池命中：{in_8pool/total*100:.1f}%")
    print(f"    差值：{(home_unbeaten - in_8pool)/total*100:.1f}%")
    
    # 4. 找出"主队不败但不在 8 池"的比分（即高比分主胜/平局）
    score_dist = {}
    cursor.execute("""
        SELECT m.score
        FROM `match` m
        JOIN asian_odds ao ON m.match_id = ao.match_id
        WHERE m.status LIKE '%完%' 
          AND m.score IS NOT NULL 
          AND TRIM(m.score) <> ''
          AND ao.initial_handicap IN ('半球', '平手/半球', '平手', '受平手', '平/半')
          AND ao.current_handicap IN ('半球', '平手/半球', '平手', '受平手', '平/半')
        GROUP BY m.match_id, m.score
        HAVING COUNT(*) > 12
    """)
    
    for (score,) in cursor.fetchall():
        parts = score.split('-')
        if len(parts) != 2:
            continue
        try:
            h, a = int(parts[0]), int(parts[1])
        except:
            continue
        
        if h >= a and score not in pool_8:  # 主队不败但不在 8 池
            if score not in score_dist:
                score_dist[score] = 0
            score_dist[score] += 1
    
    if score_dist:
        print(f"\n【主队不败但不在 8 池的比分】（即 8 池遗漏的高比分）")
        for score, cnt in sorted(score_dist.items(), key=lambda x: -x[1])[:15]:
            print(f"  {score}: {cnt} 场")
    
    cursor.close()
    conn.close()


if __name__ == '__main__':
    main()
