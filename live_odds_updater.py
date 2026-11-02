# live_odds_updater.py - 临场盘口数据更新模块
# 功能: 在预测脚本执行时，获取最新的盘口数据（亚盘、大小球、欧赔）
# 数据源: 与 scraper.py 相同，但只更新 current_* 字段
# 用法:
#   from live_odds_updater import update_live_odds
#   updated_count = update_live_odds()  # 更新今日未开赛比赛的盘口数据

import os
import re
import sys
import time
import requests
import pymysql
from datetime import datetime, timedelta
from config import DB_CONFIG

# ==================== 配置 ====================
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}
REQUEST_DELAY = 0.5  # 请求间隔(秒)

# 数据源URL模板
EURO_DATA_URL = 'http://1x2d.titan007.com/{site_match_id}.js'


def get_today_unfinished_matches(cursor):
    """获取今日未开赛的比赛列表"""
    today = datetime.now().strftime('%Y%m%d')
    
    cursor.execute("""
        SELECT match_id, site_match_id, match_time 
        FROM `match`
        WHERE match_id LIKE %s
          AND site_match_id IS NOT NULL 
          AND site_match_id != ''
          AND (score IS NULL OR score = '' OR score NOT REGEXP '^[0-9]+[-:][0-9]+$')
        ORDER BY match_id
    """, (f'{today}%',))
    
    return cursor.fetchall()


def fetch_euro_data(site_match_id):
    """获取单场比赛的最新欧赔数据"""
    url = EURO_DATA_URL.format(site_match_id=site_match_id)
    headers = {
        'User-Agent': REQUEST_HEADERS['User-Agent'],
        'Referer': f'http://1x2.titan007.com/oddslist/{site_match_id}.htm'
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return []
        text = resp.content.decode('utf-8', errors='replace')
        if text.startswith('\ufeff'):
            text = text[1:]
        return parse_euro_js(text)
    except Exception as e:
        print(f"    ✗ 欧赔请求失败: {e}")
        return []


def parse_euro_js(js_text):
    """解析欧赔JS文件中的game数组"""
    match = re.search(r'var\s+game\s*=\s*Array\((.*?)\);', js_text, re.DOTALL)
    if not match:
        return []

    array_content = match.group(1)
    companies = []
    for m in re.finditer(r'"([^"]*)"', array_content):
        raw = m.group(1)
        parts = raw.split('|')
        if len(parts) < 14:
            continue

        company_name = parts[2].strip()
        company_cn = parts[21].strip() if len(parts) > 21 else ''
        display_name = company_cn if company_cn else company_name
        display_name = display_name.replace('封', '').strip()
        if not display_name:
            continue

        cur_win = parts[10].strip()
        cur_draw = parts[11].strip()
        cur_loss = parts[12].strip()

        companies.append({
            'company': display_name,
            'current_win_odds': cur_win if cur_win and cur_win != '-' else None,
            'current_draw_odds': cur_draw if cur_draw and cur_draw != '-' else None,
            'current_loss_odds': cur_loss if cur_loss and cur_loss != '-' else None,
        })

    return companies


def update_euro_odds(cursor, match_id, site_match_id):
    """更新欧赔临场数据"""
    companies = fetch_euro_data(site_match_id)
    if not companies:
        return False
    
    # 更新数据库中的current_*字段
    updated = 0
    for c in companies:
        cursor.execute("""
            UPDATE european_odds
            SET current_win_odds = %s,
                current_draw_odds = %s,
                current_loss_odds = %s
            WHERE match_id = %s AND company = %s
        """, (
            c['current_win_odds'],
            c['current_draw_odds'],
            c['current_loss_odds'],
            match_id,
            c['company']
        ))
        updated += cursor.rowcount
    
    return updated > 0


def update_live_odds():
    """
    主函数：更新今日未开赛比赛的临场盘口数据
    返回：更新的比赛场次数
    """
    print("\n" + "="*60)
    print("开始更新临场盘口数据...")
    print("="*60)
    
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # 获取今日未开赛比赛
    matches = get_today_unfinished_matches(cursor)
    print(f"今日未开赛比赛: {len(matches)} 场")
    
    if not matches:
        print("无需更新")
        cursor.close()
        conn.close()
        return 0
    
    updated_count = 0
    failed_count = 0
    
    for i, (match_id, site_match_id, match_time) in enumerate(matches):
        print(f"[{i+1}/{len(matches)}] match_id={match_id} time={match_time}", end=' ')
        
        try:
            # 更新欧赔数据
            success = update_euro_odds(cursor, match_id, site_match_id)
            
            if success:
                conn.commit()
                print("✓ 欧赔已更新")
                updated_count += 1
            else:
                print("✗ 无数据")
                failed_count += 1
            
            # 请求间隔
            if REQUEST_DELAY > 0:
                time.sleep(REQUEST_DELAY)
                
        except Exception as e:
            print(f"✗ 更新失败: {e}")
            failed_count += 1
            conn.rollback()
    
    print("\n" + "="*60)
    print(f"临场盘口更新完成!")
    print(f"  成功: {updated_count} 场")
    print(f"  失败: {failed_count} 场")
    print("="*60 + "\n")
    
    cursor.close()
    conn.close()
    
    return updated_count


if __name__ == '__main__':
    # 直接运行时执行更新
    update_live_odds()
