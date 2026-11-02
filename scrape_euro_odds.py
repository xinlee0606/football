# scrape_euro_odds.py - 欧赔数据补录脚本
# 功能: 从 1x2d.titan007.com 批量爬取数据库中已有比赛的欧赔数据
# 数据源: http://1x2d.titan007.com/{site_match_id}.js (JS变量 game 数组)
# 用法:
#   python scrape_euro_odds.py              # 补录所有有site_match_id的比赛
#   python scrape_euro_odds.py --limit 100  # 只处理前100场
#   python scrape_euro_odds.py --date 20261012  # 只处理指定日期
import os
import re
import sys
import time
import argparse
import requests
import pymysql
from config import DB_CONFIG

# 欧赔数据JS URL模板
EURO_DATA_URL = 'http://1x2d.titan007.com/{site_match_id}.js'
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'http://1x2.titan007.com/oddslist/{site_match_id}.htm'
}
REQUEST_DELAY = 0.5  # 请求间隔(秒)


def parse_euro_js(js_text):
    """解析欧赔JS文件中的game数组
    game数组格式(每行一个公司, pipe分隔):
      companyID|oddsID|companyName|initWin|initDraw|initLoss|initWinProb|initDrawProb|initLossProb|
      initReturn|curWin|curDraw|curLoss|curWinProb|curDrawProb|curLossProb|curReturn|
      kellyWin|kellyDraw|kellyLoss|updateTime|companyNameCN|isMain|isExchange|...
    返回: [(company, init_win, init_draw, init_loss, cur_win, cur_draw, cur_loss), ...]
    """
    # 提取 game = Array("...", "...", ...)
    match = re.search(r'var\s+game\s*=\s*Array\((.*?)\);', js_text, re.DOTALL)
    if not match:
        return []

    array_content = match.group(1)
    # 提取每个公司的数据字符串
    companies = []
    for m in re.finditer(r'"([^"]*)"', array_content):
        raw = m.group(1)
        parts = raw.split('|')
        if len(parts) < 14:
            continue

        company_name = parts[2].strip()  # 英文名
        company_cn = parts[21].strip() if len(parts) > 21 else ''  # 中文名
        # 优先用中文名， fallback 英文名
        display_name = company_cn if company_cn else company_name
        # 去掉"封"等标记
        display_name = display_name.replace('封', '').strip()
        if not display_name:
            continue

        init_win = parts[3].strip()
        init_draw = parts[4].strip()
        init_loss = parts[5].strip()
        cur_win = parts[10].strip()
        cur_draw = parts[11].strip()
        cur_loss = parts[12].strip()

        # 跳过无效数据
        if not init_win or init_win in ('-', ''):
            continue

        companies.append({
            'company': display_name,
            'initial_win_odds': init_win,
            'initial_draw_odds': init_draw,
            'initial_loss_odds': init_loss,
            'current_win_odds': cur_win if cur_win and cur_win != '-' else None,
            'current_draw_odds': cur_draw if cur_draw and cur_draw != '-' else None,
            'current_loss_odds': cur_loss if cur_loss and cur_loss != '-' else None,
        })

    return companies


def fetch_euro_data(site_match_id):
    """获取单场比赛的欧赔数据"""
    url = EURO_DATA_URL.format(site_match_id=site_match_id)
    headers = {
        'User-Agent': REQUEST_HEADERS['User-Agent'],
        'Referer': REQUEST_HEADERS['Referer'].format(site_match_id=site_match_id)
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return []
        text = resp.content.decode('utf-8', errors='replace')
        # 去掉BOM
        if text.startswith('\ufeff'):
            text = text[1:]
        return parse_euro_js(text)
    except Exception as e:
        print(f"    ✗ 请求失败: {e}")
        return []


def save_european_odds(cursor, match_id, companies):
    """保存欧赔数据到european_odds表（先删后插）"""
    cursor.execute("DELETE FROM european_odds WHERE match_id = %s", (match_id,))
    sql = """INSERT INTO european_odds 
        (match_id, company, initial_win_odds, initial_draw_odds, initial_loss_odds,
         current_win_odds, current_draw_odds, current_loss_odds)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""
    for c in companies:
        cursor.execute(sql, (
            match_id, c['company'],
            c['initial_win_odds'], c['initial_draw_odds'], c['initial_loss_odds'],
            c['current_win_odds'], c['current_draw_odds'], c['current_loss_odds']
        ))


def main():
    parser = argparse.ArgumentParser(description='欧赔数据补录')
    parser.add_argument('--limit', type=int, default=0, help='最多处理场数(0=全部)')
    parser.add_argument('--date', type=str, default=None, help='指定日期(YYYYMMDD)')
    parser.add_argument('--skip-done', action='store_true', default=True, help='跳过已有欧赔数据的比赛')
    args = parser.parse_args()

    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # 查询需要处理的比赛
    if args.date:
        cursor.execute("""
            SELECT match_id, site_match_id FROM `match` 
            WHERE site_match_id IS NOT NULL AND site_match_id != ''
              AND match_id LIKE %s
            ORDER BY match_id
        """, (f'{args.date}%',))
    else:
        cursor.execute("""
            SELECT match_id, site_match_id FROM `match` 
            WHERE site_match_id IS NOT NULL AND site_match_id != ''
            ORDER BY match_id
        """)

    matches = cursor.fetchall()
    total = len(matches)
    print(f"待处理比赛: {total} 场")

    if args.skip_done:
        # 查询已有欧赔数据的match_id
        cursor.execute("SELECT DISTINCT match_id FROM european_odds")
        done_ids = set(row[0] for row in cursor.fetchall())
        matches = [(m, s) for m, s in matches if m not in done_ids]
        print(f"跳过已有数据: {total - len(matches)} 场, 待补录: {len(matches)} 场")

    if args.limit > 0:
        matches = matches[:args.limit]
        print(f"限制处理: {args.limit} 场")

    success = 0
    fail = 0
    no_data = 0

    for i, (match_id, site_match_id) in enumerate(matches):
        print(f"[{i+1}/{len(matches)}] match_id={match_id} site_id={site_match_id}", end=' ')

        companies = fetch_euro_data(site_match_id)
        if not companies:
            print(" 无数据")
            no_data += 1
            continue

        save_european_odds(cursor, match_id, companies)
        conn.commit()
        print(f"✓ {len(companies)} 家公司")
        success += 1

        # 请求间隔，避免被封
        if REQUEST_DELAY > 0:
            time.sleep(REQUEST_DELAY)

    print(f"\n完成! 成功={success}, 无数据={no_data}, 失败={fail}")
    cursor.close()
    conn.close()


if __name__ == '__main__':
    main()
