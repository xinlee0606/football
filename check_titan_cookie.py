# check_titan_cookie.py - titan007 登录Cookie健康检查(失效告警/恢复确认)
# 背景: 2026-11-24起 vip.titan007.com 亚盘/大小球专页需登录, 爬虫靠 titan_cookie.txt
#       里的 Auth Cookie 过墙; Cookie 会过期, 过期当天若没察觉, 当日亚盘数据全部抓空.
# 作用: 每日早于爬虫主动探测Cookie是否仍有效, 失效发告警邮件, 恢复发确认邮件.
# 用法: python check_titan_cookie.py    (crontab 每日 11:40, 早于 12:00 爬虫)
# 防轰炸: 状态写入 STATE_FILE, 仅在 正常<->失效 之间跳变时发邮件, 持续失效只发一次.
# 只读探测: 仅访问页面判断表格是否存在, 不写任何数据表.
import os
import sys
import time
import smtplib
from datetime import datetime
from email.header import Header
from email.mime.text import MIMEText

import pymysql
from bs4 import BeautifulSoup

from config import DB_CONFIG, TITAN_COOKIE_FILE, EMAIL_CONFIG, ENABLE_EMAIL

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, 'logs', 'titan_cookie_state.txt')
ASIAN_URL = 'https://vip.titan007.com/AsianOdds_n.aspx?id={}&l=0'
PROBE_LIMIT = 2       # 探测最近N场比赛, 全部无表格才判失效(降低单场偶发误报)
TABLE_WAIT = 12       # 单页等待亚盘表格异步渲染的秒数上限
MIN_PAGE_LEN = 20000  # 页面源码小于此值视为未真正加载(网络/浏览器异常), 不作失效判断


def send_email(subject, content):
    """邮件通知(受config.py中ENABLE_EMAIL开关控制), 与各predict脚本同款"""
    if not ENABLE_EMAIL:
        print(f"  (邮件开关ENABLE_EMAIL=False, 跳过发送: {subject})")
        return False
    try:
        msg = MIMEText(content, 'plain', 'utf-8')
        msg['Subject'] = Header(subject, 'utf-8')
        msg['From'] = EMAIL_CONFIG['sender']
        msg['To'] = EMAIL_CONFIG['receiver']
        server = smtplib.SMTP_SSL(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.login(EMAIL_CONFIG['sender'], EMAIL_CONFIG['password'])
        server.sendmail(EMAIL_CONFIG['sender'], [EMAIL_CONFIG['receiver']], msg.as_string())
        server.quit()
        print(f"✓ 邮件发送成功: {subject}")
        return True
    except Exception as e:
        print(f"✗ 邮件发送失败: {e}")
        return False


def read_prev_status():
    """读上次状态(首行OK/BAD), 文件不存在/异常返回空串(首次运行)"""
    try:
        with open(STATE_FILE, encoding='utf-8') as f:
            first = f.readline().strip()
        return first
    except Exception:
        return ''


def write_status(status, detail):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        f.write(f"{status}\n{ts}\n{detail}\n")


def get_recent_site_ids(limit):
    """取最近limit场有site_match_id的比赛(按match_id倒序), 用于探测"""
    conn = pymysql.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT site_match_id FROM `match` "
                "WHERE site_match_id IS NOT NULL AND site_match_id<>'' "
                "ORDER BY match_id DESC LIMIT %s", (limit,))
    ids = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return ids


def _asian_table_found(page_source):
    """判断页面是否含真实亚盘表格: 某table存在一行其单元格包含'公司'(与爬虫同口径)"""
    soup = BeautifulSoup(page_source, 'html.parser')
    for t in soup.find_all('table'):
        for row in t.find_all('tr'):
            texts = [c.get_text(strip=True) for c in row.find_all(['td', 'th'])]
            if texts and any(x == '公司' for x in texts):
                return True
    return False


def probe():
    """探测Cookie有效性, 返回 (status, detail): status ∈ {OK, BAD, ERROR}"""
    # Cookie文件缺失/空 → 直接判失效(未配置)
    if not TITAN_COOKIE_FILE or not os.path.isfile(TITAN_COOKIE_FILE):
        return 'BAD', 'titan_cookie.txt 文件不存在'
    with open(TITAN_COOKIE_FILE, encoding='utf-8') as f:
        raw = f.read().strip()
    if not raw:
        return 'BAD', 'titan_cookie.txt 内容为空'

    ids = get_recent_site_ids(PROBE_LIMIT)
    if not ids:
        return 'ERROR', 'DB中无带site_match_id的比赛可比对'

    # 延迟导入, 避免不需要浏览器时(文件缺失)也启动Selenium
    from scraper import TitanScraper
    s = TitanScraper()
    loaded_any = False
    found = 0
    try:
        s.init_browser()   # 内部自动注入Cookie
        for sid in ids:
            try:
                s.driver.get(ASIAN_URL.format(sid))
            except Exception:
                pass
            hit = False
            for _ in range(TABLE_WAIT):
                time.sleep(1)
                src = s.driver.page_source
                if len(src) >= MIN_PAGE_LEN:
                    loaded_any = True
                if '公司' in src and _asian_table_found(src):
                    hit = True
                    break
            if hit:
                found += 1
    finally:
        s.close_browser()

    detail = f"探测{len(ids)}场(sid={','.join(map(str, ids))}), {found}场有亚盘表格"
    if not loaded_any:
        return 'ERROR', detail + '; 页面均未加载(网络/浏览器异常, 不作失效判断)'
    if found == 0:
        return 'BAD', detail + '; 全部无表格(疑Cookie过期或命中登录墙)'
    return 'OK', detail


def alert_bad(detail):
    body = (
        "⚠ titan007 登录 Cookie 已失效, 每日爬虫将抓不到亚盘/大小球数据!\n\n"
        f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"探测结果: {detail}\n\n"
        "请立即处理:\n"
        "1) 浏览器登录 https://vip.titan007.com 任意比赛亚盘页;\n"
        "2) F12 -> Application -> Cookies -> 复制 Auth 的值;\n"
        "3) 覆盖服务器 /coding/live/titan_cookie.txt, 内容形如:\n"
        "   Auth=<你的新值>\n"
        "   loginsameinfo=<可选>\n"
        "4) 手动验证: cd /coding/live && .venv/bin/python -u check_titan_cookie.py\n"
        "5) 若当天爬虫已错过, 补爬: .venv/bin/python -u backfill_asian_ou.py <日期YYYYMMDD>\n"
    )
    send_email('【告警】titan007登录Cookie失效', body)


def alert_ok_recovery(detail):
    body = (
        "✓ titan007 登录 Cookie 已恢复正常。\n\n"
        f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"探测结果: {detail}\n"
    )
    send_email('【已恢复】titan007登录Cookie有效', body)


def main():
    status, detail = probe()
    prev = read_prev_status()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"探测结果: {status} | 上次: {prev or '(无)'} | {detail}")

    if status == 'ERROR':
        # 网络/环境异常: 不改状态、不发邮件, 避免误报
        print("  状态存疑(环境异常), 保持原状态, 不发送邮件")
        return 1

    # 状态跳变才发邮件
    if status == 'BAD' and prev != 'BAD':
        alert_bad(detail)
    elif status == 'OK' and prev == 'BAD':
        alert_ok_recovery(detail)
    else:
        print("  状态未变化, 不发送邮件")

    write_status(status, detail)
    return 0


if __name__ == '__main__':
    sys.exit(main())
