# backfill_asian_ou.py - 补爬指定日期缺失的亚盘/大小球数据(需登录Cookie)
# 背景: 2026-11-24起 vip.titan007.com 亚盘/大小球专页需登录, 每日爬虫抓到登录墙空表,
#       导致 asian_odds / over_under_odds 缺失; 而 recent_form 等非登录页正常, 故
#       history_scraper 的 _is_match_done 会误判"已完成"跳过. 本脚本绕过该判定,
#       仅对 asian/ou 缺失的场次重抓这两个表.
# 用法: python backfill_asian_ou.py 20261124 20261125 20261126
#       (不传日期则默认 20261124~20261126)
# 断点续爬: 已同时有 asian 和 ou 记录的场次自动跳过, 可安全重跑.
import sys
import time
from scraper import TitanScraper

DEFAULT_DATES = ['20261124', '20261125', '20261126']
DELAY = 6  # 场间隔秒数(小规模补爬, 兼顾速度与防封)


def has_both(cursor, match_id):
    cursor.execute("SELECT COUNT(*) FROM asian_odds WHERE match_id=%s", (match_id,))
    a = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM over_under_odds WHERE match_id=%s", (match_id,))
    o = cursor.fetchone()[0]
    return a > 0 and o > 0


def main():
    dates = sys.argv[1:] or DEFAULT_DATES
    s = TitanScraper()
    s.init_browser()   # 内部自动注入 titan_cookie.txt 登录Cookie
    s.init_db()
    cursor = s.db.cursor()

    targets = []
    for d in dates:
        cursor.execute(
            "SELECT match_id, site_match_id FROM `match` "
            "WHERE match_id LIKE %s AND site_match_id IS NOT NULL AND site_match_id<>''",
            (d + '%',))
        rows = cursor.fetchall()
        todo = [r for r in rows if not has_both(cursor, r[0])]
        print(f"{d}: 共{len(rows)}场, 待补{len(todo)}场")
        targets.extend(todo)

    print(f"总计待补 {len(targets)} 场, 场间隔{DELAY}秒, 预计约{len(targets)*(DELAY+18)//60}分钟")
    ok_a = ok_o = fail = 0
    for i, (mid, sid) in enumerate(targets, 1):
        try:
            s._scrape_asian_odds(mid, sid)
            s._scrape_over_under_odds(mid, sid)
            cursor.execute("SELECT COUNT(*) FROM asian_odds WHERE match_id=%s", (mid,))
            na = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM over_under_odds WHERE match_id=%s", (mid,))
            no = cursor.fetchone()[0]
            ok_a += int(na > 0)
            ok_o += int(no > 0)
            if na == 0 and no == 0:
                fail += 1
                print(f"  [{i}/{len(targets)}] {mid} sid={sid} ✗ 两表均空(登录墙/Cookie过期?)")
            else:
                print(f"  [{i}/{len(targets)}] {mid} sid={sid} ✓ asian={na} ou={no}")
        except Exception as e:
            fail += 1
            print(f"  [{i}/{len(targets)}] {mid} ✗ 异常: {str(e)[:80]}")
        time.sleep(DELAY)

    print("=" * 60)
    print(f"补爬完成: 亚盘有数据{ok_a}场, 大小球有数据{ok_o}场, 全失败{fail}场 / 共{len(targets)}场")
    print("=" * 60)
    cursor.close()
    s.close_browser()
    s.close_db()


if __name__ == '__main__':
    main()
