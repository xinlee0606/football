# history_scraper.py - 历史北单比赛全量回填爬虫(与scraper.py同等功能, 入口为Over完场页)
# 数据源: https://bf.titan007.com/football/Over_YYYYMMDD.htm (完场比分页, 仅北单比赛)
# 流程(与每日爬虫一致):
#   1. Over页解析北单赛程(beiDanSchedule) -> match表(含比分/状态/析链接/页面亚盘/进球数两列)
#   2. 每场进分析页: 联赛积分排名 / 对赛往绩 / 近期战绩 / 亚盘专页 / 大小球专页
# 与每日爬虫差异:
#   - 不爬技术统计: 跳过 season_stats(本赛季数据统计) 和 近期战绩逐场详情页技术统计
#   - 浏览器会话失效时自动重建(支持数天不间断长跑)
# 去重/断点续传: match基于site_match_id; 详情已爬(有亚盘或近期战绩记录)的比赛自动跳过
# 用法:
#   python -u history_scraper.py                     # 回填最近 DAYS_BACK 天
#   python -u history_scraper.py 3                   # 回填最近3天
#   python -u history_scraper.py 20250806 20250810   # 回填指定日期区间(含首尾)
#   python -u history_scraper.py 20250806 20250806 --limit 2   # 测试: 每天最多处理2场详情
import re
import sys
import time
from datetime import datetime, timedelta

from bs4 import BeautifulSoup
from selenium.common.exceptions import WebDriverException

from scraper import TitanScraper
from config import OVER_URL_TEMPLATE, MAIN_URL, REQUEST_DELAY

# ==================== 配置 ====================
DAYS_BACK = 365              # 默认回填天数(从昨天往前)
HISTORY_PAGE_WAIT = 8        # 完场存档页加载等待秒数
SKIP_STATUS = ('推迟', '取消', '腰斩', '中断')  # 无比分的异常状态直接跳过
PAGE_COLS = ['page_handicap', 'page_goals']     # match表新增两列(完场页亚盘/进球数)


class HistoryScraper(TitanScraper):
    """历史回填爬虫: 复用 TitanScraper 全部解析/入库方法, 仅跳过技术统计"""

    # ---------- 跳过技术统计(覆盖父类为空实现) ----------
    def _scrape_technical_stats_from_analysis(self, match_id, soup):
        """历史回填不爬 season_stats(本赛季数据统计比较)"""
        pass

    def _scrape_recent_form_tech_stats(self, match_id):
        """历史回填不逐场进详情页爬近期战绩技术统计(射门/射正/控球率等)"""
        pass

    # ---------- 浏览器会话自愈(长跑防崩溃) ----------
    def _safe_get(self, url, wait=HISTORY_PAGE_WAIT):
        """加载页面, 会话失效时自动重建浏览器并重试一次"""
        for attempt in (1, 2):
            try:
                self.driver.get(url)
                time.sleep(wait)
                return True
            except WebDriverException as e:
                print(f"  ⚠ 页面加载失败(第{attempt}次): {str(e).splitlines()[0]}")
                if attempt == 1:
                    print("  ↻ 重建浏览器会话...")
                    try:
                        self.driver.quit()
                    except Exception:
                        pass
                    self.driver = None
                    self.init_browser()
                    # 重建后恢复语言设置(简体队名)
                    try:
                        self._ensure_team_name_language()
                    except Exception:
                        pass
        return False

    # ---------- 数据库准备 ----------
    def ensure_page_columns(self):
        """match表自动补建完场页两列: page_handicap(亚盘) / page_goals(进球数)"""
        cursor = self.db.cursor()
        comments = {'page_handicap': '完场页亚盘(*号已转受让)', 'page_goals': '完场页进球数线'}
        for col in PAGE_COLS:
            cursor.execute(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE table_schema=DATABASE() AND table_name='match' AND column_name=%s", (col,))
            if cursor.fetchone()[0] == 0:
                cursor.execute(
                    f"ALTER TABLE `match` ADD COLUMN `{col}` VARCHAR(30) DEFAULT NULL COMMENT '{comments[col]}'")
                print(f"✓ match表新增列 {col}")
        self.db.commit()
        cursor.close()

    def cleanup_page_odds_rows(self):
        """清理早期回填写入盘口表的'完场页'标记行(该数据改存match表两列)"""
        cursor = self.db.cursor()
        n1 = cursor.execute("DELETE FROM asian_odds WHERE company = '完场页'")
        n2 = cursor.execute("DELETE FROM over_under_odds WHERE company = '完场页'")
        self.db.commit()
        cursor.close()
        if n1 or n2:
            print(f"✓ 清理盘口表'完场页'旧数据: asian_odds {n1}条, over_under_odds {n2}条")

    # ---------- 单日回填 ----------
    def backfill_date(self, date_str, detail_limit=None):
        """回填某一天的北单比赛(列表+详情), 返回 (新增, 已存在, 跳过, 详情已爬/新爬/失败)"""
        url = OVER_URL_TEMPLATE.format(date=date_str)
        if not self._safe_get(url):
            print(f"  ✗ {url} 加载失败, 跳过该日")
            return 0, 0, 0, 0, 0, 0

        try:
            html = self.driver.execute_script("return document.documentElement.outerHTML;")
        except WebDriverException as e:
            print(f"  ✗ 获取DOM失败: {str(e).splitlines()[0]}")
            return 0, 0, 0, 0, 0, 0

        # 1. 北单赛程映射 sid -> 单场编号 (Over页"选择-北单"过滤即此列表)
        sid2dan = self._parse_beidan_schedule(html)
        if not sid2dan:
            print(f"  - 无北单赛程映射(beiDanSchedule), 跳过")
            return 0, 0, 0, 0, 0, 0

        # 2. 比赛表格 sid -> 行
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table', id='table_live')
        if not table:
            print(f"  ✗ 未找到比赛表格(table_live)")
            return 0, 0, 0, 0, 0, 0
        sid2row = {}
        for r in table.find_all('tr'):
            rid = r.get('id', '')
            sid = r.get('sid')
            if rid.startswith('tr1_') and sid:
                sid2row[str(sid)] = r

        # 3. 当天match_id顺序编号(与每日爬虫规则一致: 日期+两位序号, 从已有最大序号+1续编)
        cursor = self.db.cursor()
        cursor.execute("SELECT match_id FROM `match` WHERE match_id LIKE %s", (date_str + '%',))
        existing_nums = []
        for (mid,) in cursor.fetchall():
            m = re.search(r'(\d+)$', mid)
            if m:
                existing_nums.append(int(m.group(1)))
        cursor.close()
        next_num = (max(existing_nums) + 1) if existing_nums else 1

        # 4. 解析北单比赛行 -> 入库match表
        inserted = existed = skipped = 0
        pending = []  # 待爬详情的比赛列表
        items = sorted(sid2dan.items(), key=lambda kv: self._dan_num(kv[1]) or 0)
        for sid, single_match_raw in items:
            row = sid2row.get(sid)
            if not row:
                continue
            cells = row.find_all('td')
            if len(cells) < 7:
                continue
            # Over页面列结构: 0联赛 1时间 2状态 3主队 4比分 5客队 6半场 7亚盘 8进球数 9数据
            league = cells[0].get_text(strip=True)
            match_time = cells[1].get_text(strip=True)
            status = cells[2].get_text(strip=True) or None
            score = cells[4].get_text(strip=True)
            if score in ('-', ''):
                score = None
            home_team, home_rank = self._extract_team(cells[3])
            away_team, away_rank = self._extract_team(cells[5])

            # 推迟/取消等无比分场次跳过
            if score is None and (status in SKIP_STATUS or status in (None, '-', '')):
                skipped += 1
                continue

            num = self._dan_num(single_match_raw)
            if num is None:
                skipped += 1
                continue

            data = {
                'site_match_id': sid,
                'league': league,
                'single_match': self._normalize_single_match(single_match_raw),
                'match_time': match_time,
                'status': status,
                'home_team': home_team,
                'home_rank': home_rank,
                'away_team': away_team,
                'away_rank': away_rank,
                'score': score,
                'analysis_url': f'https://zq.titan007.com/analysis/{sid}cn.htm',
                'detail_url': f'https://live.titan007.com/detail/{sid}cn.htm',
            }

            try:
                # 已存在(按site_match_id去重)复用原match_id, 新比赛顺序编号
                cursor = self.db.cursor()
                cursor.execute("SELECT match_id FROM `match` WHERE site_match_id = %s", (sid,))
                hit = cursor.fetchone()
                cursor.close()
                if hit:
                    data['match_id'] = hit[0]
                else:
                    data['match_id'] = f"{date_str}{next_num:02d}"

                saved_id, was_existing = self.save_match(data)

                # 完场页"亚盘/进球数"两列 -> match表 page_handicap / page_goals
                page_handicap = self._normalize_handicap(
                    cells[7].get_text(strip=True) if len(cells) > 7 else '')
                page_goals = (cells[8].get_text(strip=True) if len(cells) > 8 else '') or None
                cursor = self.db.cursor()
                cursor.execute(
                    "UPDATE `match` SET page_handicap=%s, page_goals=%s WHERE match_id=%s",
                    (page_handicap, page_goals, saved_id))
                self.db.commit()
                cursor.close()
            except Exception as e:
                print(f"  ✗ 保存失败 {home_team} vs {away_team}: {e}")
                continue

            if was_existing:
                existed += 1
                mark = '⊙'
            else:
                inserted += 1
                next_num += 1
                mark = '✓'
            print(f"  {mark} [{saved_id}] [{data['single_match']}] {league} "
                  f"{home_team} {score or '-'} {away_team}")
            pending.append(data)

        # 5. 详情阶段: 分析页(积分/往绩/近况) + 亚盘专页 + 大小球专页
        done_old = done_new = done_fail = 0
        total = len(pending)
        for idx, m in enumerate(pending, 1):
            if detail_limit is not None and idx > detail_limit:
                print(f"  ⚡ 测试模式: 跳过剩余 {total - detail_limit} 场详情")
                break
            mid = m['match_id']
            if self._is_match_done(mid):
                done_old += 1
                print(f"\n  [{idx}/{total}] ⊘ {mid} {m['home_team']} vs {m['away_team']} (详情已爬, 跳过)")
                continue
            print(f"\n  [{idx}/{total}] 详情: {mid} {m['home_team']} vs {m['away_team']}")
            try:
                self.scrape_analysis_page(
                    mid, m['analysis_url'], m.get('detail_url'),
                    home_team=m['home_team'], away_team=m['away_team'])
                done_new += 1
            except WebDriverException as e:
                print(f"  ✗ 详情爬取会话异常: {str(e).splitlines()[0]}, 重建浏览器...")
                try:
                    self.driver.quit()
                except Exception:
                    pass
                self.driver = None
                self.init_browser()
                try:
                    self._ensure_team_name_language()
                except Exception:
                    pass
                done_fail += 1
            except Exception as e:
                print(f"  ✗ 详情爬取失败: {e}")
                done_fail += 1
            time.sleep(REQUEST_DELAY)

        return inserted, existed, skipped, done_old, done_new, done_fail

    def backfill(self, dates, detail_limit=None):
        """按日期列表回填, 返回汇总"""
        total = dict(inserted=0, existed=0, skipped=0,
                     done_old=0, done_new=0, done_fail=0, failed_days=0)
        n = len(dates)
        for i, d in enumerate(dates, 1):
            print(f"\n{'='*60}\n[{i}/{n}] 回填 {d} ...\n{'='*60}")
            try:
                ins, exi, skp, dold, dnew, dfail = self.backfill_date(d, detail_limit)
            except Exception as e:
                print(f"  ✗ {d} 处理出错: {e}")
                import traceback
                traceback.print_exc()
                total['failed_days'] += 1
                continue
            total['inserted'] += ins
            total['existed'] += exi
            total['skipped'] += skp
            total['done_old'] += dold
            total['done_new'] += dnew
            total['done_fail'] += dfail
            print(f"  当日汇总: 新增{ins} 已存在{exi} 跳过{skp} | "
                  f"详情: 新爬{dnew} 已爬跳过{dold} 失败{dfail}")
        return total

    # ---------- 工具 ----------
    def _is_match_done(self, match_id):
        """详情是否已爬完: asian_odds 或 recent_form 有该场记录即视为已完成"""
        try:
            cursor = self.db.cursor()
            cursor.execute("SELECT COUNT(*) FROM asian_odds WHERE match_id=%s", (match_id,))
            c1 = cursor.fetchone()[0]
            cursor.execute(
                "SELECT COUNT(*) FROM recent_form WHERE match_id=%s AND is_current_match=0",
                (match_id,))
            c2 = cursor.fetchone()[0]
            cursor.close()
            return (c1 + c2) > 0
        except Exception:
            return False

    @staticmethod
    def _dan_num(single_match_raw):
        """从'单场001'提取数字编号"""
        m = re.search(r'(\d+)', single_match_raw or '')
        return int(m.group(1)) if m else None

    @staticmethod
    def _extract_team(cell):
        """提取队名与排名标记: ([7]松达尔) -> ('松达尔', '7')"""
        text = cell.get_text(strip=True)
        rank_m = re.search(r'\[(\d+)\]', text)
        rank = rank_m.group(1) if rank_m else None
        name = re.sub(r'\[[^\]]*\]', '', text).strip()
        return name, rank


def build_date_list(argv):
    """解析命令行参数生成日期列表(升序)"""
    if len(argv) >= 3:
        start = datetime.strptime(argv[1], '%Y%m%d')
        end = datetime.strptime(argv[2], '%Y%m%d')
    else:
        days = int(argv[1]) if len(argv) == 2 else DAYS_BACK
        end = datetime.now() - timedelta(days=1)
        start = end - timedelta(days=days - 1)
    dates = []
    d = start
    while d <= end:
        dates.append(d.strftime('%Y%m%d'))
        d += timedelta(days=1)
    return dates


def main():
    limit_arg = next((a for a in sys.argv[1:] if a.startswith('--limit')), None)
    detail_limit = int(limit_arg.split('=')[1]) if limit_arg and '=' in limit_arg else \
        (int(sys.argv[sys.argv.index(limit_arg) + 1]) if limit_arg else None)

    # 日期参数: 过滤掉选项本身及 --limit 后跟的数值(避免被误当日期解析)
    args = []
    skip_next = False
    for a in sys.argv[1:]:
        if a.startswith('--'):
            skip_next = ('=' not in a)
            continue
        if skip_next:
            skip_next = False
            continue
        args.append(a)

    dates = build_date_list([sys.argv[0]] + args)
    print("=" * 60)
    print(f"历史北单比赛全量回填(含详情, 无技术统计)")
    print(f"区间: {dates[0]} ~ {dates[-1]} 共{len(dates)}天")
    if detail_limit:
        print(f"测试模式: 每天最多处理 {detail_limit} 场详情")
    print("=" * 60)

    scraper = HistoryScraper()
    try:
        scraper.init_browser()
        scraper.init_db()
        scraper.ensure_page_columns()
        scraper.cleanup_page_odds_rows()
        # 统一球队名称语言(简体), 与每日爬虫保持一致
        scraper.driver.get(MAIN_URL)
        time.sleep(5)
        scraper._ensure_team_name_language()

        total = scraper.backfill(dates, detail_limit)
        print("\n" + "=" * 60)
        print(f"🎉 回填完成! 列表: 新增{total['inserted']} 已存在{total['existed']} "
              f"跳过{total['skipped']} 失败天数{total['failed_days']}")
        print(f"   详情: 新爬{total['done_new']} 已爬跳过{total['done_old']} 失败{total['done_fail']}")
        print("=" * 60)
    finally:
        scraper.close_browser()
        scraper.close_db()


if __name__ == '__main__':
    main()
