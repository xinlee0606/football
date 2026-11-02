# backfill_recent_form_tech.py
# 补爬近期战绩技术统计（按detail_url去重优化 + 断点续传 + 浏览器自愈）
# 每个历史比赛详情页只访问一次，提取主客队双方数据，批量UPDATE所有引用该URL的recent_form行
import os, sys, time, random, re, argparse, json
from datetime import datetime
from collections import defaultdict
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException
from urllib3.exceptions import ReadTimeoutError

from scraper import TitanScraper

DEFAULT_DELAY = 20
DELAY_JITTER = 10
CHECKPOINT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'backfill_recent_form_tech_checkpoint.json')


class RecentFormTechBackfiller(TitanScraper):

    def _safe_get(self, url, wait=3):
        """加载页面，会话失效/超时时自动重启浏览器"""
        for attempt in (1, 2):
            try:
                self.driver.get(url)
                time.sleep(wait)
                return True
            except (WebDriverException, ReadTimeoutError) as e:
                err_msg = str(e)
                if attempt == 1:
                    print(f"       页面加载失败: {err_msg[:80]}")
                    print(f"       重启浏览器...")
                    try:
                        self.driver.quit()
                    except Exception:
                        pass
                    self.driver = None
                    time.sleep(3)
                    self.init_browser()
                    try:
                        self._ensure_team_name_language()
                    except Exception:
                        pass
                    print(f"      ✓ 浏览器已重启，重试...")
        return False

    def _extract_both_teams_stats(self, detail_url):
        """访问详情页，提取主客队双方技术统计。返回 (home_stats, away_stats) 或 (None, None)"""
        if not self._safe_get(detail_url, wait=3):
            return None, None

        soup = BeautifulSoup(self.driver.page_source, 'html.parser')

        tech_div = soup.find('div', id='teamTechDiv_detail')
        if not tech_div:
            try:
                tabs = self.driver.find_elements(By.XPATH, '//a[contains(text(),"现场分析")]')
                if tabs:
                    tabs[0].click()
                    time.sleep(2)
                    soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                    tech_div = soup.find('div', id='teamTechDiv_detail')
            except Exception:
                pass

        if not tech_div:
            return None, None

        tech_list = tech_div.find('ul', id='teamTechDiv')
        if not tech_list:
            return None, None

        stat_field_map = {
            '角球': 'corners', '半场角球': 'half_corners',
            '射门': 'shots', '射正': 'shots_on_target',
            '进攻': 'attacks', '危险进攻': 'dangerous_attacks',
            '射门不中': 'shots_off_target', '射门被挡': 'shots_blocked',
            '任意球': 'free_kicks', '控球率': 'possession',
            '半场控球率': 'half_possession', '传球': 'passes',
            '传球成功率': 'pass_success_rate', '犯规': 'fouls',
            '越位': 'offsides', '黄牌': 'yellow_cards',
        }

        home_stats = {}
        away_stats = {}

        for li in tech_list.find_all('li', class_='lists'):
            data_div = li.find('div', class_='data')
            if not data_div:
                continue
            spans = data_div.find_all('span')
            if len(spans) < 3:
                continue

            home_val = spans[0].get_text(strip=True)
            stat_name = spans[1].get_text(strip=True)
            away_val = spans[2].get_text(strip=True)

            field = stat_field_map.get(stat_name)
            if field:
                if field in ['possession', 'half_possession', 'pass_success_rate']:
                    home_stats[field] = home_val.replace('%', '')
                    away_stats[field] = away_val.replace('%', '')
                else:
                    home_stats[field] = self._safe_int(home_val)
                    away_stats[field] = self._safe_int(away_val)

        return (home_stats, away_stats) if (home_stats or away_stats) else (None, None)

    def _resolve_team_side(self, team_name, home_team, away_team):
        """判断关注球队是主队还是客队。返回 True=主队, False=客队, None=无法匹配"""
        ft = self.clean_team_name(team_name) if team_name else ''
        ht = self.clean_team_name(home_team) if home_team else ''
        at = self.clean_team_name(away_team) if away_team else ''
        if not ft or not ht:
            return None

        n1, n2 = ft, ht
        n1 = re.sub(r'[\s\u00a0\u200b\(\)（）]', '', n1)
        n2 = re.sub(r'[\s\u00a0\u200b\(\)（）]', '', n2)
        home_match = (n1 in n2 or n2 in n1) if n1 and n2 else False
        if not home_match:
            common = sum(1 for c in n1 if c in n2)
            home_match = common / max(len(n1), len(n2)) >= 0.6 if n1 and n2 else False

        n1, n2 = ft, at
        n1 = re.sub(r'[\s\u00a0\u200b\(\)（）]', '', n1)
        n2 = re.sub(r'[\s\u00a0\u200b\(\)（）]', '', n2)
        away_match = (n1 in n2 or n2 in n1) if n1 and n2 else False
        if not away_match:
            common = sum(1 for c in n1 if c in n2)
            away_match = common / max(len(n1), len(n2)) >= 0.6 if n1 and n2 else False

        if home_match and not away_match:
            return True
        elif away_match and not home_match:
            return False
        elif home_match and away_match:
            hc = sum(1 for c in ft if c in ht)
            ac = sum(1 for c in ft if c in at)
            return hc >= ac
        return None

    def _load_checkpoint(self):
        """加载断点续传记录"""
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, encoding='utf-8') as f:
                return json.load(f)
        return {'last_url': None, 'processed': 0, 'success': 0, 'fail': 0}

    def _save_checkpoint(self, last_url, processed, success, fail):
        """保存断点续传记录"""
        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump({'last_url': last_url, 'processed': processed,
                       'success': success, 'fail': fail}, f)

    def run_backfill(self, delay=DEFAULT_DELAY):
        print("=" * 60)
        print("近期战绩技术统计补爬（去重优化版 + 断点续传）")
        print(f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"间隔: {delay}秒 +/- {DELAY_JITTER}秒")
        print("=" * 60)

        self.init_browser()
        self.init_db()

        cursor = self.db.cursor()
        cursor.execute("""
            SELECT id, match_id, detail_url, team_name, home_team, away_team
            FROM recent_form
            WHERE is_current_match = 0 AND shots IS NULL
              AND detail_url IS NOT NULL AND detail_url != ''
            ORDER BY detail_url
        """)
        rows = cursor.fetchall()
        cursor.close()

        total_rows = len(rows)
        print(f"\n待补爬: {total_rows} 行近期战绩")

        if total_rows == 0:
            print("无需补爬")
            self.close_browser()
            self.close_db()
            return

        # 按 detail_url 去重分组
        url_groups = defaultdict(list)
        for row in rows:
            url_groups[row[2]].append(row)

        unique_urls = len(url_groups)
        sorted_urls = sorted(url_groups.keys())

        # 断点续传：跳过已处理的 URL
        ckpt = self._load_checkpoint()
        start_idx = 0
        if ckpt['last_url'] and ckpt['last_url'] in url_groups:
            try:
                start_idx = sorted_urls.index(ckpt['last_url']) + 1
                print(f"断点续传: 跳过前 {start_idx} 个已处理URL (上次: {ckpt['last_url']})")
                print(f"  历史进度: 处理{ckpt['processed']} 成功{ckpt['success']} 失败{ckpt['fail']}")
            except ValueError:
                pass

        remaining = unique_urls - start_idx
        print(f"去重后: {unique_urls} 个唯一详情页, 剩余 {remaining} 个待处理")
        print(f"预计耗时: {remaining * (delay + 3) / 3600:.1f} 小时\n")

        success_rows = ckpt['success']
        fail_rows = ckpt['fail']
        processed_urls = ckpt['processed']
        last_url = ckpt['last_url']

        for i in range(start_idx, len(sorted_urls)):
            detail_url = sorted_urls[i]
            group = url_groups[detail_url]
            processed_urls += 1
            last_url = detail_url

            try:
                home_stats, away_stats = self._extract_both_teams_stats(detail_url)

                if home_stats is None:
                    fail_rows += len(group)
                    print(f"  [{processed_urls}/{unique_urls}] ✗ {detail_url[-30:]} 无数据")
                else:
                    url_success = 0
                    url_fail = 0
                    for row_id, match_id, _, team_name, home_team, away_team in group:
                        try:
                            side = self._resolve_team_side(team_name, home_team, away_team)
                            if side is None:
                                url_fail += 1
                                continue

                            stats = home_stats if side else away_stats
                            if not stats:
                                url_fail += 1
                                continue

                            tech_fields = ['corners', 'half_corners', 'yellow_cards', 'shots',
                                           'shots_on_target', 'attacks', 'dangerous_attacks',
                                           'shots_off_target', 'shots_blocked', 'free_kicks',
                                           'possession', 'half_possession', 'passes',
                                           'pass_success_rate', 'fouls', 'offsides']

                            update_parts = []
                            update_values = []
                            for field in tech_fields:
                                if field in stats:
                                    update_parts.append(f"`{field}` = %s")
                                    update_values.append(stats[field])

                            if update_parts:
                                update_values.append(row_id)
                                sql = f"UPDATE `recent_form` SET {', '.join(update_parts)} WHERE id = %s"
                                c = self.db.cursor()
                                c.execute(sql, update_values)
                                self.db.commit()
                                c.close()
                                url_success += 1
                        except Exception as e:
                            url_fail += 1

                    success_rows += url_success
                    fail_rows += url_fail
                    side_label = '主' if any(self._resolve_team_side(g[3], g[4], g[5]) for g in group[:1]) else '客'
                    print(f"  [{processed_urls}/{unique_urls}] ✓ {detail_url[-30:]} "
                          f"成功{url_success}行 失败{url_fail}行")

            except Exception as e:
                fail_rows += len(group)
                print(f"  [{processed_urls}/{unique_urls}] ✗ {detail_url[-30:]} 异常: {str(e)[:60]}")

            # 每处理10个URL保存一次断点
            if processed_urls % 10 == 0:
                self._save_checkpoint(last_url, processed_urls, success_rows, fail_rows)
                print(f"  [断点已保存] 处理{processed_urls}/{unique_urls} "
                      f"成功{success_rows} 失败{fail_rows}")

            jitter = random.uniform(-DELAY_JITTER, DELAY_JITTER)
            time.sleep(max(2, delay + jitter))

        # 最终保存
        self._save_checkpoint(last_url, processed_urls, success_rows, fail_rows)

        print(f"\n{'=' * 60}")
        print(f"补爬完成！处理 {processed_urls}/{unique_urls} 个详情页")
        print(f"  成功: {success_rows}/{total_rows} 行")
        print(f"  失败: {fail_rows}/{total_rows} 行")
        print(f"{'=' * 60}")

        self.close_browser()
        self.close_db()


def main():
    ap = argparse.ArgumentParser(description='补爬近期战绩技术统计（去重优化版 + 断点续传）')
    ap.add_argument('--delay', type=int, default=DEFAULT_DELAY, help=f'每页间隔秒数（默认{DEFAULT_DELAY}）')
    ap.add_argument('--reset', action='store_true', help='清除断点从头开始')
    args = ap.parse_args()

    if args.reset and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("已清除断点记录，将从头开始")

    backfiller = RecentFormTechBackfiller()
    backfiller.run_backfill(delay=args.delay)


if __name__ == '__main__':
    main()
