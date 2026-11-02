# backfill_tech_stats.py - 补爬历史比赛技术统计数据
# 用途：为数据库中缺少技术统计的历史比赛补爬详情页技术统计和赛季统计
# 特点：慢速爬取，每场间隔15-25秒随机延迟，避免触发反爬
# 支持断点续传：已爬过的比赛自动跳过
# 用法：
#   python -u backfill_tech_stats.py              # 补爬所有缺技术统计的比赛
#   python -u backfill_tech_stats.py 20261015     # 补爬指定日期
#   python -u backfill_tech_stats.py 20261015 --delay 30  # 自定义延迟秒数
import os
import sys
import time
import random
import argparse
from datetime import datetime
from bs4 import BeautifulSoup
from selenium.common.exceptions import WebDriverException

from scraper import TitanScraper
from config import REQUEST_DELAY

# 默认每场间隔(秒)，含随机浮动
DEFAULT_DELAY = 20
DELAY_JITTER = 10  # 随机浮动 ±10秒


class TechStatsBackfiller(TitanScraper):
    """技术统计补爬器：复用 TitanScraper 的解析/入库方法"""

    def _safe_get(self, url, wait=8):
        """加载页面，会话失效时自动重建"""
        for attempt in (1, 2):
            try:
                self.driver.get(url)
                time.sleep(wait)
                return True
            except WebDriverException as e:
                print(f"    ⚠ 页面加载失败(第{attempt}次): {str(e).splitlines()[0]}")
                if attempt == 1:
                    print("    ↻ 重建浏览器会话...")
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
        return False

    def backfill_match_tech_stats(self, match_id, detail_url, analysis_url,
                                   home_team, away_team, delay=DEFAULT_DELAY):
        """补爬单场比赛的技术统计(详情页) + 赛季统计(分析页)
        Returns: (tech_ok, season_ok)
        """
        tech_ok = season_ok = False

        # 1. 详情页技术统计 → 更新 recent_form
        if detail_url:
            print(f"    详情页技术统计...")
            try:
                self._scrape_detail_tech_stats(
                    match_id, detail_url,
                    home_team_name=home_team, away_team_name=away_team
                )
                tech_ok = True
            except Exception as e:
                print(f"    ✗ 技术统计失败: {str(e)[:80]}")

        # 间隔
        jitter = random.uniform(-DELAY_JITTER, DELAY_JITTER)
        wait = max(5, delay + jitter)
        print(f"    等待 {wait:.0f}秒...")
        time.sleep(wait)

        # 2. 分析页赛季统计 → 插入 season_stats
        if analysis_url:
            print(f"    分析页赛季统计...")
            try:
                if self._safe_get(analysis_url, wait=8):
                    html = self.driver.execute_script("return document.documentElement.outerHTML;")
                    soup = BeautifulSoup(html, 'html.parser')
                    self._scrape_technical_stats_from_analysis(match_id, soup)
                    season_ok = True
            except Exception as e:
                print(f"    ✗ 赛季统计失败: {str(e)[:80]}")

        return tech_ok, season_ok

    def run_backfill(self, target_date=None, delay=DEFAULT_DELAY):
        """执行技术统计补爬
        Args:
            target_date: 目标日期 YYYYMMDD，None则处理所有缺技术统计的比赛
            delay: 每场间隔秒数
        """
        print("=" * 60)
        print("技术统计补爬")
        print(f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"间隔: {delay}秒 ± {DELAY_JITTER}秒")
        if target_date:
            print(f"目标: {target_date}")
        else:
            print(f"目标: 所有缺技术统计的比赛")
        print("=" * 60)

        self.init_browser()
        self.init_db()

        cursor = self.db.cursor()

        # 查找缺少技术统计的比赛
        if target_date:
            date_cond = "m.match_id LIKE %s"
            date_params = (target_date + '%',)
        else:
            date_cond = "1=1"
            date_params = ()

        # 找 recent_form 中 shots 为 NULL 的比赛
        cursor.execute(f"""
            SELECT DISTINCT m.match_id, m.detail_url, m.analysis_url,
                   m.home_team, m.away_team
            FROM `match` m
            LEFT JOIN recent_form rf ON m.match_id = rf.match_id AND rf.shots IS NOT NULL
            WHERE {date_cond}
              AND m.detail_url IS NOT NULL
              AND rf.match_id IS NULL
            ORDER BY m.match_id
        """, date_params)
        matches = cursor.fetchall()
        cursor.close()

        total = len(matches)
        print(f"\n待补爬: {total} 场比赛\n")

        if total == 0:
            print("无需补爬的比赛")
            self.close_browser()
            self.close_db()
            return

        tech_ok_count = season_ok_count = fail_count = 0

        for idx, (match_id, detail_url, analysis_url, home_team, away_team) in enumerate(matches, 1):
            print(f"[{idx}/{total}] {match_id} {home_team} vs {away_team}")

            tech_ok, season_ok = self.backfill_match_tech_stats(
                match_id, detail_url, analysis_url,
                home_team, away_team, delay=delay
            )

            if tech_ok:
                tech_ok_count += 1
            if season_ok:
                season_ok_count += 1
            if not tech_ok and not season_ok:
                fail_count += 1

            # 每10场输出一次进度
            if idx % 10 == 0:
                print(f"\n--- 进度: {idx}/{total} | 技术统计成功:{tech_ok_count} 赛季统计成功:{season_ok_count} 失败:{fail_count} ---\n")

        print(f"\n{'=' * 60}")
        print(f"补爬完成！共 {total} 场")
        print(f"  详情页技术统计: {tech_ok_count}/{total} 成功")
        print(f"  分析页赛季统计: {season_ok_count}/{total} 成功")
        print(f"  完全失败: {fail_count}/{total}")
        print(f"{'=' * 60}")

        self.close_browser()
        self.close_db()


def main():
    ap = argparse.ArgumentParser(description='补爬历史比赛技术统计数据')
    ap.add_argument('date', nargs='?', default=None,
                    help='目标日期 YYYYMMDD（缺省则处理所有缺失比赛）')
    ap.add_argument('--delay', type=int, default=DEFAULT_DELAY,
                    help=f'每场间隔秒数（默认{DEFAULT_DELAY}）')
    args = ap.parse_args()

    backfiller = TechStatsBackfiller()
    backfiller.run_backfill(target_date=args.date, delay=args.delay)


if __name__ == '__main__':
    main()
