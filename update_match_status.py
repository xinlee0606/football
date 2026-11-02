# update_match_status.py - 更新比赛状态和比分 + 同步开奖SP值
# 用途：每天定时执行，从完场比分页面(Over_*.htm)获取北单比赛最新状态和比分，更新到match表
#       （仅更新已存在比赛，不爬取详情数据，不插入新比赛）；
#       随后同步更新result_sp表(500网开奖SP值)并回填match_id，
#       范围与match表保持一致：match表更新哪天的比赛，result_sp就更新哪天的开奖数据
# 数据源：https://bf.titan007.com/football/Over_YYYYMMDD.htm（选择-北单 过滤）
#         https://zx.500.com/zqdc/kaijiang.php?playid=3&expect={期号}
# 用法：python update_match_status.py            # 默认更新昨天的比赛
#       python update_match_status.py 20260726   # 指定日期更新
# 部署：通过crontab每天中午执行（见 run_daily.sh / crontab_linux.txt）
import sys
from datetime import datetime, timedelta

from scraper import TitanScraper
from sp_scraper import ResultSpScraper


def update_result_sp(target_date):
    """同步更新result_sp(500网开奖SP)并回填match_id，失败不影响主流程"""
    sp = ResultSpScraper()
    try:
        sp.ensure_table()
        total, matched, ambiguous = sp.update_by_date(target_date)
        print("\n" + "=" * 60)
        print(f"🎉 result_sp 更新完成！新入库{total['saved']}期 更新{total['updated']}期 "
              f"跳过{total['skip']}期 | SP匹配: 成功{matched}条 待核查{ambiguous}条")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ result_sp更新出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sp.close()


def main():
    # 可选命令行参数：目标日期(YYYYMMDD)，缺省为昨天
    target_date = sys.argv[1] if len(sys.argv) > 1 else None
    if not target_date:
        target_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')

    scraper = TitanScraper()
    ok = False
    try:
        scraper.init_browser()
        scraper.init_db()
        updated = scraper.update_match_status(target_date)
        print("\n" + "="*60)
        print(f"🎉 状态更新完成！共更新 {updated} 场比赛")
        print("="*60)
        ok = True
    except Exception as e:
        print(f"\n✗ 更新出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.close_browser()
        scraper.close_db()

    # 同步更新开奖SP值(与match表同一比赛日期), 500网站有EdgeOne人机验证, 内部会另起无头浏览器
    update_result_sp(target_date)

    # 主流程失败时返回非零退出码, 便于run_daily.sh/定时任务监控发现失败
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
