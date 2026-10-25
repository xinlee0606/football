# sp_scraper.py - 北单开奖SP值回填爬虫(数据源: 500彩票网)
# 数据页: https://zx.500.com/zqdc/kaijiang.php?playid=3&expect={期号}
# 反爬: 2026-08底起站点接入腾讯云EdgeOne, 首次访问返回"Security Verification"人机验证页
#   (iframe内"确认您是真人"复选框), requests直连只能拿到JS挑战页;
#   故改用无头浏览器(Selenium)加载: 隐藏webdriver特征+去UA中HeadlessChrome字样,
#   自动勾选复选框过验证, 通过后站点下发cookie, 同一浏览器会话内后续请求直接放行
# 依赖: 需Chrome/Chromium + chromedriver(与scraper.py同一环境), 无头模式即可
# 期号格式 YYMMN: 2位年份 + 2位月份 + 1位月内期序, 如 25041 = 2025年4月第1期
#   每月期数不固定(一般每周2期: 周五~周一 / 周二~周四), 月内期序一般为1~9
#   实测1年以上的旧期号仍可直接访问(如下拉框只显示最近30期, 属页面展示限制)
# 功能:
#   - 支持期号区间 / 日期区间 / 默认近365天
#   - 断点续传: 期号+场次唯一键去重; 已有数据且SP值一致则整期跳过
#   - 自动推算比赛日期(含跨年修正), 供后续匹配match表
#   - --match 将已入库SP记录与match表匹配回填match_id
#     匹配策略两级: ①北单场次号与match表single_match编号精确对应(首选)
#                   ②队名模糊+比分一致(兜底)
#   - update_by_date(date) 按比赛日期增量更新(供update_match_status.py每日串联调用)
# 用法:
#   python -u sp_scraper.py                       # 默认爬取近365天
#   python -u sp_scraper.py 25041 26077           # 指定期号区间(含首尾)
#   python -u sp_scraper.py 20250401 20250430     # 指定日期区间(按期过滤)
#   python -u sp_scraper.py --match               # 只做match_id匹配(不爬取)
#   python -u sp_scraper.py 26077 26077 --match   # 爬取后紧接着匹配
import re
import sys
import time
from datetime import datetime, timedelta

from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
import pymysql
from bs4 import BeautifulSoup

from config import DB_CONFIG

# ==================== 配置 ====================
PAGE_URL = 'https://zx.500.com/zqdc/kaijiang.php?playid=3&expect={}'
PAGE_DELAY = 1.5          # 期与期间的间隔秒数(浏览器加载本身已含等待)
MATCH_DAYS_WINDOW = 1     # 匹配match表时日期±N天窗口
CONTENT_WAIT = 30         # 单页等待内容出现的最长秒数
MAX_SOLVE = 1             # 单页最多勾选人机验证次数(重复点击会取消勾选, 靠外层重试)

# 隐藏自动化特征(页面JS会读navigator.webdriver判定机器人)
STEALTH_JS = (
    "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
    "window.chrome={runtime:{}};"
    "Object.defineProperty(navigator,'languages',{get:()=>['zh-CN','zh']});"
)

DDL = """
CREATE TABLE IF NOT EXISTS `result_sp` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `issue_no` VARCHAR(10) NOT NULL COMMENT '北单期号 如26077',
    `match_no` VARCHAR(10) NOT NULL COMMENT '期内场次序号',
    `league` VARCHAR(50) DEFAULT NULL COMMENT '赛事类型',
    `match_time` VARCHAR(20) DEFAULT NULL COMMENT '比赛时间 MM-DD HH:MM',
    `match_date` VARCHAR(10) DEFAULT NULL COMMENT '推算比赛日期 YYYYMMDD',
    `home_team` VARCHAR(100) DEFAULT NULL COMMENT '主队(500网名称)',
    `handicap` VARCHAR(10) DEFAULT NULL COMMENT '让球',
    `away_team` VARCHAR(100) DEFAULT NULL COMMENT '客队(500网名称)',
    `score` VARCHAR(20) DEFAULT NULL COMMENT '赛果比分',
    `sp_value` DECIMAL(10,2) DEFAULT NULL COMMENT '开奖SP值(比分玩法)',
    `euro_odds` VARCHAR(30) DEFAULT NULL COMMENT '平均欧赔 胜|平|负',
    `match_id` VARCHAR(20) DEFAULT NULL COMMENT '匹配match表后回填',
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_issue_match (`issue_no`, `match_no`),
    INDEX idx_match_id (`match_id`),
    INDEX idx_match_date (`match_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='北单开奖SP值(500彩票网)'
"""


# ==================== 工具函数 ====================
SCORE_RE = re.compile(r'^\d+-\d+$')


def norm_score(s):
    """比分统一为 x-y 格式: '1:1' -> '1-1'; 无效比分(如'-','-1:-1','改期')返回None"""
    s = (s or '').replace(':', '-').strip()
    return s if SCORE_RE.match(s) else None


def norm_team(name):
    """队名归一化: 去空格/全角字符, 便于跨站比较"""
    return re.sub(r'\s+', '', (name or '').replace('　', ''))


def team_similar(a, b):
    """模糊队名匹配: 子串包含 或 字符重叠度>=60% (与项目现有规则一致)"""
    a, b = norm_team(a), norm_team(b)
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    common = len(set(a) & set(b))
    return common / max(len(set(a)), len(set(b))) >= 0.6


def infer_match_date(issue_no, match_time):
    """由期号编码年份 + 页面'MM-DD HH:MM'推算比赛日期YYYYMMDD
    期号格式YYMMN: 前2位=年份, 第3~4位=月份, 第5位=月内期序
    跨年修正: 1月期可能含上年12月底比赛; 12月期可能含次年1月初比赛"""
    m = re.match(r'(\d{2})-(\d{2})', match_time or '')
    if not m:
        return None
    year = 2000 + int(issue_no[:2])
    issue_month = int(issue_no[2:4])
    month, day = int(m.group(1)), int(m.group(2))
    if issue_month == 1 and month == 12:
        year -= 1
    elif issue_month == 12 and month == 1:
        year += 1
    try:
        return datetime(year, month, day).strftime('%Y%m%d')
    except ValueError:
        return None


def generate_issues(start, end):
    """生成[start, end]内全部合法期号(升序)
    期号=YYMM+月内期序(1~9); 按月推进, 不存在的期由爬取时的空期状态消化"""
    sy, sm = int(start[:2]), int(start[2:4])
    ey, em = int(end[:2]), int(end[2:4])
    ss = min(max(int(start[4]), 1), 9)
    es = min(max(int(end[4]), 1), 9)
    issues = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        lo = ss if (y, m) == (sy, sm) else 1
        hi = es if (y, m) == (ey, em) else 9
        for s in range(lo, hi + 1):
            issues.append(f'{y:02d}{m:02d}{s}')
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return issues


# ==================== 爬虫 ====================
class ResultSpScraper:
    def __init__(self):
        self.db = pymysql.connect(**DB_CONFIG)
        self._browser = None   # 懒加载TitanScraper实例(浏览器会话)

    def close(self):
        try:
            self.db.close()
        except Exception:
            pass
        self._close_browser()

    def ensure_table(self):
        cursor = self.db.cursor()
        cursor.execute(DDL)
        self.db.commit()
        cursor.close()
        print("✓ result_sp 表就绪")

    # ---------- 浏览器会话(懒加载/自愈重建) ----------
    def _init_driver(self, user_agent=None):
        """启动一个无头浏览器(复用scraper.py的Chrome初始化,
        含服务器Chromium路径检测/chromedriver检测/加载超时), 并做反爬适配:
        - 屏蔽AutomationControlled特征 + 隐藏navigator.webdriver
        - 仅拦截图片, 保留css/字体(验证码组件依赖样式才能拿到正确复选框坐标)
        - user_agent: 启动参数级UA(必须启动时指定, CDP覆盖无效)
        返回TitanScraper实例"""
        from scraper import TitanScraper
        browser = TitanScraper()
        args = ['--disable-blink-features=AutomationControlled', '--lang=zh-CN',
                '--window-size=1366,900']
        if user_agent:
            args.append(f'--user-agent={user_agent}')
        browser.init_browser(headless=True, extra_args=args)
        driver = browser.driver
        try:
            driver.execute_cdp_cmd('Network.enable', {})
            driver.execute_cdp_cmd('Network.setBlockedURLs', {'urls': [
                '*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.ico', '*.svg', '*.webp']})
            driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {'source': STEALTH_JS})
        except WebDriverException as e:
            print(f"⚠ 反爬参数设置失败(被浏览器版本忽略): {str(e).splitlines()[0]}")
        return browser

    def _get_driver(self):
        """首次使用时初始化浏览器: 先启一次探测真实UA, 再把UA里的
        HeadlessChrome换成Chrome重启(EdgeOne见到HeadlessChrome字样则勾选无效)
        UA保留原OS标识, 避免Windows UA跑在Linux浏览器上形成新的特征矛盾"""
        if self._browser is None:
            probe = self._init_driver()
            try:
                ua = probe.driver.execute_script('return navigator.userAgent')
            finally:
                try:
                    probe.close_browser()
                except Exception:
                    pass
            self._browser = self._init_driver(user_agent=ua.replace('HeadlessChrome', 'Chrome'))
            print("✓ 已启用反爬浏览器会话")
        return self._browser.driver

    def _close_browser(self):
        if self._browser is not None:
            try:
                self._browser.close_browser()
            except Exception:
                pass
            self._browser = None

    # ---------- 人机验证 ----------
    @staticmethod
    def _has_content(html):
        """页面已含开奖表格或期号下拉框 -> 验证已通过(空期只有下拉框)"""
        return 'ld_table' in html or '<option value="' in html

    def _pass_challenge(self, driver):
        """勾选EdgeOne"确认您是真人"复选框(位于跨域iframe内)
        用CDP真实鼠标事件序列(移动->按下->释放)点击, 避免被判定为脚本点击
        返回True表示执行了勾选"""
        try:
            frames = driver.find_elements(By.CSS_SELECTOR, 'iframe[src*="captcha"]')
            if not frames:
                return False
            iframe = frames[0]
            frame_pos = iframe.rect          # 必须在切进子帧前取(主文档元素)
            driver.switch_to.frame(iframe)
            boxes = driver.find_elements(By.ID, 'verifyCheckbox')
            if not boxes:
                return False
            box = boxes[0].rect
            if not box['width'] or not box['height']:
                return False
            cx = frame_pos['x'] + box['x'] + box['width'] / 2
            cy = frame_pos['y'] + box['y'] + box['height'] / 2
            # CDP鼠标坐标基于顶层页面视口, 派发前先切回主文档上下文
            driver.switch_to.default_content()
            for etype, buttons in (('mouseMoved', 0), ('mousePressed', 1), ('mouseReleased', 0)):
                driver.execute_cdp_cmd('Input.dispatchMouseEvent', {
                    'type': etype, 'x': cx, 'y': cy, 'button': 'left',
                    'clickCount': 1, 'buttons': buttons})
                time.sleep(0.15)
            print("  ✓ 已勾选人机验证(EdgeOne Security Verification)")
            return True
        except WebDriverException as e:
            print(f"  ⚠ 勾选人机验证失败: {str(e).splitlines()[0]}")
            return False
        finally:
            try:
                driver.switch_to.default_content()
            except Exception:
                pass

    # ---------- 页面抓取与解析 ----------
    def fetch_page(self, issue_no, retries=3):
        """浏览器加载页面并返回HTML; 遇到人机验证页自动勾选后等待跳转
        返回None表示失败(重试耗尽)"""
        url = PAGE_URL.format(issue_no)
        for i in range(1, retries + 1):
            try:
                driver = self._get_driver()
                driver.get(url)
                solved = 0
                for _ in range(CONTENT_WAIT):
                    try:
                        html = driver.page_source or ''
                    except WebDriverException:
                        time.sleep(1)   # 页面正在跳转/刷新, 等下一轮
                        continue
                    if self._has_content(html):
                        return html
                    if solved < MAX_SOLVE and 'captcha' in html.lower():
                        # 验证码组件渲染完成后才点, 且一次页面加载只点一次
                        time.sleep(2)
                        if self._pass_challenge(driver):
                            solved += 1
                    time.sleep(1)
                print(f"  ⚠ 页面无开奖数据(验证未通过或被拦截?) (第{i}次)")
            except WebDriverException as e:
                print(f"  ⚠ 页面加载失败(第{i}次): {str(e).splitlines()[0]}")
                self._close_browser()   # 会话可能已失效, 下次重试时重建
            time.sleep(2 * i)
        return None

    @staticmethod
    def parse_issue_list(html):
        """解析页面下拉框中的全部可查期号(降序返回)"""
        opts = re.findall(r'<option value="(\d+)"', html)
        return sorted(set(opts), reverse=True)

    @staticmethod
    def parse_rows(html, issue_no):
        """解析开奖表格 ld_table, 返回场次字典列表
        页面每行固定15个td: 0场次 1赛事 2时间 3主队 4让球 5客队 6比分 7赛果
        8空 9SP 10空 11胜 12平 13负 14空
        注意: 若过滤空单元格, 未开赛(赛果/SP为空)的行会整体左移导致列错位
        (比分列误存欧赔胜值), 因此按固定下标取值"""
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table', class_='ld_table')
        if not table:
            return []
        rows = []
        for tr in table.find_all('tr')[1:]:  # 跳过表头
            cells = [td.get_text(strip=True) for td in tr.find_all('td')]
            if not cells or not cells[0].isdigit():
                continue
            if len(cells) >= 14:
                # 固定列结构(含空占位td)
                league, mtime = cells[1], cells[2]
                home, handicap, away = cells[3], cells[4], cells[5]
                result, sp_txt = cells[7], cells[9]
                euro_cells = cells[11:14]
            else:
                # 兜底: 异常结构按旧逻辑过滤空格
                cells = [c for c in cells if c != '']
                if len(cells) < 9:
                    continue
                league, mtime = cells[1], cells[2]
                home, handicap, away = cells[3], cells[4], cells[5]
                result, sp_txt = cells[7], cells[8]
                euro_cells = cells[9:12]
            sp = None
            try:
                sp = float(sp_txt)
            except ValueError:
                pass
            euro = '|'.join(euro_cells) if any(euro_cells) else None
            rows.append({
                'issue_no': issue_no,
                'match_no': cells[0],
                'league': league,
                'match_time': mtime,
                'match_date': infer_match_date(issue_no, mtime),
                'home_team': home,
                'handicap': handicap,
                'away_team': away,
                'score': norm_score(result),
                'sp_value': sp,
                'euro_odds': euro,
            })
        return rows

    # ---------- 断点续传判定 ----------
    def existing_issue_data(self, issue_no):
        """库中该期已有数据: {match_no: sp_value}"""
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT match_no, sp_value FROM result_sp WHERE issue_no=%s", (issue_no,))
        data = {mn: (float(sp) if sp is not None else None) for mn, sp in cursor.fetchall()}
        cursor.close()
        return data

    def crawl_issue(self, issue_no):
        """爬取一期. 返回状态: skip(已有且正确)/saved(新入库)/updated(有变更)/empty/failed"""
        html = self.fetch_page(issue_no)
        if html is None:
            return 'failed', 0
        rows = self.parse_rows(html, issue_no)
        if not rows:
            return 'empty', 0

        existing = self.existing_issue_data(issue_no)
        # 已有数据且完全一致(条数+每场SP) -> 跳过
        if len(existing) == len(rows) and all(
                r['match_no'] in existing and existing[r['match_no']] == r['sp_value']
                for r in rows):
            return 'skip', len(rows)

        cursor = self.db.cursor()
        sql = ("INSERT INTO result_sp (issue_no, match_no, league, match_time, match_date, "
               "home_team, handicap, away_team, score, sp_value, euro_odds) "
               "VALUES (%(issue_no)s, %(match_no)s, %(league)s, %(match_time)s, %(match_date)s, "
               "%(home_team)s, %(handicap)s, %(away_team)s, %(score)s, %(sp_value)s, %(euro_odds)s) "
               "ON DUPLICATE KEY UPDATE league=VALUES(league), match_time=VALUES(match_time), "
               "match_date=VALUES(match_date), home_team=VALUES(home_team), handicap=VALUES(handicap), "
               "away_team=VALUES(away_team), score=VALUES(score), sp_value=VALUES(sp_value), "
               "euro_odds=VALUES(euro_odds)")
        for r in rows:
            cursor.execute(sql, r)
        self.db.commit()
        cursor.close()
        return ('updated' if existing else 'saved'), len(rows)

    def crawl(self, issues):
        """按期号列表爬取(升序), 返回汇总"""
        total = dict(saved=0, updated=0, skip=0, empty=0, failed=0)
        n = len(issues)
        for i, issue in enumerate(issues, 1):
            status, cnt = self.crawl_issue(issue)
            total[status] += 1
            marks = {'skip': '⊘ 已有且一致,跳过', 'saved': '✓ 新入库',
                     'updated': '↻ 有变更,已更新', 'empty': '- 空期(无数据)',
                     'failed': '✗ 抓取失败'}
            print(f"[{i}/{n}] {issue}期: {marks[status]} ({cnt}场)")
            time.sleep(PAGE_DELAY)
        return total

    # ---------- match_id 匹配 ----------
    @staticmethod
    def _single_match_no(single_match):
        """从match表single_match字段(如'单场5'/'单场005')提取场次数字, 失败返回None"""
        m = re.search(r'\d+', single_match or '')
        return int(m.group()) if m else None

    @staticmethod
    def _cand_datetime(match_id, mtime):
        """match表候选的开球时间: match_id前8位为年月+大致日期, match_time如'16日03:00'"""
        m = re.match(r'(\d{1,2})日\s*(\d{1,2}):(\d{2})', mtime or '')
        if not m or not match_id or len(match_id) < 8:
            return None
        try:
            base = datetime.strptime(match_id[:8], '%Y%m%d')
        except ValueError:
            return None
        day, hh, mm = int(m.group(1)), int(m.group(2)), int(m.group(3))
        best = None
        for k in range(-3, 4):  # match_id日期与实际开球日可能相差几天
            cand = base + timedelta(days=k)
            if cand.day == day:
                if best is None or abs(k) < abs((best - base).days):
                    best = cand
        if best is None:
            return None
        try:
            return best.replace(hour=hh, minute=mm)
        except ValueError:
            return None

    def match_to_db(self, verbose=True):
        """将 result_sp 未匹配记录与 match 表关联回填 match_id
        两级匹配:
          ① 北单场次号 == match表single_match编号(日期±窗口内唯一即命中)
             500网站与北单官方场次编号一致, 不受两站译名简称差异影响;
             跨期同编号多候选时依次用队名/比分、开球时间就近消歧
          ② 队名模糊+比分一致(兜底)
        Args:
            verbose: True时逐条打印未匹配明细(--match模式); False仅输出汇总(定时任务)"""
        cursor = self.db.cursor()
        cursor.execute("SELECT id, match_no, match_date, match_time, home_team, away_team, score "
                       "FROM result_sp WHERE match_id IS NULL AND match_date IS NOT NULL")
        pending = cursor.fetchall()
        cursor.close()
        if not pending:
            print("匹配: 无待匹配记录")
            return 0, 0

        matched = ambiguous = 0
        cand_cache = {}  # 日期前缀 -> 候选列表缓存, 避免重复查库
        cursor = self.db.cursor()
        for rid, mno, mdate, mtime, home, away, score in pending:
            score = norm_score(score)  # 历史脏数据('2.17'等)归一为None
            # 日期±窗口内候选
            d0 = datetime.strptime(mdate, '%Y%m%d')
            cands = []
            for k in range(-MATCH_DAYS_WINDOW, MATCH_DAYS_WINDOW + 1):
                dp = (d0 + timedelta(days=k)).strftime('%Y%m%d')
                if dp not in cand_cache:
                    cursor.execute(
                        "SELECT match_id, single_match, home_team, away_team, score, match_time "
                        "FROM `match` WHERE match_id LIKE %s", (dp + '%',))
                    cand_cache[dp] = cursor.fetchall()
                cands.extend(cand_cache[dp])
            cdict = {c[0]: c for c in cands}

            hit = None
            # ---- 第一级: 场次编号精确匹配 ----
            if mno and mno.isdigit():
                # 先在精确日期内找(避免跨期同编号误配), 找不到再扩到窗口内
                no_hits = [c[0] for c in cands
                           if c[0].startswith(mdate) and self._single_match_no(c[1]) == int(mno)]
                if not no_hits:
                    no_hits = [c[0] for c in cands if self._single_match_no(c[1]) == int(mno)]
                no_hits = list(dict.fromkeys(no_hits))
                if len(no_hits) == 1:
                    hit = no_hits[0]
                elif len(no_hits) > 1:
                    # 消歧①: 队名双命中 或 比分一致
                    refine = [mid for mid in no_hits
                              if (team_similar(home, cdict[mid][2])
                                  and team_similar(away, cdict[mid][3]))
                              or (score and norm_score(cdict[mid][4]) == score)]
                    # 消歧②: 开球时间就近(同期编号按时间排序, 跨期同号相差约一周)
                    tm = re.search(r'(\d{2}):(\d{2})', mtime or '')
                    if tm:
                        sp_dt = d0.replace(hour=int(tm.group(1)), minute=int(tm.group(2)))
                        pool = refine if refine else no_hits
                        near = [(mid, abs((self._cand_datetime(mid, cdict[mid][5]) - sp_dt)
                                            .total_seconds()))
                                for mid in pool
                                if self._cand_datetime(mid, cdict[mid][5])]
                        near = [x for x in near if x[1] <= 2 * 3600]
                        if len(near) == 1:
                            hit = near[0][0]
                    if hit is None and len(refine) == 1:
                        hit = refine[0]
                    if hit is None:
                        ambiguous += 1
                        if verbose:
                            print(f"  ⚠ 编号多候选: {mdate} 场次{mno} {home} vs {away} -> {no_hits}")
                        continue

            # ---- 第二级: 队名模糊+比分一致(兜底) ----
            if hit is None:
                hits = []
                for mid, _sm, mh, ma, ms, _mt in cands:
                    if team_similar(home, mh) and team_similar(away, ma) \
                            and score and norm_score(ms) == score:
                        hits.append(mid)
                hits = list(dict.fromkeys(hits))
                if len(hits) == 1:
                    hit = hits[0]
                else:
                    ambiguous += 1
                    if verbose:
                        print(f"  ⚠ {'多候选' if hits else '无候选'}: {mdate} {home} vs {away} "
                              f"{score} -> {hits if hits else '未找到'}")
                    continue

            cursor.execute("UPDATE result_sp SET match_id=%s WHERE id=%s", (hit, rid))
            matched += 1
        self.db.commit()
        cursor.close()
        print(f"匹配完成: 成功{matched}条, 待人工核查{ambiguous}条")
        return matched, ambiguous

    # ---------- 每日增量更新(供 update_match_status.py 串联) ----------
    def update_by_date(self, target_date):
        """按比赛日期增量更新result_sp: 爬取含该日期比赛的全部期号(断点续传自动跳过
        已有且一致的数据), 随后执行match_id匹配回填
        Args:
            target_date: 比赛日期YYYYMMDD, 与update_match_status更新match表的日期一致
        Returns:
            (爬取汇总dict, 匹配成功数, 待核查数)"""
        print("\n" + "=" * 60)
        print(f"同步更新 result_sp (500网开奖SP): 比赛日期 {target_date}")
        print("=" * 60)
        issues = build_issue_list(self, [target_date, target_date])
        total = dict(saved=0, updated=0, skip=0, empty=0, failed=0)
        if issues:
            print(f"含{target_date}比赛的期号: {issues}")
            total = self.crawl(issues)
        else:
            print(f"未找到含 {target_date} 比赛的期号(未开奖或无此日期北单赛事)")
        matched, ambiguous = self.match_to_db(verbose=False)
        return total, matched, ambiguous


# ==================== 参数与主流程 ====================
def is_issue_arg(s):
    return s.isdigit() and len(s) == 5


def is_date_arg(s):
    return s.isdigit() and len(s) == 8


def build_issue_list(scraper, argv):
    """根据参数生成待爬期号列表(升序)"""
    # 抓默认页拿最新期号(下拉框), 避免尝试尚未开售的未来期号
    probe_html = scraper.fetch_page('') or ''
    available = scraper.parse_issue_list(probe_html)
    newest = available[0] if available else None
    if newest:
        print(f"站点最新期号: {newest}")

    # ---- 期号区间模式: 按期号编码(YYMMN)按月枚举 ----
    if len(argv) >= 2 and is_issue_arg(argv[0]) and is_issue_arg(argv[1]):
        lo, hi = sorted([argv[0], argv[1]])
        issues = generate_issues(lo, hi)
        if newest:
            issues = [i for i in issues if i <= newest]
        return issues

    # ---- 日期区间模式 / 默认近365天 ----
    if len(argv) >= 2 and is_date_arg(argv[0]) and is_date_arg(argv[1]):
        d_start, d_end = sorted([argv[0], argv[1]])
    else:
        today = datetime.now()
        d_end = today.strftime('%Y%m%d')
        d_start = (today - timedelta(days=365)).strftime('%Y%m%d')
        print(f"默认爬取区间: 近365天 {d_start} ~ {d_end}")

    ds = datetime.strptime(d_start, '%Y%m%d')
    # 从开始日期前1个月枚举起(单期可能跨月), 逐期抓取判断日期是否重叠
    y, m = ds.year, ds.month - 1
    if m < 1:
        m, y = 12, y - 1
    start_issue = f'{y - 2000:02d}{m:02d}1'
    if newest:
        end_issue = newest
    else:
        now = datetime.now()
        end_issue = f'{now.year - 2000:02d}{now.month:02d}9'

    issues = []
    for issue in generate_issues(start_issue, end_issue):
        html = scraper.fetch_page(issue)
        if not html:
            continue
        rows = scraper.parse_rows(html, issue)
        dates = [r['match_date'] for r in rows if r['match_date']]
        if not dates:
            continue                     # 空期(该月无此期序)
        if max(dates) < d_start:
            continue                     # 整期早于区间
        if min(dates) > d_end:
            break                        # 已越过区间(升序可提前结束)
        issues.append(issue)
        time.sleep(PAGE_DELAY)
    return issues


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    do_match = '--match' in sys.argv[1:]

    scraper = ResultSpScraper()
    try:
        scraper.ensure_table()
        if not do_match or args:
            issues = build_issue_list(scraper, args)
            if issues is None:
                return
            print(f"待爬期号: {issues[0] if issues else '-'} ~ {issues[-1] if issues else '-'} "
                  f"共{len(issues)}期")
            if issues:
                total = scraper.crawl(issues)
                print("=" * 50)
                print(f"爬取汇总: 新入库{total['saved']}期 更新{total['updated']}期 "
                      f"跳过{total['skip']}期 空期{total['empty']} 失败{total['failed']}")
        if do_match:
            print("=" * 50)
            scraper.match_to_db()
    finally:
        scraper.close()


if __name__ == '__main__':
    main()
