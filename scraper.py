# scraper.py - 新球体育北单比赛数据爬虫
# 功能：爬取比赛列表、分析数据、技术统计
import os
import re
import sys
import time
import shutil
import zipfile
import tempfile
import pymysql
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from config import DB_CONFIG, CHROME_HEADLESS, ELEMENT_WAIT_TIMEOUT, REQUEST_DELAY, MATCH_LIST_WAIT, MAIN_URL, TEAM_NAME_LANGUAGE, OVER_URL_TEMPLATE, CHROME_BINARY_PATH, PAGE_LOAD_TIMEOUT, CHROME_PROXY

# Linux服务器强制Selenium Manager离线模式(无法联网下载驱动); Windows本地开发不设置(需联网获取驱动)
if sys.platform.startswith('linux'):
    os.environ.setdefault('SE_MANAGER_OFFLINE', 'true')


class TitanScraper:
    def __init__(self):
        self.db = None
        self.driver = None
        self.today = datetime.now().strftime('%Y%m%d')
        self.match_counter = 0  # 单场计数器
        self._tmp_user_dir = None  # 临时用户数据目录
        self._conn_fail_streak = 0  # 连续连接失败计数（用于IP被封熔断）
        self._proxy_ext_dir = None  # 代理认证插件临时目录

    # ==================== 浏览器管理 ====================
    @staticmethod
    def _detect_browser_binary():
        """自动检测Chrome/Chromium可执行文件(主要用于Linux服务器)"""
        for name in ('chromium-browser', 'chromium', 'google-chrome', 'google-chrome-stable'):
            path = shutil.which(name)
            if path:
                return path
        return None

    @staticmethod
    def _detect_chromedriver_service():
        """优先使用系统安装的chromedriver(服务器无法联网自动下载驱动)
        未安装时返回None, 由Selenium Manager接管(本地Windows开发环境)"""
        path = shutil.which('chromedriver')
        if not path:
            # cron等极简PATH环境(不含/usr/local/bin)下which()会漏检,
            # 落到Selenium Manager需联网下载驱动在内网服务器直接失败, 故回退探测常见绝对路径
            for cand in ('/usr/local/bin/chromedriver', '/usr/bin/chromedriver'):
                if os.path.isfile(cand) and os.access(cand, os.X_OK):
                    path = cand
                    break
        return ChromeService(executable_path=path) if path else None

    def init_browser(self, proxy=None, headless=None, extra_args=None):
        """初始化Chrome浏览器
        Args:
            proxy: 代理地址, 默认取配置CHROME_PROXY
            headless: None=沿用配置CHROME_HEADLESS(行为不变);
                      True=强制无头且使用新版--headless=new
                      (旧版--headless的UA含HeadlessChrome, 易被站点WAF识别拦截)
            extra_args: 额外Chrome启动参数(反检测等场景)"""
        options = ChromeOptions()
        if headless is None:
            if CHROME_HEADLESS:
                options.add_argument('--headless')
        elif headless:
            options.add_argument('--headless=new')
        for arg in (extra_args or []):
            options.add_argument(arg)
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        # 页面加载策略改为eager：DOM解析完成即返回，不等待图片/广告/iframe等慢资源，
        # 否则部分页面资源永远加载不完会导致 driver.get() 一直阻塞直至通信超时(120秒)
        options.page_load_strategy = 'eager'
        # 代理：服务器出口IP被目标网站限制(连接超时)时，需通过代理访问
        # 支持格式: http://ip:port / socks5://ip:port / http://user:pass@ip:port
        proxy = proxy or CHROME_PROXY
        if proxy:
            proxy = self._apply_proxy(options, proxy)
        # Linux服务器: 指定Chromium路径(配置优先, 其次自动检测)
        binary = CHROME_BINARY_PATH or self._detect_browser_binary()
        if binary:
            options.binary_location = binary
            print(f"✓ 浏览器路径: {binary}")
        service = self._detect_chromedriver_service()
        try:
            self.driver = webdriver.Chrome(options=options, service=service) \
                if service else webdriver.Chrome(options=options)
        except WebDriverException as e:
            print(f"✗ 浏览器启动失败: {str(e).splitlines()[0]}")
            print("  请确认已安装: chromium + chromedriver (CentOS: sudo dnf install -y chromium chromium-driver)")
            raise
        self.driver.implicitly_wait(5)
        # 应用页面加载超时（未设置时get()会无限等待，触发chromedriver 120秒通信超时报错）
        self.driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
        # 拦截图片/样式/字体等无关资源：爬取只需HTML表格内容，
        # 分析页含大量图片，服务器带宽有限时会导致页面30秒都加载不完
        try:
            self.driver.execute_cdp_cmd("Network.enable", {})
            self.driver.execute_cdp_cmd("Network.setBlockedURLs", {
                "urls": ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.ico", "*.svg",
                         "*.webp", "*.css", "*.woff", "*.woff2", "*.ttf"]
            })
        except Exception as e:
            print(f"⚠ 设置资源拦截失败: {e}")
        print("✓ 浏览器初始化成功")
        # 保存初始化参数，供tab崩溃时重启浏览器复用
        self._browser_proxy = proxy
        self._browser_headless = headless
        self._browser_extra_args = extra_args or []

    def _restart_browser(self):
        """Tab崩溃时重启浏览器（复用初始化参数），释放内存后继续爬取"""
        print("    ⚠ 检测到浏览器tab崩溃，正在重启浏览器释放内存...")
        try:
            self.driver.quit()
        except Exception:
            pass
        self.driver = None
        time.sleep(3)
        self.init_browser(proxy=self._browser_proxy, headless=self._browser_headless,
                          extra_args=self._browser_extra_args)
        print("    ✓ 浏览器已重启，继续爬取")

    def close_browser(self):
        """关闭浏览器并清理"""
        if self.driver:
            self.driver.quit()
            print("✓ 浏览器已关闭")
        # 清理代理认证插件临时目录(仅非--headless模式生成)
        if self._proxy_ext_dir:
            shutil.rmtree(self._proxy_ext_dir, ignore_errors=True)
            try:
                os.remove(self._proxy_ext_dir + '.crx')
            except OSError:
                pass
            self._proxy_ext_dir = None

    def _apply_proxy(self, options, proxy):
        """将代理配置应用到浏览器选项，返回规范后的代理地址"""
        # 解析代理地址，支持 http://user:pass@ip:port 格式
        m = re.match(r'^(?:(?P<scheme>https?|socks5)://)?(?:(?P<user>[^:@]+):(?P<pass>[^@]*)@)?(?P<host>[^:/]+)(?::(?P<port>\d+))?$', proxy)
        if not m or not m.group('host'):
            print(f"⚠ 代理格式无效: {proxy}，已忽略。支持格式: http://ip:port 或 http://user:pass@ip:port")
        elif m.group('user'):
            # 带账号密码的代理：Chrome的--proxy-server不支持认证，
            # 通过加载浏览器插件自动填充认证信息实现(仅非--headless模式支持)
            self._build_proxy_auth_extension(options, m.group('scheme') or 'http',
                                             m.group('host'), m.group('port') or '80',
                                             m.group('user'), m.group('pass') or '')
            print(f"✓ 使用代理(账号认证): {m.group('host')}:{m.group('port') or '80'}")
            return proxy
        else:
            scheme = m.group('scheme') or 'http'
            proxy_addr = f"{scheme}://{m.group('host')}:{m.group('port') or '80'}"
            options.add_argument(f'--proxy-server={proxy_addr}')
            print(f"✓ 使用代理: {proxy_addr}")
            return proxy_addr
        return None

    def _build_proxy_auth_extension(self, options, scheme, host, port, username, password):
        """构建代理认证浏览器插件并加载(用于账号密码型代理)"""
        manifest = {
            "version": "1.0.0",
            "manifest_version": 2,
            "name": "Proxy Auth",
            "permissions": ["proxy", "tabs", "unlimitedStorage", "storage",
                             "<all_urls>", "webRequest", "webRequestBlocking"],
            "background": {"scripts": ["background.js"]},
            "minimum_chrome_version": "22.0.0"
        }
        background_js = f"""
        var config = {{
            mode: "fixed_servers",
            rules: {{
                singleProxy: {{ scheme: "{scheme}", host: "{host}", port: parseInt({port}) }},
                bypassList: ["localhost"]
            }}
        }};
        chrome.proxy.settings.set({{value: config, scope: "regular"}}, function() {{}});
        function callbackFn(details) {{
            return {{ authCredentials: {{ username: "{username}", password: "{password}" }} }};
        }}
        chrome.webRequest.onAuthRequired.addListener(callbackFn,
            {{urls: ["<all_urls>"]}}, ['blocking']);
        """
        ext_dir = tempfile.mkdtemp(prefix='proxy_auth_')
        import json
        with open(os.path.join(ext_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
            json.dump(manifest, f)
        with open(os.path.join(ext_dir, 'background.js'), 'w', encoding='utf-8') as f:
            f.write(background_js)
        # 打包为zip并加载（headless模式下插件不可用，必须移除--headless参数）
        zip_path = ext_dir + '.crx'
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.write(os.path.join(ext_dir, 'manifest.json'), 'manifest.json')
            zf.write(os.path.join(ext_dir, 'background.js'), 'background.js')
        options.add_extension(zip_path)
        self._proxy_ext_dir = ext_dir
        # 移除headless：带认证插件的代理在headless模式下无法工作，
        # 用 --disable-gpu + Xvfb 环境或直接带窗口运行(服务器无显示器时配合Xvfb)
        if '--headless' in options.arguments:
            options.arguments.remove('--headless')
            print("  ⚠ 已自动移除--headless（代理认证插件需要非headless模式，服务器可配合Xvfb运行）")

    # ==================== 数据库管理 ====================
    def init_db(self):
        """初始化数据库连接"""
        self.db = pymysql.connect(**DB_CONFIG)
        print("✓ 数据库连接成功")

    def close_db(self):
        """关闭数据库连接"""
        if self.db:
            self.db.close()
            print("✓ 数据库连接已关闭")

    def save_match(self, data):
        """保存比赛基本信息到match表（含去重：基于site_match_id）"""
        cursor = self.db.cursor()
        # 检查是否已存在相同site_match_id的记录
        cursor.execute("SELECT match_id FROM `match` WHERE site_match_id = %s", (data.get('site_match_id'),))
        existing = cursor.fetchone()
        if existing:
            # 已存在，复用原有match_id，更新基本信息
            match_id = existing[0]
            cursor.execute(
                "UPDATE `match` SET league=%s, single_match=%s, sid=%s, match_time=%s, status=%s, "
                "home_team=%s, home_rank=%s, away_team=%s, away_rank=%s, score=%s, "
                "analysis_url=%s, detail_url=%s WHERE match_id=%s",
                (data['league'], data.get('single_match'), data.get('sid'), data['match_time'], data.get('status'),
                 data['home_team'], data.get('home_rank'), data['away_team'], data.get('away_rank'),
                 data.get('score'), data.get('analysis_url'), data.get('detail_url'), match_id))
            self.db.commit()
            cursor.close()
            return match_id, True  # 返回match_id和是否已存在
        else:
            # 新比赛，插入记录
            sql = """INSERT INTO `match` 
                (match_id, site_match_id, sid, league, single_match, match_time, status, home_team, home_rank, away_team, away_rank, score, analysis_url, detail_url)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            cursor.execute(sql, (
                data['match_id'], data.get('site_match_id'), data.get('sid'), data['league'], data.get('single_match'), data['match_time'], data.get('status'),
                data['home_team'], data.get('home_rank'), data['away_team'], data.get('away_rank'),
                data.get('score'), data.get('analysis_url'), data.get('detail_url')
            ))
            self.db.commit()
            cursor.close()
            return data['match_id'], False  # 返回match_id和是否已存在

    def save_league_standings(self, data):
        """保存联赛积分排名"""
        sql = """INSERT INTO league_standings 
            (match_id, team_type, team_name, league_name, scope, played, won, drawn, lost, 
             goals_for, goals_against, goal_diff, points, ranking, win_rate)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        cursor = self.db.cursor()
        for row in data:
            cursor.execute(sql, (
                row['match_id'], row['team_type'], row.get('team_name'), row.get('league_name'),
                row['scope'], row.get('played', 0), row.get('won', 0), row.get('drawn', 0),
                row.get('lost', 0), row.get('goals_for', 0), row.get('goals_against', 0),
                row.get('goal_diff', 0), row.get('points', 0), row.get('ranking', 0),
                row.get('win_rate')
            ))
        self.db.commit()
        cursor.close()

    def save_head_to_head(self, data):
        """保存对赛往绩"""
        sql = """INSERT INTO head_to_head 
            (match_id, match_date, league, home_team, score, away_team, half_score, corners,
             initial_home_odds, initial_handicap, initial_away_odds,
             result, handicap_result, over_under_result)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        cursor = self.db.cursor()
        for row in data:
            cursor.execute(sql, (
                row['match_id'], row.get('match_date'), row.get('league'),
                row.get('home_team'), row.get('score'), row.get('away_team'),
                row.get('half_score'), row.get('corners'),
                row.get('initial_home_odds'), row.get('initial_handicap'),
                row.get('initial_away_odds'),
                row.get('result'), row.get('handicap_result'),
                row.get('over_under_result')
            ))
        self.db.commit()
        cursor.close()

    def save_recent_form(self, data):
        """保存近期战绩（含技术统计）"""
        sql = """INSERT INTO recent_form 
            (match_id, team_type, team_name, match_date, league, home_team, score, away_team, result, detail_url, is_current_match,
             initial_home_odds, initial_handicap, initial_away_odds, handicap_result, over_under_result,
             corners, half_corners, yellow_cards, shots, shots_on_target, attacks, dangerous_attacks,
             shots_off_target, shots_blocked, free_kicks, possession, half_possession, passes, pass_success_rate, fouls, offsides)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        cursor = self.db.cursor()
        for row in data:
            cursor.execute(sql, (
                row['match_id'], row['team_type'], row.get('team_name'),
                row.get('match_date'), row.get('league'), row.get('home_team'),
                row.get('score'), row.get('away_team'), row.get('result'),
                row.get('detail_url'), 0,  # is_current_match=0 (历史战绩)
                row.get('initial_home_odds'), row.get('initial_handicap'),
                row.get('initial_away_odds'),
                row.get('handicap_result'), row.get('over_under_result'),
                row.get('corners'), row.get('half_corners'), row.get('yellow_cards'),
                row.get('shots'), row.get('shots_on_target'), row.get('attacks'),
                row.get('dangerous_attacks'), row.get('shots_off_target'),
                row.get('shots_blocked'), row.get('free_kicks'), row.get('possession'),
                row.get('half_possession'), row.get('passes'), row.get('pass_success_rate'),
                row.get('fouls'), row.get('offsides')
            ))
        self.db.commit()
        cursor.close()
        return cursor.lastrowid  # 返回最后插入的ID

    def save_asian_odds(self, data):
        """保存亚盘数据"""
        sql = """INSERT INTO asian_odds 
            (match_id, company, initial_handicap, initial_home_odds, initial_away_odds,
             current_handicap, current_home_odds, current_away_odds)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""
        cursor = self.db.cursor()
        for row in data:
            cursor.execute(sql, (
                row['match_id'], row.get('company'), row.get('initial_handicap'),
                row.get('initial_home_odds'), row.get('initial_away_odds'),
                row.get('current_handicap'), row.get('current_home_odds'),
                row.get('current_away_odds')
            ))
        self.db.commit()
        cursor.close()

    def save_over_under_odds(self, data):
        """保存大小球数据"""
        sql = """INSERT INTO over_under_odds 
            (match_id, company, initial_line, initial_over_odds, initial_under_odds,
             current_line, current_over_odds, current_under_odds)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""
        cursor = self.db.cursor()
        for row in data:
            cursor.execute(sql, (
                row['match_id'], row.get('company'), row.get('initial_line'),
                row.get('initial_over_odds'), row.get('initial_under_odds'),
                row.get('current_line'), row.get('current_over_odds'),
                row.get('current_under_odds')
            ))
        self.db.commit()
        cursor.close()



    def save_season_stats(self, data):
        """保存赛季统计数据"""
        sql = """INSERT INTO season_stats 
            (match_id, team_type, section, scope, stat_name, stat_value)
            VALUES (%s, %s, %s, %s, %s, %s)"""
        cursor = self.db.cursor()
        for row in data:
            cursor.execute(sql, (
                row['match_id'], row['team_type'], row['section'],
                row['scope'], row['stat_name'], row['stat_value']
            ))
        self.db.commit()
        cursor.close()

    # ==================== 工具方法 ====================
    def extract_analysis_id(self, url):
        """从分析页面URL提取比赛ID"""
        if not url:
            return None
        match = re.search(r'/analysis/(\d+)cn\.htm', url)
        return match.group(1) if match else None

    def extract_detail_id(self, url):
        """从技术统计页面URL提取比赛ID"""
        if not url:
            return None
        match = re.search(r'/detail/(\d+)cn\.htm', url)
        return match.group(1) if match else None

    def safe_find(self, parent, selector, by=By.CSS_SELECTOR):
        """安全查找元素，找不到返回None"""
        try:
            return parent.find_element(by, selector)
        except NoSuchElementException:
            return None

    def safe_find_all(self, parent, selector, by=By.CSS_SELECTOR):
        """安全查找所有元素"""
        try:
            return parent.find_elements(by, selector)
        except NoSuchElementException:
            return []

    def wait_and_click(self, selector, by=By.CSS_SELECTOR, timeout=10):
        """等待元素可点击并点击"""
        try:
            el = WebDriverWait(self.driver, timeout).until(
                EC.element_to_be_clickable((by, selector))
            )
            el.click()
            return True
        except (TimeoutException, WebDriverException):
            return False

    def _stop_loading(self):
        """强制停止当前页面加载，避免浏览器卡在加载中影响后续导航"""
        try:
            self.driver.execute_script("window.stop();")
        except Exception:
            pass

    def _safe_get(self, url, retries=2):
        """安全加载页面：超时后停止加载并重试，重试仍失败则用已加载的DOM继续解析。
        分析页等页面常含加载很慢的资源，服务器带宽有限时容易超时，
        超时后若直接继续会导致DOM不完整(表格缺失)，所以先重试一次。
        当检测到Chrome tab崩溃(内存耗尽)时，自动重启浏览器后重试。
        Returns:
            True=页面可用, False=加载失败(含重试)
        """
        restarted = False  # 防止无限重启
        for attempt in range(1, retries + 1):
            try:
                self.driver.get(url)
                self._conn_fail_streak = 0  # 加载成功，重置连续失败计数
                return True
            except TimeoutException:
                self._stop_loading()
                if attempt < retries:
                    print(f"     页面加载超时({PAGE_LOAD_TIMEOUT}秒)，重试(第{attempt + 1}/{retries}次)...")
                    time.sleep(2)
                    continue
                print(f"     页面加载超时({PAGE_LOAD_TIMEOUT}秒)，已停止加载继续解析")
                return True
            except Exception as e:
                # 包含120秒通信超时等异常：先停止残留加载再重试，避免浏览器卡死
                self._stop_loading()
                err_msg = str(e)
                # 检测tab崩溃（Chrome内存耗尽时driver.get()抛出此异常）
                is_tab_crashed = 'tab crashed' in err_msg or 'crashed' in err_msg.lower()
                if is_tab_crashed and not restarted:
                    print(f"    ⚠ 浏览器tab崩溃(第{attempt}次): {err_msg.splitlines()[0]}")
                    self._restart_browser()
                    restarted = True
                    attempt = 0  # 重置计数器，重启后给予完整重试次数
                    continue
                print(f"    ⚠ 页面加载失败(第{attempt}/{retries}次): {err_msg.splitlines()[0]}")
                # 连接级超时(如ERR_CONNECTION_TIMED_OUT)通常是IP被目标网站限制，
                # 连续多次出现则熔断退出，避免空耗几十分钟无效重试
                if 'ERR_CONNECTION' in err_msg or 'ConnectionRefused' in err_msg:
                    self._conn_fail_streak += 1
                    if self._conn_fail_streak >= 4:
                        raise RuntimeError(
                            "连续多次连接超时，服务器IP可能已被目标网站限制。"
                            "请使用代理重试: python scraper.py --resume --proxy http://代理ip:端口"
                        ) from e
                else:
                    self._conn_fail_streak = 0
                if attempt < retries:
                    time.sleep(3)
        return False

    # ==================== 第一步：爬取比赛列表 ====================
    def _ensure_team_name_language(self):
        """确保球队名称显示方式与手动浏览器一致（默认简体）

        网站通过 SetLanguage(n) 控制球队名称显示，设置保存在Cookie("oldindex")中：
          0=简体(国语译音)  1=繁体(粤语译音)  2=Crow*名
        Selenium打开的是全新浏览器配置文件(无Cookie)，默认显示方式可能与
        手动浏览器不同，导致球队名称不一致（如"安阳" vs "FC安养"）。
        这里强制设置为配置的语言并刷新页面使其生效。
        """
        try:
            # 读取当前语言设置
            current_lang = self.driver.execute_script(
                "return (typeof Config !== 'undefined' && Config.language !== undefined) ? Config.language : -1;"
            )
            if current_lang == TEAM_NAME_LANGUAGE:
                print(f"✓ 球队名称显示已是目标设置 (language={TEAM_NAME_LANGUAGE})")
                return

            # 调用网站自身的SetLanguage函数设置语言（会自动保存Cookie）
            self.driver.execute_script(
                f"if (typeof SetLanguage === 'function') {{ SetLanguage({TEAM_NAME_LANGUAGE}); }}"
            )
            print(f"✓ 已设置球队名称显示 (language={TEAM_NAME_LANGUAGE})，刷新页面生效...")
            time.sleep(2)

            # 重新加载页面使设置生效（SetLanguage可能已触发刷新，这里确保一下）
            self._safe_get(MAIN_URL)
            time.sleep(5)
        except Exception as e:
            print(f"⚠ 设置球队名称显示失败: {e}，将使用当前显示方式")

    def _load_match_list_soup(self):
        """加载比赛列表页面并完成筛选，返回BeautifulSoup对象
        操作：加载页面→设置语言→赛事→单场→全部→全选→确定→获取当前DOM
        """
        self._safe_get(MAIN_URL)
        time.sleep(5)

        # 确保球队名称显示方式与手动浏览器一致（强制设置语言并刷新）
        self._ensure_team_name_language()

        # 通过JS函数完成筛选：赛事→单场→全部→全选→确定
        try:
            # 1. 打开赛事筛选对话框 (Layer2)
            self.driver.execute_script(
                "MM_showHideLayers('Layer2','','show','Layer3','','hidden','DivCountry','','hidden','goalDiv','','hidden','filterBox','','hidden')"
            )
            print("✓ 已打开赛事筛选对话框")
            time.sleep(1)

            # 2. 切换到"单场"标签
            self.driver.execute_script("ShowSclass(3)")
            print("✓ 已切换: 单场")
            time.sleep(1)

            # 3. 点击"全部"（显示所有比赛状态）
            self.driver.execute_script("ShowMatchByMatchState(0)")
            print("✓ 已选择: 全部")
            time.sleep(1)

            # 4. 点击"全选"（选中所有联赛）
            self.driver.execute_script("SelectAll(true,0)")
            print("✓ 已点击: 全选")
            time.sleep(1)

            # 5. 点击"确定"并关闭对话框
            self.driver.execute_script("SelectOK(1);MM_showHideLayers('Layer2','','hide');")
            print("✓ 已确认筛选")
            time.sleep(2)
        except Exception as e:
            print(f"⚠ 筛选操作失败: {e}，将使用全部比赛")

        # 等待列表加载
        print(f"等待比赛列表加载 ({MATCH_LIST_WAIT}秒)...")
        time.sleep(MATCH_LIST_WAIT)

        # 获取当前DOM的HTML（JavaScript执行后的状态）
        # 注意：driver.page_source 返回的是原始HTML，不是JS修改后的DOM
        current_html = self.driver.execute_script("return document.documentElement.outerHTML;")
        return BeautifulSoup(current_html, 'html.parser')

    def scrape_match_list(self):
        """
        爬取主页面比赛列表
        操作：赛事→单场→全部→全选→确定
        """
        print("\n" + "="*60)
        print("第一步：爬取比赛列表")
        print("="*60)

        # 加载页面并完成筛选，获取当前DOM
        soup = self._load_match_list_soup()

        # 找到比赛表格 table_live
        table = soup.find('table', id='table_live')
        if not table:
            print("✗ 未找到比赛表格(table_live)")
            return []

        # 找到所有 tr1_ 开头的行（每场比赛的主数据行）
        tr1_rows = [r for r in table.find_all('tr') if r.get('id', '').startswith('tr1_')]
        print(f"✓ 找到 {len(tr1_rows)} 场比赛")

        matches = []
        self.match_counter = 0
        for row in tr1_rows:
            # 跳过隐藏行（筛选后不可见的比赛）
            style = row.get('style', '')
            if 'display:none' in style.replace(' ', ''):
                continue

            cells = row.find_all('td')
            if len(cells) < 9:
                continue

            try:
                match_data = self._parse_match_row(row, cells)
                if match_data:
                    saved_match_id, already_exists = self.save_match(match_data)
                    # 无论是否已存在，都加入处理列表（确保技术统计等详情数据能被重新爬取）
                    match_data['match_id'] = saved_match_id
                    matches.append(match_data)
                    if already_exists:
                        print(f"  ⊙ [{saved_match_id}] {match_data['league']} "
                              f"{match_data['home_team']} vs {match_data['away_team']} (已存在，重新爬取详情)")
                    else:
                        print(f"  ✓ [{saved_match_id}] {match_data['league']} "
                              f"{match_data['home_team']} vs {match_data['away_team']}")
            except Exception as e:
                continue

        print(f"\n共爬取 {len(matches)} 场单场比赛")
        return matches

    def update_match_status(self, target_date=None):
        """从完场比分页面(Over_*.htm)更新北单比赛的状态和比分
        数据源: https://bf.titan007.com/football/Over_YYYYMMDD.htm
        原理: 页面JS变量 beiDanSchedule = "sid1,...|单场A,..." 即"选择-北单"过滤出的赛程，
              按位置将 sid 映射到 单场编号；表格 table_live 中每个 tr1_ 行含 sid 属性。
              通过 sid(稳定ID) 匹配数据库中的比赛，同时更新 single_match(动态编号)。
        Args:
            target_date: 目标日期(YYYYMMDD)，默认昨天
        Returns:
            更新的比赛数量
        """
        if not target_date:
            target_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')

        print("\n" + "="*60)
        print("更新比赛状态和比分")
        print("="*60)
        url = OVER_URL_TEMPLATE.format(date=target_date)
        print(f"目标日期: {target_date}")
        print(f"数据源: {url}")

        # 加载完场比分页面并获取JS渲染后的DOM
        self._safe_get(url)
        print(f"等待页面加载 ({MATCH_LIST_WAIT}秒)...")
        time.sleep(MATCH_LIST_WAIT)
        # 使用 page_source 获取完整 HTML（包含 sid 属性）
        html = self.driver.page_source

        # 1. 解析北单赛程映射 sid -> 单场编号
        sid2dan = self._parse_beidan_schedule(html)
        if not sid2dan:
            print("✗ 未找到北单赛程映射(beiDanSchedule)，无法过滤北单比赛")
            return 0
        print(f"✓ 北单赛程共 {len(sid2dan)} 场")

        # 2. 解析比赛表格，建立 sid -> 行 映射
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table', id='table_live')
        if not table:
            print("✗ 未找到比赛表格 (table_live)")
            return 0

        # 3. 加载该日期所有比赛的 site_match_id -> match_id 映射
        cursor = self.db.cursor()
        date_prefix = target_date + '%'
        cursor.execute(
            "SELECT match_id, site_match_id FROM `match` WHERE match_id LIKE %s AND site_match_id IS NOT NULL",
            (date_prefix,)
        )
        site_match_id_to_match_id = {}
        for row_db in cursor.fetchall():
            db_match_id, db_site_match_id = row_db[0], row_db[1]
            if db_site_match_id:
                site_match_id_to_match_id[str(db_site_match_id)] = db_match_id

        # 4. 遍历表格行，用 sid 属性匹配数据库
        # Over 页面列结构：0=联赛，1=时间，2=状态，3=主队，4=比分，5=客队，6=半场，7=亚盘，8=大小球，9=数据链接
        updated = 0
        not_found = 0
        for r in table.find_all('tr'):
            # 从行属性获取 sid（与数据库的 site_match_id 相同）
            sid = r.get('sid')
            if not sid:
                continue
            
            # 只处理北单比赛（sid 在 beiDanSchedule 中）
            if sid not in sid2dan:
                continue
            
            cells = r.find_all('td')
            if len(cells) < 6:
                continue
            
            # 从数据库查找 match_id
            match_id = site_match_id_to_match_id.get(str(sid))
            if not match_id:
                not_found += 1
                continue
            
            # 提取状态和比分（Over 页面列索引）
            status = cells[2].get_text(strip=True)
            if status in ['-', '']:
                status = None
            score = cells[4].get_text(strip=True)
            if score in ['-', '']:
                score = None
            home_team = self._clean_team_text(cells[3])
            away_team = self._clean_team_text(cells[5])
            
            # 获取单场编号
            single_match_raw = sid2dan[sid]
            single_match_norm = self._normalize_single_match(single_match_raw)
            
            try:
                cursor.execute(
                    "UPDATE `match` SET status=%s, score=%s, single_match=%s, sid=%s WHERE match_id=%s",
                    (status, score, single_match_norm, sid, match_id)
                )
                updated += 1
                print(f"  ✓ [{match_id}] [sid={sid}] [{single_match_norm}] {home_team} vs {away_team} | 状态={status} 比分={score}")
            except Exception:
                continue

        self.db.commit()
        cursor.close()
        msg = f"\n共更新 {updated} 场比赛状态/比分"
        if not_found:
            msg += f"，另有 {not_found} 场北单比赛不在数据库中(未插入)"
        print(msg)
        return updated

    def _parse_beidan_schedule(self, html):
        """从页面JS变量 beiDanSchedule 解析北单赛程映射
        格式: beiDanSchedule = "sid1,sid2,...|单场A,单场B,...";
        sid 列表与 单场 列表按位置一一对应。
        Returns:
            dict: {sid: 单场编号}，解析失败返回空字典
        """
        m = re.search(r'beiDanSchedule\s*=\s*"([^"]*)"', html)
        if not m:
            return {}
        val = m.group(1)
        if '|' not in val:
            return {}
        left, right = val.split('|', 1)
        sids = [s.strip() for s in left.split(',') if s.strip()]
        dans = [d.strip() for d in right.split(',') if d.strip()]
        if len(sids) != len(dans):
            print(f"⚠ 北单赛程 sid数({len(sids)}) 与 单场数({len(dans)}) 不一致")
        return dict(zip(sids, dans))

    @staticmethod
    def _normalize_single_match(dan):
        """规范化单场编号：去除数字前导零，与数据库存储格式一致
        例: 单场093 -> 单场93；单场100 -> 单场100
        """
        m = re.search(r'(\d+)', dan)
        if m:
            return f"单场{int(m.group(1))}"
        return dan

    @staticmethod
    def _clean_team_text(cell):
        """从球队单元格提取纯文本队名，移除 [数字]/[西数字] 等排名标记"""
        text = cell.get_text(strip=True)
        return re.sub(r'\[[^\]]*\]', '', text).strip()

    def _parse_match_row(self, row, cells):
        """解析比赛表格的一行(tr1_xxx)
        列结构:
          0: checkbox  1: 联赛  2: 时间  3: 状态  4: 主队  5: 比分  6: 客队  7: 半场  8: 数据链接(析亚大欧)  9: 资讯
        """
        # 从行ID提取网站原始比赛ID (tr1_3027598 -> 3027598)
        row_id = row.get('id', '')
        site_match_id = row_id.replace('tr1_', '')
        
        # 优先使用 sid 属性作为稳定标识符（sid 不会随比赛增减而变化）
        sid = row.get('sid')
        
        # 联赛名称 - 拆分"单场XX"
        league_raw = cells[1].get_text(strip=True)
        league_match = re.match(r'^(.+?)(单场\d+)$', league_raw)
        if league_match:
            league = league_match.group(1)
            single_match = league_match.group(2)
        else:
            league = league_raw
            single_match = None

        # 比赛时间
        match_time = cells[2].get_text(strip=True)

        # 比赛状态
        status = cells[3].get_text(strip=True)
        if status in ['-', '']:
            status = None

        # 主队 - 清理排名标记后提取纯文本
        home_team_raw = cells[4]
        # 提取排名
        home_rank_m = re.search(r'\[(\d+)\]', home_team_raw.get_text())
        home_rank = home_rank_m.group(1) if home_rank_m else None
        # 移除所有 [数字] 格式的排名标记
        for span in home_team_raw.find_all('span'):
            if re.search(r'^\d+$', span.get_text(strip=True)):
                span.decompose()
        home_team = home_team_raw.get_text(strip=True)
        # 再次清理可能的残留排名标记
        home_team = re.sub(r'\[\d+\]', '', home_team).strip()

        # 比分
        score = cells[5].get_text(strip=True)
        if score in ['-', '']:
            score = None

        # 客队 - 清理排名标记后提取纯文本
        away_team_raw = cells[6]
        # 提取排名
        away_rank_m = re.search(r'\[(\d+)\]', away_team_raw.get_text())
        away_rank = away_rank_m.group(1) if away_rank_m else None
        # 移除所有 [数字] 格式的排名标记
        for span in away_team_raw.find_all('span'):
            if re.search(r'^\d+$', span.get_text(strip=True)):
                span.decompose()
        away_team = away_team_raw.get_text(strip=True)
        # 再次清理可能的残留排名标记
        away_team = re.sub(r'\[\d+\]', '', away_team).strip()

        # 生成比赛编号 (日期+序号)
        self.match_counter += 1
        match_id = f"{self.today}{self.match_counter:02d}"

        # 分析页面URL - 从"析"链接的onclick提取
        analysis_url = None
        detail_url = None
        data_cell = cells[8] if len(cells) > 8 else None
        if data_cell:
            for link in data_cell.find_all('a'):
                onclick = link.get('onclick', '')
                link_text = link.get_text(strip=True)
                if link_text == '析' and 'analysis' in onclick:
                    # onclick="analysis(3027598)"
                    m = re.search(r'analysis\((\d+)\)', onclick)
                    if m:
                        aid = m.group(1)
                        analysis_url = f'https://zq.titan007.com/analysis/{aid}cn.htm'
                        # 从analysis_url构造detail_url（使用相同的ID）
                        detail_url = f'https://live.titan007.com/detail/{aid}cn.htm'

        return {
            'match_id': match_id,
            'site_match_id': site_match_id,
            'sid': sid,  # 稳定的网站比赛ID（不随比赛增减变化）
            'league': league,
            'single_match': single_match,  # 注意：此值为爬虫运行时网页显示的动态编号，可能随时间变化
            'match_time': match_time,
            'status': status,
            'home_team': home_team,
            'home_rank': home_rank,
            'away_team': away_team,
            'away_rank': away_rank,
            'score': score,
            'analysis_url': analysis_url,
            'detail_url': detail_url,
        }

    # ==================== 第二步：爬取分析页面 ====================
    def _switch_odds_to_36_initial(self):
        """切换对赛往绩和近期战绩表格的亚盘显示为 36* 初盘
        页面默认显示 Crow* 终盘，需要切换到 36* 初盘
        涉及的下拉框：
          - hSelect_v / hType_v (对赛往绩)
          - hSelect_hn / hType_hn (主队近期战绩)
          - hSelect_an / hType_an (客队近期战绩)
        """
        js_code = """
        // 对赛往绩 (prefix: v)
        var hSelect_v = document.getElementById('hSelect_v');
        var hType_v = document.getElementById('hType_v');
        if (hSelect_v) { hSelect_v.value = '8'; }
        if (hType_v) { hType_v.value = '0'; }
        if (typeof showOdds_h === 'function') { showOdds_h('v', 1); }

        // 主队近期战绩 (prefix: hn)
        var hSelect_hn = document.getElementById('hSelect_hn');
        var hType_hn = document.getElementById('hType_hn');
        if (hSelect_hn) { hSelect_hn.value = '8'; }
        if (hType_hn) { hType_hn.value = '0'; }
        if (typeof showOdds_h === 'function') { showOdds_h('hn', 1); }

        // 客队近期战绩 (prefix: an)
        var hSelect_an = document.getElementById('hSelect_an');
        var hType_an = document.getElementById('hType_an');
        if (hSelect_an) { hSelect_an.value = '8'; }
        if (hType_an) { hType_an.value = '0'; }
        if (typeof showOdds_h === 'function') { showOdds_h('an', 1); }
        """
        try:
            self.driver.execute_script(js_code)
            time.sleep(2)  # 等待AJAX刷新表格数据
        except Exception as e:
            print(f"    ⚠ 切换36*初盘失败: {e}")

    def scrape_analysis_page(self, match_id, analysis_url, detail_url=None, home_team=None, away_team=None):
        """
        爬取比赛分析页面数据
        包括：联赛积分排名、对赛往绩、近期战绩、亚盘、大小球
        Args:
            home_team: 当前比赛主队名称（直接传入，避免DB 查询 match_id 冲突）
            away_team: 当前比赛客队名称
        """
        print(f"\n  分析页面: {analysis_url}")

        if not self._safe_get(analysis_url):
            print(f"  ✗ 页面加载失败: {analysis_url}")
            return
        time.sleep(5)  # 等待页面加载

        # 切换亚盘显示为 36* 初盘（页面默认是 Crow* 终盘）
        self._switch_odds_to_36_initial()

        soup = BeautifulSoup(self.driver.page_source, 'html.parser')

        # 1. 爬取联赛积分排名
        self._scrape_league_standings(match_id, soup)

        # 2. 爬取对赛往绩（含角球列）
        self._scrape_head_to_head(match_id, soup, home_team, away_team)

        # 3. 爬取近期战绩
        recent_form_ids = self._scrape_recent_form(match_id, soup, home_team, away_team)

        # 4. 爬取本赛季数据统计（技术统计）
        self._scrape_technical_stats_from_analysis(match_id, soup)

        # 5. 直接导航到亚盘专页，爬取亚盘数据
        #    注意：_scrape_recent_form_tech_stats 已将浏览器导航到其他detail页面，
        #    不能再依赖点击分析页面上的"亚让"链接，必须直接构造URL访问
        site_match_id = self.extract_analysis_id(analysis_url)
        self._scrape_asian_odds(match_id, site_match_id)

        # 6. 直接导航到大小球专页，爬取大小球数据
        self._scrape_over_under_odds(match_id, site_match_id)

        return recent_form_ids

    def _scrape_league_standings(self, match_id, soup):
        """爬取联赛积分排名数据
        页面结构：h2(联赛积分排名) -> table
        表格结构：每个球队有 球队名行(colspan=11) + 表头行 + 4行数据(总/主/客/近6)
        """
        print("    爬取联赛积分排名...")
        standings_data = []

        # 找到"联赛积分排名"h2标题
        standings_h2 = None
        for h2 in soup.find_all('h2', class_='fx_title2'):
            if '联赛积分排名' in h2.get_text():
                standings_h2 = h2
                break

        if not standings_h2:
            return

        # 找到h2后面的表格
        table = standings_h2.find_next_sibling('table')
        if not table:
            return

        rows = table.find_all('tr')
        if len(rows) < 5:
            return

        # 解析表格结构
        # 每个球队的数据块：
        # - row with colspan=11 cell: team name (e.g., [中超-5]云南玉昆)
        # - header row: 全场, 赛, 胜, 平, 负, 得, 失, 净, 积分, 排名, 胜率
        # - 4 data rows: 总/主/客/近6
        current_team_name = None
        current_league_name = None
        scope_map = {'总': 'total', '主': 'home', '客': 'away', '近6': 'recent6', '近10': 'recent10'}

        for row in rows:
            cells = row.find_all(['td', 'th'])
            texts = [c.get_text(strip=True) for c in cells]

            if not texts:
                continue

            # 检查是否是球队名行 (colspan=11 或包含 [xxx] 格式)
            if len(cells) == 1 and cells[0].get('colspan') == '11':
                team_full_text = texts[0]
                # 提取 [联赛-排名]球队名
                team_match = re.match(r'\[([^\]]+)\](.+)', team_full_text)
                if team_match:
                    league_rank = team_match.group(1)  # e.g., "中超-5"
                    current_team_name = team_match.group(2).strip()
                    parts = league_rank.split('-')
                    if len(parts) >= 2:
                        current_league_name = parts[0]
                    else:
                        current_league_name = league_rank
                else:
                    current_team_name = team_full_text
                continue

            # 检查是否是表头行 (包含"全场"或"赛")
            if '全场' in texts or (len(texts) >= 5 and texts[0] == '赛'):
                continue

            # 遇到"半场"表头时停止解析（只要全场数据）
            if '半场' in texts:
                break

            # 数据行: scope(总/主/客/近6), 赛, 胜, 平, 负, 得, 失, 净, 积分, 排名, 胜率
            if len(texts) >= 11 and texts[0] in scope_map:
                scope = scope_map[texts[0]]
                try:
                    standing = {
                        'match_id': match_id,
                        'team_type': 'home' if standings_data else 'away',  # 先处理主队
                        'team_name': current_team_name,
                        'league_name': current_league_name,
                        'scope': scope,
                        'played': self._safe_int(texts[1]),
                        'won': self._safe_int(texts[2]),
                        'drawn': self._safe_int(texts[3]),
                        'lost': self._safe_int(texts[4]),
                        'goals_for': self._safe_int(texts[5]),
                        'goals_against': self._safe_int(texts[6]),
                        'goal_diff': self._safe_int(texts[7]),
                        'points': self._safe_int(texts[8]),
                        'ranking': self._safe_int(texts[9]),
                        'win_rate': texts[10] if len(texts) > 10 else None,
                    }
                    # 判断是主队还是客队（根据当前已保存的数据）
                    home_count = sum(1 for s in standings_data if s['team_type'] == 'home')
                    if home_count >= 4:
                        standing['team_type'] = 'away'
                    else:
                        standing['team_type'] = 'home'
                    standings_data.append(standing)
                except (IndexError, ValueError):
                    continue

        if standings_data:
            # 去重：先删除该match_id已有的积分排名数据
            cursor = self.db.cursor()
            cursor.execute("DELETE FROM league_standings WHERE match_id = %s", (match_id,))
            self.db.commit()
            cursor.close()
            self.save_league_standings(standings_data)
            print(f"    ✓ 保存 {len(standings_data)} 条积分排名数据")

    def _scrape_head_to_head(self, match_id, soup, home_team_name=None, away_team_name=None):
        """爬取对赛往绩数据
        页面结构：h2#003(对赛往绩) -> table
        表格结构：
          row0: 筛选行 (近123场 主客 相同)
          row1-3: 表头 (类型, 日期, 主场, 比分(半场), 角球, 客场, odds...)
          row4+: 数据行 (中超, 26-04-04, 上海海港, 2-1(1-0), 4-6, 云南玉昆, odds...)
        技术统计（角球）从对赛往绩表格的“角球”列提取
        """
        print("    爬取对赛往绩...")
        h2h_data = []

        # 找到h2#003
        h2_h2h = soup.find('h2', id='003')
        if not h2_h2h:
            return

        # 找到后面的表格
        table = h2_h2h.find_next_sibling('table')
        if not table:
            return

        rows = table.find_all('tr')
        if len(rows) < 5:
            return

        # 数据行从第4行开始（跳过筛选行和表头）
        for i, row in enumerate(rows):
            if i < 4:  # 跳过前4行（筛选+表头）
                continue

            cells = row.find_all(['td', 'th'])
            texts = [c.get_text(strip=True) for c in cells]

            if len(texts) < 6:
                continue

            # 数据行格式: 类型, 日期, 主场, 比分(半场), 角球, 客场, 主水, 盘口, 客水, 胜平负..., 胜负, 让球, 进球数
            league = texts[0] if texts else None
            match_date = texts[1] if len(texts) > 1 else None
            home_team = texts[2] if len(texts) > 2 else None
            score_full = texts[3] if len(texts) > 3 else None  # e.g., "2-1(1-0)"
            corners = texts[4] if len(texts) > 4 else None
            away_team = texts[5] if len(texts) > 5 else None

            # 解析比分和半场比分
            score = None
            half_score = None
            if score_full:
                score_match = re.match(r'(\d+-\d+)\((\d+-\d+)\)', score_full)
                if score_match:
                    score = score_match.group(1)
                    half_score = score_match.group(2)
                else:
                    score = score_full

            # 提取36*初盘数据 (列6-8: 主水/盘口/客水)
            initial_home_odds = texts[6] if len(texts) > 6 else None
            # 盘口带*号表示受让，如 "*半球" -> "受让半球"
            initial_handicap = self._normalize_handicap(texts[7]) if len(texts) > 7 else None
            initial_away_odds = texts[8] if len(texts) > 8 else None

            # 提取结果数据 (列12-14: 胜负/让球/进球数)
            result = texts[12] if len(texts) > 12 else None
            handicap_result = texts[13] if len(texts) > 13 else None
            over_under_result = texts[14] if len(texts) > 14 else None

            h2h_data.append({
                'match_id': match_id,
                'match_date': match_date,
                'league': league,
                'home_team': home_team,
                'score': score,
                'away_team': away_team,
                'half_score': half_score,
                'corners': corners,
                'initial_home_odds': initial_home_odds,
                'initial_handicap': initial_handicap,
                'initial_away_odds': initial_away_odds,
                'result': result,
                'handicap_result': handicap_result,
                'over_under_result': over_under_result,
            })

        if h2h_data:
            # 去重：先删除该match_id已有的对赛往绩数据
            cursor = self.db.cursor()
            cursor.execute("DELETE FROM head_to_head WHERE match_id = %s", (match_id,))
            self.db.commit()
            cursor.close()
            self.save_head_to_head(h2h_data)
            print(f"    ✓ 保存 {len(h2h_data)} 条对赛往绩数据")

    def _extract_tech_stats_from_h2h(self, match_id, h2h_data, home_team_name=None, away_team_name=None):
        """从对赛往绩数据提取技术统计（角球）保存到recent_form表
        使用第一条对赛往绩记录的角球数据作为当前比赛的技术统计
        角球格式: "主队角球-客队角球" 如 "4-6"
        """
        if not h2h_data:
            return

        # 获取第一条对赛往绩记录的角球数据
        first_h2h = h2h_data[0]
        corners_str = first_h2h.get('corners')
        if not corners_str or '-' not in corners_str:
            print("    ⚠ 对赛往绩无角球数据")
            return

        # 解析角球: "4-6" -> home_corners=4, away_corners=6
        try:
            parts = corners_str.split('-')
            h2h_home_corners = int(parts[0])
            h2h_away_corners = int(parts[1])
        except (ValueError, IndexError):
            print(f"    ⚠ 角球数据解析失败: {corners_str}")
            return

        # 确定当前比赛主客队的角球
        # 对赛往绩的主队可能与当前比赛的主队不同，需要匹配
        h2h_home_team = first_h2h.get('home_team', '')
        h2h_away_team = first_h2h.get('away_team', '')

        if home_team_name and home_team_name in h2h_home_team:
            # 当前主队 = 对赛往绩主队
            home_corners = h2h_home_corners
            away_corners = h2h_away_corners
        elif home_team_name and home_team_name in h2h_away_team:
            # 当前主队 = 对赛往绩客队
            home_corners = h2h_away_corners
            away_corners = h2h_home_corners
        else:
            # 无法匹配，默认按对赛往绩的主客队顺序
            home_corners = h2h_home_corners
            away_corners = h2h_away_corners

        # 构建技术统计数据
        home_stats = {'corners': home_corners}
        away_stats = {'corners': away_corners}

        # 保存到recent_form表（is_current_match=1）
        if not home_team_name or not away_team_name:
            home_team_name, away_team_name = self._get_team_names_from_db(match_id)
        self.save_tech_stats_to_recent_form(match_id, home_stats, away_stats, home_team_name, away_team_name)
        print(f"    ✓ 保存技术统计: 主队角球={home_corners}, 客队角球={away_corners}")

    def _scrape_recent_form(self, match_id, soup, home_team_name=None, away_team_name=None):
        """爬取近期战绩数据
        页面结构：table_hn(主队近期战绩) 和 table_an(客队近期战绩)
        表格结构：
          row0-1: 表头 (类型, 日期, 主场, 比分(半场), 角球, 客场, odds...)
          row2+: 数据行 (中超, 26-07-10, 山东泰山, 4-3(3-1), 2-14, 云南玉昆, odds...)
        Args:
            home_team_name: 当前比赛主队名称（直接传入）
            away_team_name: 当前比赛客队名称（直接传入）
        """
        print("    爬取近期战绩...")
        form_data = []

        # 优先使用直接传入的球队名称，避免 DB 查询 match_id 冲突
        if not home_team_name or not away_team_name:
            home_team_name, away_team_name = self._get_team_names_from_db(match_id)

        # 主队近期战绩 - table_hn
        table_hn = soup.find('table', id='table_hn')
        if table_hn:
            hn_data = self._parse_recent_form_table(table_hn, match_id, 'home', home_team_name)
            form_data.extend(hn_data)

        # 客队近期战绩 - table_an
        table_an = soup.find('table', id='table_an')
        if table_an:
            an_data = self._parse_recent_form_table(table_an, match_id, 'away', away_team_name)
            form_data.extend(an_data)

        if form_data:
            # 去重：先删除该match_id已有的近期战绩记录（is_current_match=0）
            cursor = self.db.cursor()
            cursor.execute("DELETE FROM `recent_form` WHERE match_id = %s AND is_current_match = 0", (match_id,))
            self.db.commit()
            cursor.close()
            self.save_recent_form(form_data)
            print(f"    ✓ 保存 {len(form_data)} 条近期战绩数据")

            # 爬取每场近期战绩的技术统计（跳转现场分析页面）
            self._scrape_recent_form_tech_stats(match_id)

        return form_data

    def _extract_team_name_from_data_row(self, table, team_type):
        """从近期战绩表格的第一条数据行提取球队名称
        表格结构：row0=表头行1, row1=表头行2, row2+=数据行
        主队表格：球队名在第3列(主场列, index=2)
        客队表格：球队名在第6列(客场列, index=5)
        """
        rows = table.find_all('tr')
        # 跳过前2行表头，找第一条数据行
        for i, row in enumerate(rows):
            if i < 2:
                continue
            cells = row.find_all(['td', 'th'])
            if len(cells) < 6:
                continue
            # 提取球队名并清理排名标记（如"1济州SK" -> "济州SK"）
            if team_type == 'home':
                raw_name = cells[2].get_text(strip=True)  # 主场列
            else:
                raw_name = cells[5].get_text(strip=True)  # 客场列
            clean_name = self.clean_team_name(raw_name)
            return clean_name if clean_name else None
        return None

    def _normalize_handicap(self, raw):
        """规范化盘口值：页面用*号表示受让，如 "*半球" -> "受让半球" """
        if not raw:
            return None
        raw = raw.strip()
        if raw.startswith('*'):
            return '受让' + raw[1:]
        return raw

    def _parse_recent_form_table(self, table, match_id, team_type, team_name=None):
        """解析近期战绩表格(table_hn或table_an)"""
        data = []
        rows = table.find_all('tr')

        # 跳过前2行（表头）
        for i, row in enumerate(rows):
            if i < 2:  # 跳过表头
                continue

            cells = row.find_all(['td', 'th'])
            texts = [c.get_text(strip=True) for c in cells]

            if len(texts) < 6:
                continue

            # 数据行格式: 类型, 日期, 主场, 比分(半场), 角球, 客场, 主水, 盘口, 客水, 胜平负..., 胜负, 让球, 进球数
            league = texts[0]
            match_date = texts[1]
            home_team = self.clean_team_name(texts[2])
            score_cell = cells[3]  # 比分(半场)列的cell元素
            score_full = score_cell.get_text(strip=True)  # e.g., "4-3(3-1)"
            corners = texts[4]
            away_team = self.clean_team_name(texts[5])

            # 解析比分
            score = None
            half_score = None
            if score_full:
                score_match = re.match(r'(\d+-\d+)\((\d+-\d+)\)', score_full)
                if score_match:
                    score = score_match.group(1)
                    half_score = score_match.group(2)
                else:
                    score = score_full

            # 提取36*初盘数据 (列6-8: 主水/盘口/客水)
            initial_home_odds = texts[6] if len(texts) > 6 else None
            # 盘口带*号表示受让，如 "*半球" -> "受让半球"
            initial_handicap = self._normalize_handicap(texts[7]) if len(texts) > 7 else None
            initial_away_odds = texts[8] if len(texts) > 8 else None

            # 提取结果数据 (列12-14: 胜负/让球/进球数)
            handicap_result = texts[13] if len(texts) > 13 else None
            over_under_result = texts[14] if len(texts) > 14 else None

            # 判断胜负
            result = None
            if score:
                parts = re.split(r'-', score)
                if len(parts) == 2:
                    try:
                        home_score, away_score = int(parts[0]), int(parts[1])
                        if team_type == 'home':
                            # 主队视角
                            if home_score > away_score:
                                result = 'W'
                            elif home_score < away_score:
                                result = 'L'
                            else:
                                result = 'D'
                        else:
                            # 客队视角
                            if away_score > home_score:
                                result = 'W'
                            elif away_score < home_score:
                                result = 'L'
                            else:
                                result = 'D'
                    except ValueError:
                        pass

            data.append({
                'match_id': match_id,
                'team_type': team_type,
                'team_name': team_name,
                'match_date': match_date,
                'league': league,
                'home_team': home_team,
                'score': score,
                'away_team': away_team,
                'result': result,
                'detail_url': self._extract_detail_url(score_cell),
                'initial_home_odds': initial_home_odds,
                'initial_handicap': initial_handicap,
                'initial_away_odds': initial_away_odds,
                'handicap_result': handicap_result,
                'over_under_result': over_under_result,
            })

        return data

    def _extract_detail_url(self, score_cell):
        """从比分单元格提取现场分析页面URL
        处理方式：
        1. 查找<a>标签的href属性
        2. 解析onclick属性中的URL或比赛ID
        3. 在单元格HTML中搜索比赛ID模式
        URL格式: https://live.titan007.com/detail/{match_id}cn.htm
        """
        # 方式1: 查找<a>标签
        a_tag = score_cell.find('a')
        if a_tag:
            href = a_tag.get('href', '')
            onclick = a_tag.get('onclick', '')
            # 从href提取
            if href and href not in ('javascript:void(0)', 'javascript:;', '#'):
                id_match = re.search(r'(\d{5,})', href)
                if id_match:
                    return f"https://live.titan007.com/detail/{id_match.group(1)}cn.htm"
            # 从onclick提取
            if onclick:
                id_match = re.search(r'(\d{5,})', onclick)
                if id_match:
                    return f"https://live.titan007.com/detail/{id_match.group(1)}cn.htm"

        # 方式2: 检查cell本身的onclick
        onclick = score_cell.get('onclick', '')
        if onclick:
            id_match = re.search(r'(\d{5,})', onclick)
            if id_match:
                return f"https://live.titan007.com/detail/{id_match.group(1)}cn.htm"

        # 方式3: 在单元格HTML中搜索比赛ID模式
        cell_html = str(score_cell)
        id_match = re.search(r'(?:detail|analysis)[/\\](\d{5,})', cell_html)
        if id_match:
            return f"https://live.titan007.com/detail/{id_match.group(1)}cn.htm"

        # 方式4: 查找任意包含数字ID的链接模式
        id_match = re.search(r'["\'/](\d{6,})["\'cn]', cell_html)
        if id_match:
            return f"https://live.titan007.com/detail/{id_match.group(1)}cn.htm"

        return None

    def _scrape_recent_form_tech_stats(self, match_id):
        """爬取近期战绩每场比赛的技术统计
        依次访问每场近期战绩的现场分析页面，抓取关注球队的技术统计数据
        关注球队 = 近期战绩所属球队（team_name字段）
        """
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT id, detail_url, team_name, home_team, away_team "
            "FROM `recent_form` WHERE match_id = %s AND is_current_match = 0 "
            "AND detail_url IS NOT NULL AND detail_url != ''",
            (match_id,)
        )
        records = cursor.fetchall()
        cursor.close()

        if not records:
            print("    ⚠ 近期战绩无比分链接，跳过技术统计爬取")
            return

        print(f"    爬取近期战绩技术统计 ({len(records)} 场)...")
        success_count = 0

        for record_id, detail_url, team_name, home_team, away_team in records:
            if not detail_url:
                continue

            stats = self._scrape_match_detail_stats(detail_url, team_name, home_team, away_team)
            if stats:
                # 更新recent_form记录的技术统计字段
                tech_fields = ['corners', 'half_corners', 'yellow_cards', 'shots', 'shots_on_target',
                               'attacks', 'dangerous_attacks', 'shots_off_target', 'shots_blocked',
                               'free_kicks', 'possession', 'half_possession', 'passes',
                               'pass_success_rate', 'fouls', 'offsides']
                update_parts = []
                update_values = []
                for field in tech_fields:
                    if field in stats:
                        update_parts.append(f"`{field}` = %s")
                        update_values.append(stats[field])

                if update_parts:
                    update_values.append(record_id)
                    sql = f"UPDATE `recent_form` SET {', '.join(update_parts)} WHERE id = %s"
                    cursor = self.db.cursor()
                    cursor.execute(sql, update_values)
                    self.db.commit()
                    cursor.close()
                    success_count += 1

            time.sleep(1)  # 请求间隔，避免被封

        print(f"    ✓ 完成技术统计: {success_count}/{len(records)} 场成功")

    def _team_name_matches(self, name1, name2):
        """检查两个球队名是否匹配（模糊匹配）
        处理同一球队在不同页面名称略有差异的情况（如 索菲亚 vs 索非亚）
        匹配策略：
        1. 子串包含
        2. 字符重叠度 >= 60%
        """
        if not name1 or not name2:
            return False
        n1 = re.sub(r'[\s\u00a0\u200b\(\)（）]', '', name1)
        n2 = re.sub(r'[\s\u00a0\u200b\(\)（）]', '', name2)
        if not n1 or not n2:
            return False
        # 方式1: 子串包含
        if n1 in n2 or n2 in n1:
            return True
        # 方式2: 字符重叠度（处理同音不同字的情况）
        common = sum(1 for c in n1 if c in n2)
        overlap = common / max(len(n1), len(n2))
        return overlap >= 0.6

    def _scrape_match_detail_stats(self, detail_url, followed_team, home_team, away_team):
        """访问单场比赛的现场分析页面，提取关注球队的技术统计
        页面结构：div#teamTechDiv_detail > ul#teamTechDiv > li.lists > div.data
        每个li包含: span[主队值] + span[统计项名称] + span[客队值]
        Args:
            detail_url: 现场分析页面URL (live.titan007.com/detail/{id}cn.htm)
            followed_team: 关注球队名称（近期战绩所属球队）
            home_team: 该场比赛的主队名称
            away_team: 该场比赛的客队名称
        Returns:
            dict: 关注球队的技术统计 {field: value}，失败返回None
        """
        if not self._safe_get(detail_url):
            print(f"      ✗ 页面加载失败: {detail_url}")
            return None
        time.sleep(3)

        soup = BeautifulSoup(self.driver.page_source, 'html.parser')

        # 确定关注球队是主队还是客队（使用模糊匹配）
        followed_is_home = None
        if followed_team and home_team:
            ft = self.clean_team_name(followed_team)
            ht = self.clean_team_name(home_team)
            at = self.clean_team_name(away_team) if away_team else ''
            home_match = self._team_name_matches(ft, ht)
            away_match = self._team_name_matches(ft, at) if at else False
            if home_match and not away_match:
                followed_is_home = True
            elif away_match and not home_match:
                followed_is_home = False
            elif home_match and away_match:
                # 两个都匹配，选重叠度更高的
                home_common = sum(1 for c in ft if c in ht)
                away_common = sum(1 for c in ft if c in at)
                followed_is_home = home_common >= away_common
            else:
                print(f"      ⚠ 无法匹配关注球队: {followed_team} (主:{home_team} 客:{away_team})")
                return None
        else:
            return None

        # 查找技术统计容器
        tech_div = soup.find('div', id='teamTechDiv_detail')
        if not tech_div:
            # 尝试点击"现场分析"标签后再查找
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
            print(f"      ⚠ 未找到技术统计区域: {detail_url}")
            return None

        tech_list = tech_div.find('ul', id='teamTechDiv')
        if not tech_list:
            return None

        # 统计项名称到字段名的映射
        stat_field_map = {
            '角球': 'corners',
            '半场角球': 'half_corners',
            '射门': 'shots',
            '射正': 'shots_on_target',
            '进攻': 'attacks',
            '危险进攻': 'dangerous_attacks',
            '射门不中': 'shots_off_target',
            '射门被挡': 'shots_blocked',
            '任意球': 'free_kicks',
            '控球率': 'possession',
            '半场控球率': 'half_possession',
            '传球': 'passes',
            '传球成功率': 'pass_success_rate',
            '犯规': 'fouls',
            '越位': 'offsides',
            '黄牌': 'yellow_cards',
        }

        stats = {}
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
                val = home_val if followed_is_home else away_val
                if field in ['possession', 'half_possession', 'pass_success_rate']:
                    stats[field] = val.replace('%', '')
                else:
                    stats[field] = self._safe_int(val)

        if stats:
            side = '主队' if followed_is_home else '客队'
            print(f"      ✓ {followed_team}({side}): 角球={stats.get('corners', '-')}, "
                  f"射门={stats.get('shots', '-')}, 射正={stats.get('shots_on_target', '-')}, "
                  f"进攻={stats.get('attacks', '-')}, 危险进攻={stats.get('dangerous_attacks', '-')}")
            return stats

        return None

    def _scrape_asian_odds(self, match_id, site_match_id=None):
        """导航到亚盘指数专页，爬取亚盘数据
        直接通过URL访问亚盘专页（AsianOdds_n.aspx），不再依赖点击"亚让"链接
        （因为爬取近期战绩技术统计后浏览器已不在分析页面，点击会导航到错误比赛）
        页面结构：
          主数据表(T[5])：表头2行 + 每个公司一组（主行13cells + 分隔行1cell + 可选附加盘口行）
          列映射(13 cells):
            [0]空 [1]公司名 [2]空/盘号
            [3]初主水 [4]初盘口 [5]初客水
            [6-8]隐藏列(欧转亚盘, display:none)
            [9]终主水 [10]终盘口 [11]终客水
            [12]操作按钮
        """
        print("    爬取亚盘数据...")

        if not site_match_id:
            print("    ✗ 无site_match_id，无法构造亚盘专页URL")
            return

        # 直接导航到亚盘专页
        asian_url = f'https://vip.titan007.com/AsianOdds_n.aspx?id={site_match_id}&l=0'
        if not self._safe_get(asian_url):
            print("    ✗ 亚盘专页加载失败")
            return
        time.sleep(5)

        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        odds_data = []

        # 找到包含"公司"表头的表格（亚盘专页的主数据表）
        target_table = None
        for table in soup.find_all('table'):
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                texts = [c.get_text(strip=True) for c in cells]
                if texts and '公司' in texts:
                    target_table = table
                    break
            if target_table:
                break

        if not target_table:
            print("    ✗ 未找到亚盘表格")
            return

        # 解析亚盘专页表格
        rows = target_table.find_all('tr')

        for row in rows:
            cells = row.find_all(['td', 'th'])
            texts = [c.get_text(strip=True) for c in cells]

            # 只处理13列的主数据行（公司数据行）
            if len(texts) != 13:
                continue

            # 跳过表头行
            if texts[1] in ('公司', ''):
                continue

            # 跳过附加盘口行（盘2/盘3/盘4，无公司名）
            company_raw = texts[1]
            if not company_raw:
                continue

            # 清理公司名（去掉"封"等标记）
            company = company_raw.replace('封', '').strip()
            if not company:
                continue

            # 提取初盘数据: [3]=主水 [4]=盘口 [5]=客水
            initial_home = texts[3]
            initial_handicap = texts[4]
            initial_away = texts[5]

            # 提取终盘数据: [9]=主水 [10]=盘口 [11]=客水
            current_home = texts[9]
            current_handicap = texts[10]
            current_away = texts[11]

            # 跳过无效数据（终盘可能为"-"表示无变化）
            if current_home == '-' or current_handicap == '-':
                current_home = None
                current_handicap = None
                current_away = None

            odds_data.append({
                'match_id': match_id,
                'company': company,
                'initial_handicap': initial_handicap,
                'initial_home_odds': initial_home,
                'initial_away_odds': initial_away,
                'current_handicap': current_handicap,
                'current_home_odds': current_home,
                'current_away_odds': current_away,
            })

        if odds_data:
            # 去重：先删除该match_id已有的亚盘数据
            cursor = self.db.cursor()
            cursor.execute("DELETE FROM asian_odds WHERE match_id = %s", (match_id,))
            self.db.commit()
            cursor.close()
            self.save_asian_odds(odds_data)
            print(f"    ✓ 保存 {len(odds_data)} 条亚盘数据")

    def _scrape_over_under_odds(self, match_id, site_match_id=None):
        """导航到大小球指数专页，爬取大小球数据
        直接通过URL访问大小球专页（OverDown_n.aspx），不再依赖点击"进球数"链接
        页面结构：
          主数据表：表头2行 + 每个公司一组（主行13cells + 分隔行1cell + 可选附加盘口行）
          列映射(13 cells):
            [0]空 [1]公司名 [2]空/盘号
            [3]初大球水 [4]初进球数 [5]初小球水
            [6-8]隐藏列(display:none)
            [9]终大球水 [10]终进球数 [11]终小球水
            [12]操作按钮
        """
        print("    爬取大小球数据...")

        if not site_match_id:
            print("    ✗ 无site_match_id，无法构造大小球专页URL")
            return

        # 直接导航到大小球专页
        over_url = f'https://vip.titan007.com/OverDown_n.aspx?id={site_match_id}&l=0'
        if not self._safe_get(over_url):
            print("    ✗ 大小球专页加载失败")
            return
        time.sleep(5)

        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        odds_data = []

        # 找到包含"公司"表头的表格（大小球专页的主数据表）
        target_table = None
        for table in soup.find_all('table'):
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                texts = [c.get_text(strip=True) for c in cells]
                if texts and '公司' in texts:
                    target_table = table
                    break
            if target_table:
                break

        if not target_table:
            print("    ✗ 未找到大小球表格")
            return

        # 解析大小球专页表格
        rows = target_table.find_all('tr')

        for row in rows:
            cells = row.find_all(['td', 'th'])
            texts = [c.get_text(strip=True) for c in cells]

            # 只处理13列的主数据行（公司数据行）
            if len(texts) != 13:
                continue

            # 跳过表头行
            if texts[1] in ('公司', ''):
                continue

            # 跳过附加盘口行（盘2/盘3/盘4，无公司名）
            company_raw = texts[1]
            if not company_raw:
                continue

            # 清理公司名（去掉"封"等标记）
            company = company_raw.replace('封', '').strip()
            if not company:
                continue

            # 提取初盘数据: [3]=大球水 [4]=进球数 [5]=小球水
            initial_over_odds = texts[3]
            initial_line = texts[4]
            initial_under_odds = texts[5]

            # 提取终盘数据: [9]=大球水 [10]=进球数 [11]=小球水
            current_over_odds = texts[9]
            current_line = texts[10]
            current_under_odds = texts[11]

            # 跳过无效数据（终盘可能为"-"表示无变化）
            if current_over_odds == '-' or current_line == '-':
                current_over_odds = None
                current_line = None
                current_under_odds = None

            odds_data.append({
                'match_id': match_id,
                'company': company,
                'initial_line': initial_line,
                'initial_over_odds': initial_over_odds,
                'initial_under_odds': initial_under_odds,
                'current_line': current_line,
                'current_over_odds': current_over_odds,
                'current_under_odds': current_under_odds,
            })

        if odds_data:
            # 去重：先删除该match_id已有的大小球数据
            cursor = self.db.cursor()
            cursor.execute("DELETE FROM over_under_odds WHERE match_id = %s", (match_id,))
            self.db.commit()
            cursor.close()
            self.save_over_under_odds(odds_data)
            print(f"    ✓ 保存 {len(odds_data)} 条大小球数据")

    def _scrape_detail_tech_stats(self, match_id, detail_url, home_team_name=None, away_team_name=None):
        """从详情页爬取技术统计数据（角球、射门、黄牌等）
        页面结构：div#teamTechDiv_detail > ul#teamTechDiv > li.lists > div.data
        每个li包含: span[主队值] + span[统计项名称] + span[客队值]
        """
        print("    爬取详情页技术统计...")
        
        if not self._safe_get(detail_url):
            print("    ✗ 详情页加载失败")
            return
        time.sleep(5)
        
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        
        # 查找技术统计容器
        tech_div = soup.find('div', id='teamTechDiv_detail')
        if not tech_div:
            print("    ✗ 未找到技术统计区域")
            return
        
        tech_list = tech_div.find('ul', id='teamTechDiv')
        if not tech_list:
            return
        
        # 统计项名称到字段名的映射
        stat_field_map = {
            '角球': 'corners',
            '半场角球': 'half_corners',
            '射门': 'shots',
            '射正': 'shots_on_target',
            '进攻': 'attacks',
            '危险进攻': 'dangerous_attacks',
            '射门不中': 'shots_off_target',
            '射门被挡': 'shots_blocked',
            '任意球': 'free_kicks',
            '控球率': 'possession',
            '半场控球率': 'half_possession',
            '传球': 'passes',
            '传球成功率': 'pass_success_rate',
            '犯规': 'fouls',
            '越位': 'offsides',
            '黄牌': 'yellow_cards',
        }
        
        # 解析每个统计项
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
            
            # 映射到字段名
            field = stat_field_map.get(stat_name)
            if field:
                # 根据字段类型转换值
                if field in ['possession', 'half_possession', 'pass_success_rate']:
                    home_stats[field] = home_val.replace('%', '')
                    away_stats[field] = away_val.replace('%', '')
                else:
                    home_stats[field] = self._safe_int(home_val)
                    away_stats[field] = self._safe_int(away_val)
        
        # 保存到recent_form表（插入当前比赛的独立记录）
        if home_stats or away_stats:
            # 优先使用直接传入的球队名称
            if not home_team_name or not away_team_name:
                home_team_name, away_team_name = self._get_team_names_from_db(match_id)
            self.save_tech_stats_to_recent_form(match_id, home_stats, away_stats, home_team_name, away_team_name)
            print(f"    ✓ 保存技术统计: 主队{len(home_stats)}项, 客队{len(away_stats)}项")
    
    def _get_team_names_from_db(self, match_id):
        """从match表查询主队和客队名称"""
        try:
            cursor = self.db.cursor()
            cursor.execute("SELECT home_team, away_team FROM `match` WHERE match_id = %s", (match_id,))
            row = cursor.fetchone()
            cursor.close()
            if row:
                return row[0], row[1]
        except Exception as e:
            print(f"    ⚠ 查询球队名失败: {e}")
        return None, None
    
    def save_tech_stats_to_recent_form(self, match_id, home_stats, away_stats, home_team_name=None, away_team_name=None):
        """将技术统计保存到recent_form表
        为当前比赛插入独立记录（is_current_match=1）
        去重：先删除该match_id已有的is_current_match=1记录，再重新插入
        """
        cursor = self.db.cursor()
        
        # 去重：删除旧的技术统计记录，确保重新爬取时数据正确
        cursor.execute("DELETE FROM `recent_form` WHERE match_id = %s AND is_current_match = 1", (match_id,))
        
        tech_fields = ['corners', 'half_corners', 'yellow_cards', 'shots', 'shots_on_target',
                       'attacks', 'dangerous_attacks', 'shots_off_target', 'shots_blocked',
                       'free_kicks', 'possession', 'half_possession', 'passes', 'pass_success_rate',
                       'fouls', 'offsides']
        
        for team_type, stats, team_name in [('home', home_stats, home_team_name),
                                              ('away', away_stats, away_team_name)]:
            if not stats:
                continue
            
            # 构建INSERT语句，插入当前比赛的独立记录
            columns = ['match_id', 'team_type', 'team_name', 'home_team', 'away_team', 'is_current_match']
            values = [match_id, team_type, team_name, home_team_name, away_team_name, 1]
            
            for field in tech_fields:
                if field in stats:
                    columns.append(field)
                    values.append(stats[field])
            
            col_str = ', '.join(f'`{c}`' for c in columns)
            placeholders = ', '.join(['%s'] * len(values))
            sql = f"INSERT INTO `recent_form` ({col_str}) VALUES ({placeholders})"
            cursor.execute(sql, values)
        
        self.db.commit()
        cursor.close()

    def _scrape_technical_stats_from_analysis(self, match_id, soup):
        """从分析页面爬取赛季数据统计
        页面结构：h2#017(本赛季数据统计比较) -> table
        表格包含6个区域：
          主队战绩统计 / 主队得失球统计 / 主队净胜球统计
          客队战绩统计 / 客队得失球统计 / 客队净胜球统计
        每个区域有 总/主场(或客场)/近六场 三个范围
        """
        print("    爬取赛季统计...")
        stats_data = []

        # 找到h2#017 "本赛季数据统计比较"
        h2_stats = soup.find('h2', id='017')
        if not h2_stats:
            return

        table = h2_stats.find_next_sibling('table')
        if not table:
            table = h2_stats.find_next('table')
        if not table:
            return

        rows = table.find_all('tr')
        if len(rows) < 10:
            return

        # 区域标题映射
        section_map = {
            '主队战绩统计': ('home', 'record'),
            '客队战绩统计': ('away', 'record'),
            '主队得失球统计': ('home', 'goals'),
            '客队得失球统计': ('away', 'goals'),
            '主队净胜球统计': ('home', 'goal_diff'),
            '客队净胜球统计': ('away', 'goal_diff'),
        }

        current_team_type = None
        current_section = None
        current_scope = 'total'

        # 范围关键词映射
        scope_keywords = {
            '总': 'total',
            '主场': 'home',
            '客场': 'away',
            '近六场': 'recent6',
            '近10': 'recent10',
        }

        for row in rows:
            cells = row.find_all(['td', 'th'])
            texts = [c.get_text(strip=True) for c in cells]

            if not texts:
                continue

            # 检查是否是区域标题行（单cell，包含"统计"关键词）
            full_text = texts[0]
            if len(cells) == 1 and full_text in section_map:
                current_team_type, current_section = section_map[full_text]
                current_scope = 'total'
                continue

            # 跳过标题行（单cell但不是已知区域）
            if len(cells) == 1:
                continue

            if current_team_type is None:
                continue

            # 检查是否有范围变化（cell带rowspan=3或4，文本为范围关键词）
            for c in cells:
                c_text = c.get_text(strip=True)
                if c_text in scope_keywords and c.get('rowspan'):
                    current_scope = scope_keywords[c_text]
                    break

            # 解析数据行：结构通常是 stat_name | value 或 value | stat_name
            if len(texts) >= 2 and current_section:
                self._parse_season_stats_row(
                    stats_data, match_id, current_team_type,
                    current_section, current_scope, texts, cells
                )

        if stats_data:
            # 去重：先删除该match_id已有的赛季统计数据
            cursor = self.db.cursor()
            cursor.execute("DELETE FROM season_stats WHERE match_id = %s", (match_id,))
            self.db.commit()
            cursor.close()
            self.save_season_stats(stats_data)
            print(f"    ✓ 保存 {len(stats_data)} 条赛季统计数据")

    def _parse_season_stats_row(self, stats_data, match_id, team_type, section, scope, texts, cells):
        """解析赛季统计的一行数据
        规则：纯中文=统计项名称，包含数字=统计值
        跳过汇总行（第一个cell很长，包含多个统计项拼接）
        """
        # 过滤掉空文本和范围标记
        filtered = []
        for i, t in enumerate(texts):
            if t in ['总', '主场', '客场', '近六场', '近10']:
                continue
            if t == '':
                continue
            filtered.append((i, t))

        if len(filtered) < 2:
            return

        # 跳过汇总行：第一个cell文本很长（>30字符），说明是多个统计项拼接
        if len(filtered) >= 2 and len(filtered[0][1]) > 30:
            return

        # 简单规则：纯中文(含少量标奌)=名称，包含数字但不是名称的就是值（如"38% [7场]""[5场] 29%""16.67%[3场]"）
        stat_name = None
        stat_value = None

        for idx, (i, t) in enumerate(filtered):
            # 名称判断：中文字符占文本50%以上（如"胜""进球总数""净胜2球或以上"）
            chinese_chars = re.findall(r'[\u4e00-\u9fff]', t)
            is_name = len(chinese_chars) >= 1 and len(t) <= 15 and (len(chinese_chars) / len(t) >= 0.5)
            # 包含数字但不是名称的就是值（如"38% [7场]""[5场] 29%""16.67%[3场]"）
            has_digit = any(c.isdigit() for c in t)

            if is_name and stat_name is None:
                stat_name = t
            elif has_digit and not is_name and stat_value is None:
                stat_value = t

        if stat_name and stat_value:
            # 清理value格式：去掉方括号
            stat_value = stat_value.replace('[', '').replace(']', '').strip()
            stats_data.append({
                'match_id': match_id,
                'team_type': team_type,
                'section': section,
                'scope': scope,
                'stat_name': stat_name,
                'stat_value': stat_value,
            })

    def _extract_stat_value(self, text):
        """从文本中提取统计数值"""
        if not text:
            return None
        # 匹配数字或百分比
        match = re.match(r'^(\d+%?)$', text.strip())
        if match:
            return match.group(1)
        return None



    # ==================== 辅助方法 ====================
    def clean_team_name(self, name):
        """清理球队名称：去掉排名数字前缀和(中)后缀，以及尾部数字
        例: '1济州SK' -> '济州SK', '米阿尔比(中)' -> '米阿尔比', '迪弗当日2' -> '迪弗当日'
        """
        if not name:
            return name
        # 去掉开头的排名数字（如"1济州SK" -> "济州SK"）
        name = re.sub(r'^\d+', '', name).strip()
        # 去掉结尾的(中)标记
        name = re.sub(r'[\(（]中[\)）]$', '', name).strip()
        # 去掉尾部的数字（如"迪弗当日2" -> "迪弗当日"）
        name = re.sub(r'\d+$', '', name).strip()
        return name

    def _safe_int(self, val):
        """安全转换为整数"""
        try:
            return int(val)
        except (ValueError, TypeError):
            return 0

    def _safe_float(self, val):
        """安全转换为浮点数"""
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    def _is_match_scraped(self, match_id):
        """检查比赛是否已经爬取过详情数据（用于断点续传）
        判断依据：recent_form表中存在该match_id且is_current_match=0的记录
        说明该比赛的近期战绩（含技术统计）已经爬取过
        """
        try:
            cursor = self.db.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM `recent_form` WHERE match_id = %s AND is_current_match = 0",
                (match_id,)
            )
            count = cursor.fetchone()[0]
            cursor.close()
            return count > 0
        except Exception:
            return False

    # ==================== 主流程 ====================
    def run(self, max_matches=None, resume=False, proxy=None):
        """执行完整爬取流程
        Args:
            max_matches: 最多处理的比赛数量，None表示全部
            resume: 是否启用断点续传，True时跳过已爬取详情的比赛
            proxy: 代理地址(如 http://ip:port)，None时使用config中的CHROME_PROXY
        返回: True=爬取成功, False=失败(供入口脚本返回非零退出码, 便于定时任务监控)
        """
        print("="*60)
        print("新球体育北单比赛数据爬虫")
        print(f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if max_matches:
            print(f"模式: 限制爬取 {max_matches} 场比赛")
        if resume:
            print("模式: 断点续传（跳过已爬取详情的比赛）")
        print("="*60)

        success = False
        try:
            self.init_browser(proxy=proxy)
            self.init_db()

            # 第一步：爬取比赛列表
            matches = self.scrape_match_list()

            if not matches:
                print("\n⚠ 未获取到比赛数据，请检查：")
                print("  1. 网络连接是否正常")
                print("  2. 页面筛选是否成功")
                print("  3. 页面结构是否变化")
                return False

            # 限制处理数量
            if max_matches:
                matches = matches[:max_matches]
                print(f"\n⚡ 限制处理前 {max_matches} 场比赛")

            # 第二步：对每场比赛爬取详细数据（包括技术统计）
            total = len(matches)
            skipped = 0
            for idx, match in enumerate(matches, 1):
                # 断点续传：跳过已爬取过详情的比赛
                if resume and self._is_match_scraped(match['match_id']):
                    skipped += 1
                    print(f"\n[{idx}/{total}] ⊘ 跳过: {match['match_id']} {match['home_team']} vs {match['away_team']} (已爬取)")
                    continue

                print(f"\n[{idx}/{total}] 处理比赛: {match['match_id']} {match['home_team']} vs {match['away_team']}")

                analysis_url = match.get('analysis_url')
                detail_url = match.get('detail_url')
                if analysis_url:
                    self.scrape_analysis_page(
                        match['match_id'], analysis_url, detail_url,
                        home_team=match.get('home_team'),
                        away_team=match.get('away_team')
                    )
                    time.sleep(REQUEST_DELAY)

            if resume and skipped > 0:
                print(f"\n 断点续传: 跳过 {skipped} 场已爬取的比赛")

            print("\n" + "="*60)
            print("🎉 爬取完成！")
            print("="*60)
            success = True

        except Exception as e:
            print(f"\n✗ 爬取出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.close_browser()
            self.close_db()
        return success


if __name__ == '__main__':
    import sys
    scraper = TitanScraper()
    # 支持命令行参数:
    #   python scraper.py              -> 爬取全部比赛
    #   python scraper.py 2            -> 只爬取前2场比赛
    #   python scraper.py --resume     -> 断点续传（跳过已爬取的比赛）
    #   python scraper.py 2 --resume   -> 断点续传 + 限制数量
    #   python scraper.py --proxy http://1.2.3.4:8080 -> 通过代理爬取(服务器IP被限制时使用)
    max_matches = None
    resume = False
    proxy = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--resume':
            resume = True
        elif arg == '--proxy':
            if i + 1 < len(args):
                proxy = args[i + 1]
                i += 1
            else:
                print(" --proxy 需要指定代理地址，如: --proxy http://1.2.3.4:8080")
        else:
            try:
                max_matches = int(arg)
            except ValueError:
                print(f" 未知参数: {arg}，忽略")
        i += 1
    ok = scraper.run(max_matches=max_matches, resume=resume, proxy=proxy)
    sys.exit(0 if ok else 1)
