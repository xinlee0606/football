# config.py - 爬虫配置文件

# ==================== 数据库配置 ====================
DB_CONFIG = {
    'host': 'rm-0jlt516knd10vpy18ko.mysql.rds.aliyuncs.com',
    'port': 3306,
    'user': 'root',
    'password': 'Xin7231013@@@',       # ← 请修改为你的MySQL密码
    'database': 'match_live',
    'charset': 'utf8mb4'
}

# ==================== 爬虫配置 ====================
CHROME_HEADLESS = True          # False=显示浏览器窗口(调试用), True=无头模式
# Chrome/Chromium可执行文件路径(空=自动检测)
# Linux服务器一般为 /usr/bin/chromium-browser 或 /usr/bin/google-chrome
CHROME_BINARY_PATH = ''
# 代理服务器(空=不使用)。服务器IP被目标网站限制时必需配置，
# 格式: http://ip:port 或 socks5://ip:port (不带账号密码)
# 也可用命令行参数覆盖: python scraper.py --proxy http://1.2.3.4:8080
CHROME_PROXY = ''
PAGE_LOAD_TIMEOUT = 30           # 页面加载超时(秒)
ELEMENT_WAIT_TIMEOUT = 15        # 元素等待超时(秒)
REQUEST_DELAY = 2                # 请求间隔(秒), 避免被封
MATCH_LIST_WAIT = 10             # 等待比赛列表加载时间(秒)

# ==================== 目标URL ====================
MAIN_URL = 'https://live.titan007.com/oldIndexall.aspx'
# 完场比分页面模板（{date}为YYYYMMDD格式日期），用于更新比赛状态和比分
OVER_URL_TEMPLATE = 'https://bf.titan007.com/football/Over_{date}.htm'

# ==================== 球队名称显示设置 ====================
# 网站通过语言设置(SetLanguage函数)控制球队名称显示方式，设置保存在Cookie("oldindex")中。
# Selenium打开的是全新浏览器配置文件(无Cookie)，默认显示方式与手动浏览器不同，
# 导致球队名称不一致（如手动浏览器显示"FC安养"，爬虫显示"安阳"）。
# 爬虫加载页面后会强制设置为以下语言并刷新页面：
#   0 = 简体（国语译音，如"FC安养"）← 与手动浏览器一致，推荐
#   1 = 繁体（粤语译音）
#   2 = Crow*名
TEAM_NAME_LANGUAGE = 0

# ==================== 邮件通知配置 ====================
ENABLE_EMAIL = True              # 邮件发送总开关（True=启用, False=关闭）
EMAIL_CONFIG = {
    'smtp_server': 'smtp.163.com',
    'smtp_port': 465,               # SSL端口
    'sender': 'xinlee0606@163.com',
    'password': 'GQ5CsCJA4FGjtJA2',  # 163邮箱SMTP授权码
    'receiver': '15802967941@139.com',
}
