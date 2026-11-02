# init_db.py - 数据库初始化脚本
import pymysql
from config import DB_CONFIG


def init_database():
    """创建数据库和所有表"""
    # 先连接MySQL创建数据库
    conn = pymysql.connect(
        host=DB_CONFIG['host'],
        port=DB_CONFIG['port'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
        charset='utf8mb4'
    )
    cursor = conn.cursor()
    cursor.execute("CREATE DATABASE IF NOT EXISTS match_live DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    cursor.execute("USE match_live")
    conn.commit()
    cursor.close()
    conn.close()
    print("✓ 数据库 match_live 创建成功")

    # 连接数据库创建表
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    tables = {
        'match': """
            CREATE TABLE IF NOT EXISTS `match` (
                `id` INT AUTO_INCREMENT PRIMARY KEY,
                `match_id` VARCHAR(20) NOT NULL COMMENT '比赛编号 如2026071701',
                `site_match_id` VARCHAR(20) DEFAULT NULL COMMENT '网站原始比赛ID 如3027598',
                `sid` VARCHAR(20) DEFAULT NULL COMMENT '网站稳定比赛ID(tr行sid属性,不随比赛增减变化)',
                `league` VARCHAR(100) COMMENT '联赛名称',
                `single_match` VARCHAR(50) DEFAULT NULL COMMENT '单场编号 如单场83',
                `match_time` VARCHAR(50) COMMENT '比赛时间',
                `status` VARCHAR(50) DEFAULT NULL COMMENT '比赛状态',
                `home_team` VARCHAR(100) COMMENT '主队名称',
                `home_rank` VARCHAR(10) DEFAULT NULL COMMENT '主队排名标记',
                `away_team` VARCHAR(100) COMMENT '客队名称',
                `away_rank` VARCHAR(10) DEFAULT NULL COMMENT '客队排名标记',
                `score` VARCHAR(20) DEFAULT NULL COMMENT '比分',
                `analysis_url` VARCHAR(500) DEFAULT NULL COMMENT '分析页面URL',
                `detail_url` VARCHAR(500) DEFAULT NULL COMMENT '技术统计页面URL',
                `prediction` VARCHAR(20) DEFAULT NULL COMMENT '预测结果: 胜平/平负',
                `page_handicap` VARCHAR(30) DEFAULT NULL COMMENT '完场页亚盘(*号已转受让)',
                `page_goals` VARCHAR(30) DEFAULT NULL COMMENT '完场页进球数线',
                `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE INDEX idx_site_match_id (`site_match_id`),
                INDEX idx_match_id (`match_id`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='比赛基本信息表'
        """,

        'league_standings': """
            CREATE TABLE IF NOT EXISTS `league_standings` (
                `id` INT AUTO_INCREMENT PRIMARY KEY,
                `match_id` VARCHAR(20) NOT NULL COMMENT '关联比赛编号',
                `team_type` VARCHAR(10) NOT NULL COMMENT '主队home/客队away',
                `team_name` VARCHAR(100) COMMENT '球队名称',
                `league_name` VARCHAR(100) COMMENT '联赛名称',
                `scope` VARCHAR(10) NOT NULL COMMENT '总total/主home/客away/近6recent6',
                `played` INT DEFAULT 0 COMMENT '赛',
                `won` INT DEFAULT 0 COMMENT '胜',
                `drawn` INT DEFAULT 0 COMMENT '平',
                `lost` INT DEFAULT 0 COMMENT '负',
                `goals_for` INT DEFAULT 0 COMMENT '得',
                `goals_against` INT DEFAULT 0 COMMENT '失',
                `goal_diff` INT DEFAULT 0 COMMENT '净',
                `points` INT DEFAULT 0 COMMENT '积分',
                `ranking` INT DEFAULT 0 COMMENT '排名',
                `win_rate` VARCHAR(20) DEFAULT NULL COMMENT '胜率',
                `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_match (`match_id`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='联赛积分排名表'
        """,

        'head_to_head': """
            CREATE TABLE IF NOT EXISTS `head_to_head` (
                `id` INT AUTO_INCREMENT PRIMARY KEY,
                `match_id` VARCHAR(20) NOT NULL COMMENT '关联比赛编号',
                `match_date` VARCHAR(50) DEFAULT NULL COMMENT '比赛日期',
                `league` VARCHAR(100) DEFAULT NULL COMMENT '联赛',
                `home_team` VARCHAR(100) DEFAULT NULL COMMENT '主队',
                `score` VARCHAR(20) DEFAULT NULL COMMENT '比分',
                `away_team` VARCHAR(100) DEFAULT NULL COMMENT '客队',
                `half_score` VARCHAR(20) DEFAULT NULL COMMENT '半场比分',
                `corners` VARCHAR(20) DEFAULT NULL COMMENT '角球(主队-客队)',
                `initial_home_odds` VARCHAR(20) DEFAULT NULL COMMENT '36*初盘主队水位',
                `initial_handicap` VARCHAR(50) DEFAULT NULL COMMENT '36*初盘盘口',
                `initial_away_odds` VARCHAR(20) DEFAULT NULL COMMENT '36*初盘客队水位',
                `result` VARCHAR(10) DEFAULT NULL COMMENT '胜负: 胜/平/负',
                `handicap_result` VARCHAR(10) DEFAULT NULL COMMENT '让球: 赢/输',
                `over_under_result` VARCHAR(10) DEFAULT NULL COMMENT '进球数: 大/小',
                `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_match (`match_id`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='对赛往绩表'
        """,

        'recent_form': """
            CREATE TABLE IF NOT EXISTS `recent_form` (
                `id` INT AUTO_INCREMENT PRIMARY KEY,
                `match_id` VARCHAR(20) NOT NULL COMMENT '关联比赛编号',
                `team_type` VARCHAR(10) NOT NULL COMMENT '主队home/客队away',
                `team_name` VARCHAR(100) DEFAULT NULL COMMENT '球队名称',
                `match_date` VARCHAR(50) DEFAULT NULL COMMENT '比赛日期',
                `league` VARCHAR(100) DEFAULT NULL COMMENT '联赛',
                `home_team` VARCHAR(100) DEFAULT NULL COMMENT '主队',
                `score` VARCHAR(20) DEFAULT NULL COMMENT '比分',
                `away_team` VARCHAR(100) DEFAULT NULL COMMENT '客队',
                `result` VARCHAR(10) DEFAULT NULL COMMENT '胜W/平D/负L',
                `detail_url` VARCHAR(500) DEFAULT NULL COMMENT '技术统计详情页URL',
                `is_current_match` TINYINT DEFAULT 0 COMMENT '是否当前比赛: 1=是 0=否(历史战绩)',
                `initial_home_odds` VARCHAR(20) DEFAULT NULL COMMENT '36*初盘主队水位',
                `initial_handicap` VARCHAR(50) DEFAULT NULL COMMENT '36*初盘盘口',
                `initial_away_odds` VARCHAR(20) DEFAULT NULL COMMENT '36*初盘客队水位',
                `handicap_result` VARCHAR(10) DEFAULT NULL COMMENT '让球: 赢/输/走',
                `over_under_result` VARCHAR(10) DEFAULT NULL COMMENT '进球数: 大/小/走',
                `corners` INT DEFAULT NULL COMMENT '角球',
                `half_corners` INT DEFAULT NULL COMMENT '半场角球',
                `yellow_cards` INT DEFAULT NULL COMMENT '黄牌',
                `shots` INT DEFAULT NULL COMMENT '射门',
                `shots_on_target` INT DEFAULT NULL COMMENT '射正',
                `attacks` INT DEFAULT NULL COMMENT '进攻',
                `dangerous_attacks` INT DEFAULT NULL COMMENT '危险进攻',
                `shots_off_target` INT DEFAULT NULL COMMENT '射门不中',
                `shots_blocked` INT DEFAULT NULL COMMENT '射门被挡',
                `free_kicks` INT DEFAULT NULL COMMENT '任意球',
                `possession` VARCHAR(10) DEFAULT NULL COMMENT '控球率',
                `half_possession` VARCHAR(10) DEFAULT NULL COMMENT '半场控球率',
                `passes` INT DEFAULT NULL COMMENT '传球',
                `pass_success_rate` VARCHAR(10) DEFAULT NULL COMMENT '传球成功率',
                `fouls` INT DEFAULT NULL COMMENT '犯规',
                `offsides` INT DEFAULT NULL COMMENT '越位',
                `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_match (`match_id`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='近期战绩表（含技术统计）'
        """,

        'asian_odds': """
            CREATE TABLE IF NOT EXISTS `asian_odds` (
                `id` INT AUTO_INCREMENT PRIMARY KEY,
                `match_id` VARCHAR(20) NOT NULL COMMENT '关联比赛编号',
                `company` VARCHAR(100) DEFAULT NULL COMMENT '博彩公司',
                `initial_handicap` VARCHAR(50) DEFAULT NULL COMMENT '初盘让球',
                `initial_home_odds` VARCHAR(20) DEFAULT NULL COMMENT '初盘主队水位',
                `initial_away_odds` VARCHAR(20) DEFAULT NULL COMMENT '初盘客队水位',
                `current_handicap` VARCHAR(50) DEFAULT NULL COMMENT '即时让球',
                `current_home_odds` VARCHAR(20) DEFAULT NULL COMMENT '即时主队水位',
                `current_away_odds` VARCHAR(20) DEFAULT NULL COMMENT '即时客队水位',
                `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_match (`match_id`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='亚盘数据表'
        """,

        'over_under_odds': """
            CREATE TABLE IF NOT EXISTS `over_under_odds` (
                `id` INT AUTO_INCREMENT PRIMARY KEY,
                `match_id` VARCHAR(20) NOT NULL COMMENT '关联比赛编号',
                `company` VARCHAR(100) DEFAULT NULL COMMENT '博彩公司',
                `initial_line` VARCHAR(50) DEFAULT NULL COMMENT '初盘大小球线',
                `initial_over_odds` VARCHAR(20) DEFAULT NULL COMMENT '初盘大球水位',
                `initial_under_odds` VARCHAR(20) DEFAULT NULL COMMENT '初盘小球水位',
                `current_line` VARCHAR(50) DEFAULT NULL COMMENT '即时大小球线',
                `current_over_odds` VARCHAR(20) DEFAULT NULL COMMENT '即时大球水位',
                `current_under_odds` VARCHAR(20) DEFAULT NULL COMMENT '即时小球水位',
                `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_match (`match_id`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='大小球数据表'
        """,

        'european_odds': """
            CREATE TABLE IF NOT EXISTS `european_odds` (
                `id` INT AUTO_INCREMENT PRIMARY KEY,
                `match_id` VARCHAR(20) NOT NULL COMMENT '关联比赛编号',
                `company` VARCHAR(100) DEFAULT NULL COMMENT '博彩公司',
                `initial_win_odds` VARCHAR(20) DEFAULT NULL COMMENT '初盘主胜赔率',
                `initial_draw_odds` VARCHAR(20) DEFAULT NULL COMMENT '初盘平局赔率',
                `initial_loss_odds` VARCHAR(20) DEFAULT NULL COMMENT '初盘客胜赔率',
                `current_win_odds` VARCHAR(20) DEFAULT NULL COMMENT '即时主胜赔率',
                `current_draw_odds` VARCHAR(20) DEFAULT NULL COMMENT '即时平局赔率',
                `current_loss_odds` VARCHAR(20) DEFAULT NULL COMMENT '即时客胜赔率',
                `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_match (`match_id`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='欧赔数据表'
        """,



        'season_stats': """
            CREATE TABLE IF NOT EXISTS `season_stats` (
                `id` INT AUTO_INCREMENT PRIMARY KEY,
                `match_id` VARCHAR(20) NOT NULL COMMENT '关联比赛编号',
                `team_type` VARCHAR(10) NOT NULL COMMENT '主队home/客队away',
                `section` VARCHAR(20) NOT NULL COMMENT '统计区域: record战绩/goals得失球/goal_diff净胜球',
                `scope` VARCHAR(20) NOT NULL COMMENT '范围: total总/home主场/away客场/recent6近六场',
                `stat_name` VARCHAR(50) NOT NULL COMMENT '统计项名称',
                `stat_value` VARCHAR(50) DEFAULT NULL COMMENT '统计值',
                `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_match (`match_id`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='赛季统计数据表'
        """,

        'result_sp': """
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
    }

    for table_name, sql in tables.items():
        cursor.execute(sql)
        print(f"✓ 表 {table_name} 创建成功")

    conn.commit()
    cursor.close()
    conn.close()
    print("\n🎉 数据库初始化完成！所有表已创建。")


if __name__ == '__main__':
    init_database()
