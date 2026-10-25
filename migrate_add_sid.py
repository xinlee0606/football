#!/usr/bin/env python3
"""数据库迁移：为 match 表添加 sid 列
sid 来自网页 HTML 行的 sid 属性，是稳定的比赛标识符，不随比赛增减而变化。
已有的 single_match（单场XX）是动态编号，会随网站增删比赛而偏移。

执行方式：
    python migrate_add_sid.py

幂等安全：如果 sid 列已存在则跳过。
"""
import pymysql
from config import DB_CONFIG


def migrate():
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # 检查 sid 列是否已存在
    cursor.execute(
        "SELECT COUNT(*) FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'match' AND COLUMN_NAME = 'sid'",
        (DB_CONFIG['database'],)
    )
    exists = cursor.fetchone()[0]

    if exists:
        print("✓ sid 列已存在，无需迁移")
    else:
        cursor.execute(
            "ALTER TABLE `match` ADD COLUMN `sid` VARCHAR(20) DEFAULT NULL "
            "COMMENT '网站稳定比赛ID(tr行sid属性,不随比赛增减变化)' AFTER `site_match_id`"
        )
        conn.commit()
        print("✓ 已添加 sid 列")

    # 回填已有比赛的 sid（从 site_match_id 复制，因为旧数据中两者相同）
    cursor.execute(
        "UPDATE `match` SET sid = site_match_id WHERE sid IS NULL AND site_match_id IS NOT NULL"
    )
    conn.commit()
    print(f"✓ 回填 sid 完成（{cursor.rowcount} 条记录）")

    cursor.close()
    conn.close()
    print("\n🎉 迁移完成！")


if __name__ == '__main__':
    migrate()
