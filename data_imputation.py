"""
数据补全模块 - Data Imputation Module
提供三级补全策略：联赛均值 -> 球队历史均值 -> 全局均值
"""

import pandas as pd
import numpy as np
import pymysql
from typing import Dict, List, Optional, Any
from config import DB_CONFIG


class DataImputer:
    """数据补全器：支持角球、进球等数据的智能补全"""
    
    def __init__(self, conn=None):
        """初始化补全器
        
        Args:
            conn: 数据库连接，如果为None则自动创建
        """
        self.conn = conn or self._create_connection()
        self._cache = {}  # 缓存计算结果
        
    def _create_connection(self):
        """创建数据库连接"""
        return pymysql.connect(**DB_CONFIG)
    
    def get_league_avg_corners(self, league: str) -> Optional[float]:
        """获取联赛平均角球数
        
        Args:
            league: 联赛名称
            
        Returns:
            平均角球数，如果无数据返回None
        """
        cache_key = f"league_corners_{league}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            query = """
                SELECT AVG(home_corners + away_corners) as avg_corners
                FROM match_results
                WHERE league = %s 
                AND home_corners IS NOT NULL 
                AND away_corners IS NOT NULL
                AND match_time >= DATE_SUB(NOW(), INTERVAL 1 YEAR)
            """
            df = pd.read_sql(query, self.conn, params=(league,))
            if len(df) > 0 and pd.notna(df['avg_corners'].iloc[0]):
                result = float(df['avg_corners'].iloc[0])
                self._cache[cache_key] = result
                return result
        except Exception as e:
            print(f"Warning: Failed to get league avg corners for {league}: {e}")
        
        return None
    
    def get_team_avg_corners(self, team: str, is_home: bool = True) -> Optional[float]:
        """获取球队平均角球数
        
        Args:
            team: 球队名称
            is_home: 是否为主场
            
        Returns:
            平均角球数，如果无数据返回None
        """
        cache_key = f"team_corners_{team}_{is_home}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            if is_home:
                query = """
                    SELECT AVG(home_corners) as avg_corners
                    FROM match_results
                    WHERE home_team = %s 
                    AND home_corners IS NOT NULL
                    AND match_time >= DATE_SUB(NOW(), INTERVAL 1 YEAR)
                """
            else:
                query = """
                    SELECT AVG(away_corners) as avg_corners
                    FROM match_results
                    WHERE away_team = %s 
                    AND away_corners IS NOT NULL
                    AND match_time >= DATE_SUB(NOW(), INTERVAL 1 YEAR)
                """
            
            df = pd.read_sql(query, self.conn, params=(team,))
            if len(df) > 0 and pd.notna(df['avg_corners'].iloc[0]):
                result = float(df['avg_corners'].iloc[0])
                self._cache[cache_key] = result
                return result
        except Exception as e:
            print(f"Warning: Failed to get team avg corners for {team}: {e}")
        
        return None
    
    def get_global_avg_corners(self) -> float:
        """获取全局平均角球数（兜底值）
        
        Returns:
            全局平均角球数
        """
        cache_key = "global_corners"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            query = """
                SELECT AVG(home_corners + away_corners) as avg_corners
                FROM match_results
                WHERE home_corners IS NOT NULL 
                AND away_corners IS NOT NULL
                AND match_time >= DATE_SUB(NOW(), INTERVAL 1 YEAR)
            """
            df = pd.read_sql(query, self.conn)
            if len(df) > 0 and pd.notna(df['avg_corners'].iloc[0]):
                result = float(df['avg_corners'].iloc[0])
                self._cache[cache_key] = result
                return result
        except Exception as e:
            print(f"Warning: Failed to get global avg corners: {e}")
        
        # 默认值：9.5个角球（行业经验值）
        return 9.5
    
    def impute_corners(self, league: str, home_team: str, away_team: str) -> float:
        """三级补全策略：联赛均值 -> 球队均值 -> 全局均值
        
        Args:
            league: 联赛名称
            home_team: 主队名称
            away_team: 客队名称
            
        Returns:
            补全后的角球数
        """
        # 策略1：联赛平均值
        league_avg = self.get_league_avg_corners(league)
        if league_avg is not None:
            return league_avg
        
        # 策略2：球队历史均值（取主客队均值）
        home_avg = self.get_team_avg_corners(home_team, is_home=True)
        away_avg = self.get_team_avg_corners(away_team, is_home=False)
        
        if home_avg is not None and away_avg is not None:
            return (home_avg + away_avg) / 2
        elif home_avg is not None:
            return home_avg
        elif away_avg is not None:
            return away_avg
        
        # 策略3：全局均值（兜底）
        return self.get_global_avg_corners()
    
    def impute_feature(self, feature_name: str, league: str, 
                       home_team: str, away_team: str) -> float:
        """通用特征补全接口
        
        Args:
            feature_name: 特征名称（如 'corners', 'goals' 等）
            league: 联赛名称
            home_team: 主队名称
            away_team: 客队名称
            
        Returns:
            补全后的特征值
        """
        if feature_name == 'corners' or feature_name.endswith('_corners'):
            return self.impute_corners(league, home_team, away_team)
        
        # 其他特征可以在这里扩展
        # elif feature_name == 'goals':
        #     return self.impute_goals(league, home_team, away_team)
        
        # 默认返回NaN
        return float('nan')
    
    def batch_impute(self, df: pd.DataFrame, feature_name: str,
                     league_col: str = 'league',
                     home_col: str = 'home_team',
                     away_col: str = 'away_team') -> pd.DataFrame:
        """批量补全DataFrame中的缺失值
        
        Args:
            df: 输入DataFrame
            feature_name: 要补全的特征名称
            league_col: 联赛列名
            home_col: 主队列名
            away_col: 客队列名
            
        Returns:
            补全后的DataFrame
        """
        df = df.copy()
        
        # 找出缺失的行
        missing_mask = df[feature_name].isna()
        missing_count = missing_mask.sum()
        
        if missing_count == 0:
            return df
        
        print(f"Imputing {missing_count} missing values for {feature_name}...")
        
        # 逐行补全
        for idx in df[missing_mask].index:
            league = df.loc[idx, league_col]
            home = df.loc[idx, home_col]
            away = df.loc[idx, away_col]
            
            imputed_value = self.impute_feature(feature_name, league, home, away)
            df.loc[idx, feature_name] = imputed_value
        
        return df
    
    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()
    
    def __del__(self):
        """析构时关闭连接"""
        if hasattr(self, 'conn') and self.conn:
            try:
                self.conn.close()
            except:
                pass


# 便捷函数
def impute_corners_for_match(league: str, home_team: str, away_team: str, 
                             conn=None) -> float:
    """为单场比赛补全角球数据
    
    Args:
        league: 联赛名称
        home_team: 主队名称
        away_team: 客队名称
        conn: 数据库连接（可选）
        
    Returns:
        补全后的角球数
    """
    imputer = DataImputer(conn)
    return imputer.impute_corners(league, home_team, away_team)


if __name__ == '__main__':
    # 测试代码
    imputer = DataImputer()
    
    # 测试单场比赛补全
    test_league = '英超'
    test_home = '曼联'
    test_away = '利物浦'
    
    corners = imputer.impute_corners(test_league, test_home, test_away)
    print(f"Imputed corners for {test_home} vs {test_away}: {corners}")
    
    # 测试批量补全
    test_data = {
        'league': ['英超', '西甲', '德甲'],
        'home_team': ['曼联', '皇马', '拜仁'],
        'away_team': ['利物浦', '巴萨', '多特'],
        'corners': [None, 8.5, None]
    }
    df = pd.DataFrame(test_data)
    print("\nBefore imputation:")
    print(df)
    
    df_imputed = imputer.batch_impute(df, 'corners')
    print("\nAfter imputation:")
    print(df_imputed)
