"""
数据质量监控模块 - Data Quality Monitor
提供特征分布漂移检测、异常值标记、数据新鲜度监控、质量报告生成
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats


class DataQualityMonitor:
    """数据质量监控器"""
    
    def __init__(self, report_path: str = 'data_quality_report.json'):
        """初始化监控器
        
        Args:
            report_path: 质量报告输出路径
        """
        self.report_path = report_path
        self.thresholds = {
            'missing_rate': 0.10,      # 缺失率阈值10%
            'psi': 0.20,               # PSI阈值0.2
            'outlier_std': 3.0,        # 异常值标准差倍数
            'max_data_age_days': 7,    # 数据最大年龄（天）
            'high_correlation': 0.95   # 高相关性阈值
        }
        self.report = {}
    
    def calculate_psi(self, expected: pd.Series, actual: pd.Series, 
                      bins: int = 10) -> float:
        """计算PSI（Population Stability Index）
        
        Args:
            expected: 期望分布（训练集）
            actual: 实际分布（测试集/新数据）
            bins: 分箱数量
            
        Returns:
            PSI值
        """
        # 移除NaN
        expected = expected.dropna()
        actual = actual.dropna()
        
        if len(expected) == 0 or len(actual) == 0:
            return 0.0
        
        # 创建分箱边界
        breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
        breakpoints = np.unique(breakpoints)  # 移除重复值
        
        if len(breakpoints) < 3:
            return 0.0
        
        # 计算每个bin的占比
        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]
        
        # 转换为百分比，避免除零
        expected_pct = (expected_counts + 1) / (len(expected) + bins)
        actual_pct = (actual_counts + 1) / (len(actual) + bins)
        
        # 计算PSI
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        
        return psi
    
    def detect_outliers(self, series: pd.Series, method: str = 'zscore') -> pd.Series:
        """检测异常值
        
        Args:
            series: 输入序列
            method: 检测方法 ('zscore' 或 'iqr')
            
        Returns:
            布尔序列，True表示异常值
        """
        series = series.dropna()
        
        if method == 'zscore':
            # Z-score方法
            z_scores = np.abs(stats.zscore(series))
            return z_scores > self.thresholds['outlier_std']
        
        elif method == 'iqr':
            # IQR方法
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (series < lower_bound) | (series > upper_bound)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def check_missing_rate(self, df: pd.DataFrame, 
                          feature_cols: List[str]) -> Dict[str, Any]:
        """检查缺失率
        
        Args:
            df: 输入DataFrame
            feature_cols: 特征列名列表
            
        Returns:
            缺失率检查结果
        """
        missing_rates = df[feature_cols].isnull().mean()
        
        high_missing = missing_rates[missing_rates > self.thresholds['missing_rate']]
        
        result = {
            'avg_missing_rate': float(missing_rates.mean()),
            'max_missing_rate': float(missing_rates.max()),
            'features_above_threshold': len(high_missing),
            'high_missing_features': {
                col: float(rate) 
                for col, rate in high_missing.items()
            }
        }
        
        return result
    
    def check_distribution_drift(self, train_df: pd.DataFrame, 
                                test_df: pd.DataFrame,
                                feature_cols: List[str]) -> Dict[str, Any]:
        """检查分布漂移
        
        Args:
            train_df: 训练集
            test_df: 测试集/新数据
            feature_cols: 特征列名列表
            
        Returns:
            漂移检查结果
        """
        psi_values = {}
        drifted_features = []
        
        for col in feature_cols:
            if col in train_df.columns and col in test_df.columns:
                psi = self.calculate_psi(train_df[col], test_df[col])
                psi_values[col] = float(psi)
                
                if psi > self.thresholds['psi']:
                    drifted_features.append(col)
        
        result = {
            'avg_psi': float(np.mean(list(psi_values.values()))) if psi_values else 0.0,
            'max_psi': float(max(psi_values.values())) if psi_values else 0.0,
            'features_above_threshold': len(drifted_features),
            'drifted_features': drifted_features,
            'psi_values': psi_values
        }
        
        return result
    
    def check_outliers(self, df: pd.DataFrame, 
                      feature_cols: List[str]) -> Dict[str, Any]:
        """检查异常值
        
        Args:
            df: 输入DataFrame
            feature_cols: 特征列名列表
            
        Returns:
            异常值检查结果
        """
        outlier_counts = {}
        total_outliers = 0
        
        for col in feature_cols:
            outliers = self.detect_outliers(df[col])
            count = int(outliers.sum())
            outlier_counts[col] = count
            total_outliers += count
        
        result = {
            'total_outliers': total_outliers,
            'avg_outliers_per_feature': total_outliers / len(feature_cols) if feature_cols else 0,
            'outlier_counts': outlier_counts
        }
        
        return result
    
    def check_data_freshness(self, df: pd.DataFrame, 
                            date_col: str = 'match_time') -> Dict[str, Any]:
        """检查数据新鲜度
        
        Args:
            df: 输入DataFrame
            date_col: 日期列名
            
        Returns:
            新鲜度检查结果
        """
        if date_col not in df.columns:
            return {'error': f'Date column {date_col} not found'}
        
        # 转换为datetime
        dates = pd.to_datetime(df[date_col], errors='coerce')
        
        if dates.isna().all():
            return {'error': 'No valid dates found'}
        
        latest_date = dates.max()
        data_age = (datetime.now() - latest_date).days
        
        result = {
            'latest_date': latest_date.strftime('%Y-%m-%d'),
            'data_age_days': int(data_age),
            'is_fresh': data_age <= self.thresholds['max_data_age_days']
        }
        
        return result
    
    def check_high_correlations(self, df: pd.DataFrame, 
                               feature_cols: List[str]) -> Dict[str, Any]:
        """检查高相关性特征
        
        Args:
            df: 输入DataFrame
            feature_cols: 特征列名列表
            
        Returns:
            高相关性检查结果
        """
        # 计算相关矩阵
        corr_matrix = df[feature_cols].corr()
        
        # 找出高相关性特征对
        high_corr_pairs = []
        
        for i in range(len(feature_cols)):
            for j in range(i + 1, len(feature_cols)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > self.thresholds['high_correlation']:
                    high_corr_pairs.append({
                        'feature1': feature_cols[i],
                        'feature2': feature_cols[j],
                        'correlation': float(corr)
                    })
        
        result = {
            'high_correlation_pairs': len(high_corr_pairs),
            'pairs': high_corr_pairs
        }
        
        return result
    
    def generate_quality_report(self, df: pd.DataFrame, 
                               feature_cols: List[str],
                               model_name: str = 'unknown',
                               train_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """生成完整的质量报告
        
        Args:
            df: 输入DataFrame
            feature_cols: 特征列名列表
            model_name: 模型名称
            train_df: 训练集（用于漂移检测）
            
        Returns:
            完整质量报告
        """
        print(f"Generating quality report for {model_name}...")
        
        report = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(df),
            'total_features': len(feature_cols)
        }
        
        # 1. 缺失率检查
        print("  Checking missing rates...")
        report['missing_rate'] = self.check_missing_rate(df, feature_cols)
        
        # 2. 异常值检查
        print("  Checking outliers...")
        report['outliers'] = self.check_outliers(df, feature_cols)
        
        # 3. 数据新鲜度检查
        print("  Checking data freshness...")
        report['freshness'] = self.check_data_freshness(df)
        
        # 4. 高相关性检查
        print("  Checking high correlations...")
        report['correlations'] = self.check_high_correlations(df, feature_cols)
        
        # 5. 分布漂移检查（如果有训练集）
        if train_df is not None:
            print("  Checking distribution drift...")
            report['drift'] = self.check_distribution_drift(train_df, df, feature_cols)
        
        # 6. 总体评分
        print("  Calculating overall score...")
        report['overall_score'] = self._calculate_overall_score(report)
        
        # 保存报告
        self.report[model_name] = report
        self._save_report()
        
        print(f"Quality report saved to {self.report_path}")
        
        return report
    
    def _calculate_overall_score(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """计算总体质量评分
        
        Args:
            report: 质量报告
            
        Returns:
            总体评分
        """
        score = 100
        issues = []
        
        # 缺失率扣分
        avg_missing = report['missing_rate']['avg_missing_rate']
        if avg_missing > 0.05:
            score -= 20
            issues.append(f"High average missing rate: {avg_missing:.1%}")
        
        # 异常值扣分
        avg_outliers = report['outliers']['avg_outliers_per_feature']
        if avg_outliers > 10:
            score -= 15
            issues.append(f"Too many outliers: {avg_outliers:.1f} per feature")
        
        # 数据新鲜度扣分
        if not report['freshness'].get('is_fresh', True):
            score -= 25
            issues.append(f"Data is stale: {report['freshness'].get('data_age_days', 'N/A')} days old")
        
        # 分布漂移扣分
        if 'drift' in report:
            avg_psi = report['drift']['avg_psi']
            if avg_psi > 0.1:
                score -= 20
                issues.append(f"Significant distribution drift: PSI={avg_psi:.3f}")
        
        # 高相关性扣分
        high_corr_count = report['correlations']['high_correlation_pairs']
        if high_corr_count > 5:
            score -= 10
            issues.append(f"Too many highly correlated features: {high_corr_count} pairs")
        
        return {
            'score': max(0, score),
            'grade': self._score_to_grade(score),
            'issues': issues
        }
    
    def _score_to_grade(self, score: int) -> str:
        """将分数转换为等级
        
        Args:
            score: 分数
            
        Returns:
            等级（A/B/C/D/F）
        """
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _save_report(self):
        """保存报告到JSON文件"""
        with open(self.report_path, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, ensure_ascii=False, indent=2)
    
    def load_report(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """加载报告
        
        Args:
            model_name: 模型名称，如果为None则返回所有报告
            
        Returns:
            报告数据
        """
        if not os.path.exists(self.report_path):
            return {}
        
        with open(self.report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        if model_name:
            return report.get(model_name, {})
        
        return report
    
    def print_summary(self, model_name: str):
        """打印模型质量摘要
        
        Args:
            model_name: 模型名称
        """
        report = self.load_report(model_name)
        
        if not report:
            print(f"No report found for {model_name}")
            return
        
        print(f"\n{'='*60}")
        print(f"Data Quality Report: {model_name}")
        print(f"{'='*60}")
        print(f"Timestamp: {report.get('timestamp', 'N/A')}")
        print(f"Samples: {report.get('total_samples', 'N/A')}")
        print(f"Features: {report.get('total_features', 'N/A')}")
        print(f"\nOverall Score: {report['overall_score']['score']}/100 ({report['overall_score']['grade']})")
        
        if report['overall_score']['issues']:
            print("\nIssues:")
            for issue in report['overall_score']['issues']:
                print(f"  - {issue}")
        
        print(f"\nMissing Rate: {report['missing_rate']['avg_missing_rate']:.1%} avg, {report['missing_rate']['max_missing_rate']:.1%} max")
        print(f"Outliers: {report['outliers']['total_outliers']} total")
        print(f"Data Age: {report['freshness'].get('data_age_days', 'N/A')} days")
        
        if 'drift' in report:
            print(f"Distribution Drift: PSI={report['drift']['avg_psi']:.3f} avg")
        
        print(f"High Correlations: {report['correlations']['high_correlation_pairs']} pairs")
        print(f"{'='*60}\n")


# 便捷函数
def monitor_data_quality(df: pd.DataFrame, feature_cols: List[str],
                         model_name: str = 'unknown',
                         train_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """便捷函数：监控数据质量
    
    Args:
        df: 输入DataFrame
        feature_cols: 特征列名列表
        model_name: 模型名称
        train_df: 训练集（用于漂移检测）
        
    Returns:
        质量报告
    """
    monitor = DataQualityMonitor()
    return monitor.generate_quality_report(df, feature_cols, model_name, train_df)


if __name__ == '__main__':
    # 测试代码
    import pandas as pd
    import numpy as np
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 1000
    
    test_data = {
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'match_time': pd.date_range('2026-10-01', periods=n_samples, freq='H')
    }
    
    # 添加一些缺失值
    test_data['feature1'][np.random.choice(n_samples, 50)] = np.nan
    test_data['feature2'][np.random.choice(n_samples, 100)] = np.nan
    
    df = pd.DataFrame(test_data)
    
    # 运行监控
    monitor = DataQualityMonitor()
    report = monitor.generate_quality_report(
        df=df,
        feature_cols=['feature1', 'feature2', 'feature3'],
        model_name='test_model'
    )
    
    # 打印摘要
    monitor.print_summary('test_model')
