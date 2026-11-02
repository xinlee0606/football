"""
特征选择模块 - Feature Selector
提供SHAP值分析、特征重要性排名、高相关特征PCA降维
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class FeatureSelector:
    """特征选择器"""
    
    def __init__(self, importance_threshold: float = 0.001,
                 correlation_threshold: float = 0.95):
        """初始化特征选择器
        
        Args:
            importance_threshold: 特征重要性阈值（低于此值将被移除）
            correlation_threshold: 高相关性阈值（高于此值的特征对将进行PCA）
        """
        self.importance_threshold = importance_threshold
        self.correlation_threshold = correlation_threshold
        self.selected_features = []
        self.pca_components = {}
        self.importance_report = {}
    
    def calculate_feature_importance(self, model: Any, X: pd.DataFrame,
                                    method: str = 'shap') -> Dict[str, float]:
        """计算特征重要性
        
        Args:
            model: 训练好的模型
            X: 特征DataFrame
            method: 计算方法 ('shap' 或 'gain')
            
        Returns:
            特征重要性字典
        """
        if method == 'shap':
            try:
                import shap
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                
                # 处理多分类情况
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # 取正类的SHAP值
                
                # 计算平均绝对SHAP值
                importance = np.abs(shap_values).mean(axis=0)
                importance_dict = dict(zip(X.columns, importance))
                
            except ImportError:
                print("Warning: shap not installed, falling back to gain method")
                return self.calculate_feature_importance(model, X, method='gain')
        
        elif method == 'gain':
            # 使用模型自带的特征重要性
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'get_score'):
                # LightGBM
                importance_dict = model.get_score(importance_type='gain')
                return importance_dict
            else:
                raise ValueError("Model does not support feature importance calculation")
            
            importance_dict = dict(zip(X.columns, importance))
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return importance_dict
    
    def select_important_features(self, importance_dict: Dict[str, float],
                                 threshold: Optional[float] = None) -> List[str]:
        """选择重要特征
        
        Args:
            importance_dict: 特征重要性字典
            threshold: 重要性阈值，如果为None则使用默认值
            
        Returns:
            选中的特征列表
        """
        if threshold is None:
            threshold = self.importance_threshold
        
        # 按重要性排序
        sorted_features = sorted(importance_dict.items(), 
                                key=lambda x: x[1], 
                                reverse=True)
        
        # 选择重要性大于阈值的特征
        selected = [feat for feat, imp in sorted_features if imp >= threshold]
        
        self.selected_features = selected
        self.importance_report = {
            'total_features': len(importance_dict),
            'selected_features': len(selected),
            'removed_features': len(importance_dict) - len(selected),
            'threshold': threshold,
            'feature_importance': importance_dict,
            'selected_feature_list': selected,
            'removed_features_list': [feat for feat, imp in sorted_features if imp < threshold]
        }
        
        return selected
    
    def find_high_correlations(self, X: pd.DataFrame, 
                              threshold: Optional[float] = None) -> List[Tuple[str, str, float]]:
        """找出高相关性特征对
        
        Args:
            X: 特征DataFrame
            threshold: 相关性阈值，如果为None则使用默认值
            
        Returns:
            高相关性特征对列表 [(feat1, feat2, correlation), ...]
        """
        if threshold is None:
            threshold = self.correlation_threshold
        
        # 计算相关矩阵
        corr_matrix = X.corr()
        
        # 找出高相关性特征对
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        float(corr)
                    ))
        
        return high_corr_pairs
    
    def apply_pca(self, X: pd.DataFrame, 
                 high_corr_pairs: List[Tuple[str, str, float]],
                 variance_threshold: float = 0.95) -> Tuple[pd.DataFrame, Dict]:
        """对高相关性特征应用PCA
        
        Args:
            X: 特征DataFrame
            high_corr_pairs: 高相关性特征对列表
            variance_threshold: 方差解释率阈值
            
        Returns:
            (降维后的DataFrame, PCA组件字典)
        """
        if not high_corr_pairs:
            return X, {}
        
        # 找出所有需要PCA的特征
        features_to_pca = set()
        for feat1, feat2, corr in high_corr_pairs:
            features_to_pca.add(feat1)
            features_to_pca.add(feat2)
        
        features_to_pca = list(features_to_pca)
        
        if len(features_to_pca) < 2:
            return X, {}
        
        # 提取需要PCA的特征
        X_pca = X[features_to_pca].copy()
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_pca)
        
        # 应用PCA
        pca = PCA(n_components=variance_threshold)
        X_pca_result = pca.fit_transform(X_scaled)
        
        # 创建PCA特征名
        pca_feature_names = [f'pca_comp_{i}' for i in range(X_pca_result.shape[1])]
        X_pca_df = pd.DataFrame(X_pca_result, 
                               index=X.index, 
                               columns=pca_feature_names)
        
        # 保存PCA组件信息
        self.pca_components = {
            'original_features': features_to_pca,
            'n_components': X_pca_result.shape[1],
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'total_explained_variance': float(pca.explained_variance_ratio_.sum()),
            'components': pca.components_.tolist(),
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist()
        }
        
        # 移除原始特征，添加PCA特征
        X_result = X.drop(columns=features_to_pca)
        X_result = pd.concat([X_result, X_pca_df], axis=1)
        
        return X_result
    
    def optimize_features(self, model: Any, X: pd.DataFrame, y: pd.Series,
                         method: str = 'shap') -> Tuple[pd.DataFrame, Dict]:
        """完整的特征优化流程
        
        Args:
            model: 训练好的模型
            X: 特征DataFrame
            y: 目标变量
            method: 重要性计算方法
            
        Returns:
            (优化后的DataFrame, 优化报告)
        """
        print("Starting feature optimization...")
        
        # 1. 计算特征重要性
        print("  Calculating feature importance...")
        importance_dict = self.calculate_feature_importance(model, X, method)
        
        # 2. 选择重要特征
        print("  Selecting important features...")
        selected_features = self.select_important_features(importance_dict)
        print(f"    Selected {len(selected_features)}/{len(importance_dict)} features")
        
        # 3. 找出高相关性特征
        print("  Finding high correlations...")
        X_selected = X[selected_features]
        high_corr_pairs = self.find_high_correlations(X_selected)
        print(f"    Found {len(high_corr_pairs)} highly correlated pairs")
        
        # 4. 应用PCA（如果有高相关性特征）
        if high_corr_pairs:
            print("  Applying PCA...")
            X_optimized = self.apply_pca(X_selected, high_corr_pairs)
            print(f"    Reduced to {X_optimized.shape[1]} features")
        else:
            X_optimized = X_selected
        
        # 5. 生成优化报告
        report = {
            'original_features': len(X.columns),
            'after_selection': len(selected_features),
            'after_pca': X_optimized.shape[1],
            'high_correlation_pairs': len(high_corr_pairs),
            'pca_applied': len(high_corr_pairs) > 0,
            'importance_report': self.importance_report,
            'pca_components': self.pca_components if self.pca_components else None
        }
        
        self.importance_report = report
        
        return X_optimized, report
    
    def save_report(self, filepath: str):
        """保存优化报告到JSON文件
        
        Args:
            filepath: 输出文件路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.importance_report, f, ensure_ascii=False, indent=2)
        
        print(f"Feature importance report saved to {filepath}")
    
    def load_report(self, filepath: str) -> Dict:
        """加载优化报告
        
        Args:
            filepath: 报告文件路径
            
        Returns:
            报告字典
        """
        if not os.path.exists(filepath):
            return {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def print_summary(self):
        """打印优化摘要"""
        report = self.importance_report
        
        if not report:
            print("No optimization report available")
            return
        
        print("\n" + "="*60)
        print("Feature Selection Summary")
        print("="*60)
        print(f"Original features: {report['original_features']}")
        print(f"After importance selection: {report['after_selection']}")
        print(f"After PCA: {report['after_pca']}")
        print(f"High correlation pairs: {report['high_correlation_pairs']}")
        print(f"PCA applied: {report['pca_applied']}")
        
        if 'importance_report' in report:
            imp_report = report['importance_report']
            print(f"\nImportance threshold: {imp_report['threshold']}")
            print(f"Features removed: {imp_report['removed_features']}")
            
            if imp_report['removed_features_list']:
                print("\nRemoved features:")
                for feat in imp_report['removed_features_list'][:10]:  # 只显示前10个
                    print(f"  - {feat}")
                if len(imp_report['removed_features_list']) > 10:
                    print(f"  ... and {len(imp_report['removed_features_list']) - 10} more")
        
        if report['pca_applied'] and report['pca_components']:
            pca_info = report['pca_components']
            print(f"\nPCA Details:")
            print(f"  Original features: {len(pca_info['original_features'])}")
            print(f"  PCA components: {pca_info['n_components']}")
            print(f"  Explained variance: {pca_info['total_explained_variance']:.2%}")
        
        print("="*60 + "\n")


# 便捷函数
def optimize_model_features(model: Any, X: pd.DataFrame, y: pd.Series,
                           model_name: str = 'unknown',
                           output_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """便捷函数：优化模型特征
    
    Args:
        model: 训练好的模型
        X: 特征DataFrame
        y: 目标变量
        model_name: 模型名称
        output_path: 报告输出路径
        
    Returns:
        (优化后的DataFrame, 优化报告)
    """
    selector = FeatureSelector()
    X_optimized, report = selector.optimize_features(model, X, y)
    
    if output_path:
        selector.save_report(output_path)
    
    selector.print_summary()
    
    return X_optimized, report


if __name__ == '__main__':
    # 测试代码
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 1000
    
    # 生成相关特征
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'feature4': np.random.normal(0, 1, n_samples),
        'feature5': np.random.normal(0, 1, n_samples),
    })
    
    # 添加一些高相关性特征
    X['feature6'] = X['feature1'] * 0.9 + np.random.normal(0, 0.1, n_samples)
    X['feature7'] = X['feature2'] * 0.85 + np.random.normal(0, 0.15, n_samples)
    
    # 生成目标变量
    y = (X['feature1'] + X['feature3'] > 0).astype(int)
    
    # 训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # 运行特征选择
    selector = FeatureSelector()
    X_optimized, report = selector.optimize_features(model, X, y)
    
    # 打印摘要
    selector.print_summary()
    
    print(f"\nOriginal shape: {X.shape}")
    print(f"Optimized shape: {X_optimized.shape}")
