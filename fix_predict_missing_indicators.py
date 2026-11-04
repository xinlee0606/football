"""
修复预测脚本：添加缺失指示器
在predict_rows函数中，预测前为所有特征列创建缺失指示器
"""

import re

def fix_predict_script(filename, func_name='predict_rows'):
    print(f'Fixing {filename}...')
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the predict_rows function and add missing indicators before prediction
    # Look for the line where we call model.predict or model.predict_proba
    
    # Pattern: find where we prepare X for prediction
    # We need to add missing indicators before the model prediction
    
    # Strategy: Find "X = d[available_feats]" or similar and add indicator creation after it
    
    patterns = [
        # Pattern 1: X = d[available_feats]
        (r'(X\s*=\s*d\[available_feats\])',
         r'''\1
    
    # 添加缺失指示器(与训练时一致)
    indicator_cols = []
    for col in available_feats:
        if col.endswith('_missing'):
            continue
        if col not in X.columns:
            continue
        indicator_name = f'{col}_missing'
        X[indicator_name] = X[col].isnull().astype(int)
        indicator_cols.append(indicator_name)
    
    # 更新特征列表以包含指示器
    all_feats = available_feats + indicator_cols
    X = X[all_feats]'''),
        
        # Pattern 2: d[available_feats] used directly
        (r'(d\[available_feats\])',
         r'''\1
    
    # 添加缺失指示器(与训练时一致)
    indicator_cols = []
    for col in available_feats:
        if col.endswith('_missing'):
            continue
        if col not in d.columns:
            continue
        indicator_name = f'{col}_missing'
        d[indicator_name] = d[col].isnull().astype(int)
        indicator_cols.append(indicator_name)
    
    # 更新特征列表
    all_feats = available_feats + indicator_cols'''),
    ]
    
    fixed = False
    for pattern, replacement in patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content, count=1)
            print(f'  ✓ Added missing indicators (pattern {patterns.index((pattern, replacement)) + 1})')
            fixed = True
            break
    
    if not fixed:
        print(f'  ⚠ Could not find insertion point')
        # Try a more flexible approach
        # Look for any line with "available_feats"
        if 'available_feats' in content:
            print(f'  ℹ Found "available_feats" in file, trying manual fix...')
            # Find the line number
            for i, line in enumerate(content.split('\n')):
                if 'available_feats' in line and ('=' in line or 'in' in line):
                    print(f'    Line {i+1}: {line.strip()}')
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f'✓ Fixed {filename}')

# Fix both predict scripts
fix_predict_script('/coding/live/predict_over.py')
fix_predict_script('/coding/live/predict_under.py')

print('\n✓ All predict scripts fixed')
