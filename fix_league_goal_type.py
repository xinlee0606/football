"""
修复 league_goal_strength 类型问题
在训练前确保所有特征列为数值类型
"""

import pandas as pd

# Fix for train_over.py
filename = '/coding/live/train_over.py'
print(f'Fixing {filename}...')

with open(filename, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the line where we add missing indicators and add type conversion before it
old_line = '    # 添加缺失指示器(与训练时一致)'
new_code = '''    # 确保所有特征列为数值类型(防止object类型导致LightGBM报错)
    for col in feats:
        if col.endswith('_missing'):
            continue
        if col not in tr.columns:
            continue
        if tr[col].dtype == object:
            tr[col] = pd.to_numeric(tr[col], errors='coerce')
        if col in va.columns and va[col].dtype == object:
            va[col] = pd.to_numeric(va[col], errors='coerce')
    
    # 添加缺失指示器(与训练时一致)'''

if old_line in content:
    content = content.replace(old_line, new_code)
    print('✓ Added type conversion before missing indicators')
else:
    print('⚠ Could not find the line to replace')

with open(filename, 'w', encoding='utf-8') as f:
    f.write(content)

print(f'✓ Fixed {filename}')

# Fix for train_under.py
filename2 = '/coding/live/train_under.py'
print(f'\nFixing {filename2}...')

with open(filename2, 'r', encoding='utf-8') as f:
    content2 = f.read()

if old_line in content2:
    content2 = content2.replace(old_line, new_code)
    print('✓ Added type conversion before missing indicators')
else:
    print('⚠ Could not find the line to replace')

with open(filename2, 'w', encoding='utf-8') as f:
    f.write(content2)

print(f'✓ Fixed {filename2}')
