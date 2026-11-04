"""
修复 train_under.py 的 league_goal_strength 类型问题
在调用 _add_missing_indicators 之前确保所有特征列为数值类型
"""

filename = '/coding/live/train_under.py'
print(f'Fixing {filename}...')

with open(filename, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find line 227: df_train, feats = _add_missing_indicators(df_train, feats)
# Insert type conversion before it
insert_idx = None
for i, line in enumerate(lines):
    if 'df_train, feats = _add_missing_indicators(df_train, feats)' in line:
        insert_idx = i
        break

if insert_idx is not None:
    # Insert type conversion code
    indent = '    '
    new_code = [
        f'{indent}# 确保所有特征列为数值类型(防止object类型导致LightGBM报错)\n',
        f'{indent}for col in feats:\n',
        f'{indent}    if col in df_train.columns and df_train[col].dtype == object:\n',
        f'{indent}        df_train[col] = pd.to_numeric(df_train[col], errors=\'coerce\')\n',
        f'{indent}    if col in df_valid.columns and df_valid[col].dtype == object:\n',
        f'{indent}        df_valid[col] = pd.to_numeric(df_valid[col], errors=\'coerce\')\n',
        '\n',
    ]
    
    lines = lines[:insert_idx] + new_code + lines[insert_idx:]
    print(f'✓ Inserted type conversion at line {insert_idx + 1}')
else:
    print('✗ Could not find insertion point')

with open(filename, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f'✓ Fixed {filename}')
