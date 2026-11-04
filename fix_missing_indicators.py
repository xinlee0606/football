# 修复 _add_missing_indicators 函数，跳过已经是指示器的列

with open('/coding/live/train_over.py', 'r') as f:
    lines = f.readlines()

# 找到函数位置并修复
new_lines = []
in_function = False
fixed = False
for i, line in enumerate(lines):
    if 'def _add_missing_indicators(df, feats):' in line:
        in_function = True
        new_lines.append(line)
    elif in_function and not fixed and 'for col in feats:' in line:
        # 在这行之前添加过滤逻辑
        indent = '    '
        new_lines.append(f'{indent}# 只为原始特征创建指示器，跳过已经是指示器的列\n')
        new_lines.append(f'{indent}original_feats = [f for f in feats if not f.endswith(\'_missing\')]\n')
        new_lines.append(f'{indent}for col in original_feats:\n')
        fixed = True
    elif in_function and 'return df, new_feats' in line:
        in_function = False
        new_lines.append(line)
    else:
        new_lines.append(line)

with open('/coding/live/train_over.py', 'w') as f:
    f.writelines(new_lines)

print("✓ 修复 train_over.py")

# 同样修复 train_under.py
with open('/coding/live/train_under.py', 'r') as f:
    lines = f.readlines()

new_lines = []
in_function = False
fixed = False
for i, line in enumerate(lines):
    if 'def _add_missing_indicators(df, feats):' in line:
        in_function = True
        new_lines.append(line)
    elif in_function and not fixed and 'for col in feats:' in line:
        indent = '    '
        new_lines.append(f'{indent}# 只为原始特征创建指示器，跳过已经是指示器的列\n')
        new_lines.append(f'{indent}original_feats = [f for f in feats if not f.endswith(\'_missing\')]\n')
        new_lines.append(f'{indent}for col in original_feats:\n')
        fixed = True
    elif in_function and 'return df, new_feats' in line:
        in_function = False
        new_lines.append(line)
    else:
        new_lines.append(line)

with open('/coding/live/train_under.py', 'w') as f:
    f.writelines(new_lines)

print("✓ 修复 train_under.py")
