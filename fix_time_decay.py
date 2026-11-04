"""
短期改进1: 降低时间衰减参数
将 TIME_DECAY_LAMBDA 从 0.08 调低至 0.05
"""

files = [
    '/coding/live/features_over.py',
    '/coding/live/features_under.py'
]

for filepath in files:
    print(f'Processing {filepath}...')
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace TIME_DECAY_LAMBDA value
    old_line = 'TIME_DECAY_LAMBDA = 0.08'
    new_line = 'TIME_DECAY_LAMBDA = 0.05'
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        print(f'  ✓ Changed {old_line} -> {new_line}')
    else:
        print(f'  ⚠ Line not found: {old_line}')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

print('\n✓ Time decay parameter updated in both files')
