"""
短期改进2: 更新 MAX_POOL_PROB_THRESHOLD
将 predict_over.py 中的 MAX_POOL_PROB_THRESHOLD 从 0.50 提高到 0.85
"""

filepath = '/coding/live/predict_over.py'
print(f'Processing {filepath}...')

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace MAX_POOL_PROB_THRESHOLD
old_line = 'MAX_POOL_PROB_THRESHOLD = 0.50  # 阈值上限 (2026-11-01 修改，降低阈值以增加推荐场次)'
new_line = 'MAX_POOL_PROB_THRESHOLD = 0.85  # 阈值上限 (2026-11-03 修改，提高阈值以保证命中率)'

if old_line in content:
    content = content.replace(old_line, new_line)
    print(f'✓ Changed MAX_POOL_PROB_THRESHOLD from 0.50 to 0.85')
else:
    print(f'⚠ Line not found, trying alternative...')
    # Try a more flexible pattern
    import re
    pattern = r'MAX_POOL_PROB_THRESHOLD\s*=\s*0\.50'
    if re.search(pattern, content):
        content = re.sub(pattern, 'MAX_POOL_PROB_THRESHOLD = 0.85', content)
        print(f'✓ Changed MAX_POOL_PROB_THRESHOLD from 0.50 to 0.85 (regex)')
    else:
        print(f'✗ Could not find MAX_POOL_PROB_THRESHOLD')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print(f'✓ Updated {filepath}')
