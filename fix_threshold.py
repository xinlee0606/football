"""
短期改进2: 提高推荐阈值
将 picked 阈值从 0.80 提高到 0.85
"""

files = [
    '/coding/live/predict_over.py',
    '/coding/live/predict_under.py',
    '/coding/live/predict_upside.py'
]

for filepath in files:
    print(f'Processing {filepath}...')
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace the threshold
    # Looking for patterns like: picked = p >= 0.80 or picked = p_pool >= 0.8
    import re
    
    # Pattern 1: picked = p >= 0.80
    pattern1 = r"(picked\s*=\s*p[^>]*>=\s*)0\.80"
    if re.search(pattern1, content):
        content = re.sub(pattern1, r"\g<1>0.85", content)
        print(f'  ✓ Changed threshold from 0.80 to 0.85 (pattern 1)')
    else:
        # Pattern 2: picked = p_pool >= 0.80
        pattern2 = r"(picked\s*=\s*p_pool\s*>=\s*)0\.80"
        if re.search(pattern2, content):
            content = re.sub(pattern2, r"\g<1>0.85", content)
            print(f'  ✓ Changed threshold from 0.80 to 0.85 (pattern 2)')
        else:
            # Pattern 3: Look for any 0.8 or 0.80 in picked assignment
            pattern3 = r"(picked\s*=\s*[^#\n]*?)0\.8(?!\d)"
            if re.search(pattern3, content):
                content = re.sub(pattern3, r"\g<1>0.85", content)
                print(f'  ✓ Changed threshold from 0.8 to 0.85 (pattern 3)')
            else:
                print(f'  ⚠ Could not find threshold pattern')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

print('\n✓ Recommendation threshold updated')
