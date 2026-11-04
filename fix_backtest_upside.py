"""Fix backtest_upside.py to add missing indicators before prediction"""
import re

filename = '/coding/live/backtest_upside.py'
print(f'Fixing {filename}...')

with open(filename, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the line: df_test['p_upside'] = model.predict_proba(df_test[feats])[:, 1]
# Add indicator creation before it

old_line = "    df_test['p_upside'] = model.predict_proba(df_test[feats])[:, 1]"

new_code = """    # 添加缺失指示器(与训练一致)
    indicator_cols = []
    for col in feats:
        if col.endswith('_missing'):
            continue
        if col not in df_test.columns:
            continue
        indicator_name = f'{col}_missing'
        df_test[indicator_name] = df_test[col].isnull().astype(int)
        indicator_cols.append(indicator_name)
    feats_with_indicators = feats + indicator_cols
    
    df_test['p_upside'] = model.predict_proba(df_test[feats_with_indicators])[:, 1]"""

content = content.replace(old_line, new_code)

# Also check if there are other predict_proba calls that need fixing
# Look for attach_pool_probs function
if 'def attach_pool_probs' in content:
    # Find pool_clf.predict_proba calls
    content = re.sub(
        r"(pool_clf\.predict_proba\()df_test\[feats\](\))",
        r"\1df_test[feats_with_indicators]\2",
        content
    )

with open(filename, 'w', encoding='utf-8') as f:
    f.write(content)

print('✓ Fixed backtest_upside.py')
