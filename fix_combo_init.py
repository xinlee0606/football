"""Fix incremental_features.py to always include all combo features"""

filename = '/coding/live/incremental_features.py'
print(f'Fixing {filename}...')

with open(filename, 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the combo section to initialize all keys first
old_combo = """        # Compute combination features
        combo = {}
        h_sot = home_recent.get('rf_sot_mean')"""

new_combo = """        # Compute combination features
        combo = {
            'attack_eff_diff': None,
            'possession_diff': None,
            'corners_diff': None,
            'fouls_diff': None,
            'yellow_diff': None,
            'form_momentum_h': None,
            'form_momentum_a': None
        }
        h_sot = home_recent.get('rf_sot_mean')"""

if old_combo in content:
    content = content.replace(old_combo, new_combo)
    print('✓ Initialized all combo keys to None')
else:
    print('⚠ Could not find combo section')

with open(filename, 'w', encoding='utf-8') as f:
    f.write(content)

print(f'✓ Fixed {filename}')
