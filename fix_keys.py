
import re

filepath = '/coding/live/features_ml.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix asian_keys - match the actual pattern in the file
old_asian = """asian_keys = ['id', 'match_id', 'bookmaker', 'initial_handicap', 'current_handicap',
                      'initial_home_odds', 'initial_away_odds', 'current_home_odds', 'current_away_odds']"""
new_asian = """asian_keys = ['id', 'match_id', 'company', 'initial_handicap',
                      'initial_home_odds', 'initial_away_odds',
                      'current_handicap', 'current_home_odds', 'current_away_odds']"""

if old_asian in content:
    content = content.replace(old_asian, new_asian)
    print("Fixed asian_keys")
else:
    print("WARNING: old asian_keys pattern not found, checking current state...")
    # Check if already fixed
    if 'company' in content and 'initial_handicap' in content:
        print("  -> Already fixed or different format")
    else:
        print("  -> Need manual inspection")

# Fix ou_keys
old_ou = """ou_keys = ['id', 'match_id', 'bookmaker', 'initial_line', 'current_line',
                   'initial_over_odds', 'initial_under_odds', 'current_over_odds', 'current_under_odds']"""
new_ou = """ou_keys = ['id', 'match_id', 'company', 'initial_line',
                   'initial_over_odds', 'initial_under_odds',
                   'current_line', 'current_over_odds', 'current_under_odds']"""

if old_ou in content:
    content = content.replace(old_ou, new_ou)
    print("Fixed ou_keys")
else:
    print("WARNING: old ou_keys pattern not found, checking current state...")
    if 'company' in content and 'initial_line' in content:
        print("  -> Already fixed or different format")
    else:
        print("  -> Need manual inspection")

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Done writing fixed file")
