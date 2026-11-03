import re

def fix_file(filename):
    print(f'\n=== Fixing {filename} ===')
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # 1. Remove exp_total from keys list
    content = content.replace("'over_prob_cur', 'exp_total']", "'over_prob_cur']")
    # Remove exp_total computation line (multi-line)
    content = re.sub(r"\n    feats\['exp_total'\] = feats\['ou_line_cur_mean'\].*?(?=\n    return|\n\n)", '', content, flags=re.DOTALL)
    
    # 2. Remove recent_loss_rate from keys list  
    content = content.replace("'recent_draw_rate', 'recent_loss_rate',", "'recent_draw_rate',")
    # Remove recent_loss_rate computation
    content = re.sub(r"\n        feats\['recent_loss_rate'\] = losses / n", '', content)
    
    # 3. Remove cover_prob_cur from keys list
    content = content.replace("'cover_prob_init', 'cover_prob_cur',", "'cover_prob_init',")
    # Remove cover_prob_cur computation
    content = re.sub(r"\n        feats\['cover_prob_cur'\] = _mean\(\[a / \(h \+ a\) for h, a in zip\(ho_c, ao_c\)\]\)", '', content)
    
    if content != original:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'✓ {filename} modified')
        return True
    else:
        print(f'⚠ No changes to {filename}')
        return False

# Fix both files
fix_file('/coding/live/features_over.py')
fix_file('/coding/live/features_under.py')

print('\n=== Done ===')
