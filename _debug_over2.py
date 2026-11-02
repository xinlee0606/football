import os, urllib.request
os.environ['SE_MANAGER_OFFLINE'] = '1'

# First, try raw HTTP to see what the Over page returns
url = 'https://bf.titan007.com/football/Over_20261016.htm'
print(f"=== Raw HTTP: {url} ===")
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urllib.request.urlopen(req, timeout=15)
    data = resp.read()
    print(f"Status: {resp.status}, Length: {len(data)}")
    print(f"Content (first 2000 chars):")
    print(data.decode('utf-8', errors='replace')[:2000])
except Exception as e:
    print(f"Error: {e}")

# Try alternative URL patterns
for alt_url in [
    'https://bf.titan007.com/football/Over_20261016.htm',
    'https://bf.titan007.com/football/over_20261016.htm',
    'https://bf.titan007.com/football/Over/Over_20261016.htm',
    'https://live.titan007.com/football/Over_20261016.htm',
]:
    print(f"\n=== Trying: {alt_url} ===")
    try:
        req = urllib.request.Request(alt_url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req, timeout=15)
        data = resp.read()
        print(f"Status: {resp.status}, Length: {len(data)}")
        if len(data) > 1000:
            # Check for beiDanSchedule
            text = data.decode('utf-8', errors='replace')
            import re
            m = re.search(r'beiDanSchedule\s*=\s*"([^"]*)"', text)
            if m:
                val = m.group(1)
                sids = val.split('|')[0].split(',') if '|' in val else []
                print(f"  FOUND beiDanSchedule: {len(sids)} matches")
            else:
                print(f"  No beiDanSchedule. First 500 chars: {text[:500]}")
        else:
            print(f"  Content: {data.decode('utf-8', errors='replace')[:500]}")
    except Exception as e:
        print(f"  Error: {e}")

# Also try: maybe the date format is different (with leading zeros removed?)
for alt_date in ['20261016', '202601016', '261016']:
    alt_url = f'https://bf.titan007.com/football/Over_{alt_date}.htm'
    print(f"\n=== Trying date format: {alt_date} ===")
    try:
        req = urllib.request.Request(alt_url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req, timeout=10)
        data = resp.read()
        print(f"  Status: {resp.status}, Length: {len(data)}")
    except Exception as e:
        print(f"  Error: {e}")

print("\nDONE")
