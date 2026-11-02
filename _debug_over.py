import os
os.environ['SE_MANAGER_OFFLINE'] = '1'
from selenium import webdriver
import re

opts = webdriver.ChromeOptions()
opts.add_argument('--headless')
opts.add_argument('--no-sandbox')
opts.add_argument('--disable-dev-shm-usage')
opts.binary_location = '/usr/bin/google-chrome'
d = webdriver.Chrome(options=opts)
d.set_page_load_timeout(30)

url = 'https://bf.titan007.com/football/Over_20261016.htm'
print(f"Loading: {url}")
d.get(url)
import time; time.sleep(8)

html = d.page_source
print(f"Page length: {len(html)} chars")

# Check for beiDanSchedule
m = re.search(r'beiDanSchedule\s*=\s*"([^"]*)"', html)
if m:
    val = m.group(1)
    print(f"beiDanSchedule found: '{val[:200]}...'")
    if '|' in val:
        sids = val.split('|')[0].split(',')
        dans = val.split('|')[1].split(',')
        print(f"  sids: {len(sids)}, dans: {len(dans)}")
    else:
        print(f"  No | separator, raw value length: {len(val)}")
else:
    print("beiDanSchedule NOT found by regex")
    # Try broader search
    m2 = re.search(r'beiDan', html)
    if m2:
        start = max(0, m2.start() - 50)
        end = min(len(html), m2.end() + 200)
        print(f"  Found 'beiDan' at pos {m2.start()}: ...{html[start:end]}...")
    else:
        print("  'beiDan' not found anywhere in page")

# Check for table_live
table = d.find_elements('id', 'table_live')
print(f"table_live elements: {len(table)}")

# Check for tr1_ rows
tr1 = d.find_elements('xpath', "//tr[starts-with(@id,'tr1_')]")
print(f"tr1_ rows: {len(tr1)}")

# Check for sid attributes
sids = d.find_elements('xpath', "//tr[@sid]")
print(f"Rows with sid attribute: {len(sids)}")

# Get first few sids
for s in sids[:5]:
    print(f"  sid={s.get_attribute('sid')}")

# Also check: maybe the page uses a different date format or URL
# Try the main oldIndexall page for 20261016
print("\n--- Checking oldIndexall.aspx ---")
url2 = 'https://live.titan007.com/oldIndexall.aspx?date=20261016'
try:
    d.get(url2)
    time.sleep(5)
    html2 = d.page_source
    print(f"oldIndexall page length: {len(html2)} chars")
    m3 = re.search(r'beiDanSchedule\s*=\s*"([^"]*)"', html2)
    if m3:
        val = m3.group(1)
        sids3 = val.split('|')[0].split(',') if '|' in val else []
        print(f"beiDanSchedule on oldIndexall: {len(sids3)} matches")
    else:
        print("beiDanSchedule NOT on oldIndexall either")
    
    # Check for 单场 text
    count = html2.count('单场')
    print(f"'单场' occurrences on oldIndexall: {count}")
except Exception as e:
    print(f"oldIndexall error: {e}")

d.quit()
print("DONE")
