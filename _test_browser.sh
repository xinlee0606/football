#!/bin/bash
cd /coding/live
export SE_MANAGER_OFFLINE=1
/coding/live/.venv/bin/python -c "
from selenium import webdriver
opts = webdriver.ChromeOptions()
opts.add_argument('--headless')
opts.add_argument('--no-sandbox')
opts.add_argument('--disable-dev-shm-usage')
opts.binary_location = '/usr/bin/google-chrome'
d = webdriver.Chrome(options=opts)
print('Browser OK:', d.capabilities['browserVersion'])
d.quit()
print('DONE')
" 2>&1 | tee /coding/live/_browser_test.log

