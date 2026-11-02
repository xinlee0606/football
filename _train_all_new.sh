#!/bin/bash
cd /coding/live
echo "=== $(date) Starting ==="

# Build Over cache
echo "=== Building OVER cache ==="
.venv/bin/python -u -c "
from features_over import load_or_build_over
load_or_build_over(rebuild=True)
print('OVER CACHE DONE')
" 2>&1 | tee logs/build_over_cache.log
echo "OVER cache exit=$?"

# Build Under cache
echo "=== Building UNDER cache ==="
.venv/bin/python -u -c "
from features_under import load_or_build_under
load_or_build_under(rebuild=True)
print('UNDER CACHE DONE')
" 2>&1 | tee logs/build_under_cache.log
echo "UNDER cache exit=$?"

# Build Upside cache
echo "=== Building UPSIDE cache ==="
.venv/bin/python -u -c "
from features_upside_ext import build_features_upside_ext
build_features_upside_ext(rebuild=True)
print('UPSIDE CACHE DONE')
" 2>&1 | tee logs/build_upside_cache.log
echo "UPSIDE cache exit=$?"

# Train Over
echo "=== Training OVER ==="
.venv/bin/python -u train_over.py 2>&1 | tee logs/train_over.log
echo "OVER train exit=$?"

# Train Under
echo "=== Training UNDER ==="
.venv/bin/python -u train_under.py 2>&1 | tee logs/train_under.log
echo "UNDER train exit=$?"

# Train Upside
echo "=== Training UPSIDE ==="
.venv/bin/python -u train_upside.py 2>&1 | tee logs/train_upside.log
echo "UPSIDE train exit=$?"

echo "=== $(date) ALL DONE ==="
