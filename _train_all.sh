cd /coding/live
# delete old caches to force rebuild
rm -f features_9p_cache.csv features_ml_cache.csv features_cache.csv
echo "caches deleted"

# retrain all three sequentially
echo "=== Training 9P ==="
.venv/bin/python -u train_9p.py --rebuild > logs/retrain_9p_20261014.log 2>&1
echo "9P exit=$?"

echo "=== Training ML ==="
.venv/bin/python -u train_model.py --rebuild > logs/retrain_ml_20261014.log 2>&1
echo "ML exit=$?"

echo "=== Training 1X ==="
.venv/bin/python -u train_1x.py --rebuild > logs/retrain_1x_20261014.log 2>&1
echo "1X exit=$?"

echo "ALL DONE"
