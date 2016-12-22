# Santander Product Recommendation - 8th place

### Caution
make_data() step in main.py needs 30GB of memory but it can be optimized.

### This code produces 3 submissions
* xgboost - 0.03061 public LB
* lightgbm - 0.03059 public LB
* xgboost+lightgbm - 0.03063 public LB

### Steps
* place train_ver2.csv, test_ver2.csv to ../input/
* install pandas, scikit-learn, numpy, xgboost, lightgbm (or comment out lightgbm part) libs for python3
* set proper number of threads in engines.py
* ./run.sh
