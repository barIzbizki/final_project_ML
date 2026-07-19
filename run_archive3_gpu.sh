#!/usr/bin/env bash
# Full archive3 pipeline.
#
# GPU coverage: KNN and SVM run on cuML, XGBoost via device="cuda" (all wired up in
# models_gpu.py). MLP has no GPU implementation in sklearn or cuML and runs on CPU.
# PCA stays on sklearn -- cuML's PCA cannot take the fractional variance targets in
# PCA_VARIANCE_OPTIONS, and at 14-21 features it is not worth accelerating anyway.
set -euo pipefail
cd "$(dirname "$0")"

PY="./venv/bin/python"
LOG_DIR="outputs/archive3/logs"
mkdir -p "$LOG_DIR"

run() {
  echo "=== [$(date +%H:%M:%S)] $1 ==="
  $PY "$1" 2>&1 | tee "$LOG_DIR/${1%.py}.log"
}

run 01_eda_archive3.py
run 02_split_dataset_archive3.py
run 03_clinical_models_pca_archive3.py
run 04_lifestyle_models_pca_archive3.py
run 05_fusion_archive3.py
run 06_merged_models_pca_archive3.py
run 07_compare_results_pca_archive3.py
run 08_pca_ablation_archive3.py

echo "=== [$(date +%H:%M:%S)] pipeline complete ==="
