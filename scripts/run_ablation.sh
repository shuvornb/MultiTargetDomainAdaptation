#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

# Ablation study:
# 1. Full: BCE + MM + CA
# 2. No MM: BCE + CA
# 3. No CA: BCE + MM
# 4. BCE only

for seed in 123 456; do
  for task in "DF FS F2F" "FS F2F DF"; do
    read -r source t1 t2 <<< "$task"

    (
      T_LABELED_TOTAL=1000
      T_UNLABELED_TOTAL=9000
      LAMBDA_MM=1.0
      LAMBDA_CA=0.1
      run_experiment "$source" "$t1" "$t2" "$seed" "full_bce_mm_ca"
    )

    (
      T_LABELED_TOTAL=1000
      T_UNLABELED_TOTAL=9000
      LAMBDA_MM=0.0
      LAMBDA_CA=0.1
      run_experiment "$source" "$t1" "$t2" "$seed" "no_mm_bce_ca"
    )

    (
      T_LABELED_TOTAL=1000
      T_UNLABELED_TOTAL=9000
      LAMBDA_MM=1.0
      LAMBDA_CA=0.0
      run_experiment "$source" "$t1" "$t2" "$seed" "no_ca_bce_mm"
    )

    (
      T_LABELED_TOTAL=1000
      T_UNLABELED_TOTAL=9000
      LAMBDA_MM=0.0
      LAMBDA_CA=0.0
      run_experiment "$source" "$t1" "$t2" "$seed" "bce_only"
    )
  done
done