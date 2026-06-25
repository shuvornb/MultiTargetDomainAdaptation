#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

# Main paper experiments.
# Each run is executed in a subshell so logging remains separate.

for seed in 123 456; do
  (
    T_LABELED_TOTAL=1000
    T_UNLABELED_TOTAL=9000
    LAMBDA_MM=1.0
    LAMBDA_CA=0.1
    run_experiment "DF" "FS" "F2F" "$seed" "main"
  )

  (
    T_LABELED_TOTAL=1000
    T_UNLABELED_TOTAL=9000
    LAMBDA_MM=1.0
    LAMBDA_CA=0.1
    run_experiment "FS" "F2F" "DF" "$seed" "main"
  )
done