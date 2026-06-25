#!/usr/bin/env bash

# Common utilities for SSMT-DA experiment scripts.
#
# This file is intended to be sourced from other scripts:
#
#   source "$(dirname "$0")/common.sh"
#
# Expected repository layout:
#
# multi-target-da-deepfake/
#   src/
#     gen_ssmt_dataset.py
#     train_ssmt_xception.py
#     test_ssmt_xception.py
#   scripts/
#     common.sh
#     run_main_results.sh
#     ...

set -euo pipefail

# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$PROJECT_ROOT/src"

# Users can override these from the calling script or environment.
RAW_DIR="${RAW_DIR:-$PROJECT_ROOT/raw_data/Images_128}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/datasets}"
RUNS_ROOT="${RUNS_ROOT:-$PROJECT_ROOT/runs}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/results}"

COMBINED_NAME="${COMBINED_NAME:-TARGET_COMBINED}"

# ------------------------------------------------------------
# Default experiment settings
# ------------------------------------------------------------
IMG_SIZE="${IMG_SIZE:-128}"
EPOCHS="${EPOCHS:-10}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"

BATCH_SOURCE="${BATCH_SOURCE:-16}"
BATCH_T_L="${BATCH_T_L:-8}"
BATCH_T_U="${BATCH_T_U:-16}"

LR="${LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"

LAMBDA_MM="${LAMBDA_MM:-1.0}"
LAMBDA_CA="${LAMBDA_CA:-0.1}"

NUM_WORKERS="${NUM_WORKERS:-4}"
BACKBONE="${BACKBONE:-xception}"

T_LABELED_TOTAL="${T_LABELED_TOTAL:-1000}"
T_UNLABELED_TOTAL="${T_UNLABELED_TOTAL:-9000}"

AMP_FLAG="${AMP_FLAG:-}"   # set to "--amp" if wanted

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
now() {
  date '+%Y-%m-%d %H:%M:%S'
}

join_by_underscore() {
  local IFS="_"
  echo "$*"
}

print_header() {
  local title="$1"
  echo
  echo "============================================================"
  echo "$title"
  echo "============================================================"
}

make_dataset_dir() {
  local source="$1"
  shift
  local seed="${@: -1}"
  local targets=("${@:1:$#-1}")

  local target_str
  target_str="$(join_by_underscore "${targets[@]}")"

  echo "$DATASET_ROOT/dataset_ssmt_${source}__${target_str}_seed${seed}_lab${T_LABELED_TOTAL}_unlab${T_UNLABELED_TOTAL}"
}

make_run_dir() {
  local source="$1"
  shift
  local seed="${@: -2:1}"
  local tag="${@: -1}"
  local targets=("${@:1:$#-2}")

  local target_str
  target_str="$(join_by_underscore "${targets[@]}")"

  echo "$RUNS_ROOT/ssmt_${source}__${target_str}_seed${seed}_lab${T_LABELED_TOTAL}_unlab${T_UNLABELED_TOTAL}__${tag}"
}

start_log() {
  local out_dir="$1"

  mkdir -p "$out_dir/logs"
  local log_file="$out_dir/logs/run_$(date '+%Y%m%d_%H%M%S').log"

  # Redirect everything after this point to both console and log file.
  exec > >(tee -a "$log_file") 2>&1

  echo "LOG_FILE: $log_file"
}

print_config() {
  local source="$1"
  shift
  local seed="${@: -1}"
  local targets=("${@:1:$#-1}")

  echo "Started at      : $(now)"
  echo "PROJECT_ROOT    : $PROJECT_ROOT"
  echo "RAW_DIR         : $RAW_DIR"
  echo "DATASET_ROOT    : $DATASET_ROOT"
  echo "RUNS_ROOT       : $RUNS_ROOT"
  echo "RESULTS_ROOT    : $RESULTS_ROOT"
  echo "SOURCE          : $source"
  echo "TARGETS         : ${targets[*]}"
  echo "SEED            : $seed"
  echo "IMG_SIZE        : $IMG_SIZE"
  echo "EPOCHS          : $EPOCHS"
  echo "STEPS/EPOCH     : $STEPS_PER_EPOCH"
  echo "BATCH_SOURCE    : $BATCH_SOURCE"
  echo "BATCH_T_L       : $BATCH_T_L"
  echo "BATCH_T_U       : $BATCH_T_U"
  echo "LR              : $LR"
  echo "WEIGHT_DECAY    : $WEIGHT_DECAY"
  echo "LAMBDA_MM       : $LAMBDA_MM"
  echo "LAMBDA_CA       : $LAMBDA_CA"
  echo "T_LABELED_TOTAL : $T_LABELED_TOTAL"
  echo "T_UNLAB_TOTAL   : $T_UNLABELED_TOTAL"
  echo "BACKBONE        : $BACKBONE"
  echo "NUM_WORKERS     : $NUM_WORKERS"
  echo "COMBINED_NAME   : $COMBINED_NAME"
}

generate_dataset() {
  local source="$1"
  shift
  local seed="${@: -1}"
  local targets=("${@:1:$#-1}")

  local dataset_dir
  dataset_dir="$(make_dataset_dir "$source" "${targets[@]}" "$seed")"

  mkdir -p "$dataset_dir"

  print_header "Generate dataset: ${source} -> ${targets[*]} | seed=${seed}"

  python3 "$SRC_DIR/gen_ssmt_dataset.py" \
    --raw_dir "$RAW_DIR" \
    --out_dir "$dataset_dir" \
    --seed "$seed" \
    --source "$source" \
    --targets "${targets[@]}" \
    --combined_name "$COMBINED_NAME" \
    --t_labeled_total "$T_LABELED_TOTAL" \
    --t_unlabeled_total "$T_UNLABELED_TOTAL"
}

train_model() {
  local source="$1"
  shift
  local seed="${@: -2:1}"
  local tag="${@: -1}"
  local targets=("${@:1:$#-2}")

  local dataset_dir
  local out_dir

  dataset_dir="$(make_dataset_dir "$source" "${targets[@]}" "$seed")"
  out_dir="$(make_run_dir "$source" "${targets[@]}" "$seed" "$tag")"

  mkdir -p "$out_dir"

  print_header "Train: ${source} -> ${targets[*]} | seed=${seed} | tag=${tag}"

  python3 "$SRC_DIR/train_ssmt_xception.py" \
    --dataset_dir "$dataset_dir" \
    --out_dir "$out_dir" \
    --seed "$seed" \
    --source "$source" \
    --targets "${targets[@]}" \
    --epochs "$EPOCHS" \
    --steps_per_epoch "$STEPS_PER_EPOCH" \
    --img_size "$IMG_SIZE" \
    --batch_source "$BATCH_SOURCE" \
    --batch_t_l "$BATCH_T_L" \
    --batch_t_u "$BATCH_T_U" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --lambda_mm "$LAMBDA_MM" \
    --lambda_ca "$LAMBDA_CA" \
    --backbone "$BACKBONE" \
    --num_workers "$NUM_WORKERS" \
    $AMP_FLAG
}

test_model() {
  local source="$1"
  shift
  local seed="${@: -2:1}"
  local tag="${@: -1}"
  local targets=("${@:1:$#-2}")

  local dataset_dir
  local out_dir

  dataset_dir="$(make_dataset_dir "$source" "${targets[@]}" "$seed")"
  out_dir="$(make_run_dir "$source" "${targets[@]}" "$seed" "$tag")"

  print_header "Test: ${source} -> ${targets[*]} | seed=${seed} | tag=${tag}"

  python3 "$SRC_DIR/test_ssmt_xception.py" \
    --dataset_dir "$dataset_dir" \
    --ckpt "$out_dir/checkpoint_last.pt" \
    --source "$source" \
    --targets "${targets[@]}" \
    --combined_name "$COMBINED_NAME" \
    --backbone "$BACKBONE" \
    --img_size "$IMG_SIZE" \
    --batch_size 128 \
    --num_workers "$NUM_WORKERS"
}

run_experiment() {
  local source="$1"
  shift
  local seed="${@: -2:1}"
  local tag="${@: -1}"
  local targets=("${@:1:$#-2}")

  local out_dir
  out_dir="$(make_run_dir "$source" "${targets[@]}" "$seed" "$tag")"

  mkdir -p "$out_dir"

  start_log "$out_dir"
  print_header "Run experiment: ${source} -> ${targets[*]} | seed=${seed} | tag=${tag}"
  print_config "$source" "${targets[@]}" "$seed"

  local start_epoch
  start_epoch="$(date +%s)"

  generate_dataset "$source" "${targets[@]}" "$seed"
  train_model "$source" "${targets[@]}" "$seed" "$tag"
  test_model "$source" "${targets[@]}" "$seed" "$tag"

  local end_epoch
  end_epoch="$(date +%s)"
  local elapsed
  elapsed="$((end_epoch - start_epoch))"

  print_header "Completed"
  echo "Ended at          : $(now)"
  echo "Total runtime (s): $elapsed"
  echo "Total runtime (m): $((elapsed / 60))"
}