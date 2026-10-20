#!/usr/bin/env bash
# One lane of the final protocol: for each seed -> train (resumable) -> evaluate -> CDM.
#   scripts/ops/final_lane.sh <seed> [<seed> ...]
# CODE: a clean checkout/worktree of the commit to report (run.json records its hash; default: this repo).
# Data and run dirs: the repo (data/processed, runs/final_s<seed>); checkpoints under $STORE.
# Memory: expandable allocator segments + 2 loader workers (training peak ~5.6 GB of tensors);
# launches only with >= MIN_AVAIL_GB available.
set -uo pipefail
ROOT=${MATHOCR_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
CODE=${FINAL_CODE:-$ROOT}
STORE=${STORE:-/mnt/SSD/MathOCR}
CONFIG=${FINAL_CONFIG:-$CODE/configs/final.yaml}
MIN_AVAIL_GB=${MIN_AVAIL_GB:-35}
cd "$ROOT"
export PYTHONPATH=$CODE/src CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
py=${PYTHON:-$ROOT/.venv/bin/python}
for seed in "$@"; do
  run=runs/final_s$seed
  mkdir -p "$STORE/runs/final_s$seed/checkpoints" "$run"
  ln -sfn "$STORE/runs/final_s$seed/checkpoints" "$run/checkpoints"
  until (( $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) >= MIN_AVAIL_GB )); do sleep 30; done
  echo "$(date '+%F %T') seed $seed: train"
  nice -n 10 "$py" -m mathocr.train --config "$CONFIG" --seed "$seed" --run-dir "$run" --resume --set train.workers=2 \
    || { echo "train failed seed $seed"; exit 4; }
  echo "$(date '+%F %T') seed $seed: evaluate"
  nice -n 10 "$py" -m mathocr.evaluate --run-dir "$run" || { echo "evaluate failed seed $seed"; exit 5; }
  echo "$(date '+%F %T') seed $seed: cdm"
  "$py" -m mathocr.cdm --run-dir "$run" --cpus 4 || echo "cdm failed seed $seed"
  echo "$(date '+%F %T') seed $seed: done"
done
