#!/usr/bin/env bash
# Full protocol for one seed: train -> evaluate (text metrics) -> CDM.
#   scripts/run_seed.sh <seed> [config] [extra train args...]
# Re-running continues an interrupted run from its last epoch (--resume).
set -euo pipefail
seed=${1:?seed}; config=${2:-configs/base.yaml}; shift $(( $# > 1 ? 2 : 1 ))
name=$(basename "$config" .yaml)
run=runs/${name}_s${seed}
py=${PYTHON:-.venv/bin/python}
export PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8
nice -n 10 "$py" -m mathocr.train --config "$config" --seed "$seed" --run-dir "$run" --resume "$@"
nice -n 10 "$py" -m mathocr.evaluate --run-dir "$run"
if docker image inspect mathocr-cdm >/dev/null 2>&1; then
  "$py" -m mathocr.cdm --run-dir "$run"
else
  echo "docker image mathocr-cdm missing: CDM skipped (docker build -f docker/cdm.Dockerfile -t mathocr-cdm docker/)" >&2
fi
