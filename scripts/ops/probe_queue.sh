#!/usr/bin/env bash
# Sequential probe queue (scripts/probe.py).  Jobs file lines:  name|seed|override override ...
# Special names: reeval_<run>, trace_<run>, decode_<run>, knn_<run>, soft_<self|knn|knnNN>_<run>,
# knnp128_<run>, fknn_<final run>  (see scripts/probe.py main()).
# Re-reads the jobs file after every job (append to add work); skips finished probes
# (runs/<name>/probe.json or runs/<name>.failed); launches only with >= MIN_AVAIL_GB available.
#   PROBE_ROOT=/path/to/probe MATHOCR_ROOT=/path/to/repo PYTHON=... scripts/ops/probe_queue.sh
ROOT=${MATHOCR_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
PROBE_ROOT=${PROBE_ROOT:-/mnt/SSD/MathOCR/probe}
py=${PYTHON:-$ROOT/.venv/bin/python}
MIN_AVAIL_GB=${MIN_AVAIL_GB:-25}
jobs=$PROBE_ROOT/jobs.txt
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PROBE_ROOT
mkdir -p "$PROBE_ROOT/logs" "$PROBE_ROOT/runs"
cd "$ROOT"
if ! git diff --quiet -- data/processed/manifest.json data/processed/vocab.json || [[ ! -s data/processed/train.parquet ]]; then
  echo "data not ready or not the committed version; queue stops"; exit 3
fi
while true; do
  next=""
  while IFS='|' read -r name seed sets; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    [[ -f $PROBE_ROOT/runs/$name/probe.json || -f $PROBE_ROOT/runs/$name.failed ]] && continue
    next="$name|$seed|$sets"; break
  done < "$jobs"
  if [[ -z "$next" ]]; then sleep 60; continue; fi
  IFS='|' read -r name seed sets <<< "$next"
  until (( $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) >= MIN_AVAIL_GB )); do sleep 30; done
  echo "$(date '+%F %T') start $name seed=$seed $sets"
  if nice -n 10 "$py" scripts/probe.py --name "$name" --seed "$seed" --set $sets > "$PROBE_ROOT/logs/$name.out" 2>&1; then
    echo "$(date '+%F %T') done $name"
  else
    echo "$(date '+%F %T') FAILED $name"; touch "$PROBE_ROOT/runs/$name.failed"
  fi
done
