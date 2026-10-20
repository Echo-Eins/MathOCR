#!/usr/bin/env bash
# Copied from /mnt/SSD/MathOCR/final/watchdog.sh (GB10, shared 121 GB unified memory).  Run it with nohup next to
# long MathOCR jobs on a shared machine; it only ever stops MathOCR processes.
# Memory safety valve for MathOCR jobs only (never touches other jobs).
# Two consecutive samples below 12 GB available -> stop, in this order, the first that exists:
# every experiment process (probe.py and its decode_study.py child), final seed 3, final seed 2, final seed 1.
# Final runs resume from the last epoch boundary (lane.sh, --resume).
low=0
experiment() { ps -eo pid,args | awk '($0 ~ /scripts\/probe\.py/ || $0 ~ /scripts\/decode_study\.py/ || $0 ~ /scripts\/knn_decode\.py/ || $0 ~ /scripts\/soft_targets\.py/) && /python/ && !/awk/ {print $1}' | tr '\n' ' '; }
final() { ps -eo pid,args | awk -v r="runs/final_s$1" '$0 ~ /mathocr\.train/ && index($0, r) && !/awk/ {print $1}' | head -1; }
while true; do
  a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if (( a < 12 )); then low=$((low + 1)); else low=0; fi
  if (( low >= 2 )); then
    for v in experiment "final 3" "final 2" "final 1"; do
      p=$($v)
      if [[ -n "$p" ]]; then
        echo "$(date '+%F %T') available ${a} GB: stopping $v (pids $p)"; kill $p; sleep 60; break
      fi
    done
    low=0
  fi
  sleep 15
done
