"""Run the official CDM evaluator (docker image ``mathocr-cdm``) on a run.

    docker build -f docker/cdm.Dockerfile -t mathocr-cdm docker/
    python -m mathocr.cdm --run-dir runs/base_s1

Scores every ``eval/<split>.cdm.json`` written by ``mathocr.evaluate`` and
collects ``mean_score`` (CDM F1) and ``exp_rate`` (share of formulas with
CDM F1 = 1) into ``eval/cdm.json``.  CDM skips samples whose ground truth
fails to render; the counts of scored samples are recorded.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

IMAGE = "mathocr-cdm"


def run_cdm(run_dir: Path, splits: list[str] | None, cpus: int, keep: bool) -> dict:
    eval_dir = (run_dir / "eval").resolve()
    files = sorted(eval_dir.glob("*.cdm.json"))
    if splits:
        files = [f for f in files if f.name.removesuffix(".cdm.json") in splits]
    summary_path = eval_dir / "cdm.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for f in files:
        split = f.name.removesuffix(".cdm.json")
        out = eval_dir / "cdm_work"
        # The container writes as the host user on POSIX; Docker Desktop (Windows) maps the mount itself.
        user = ["-u", f"{os.getuid()}:{os.getgid()}"] if hasattr(os, "getuid") else []
        cmd = ["docker", "run", "--rm", f"--cpus={cpus}", *user,
               "-e", "HOME=/tmp", "-v", f"{eval_dir}:/work", IMAGE,
               "-i", f"/work/{f.name}", "-o", "/work/cdm_work", "-p", str(cpus)]
        # nice only on POSIX: on Windows a `nice` on PATH is MSYS (Git for Windows), whose argument
        # conversion rewrites the container paths (/work/... -> C:/Program Files/Git/work/...)
        nice = ["nice", "-n", "19"] if os.name == "posix" and shutil.which("nice") else []
        subprocess.run([*nice, *cmd], check=True)
        # evaluation.py names its output dir after the input basename up to the first dot.
        res = json.loads((out / split / "metrics_res.json").read_text())
        details = res.get("details", {})
        summary[split] = {"cdm": res["mean_score"], "cdm_exp_rate": res["exp_rate"],
                          "scored": len(details), "submitted": len(json.loads(f.read_text()))}
        print(json.dumps({"split": split, **summary[split]}), flush=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
        if not keep:
            shutil.rmtree(out / split)
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Score a run's predictions with the official CDM evaluator.")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--splits", nargs="*")
    ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--keep", action="store_true", help="keep CDM renders and match visualizations")
    args = ap.parse_args(argv)
    run_cdm(args.run_dir, args.splits, args.cpus, args.keep)


if __name__ == "__main__":
    main()
