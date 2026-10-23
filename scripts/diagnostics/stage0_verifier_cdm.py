"""(Paths below are those of the Windows laptop; edit them for another machine.)
Stage 0c, step 3: the verifier's picks scored with the official CDM (render-based, notation-blind).  Only the samples
where a pick differs from the greedy output are scored (elsewhere the difference is zero by construction): for each
split, the greedy output and every method's pick on the union of changed samples.  Reported: change of the split's
mean CDM and of the number of formulas with CDM = 1, against greedy.
Output: docs/probe_results/stage0_verifier_cdm.json."""

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mathocr.cdm import run_cdm  # noqa: E402

WORK = Path(r"D:\MathOCR\verifier")
METHODS = ("chamfer", "deep_final", "deep_f1", "size")


def main() -> None:
    picks = [json.loads(x) for x in (WORK / "picks.jsonl").read_text(encoding="utf-8").splitlines()]
    by_split = defaultdict(list)
    for p in picks:
        by_split[p["split"]].append(p)
    run = WORK / "cdm_run"
    shutil.rmtree(run, ignore_errors=True)
    (run / "eval").mkdir(parents=True)
    changed = {}
    for split, ps in by_split.items():
        idx = [i for i, p in enumerate(ps) if any(p[f"pick_{m}"] != p["greedy"] for m in METHODS)]
        changed[split] = idx
        for method in ("greedy",) + METHODS:
            key = "greedy" if method == "greedy" else f"pick_{method}"
            items = [{"img_id": f"{split}_{i}", "gt": ps[i]["gt"], "pred": ps[i][key]} for i in idx]
            (run / "eval" / f"{split}__{method}.cdm.json").write_text(json.dumps(items, indent=0), encoding="utf-8")
        print(split, len(ps), "samples,", len(idx), "changed by some method", flush=True)
    run_cdm(run, None, cpus=8, keep=True)
    res = {}
    for split, ps in by_split.items():
        scores = {}
        for method in ("greedy",) + METHODS:
            mr = json.loads((run / "eval" / "cdm_work" / f"{split}__{method}" / "metrics_res.json").read_text())
            det = mr.get("details", {})
            scores[method] = {k: (v.get("F1_score", v) if isinstance(v, dict) else v) for k, v in det.items()}
        row = {"n": len(ps), "changed": len(changed[split])}
        g = scores["greedy"]
        for method in METHODS:
            s = scores[method]
            common = [k for k in g if k in s]
            d = sum(float(s[k]) - float(g[k]) for k in common) / len(ps)
            ones = sum((float(s[k]) == 1.0) - (float(g[k]) == 1.0) for k in common)
            row[method] = {"delta_mean_cdm_over_split": d, "delta_cdm1_formulas": int(ones), "scored": len(common)}
        res[split] = row
        print(f"{split:13} changed {row['changed']:4}: " + " | ".join(
            f"{m} dCDM {row[m]['delta_mean_cdm_over_split']:+.4f} dCDM=1 {row[m]['delta_cdm1_formulas']:+d}" for m in METHODS), flush=True)
    (ROOT / "docs" / "probe_results" / "stage0_verifier_cdm.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
