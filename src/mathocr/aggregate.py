"""Aggregate several seeds: mean ± sample standard deviation per split/metric.

    python -m mathocr.aggregate runs/base_s1 runs/base_s2 runs/base_s3 --out results/base

Reads ``eval/metrics.json`` (and ``eval/cdm.json`` when present) of every run;
writes ``<out>.json`` and a Markdown table ``<out>.md``.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

METRICS = (("exact_match", "EM", 100), ("edit_distance", "Edit dist ↓", 1), ("bleu", "BLEU", 100),
           ("cdm", "CDM", 100), ("cdm_exp_rate", "ExpRate@CDM", 100))


def collect(run_dirs: list[Path]) -> dict:
    per_split: dict[str, dict[str, list[float]]] = {}
    seeds = []
    for run in run_dirs:
        m = json.loads((run / "eval" / "metrics.json").read_text())
        seeds.append(m["seed"])
        cdm_path = run / "eval" / "cdm.json"
        cdm = json.loads(cdm_path.read_text()) if cdm_path.exists() else {}
        for split, vals in m["splits"].items():
            merged = {**vals, **cdm.get(split, {})}
            for key, *_ in METRICS:
                if key in merged:
                    per_split.setdefault(split, {}).setdefault(key, []).append(float(merged[key]))
    out = {"runs": [str(r) for r in run_dirs], "seeds": seeds, "splits": {}}
    for split, metrics in per_split.items():
        out["splits"][split] = {
            k: {"mean": statistics.fmean(v), "std": statistics.stdev(v) if len(v) > 1 else 0.0, "n": len(v),
                "values": v}
            for k, v in metrics.items()
        }
    return out


def markdown(agg: dict) -> str:
    cols = [m for m in METRICS if any(m[0] in s for s in agg["splits"].values())]
    lines = ["| Split | " + " | ".join(c[1] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for split, metrics in agg["splits"].items():
        cells = []
        for key, _, scale in cols:
            if key in metrics:
                m = metrics[key]
                digits = 1 if scale == 100 else 3
                cells.append(f"{m['mean'] * scale:.{digits}f} ± {m['std'] * scale:.{digits}f}")
            else:
                cells.append("—")
        lines.append(f"| {split} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"Seeds: {agg['seeds']}; mean ± sample std over {len(agg['seeds'])} runs.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Mean ± std of evaluation metrics over seeds.")
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    agg = collect(args.runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(agg, indent=2) + "\n")
    args.out.with_suffix(".md").write_text(markdown(agg))
    print(markdown(agg))


if __name__ == "__main__":
    main()
