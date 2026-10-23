"""(Paths below are those of the Windows laptop; edit them for another machine.)
Stage 0c, step 1: candidates for the render-and-compare verifier.  From the n-best dumps (nbest_dump.py): per sample
the greedy output plus the beam hypotheses, de-duplicated, with the model score (normalized beam score; the greedy
output takes its beam's score, or the best one when no beam equals it).  Writes WORK/candidates.jsonl and the render
lists WORK/<group>/labels.jsonl (group "ind": CM only; group "ood": rendered in four fonts by scripts/fonts/render.py);
the reference of every sample is rendered too (id <split>__<index>__gt), only for the sanity check of the measure."""

import json
from pathlib import Path

NB = Path(r"D:\MathOCR\nbest_final_v2_s1")
WORK = Path(r"D:\MathOCR\verifier")
GROUPS = {"val": "ind", "test_im2latex": "ind", "test_oleehyo": "ind", "ood_spe": "ood", "ood_sce": "ood", "ood_cpe": "ood"}


def main() -> None:
    lists = {"ind": {}, "ood": {}}
    n_rows = 0
    with open(WORK / "candidates.jsonl", "w", encoding="utf-8", newline="\n") as out:
        for split, group in GROUPS.items():
            path = NB / f"{split}.jsonl"
            if not path.exists():
                print("missing", split)
                continue
            for r in (json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()):
                cands = {}
                for s, h in r["hyps"]:
                    cands.setdefault(h, s)
                if r["greedy"] not in cands:
                    cands[r["greedy"]] = max((s for s, _ in r["hyps"]), default=0.0)
                items = []
                for k, (h, s) in enumerate(sorted(cands.items(), key=lambda kv: -kv[1])):
                    cid = f"{split}__{r['index']}__{k}"
                    items.append({"id": cid, "latex": h, "score": s, "greedy": h == r["greedy"]})
                    lists[group][cid] = h
                lists[group][f"{split}__{r['index']}__gt"] = r["gt"]  # the reference, rendered for the sanity check only
                out.write(json.dumps({"split": split, "index": r["index"], "gt": r["gt"], "cands": items}) + "\n")
                n_rows += 1
    for group, ids in lists.items():
        (WORK / group).mkdir(parents=True, exist_ok=True)
        with open(WORK / group / "labels.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for cid, latex in ids.items():
                f.write(json.dumps({"id": cid, "latex": latex}) + "\n")
        print(group, len(ids), "candidates to render")
    print(n_rows, "samples")


if __name__ == "__main__":
    main()
