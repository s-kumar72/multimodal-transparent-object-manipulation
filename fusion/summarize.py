"""
Tables comparing segmentation runs (data/fusion/runs/*/final.json) -> markdown.

  python -m fusion.summarize --runs data/fusion/runs --out results/fusion/summary.md
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SIM_TESTS = {"a_in_distribution": "sim a (in-dist.)", "g_heldout_all": "sim g (held out)", "f_crossover": "sim f (crossover)",
             "main": "sim main"}


def fmt(v, pct=False):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{100 * v:.1f}" if pct else f"{v:.2f}")


def mean(xs):
    xs = [x for x in xs if x is not None and not np.isnan(x)]
    return float(np.mean(xs)) if xs else None


def group(test):
    """{column: {metric: value}} with sim test sets averaged over levels and L4b also shown alone."""
    cols = defaultdict(lambda: defaultdict(list))
    for name, r in test.items():
        base, _, corr = name.partition("+")
        if base.startswith("real_"):
            col = base.removeprefix("real_").removesuffix("_test")
            keys = [col]
        else:
            level, _, ts = base.partition("_test_")
            keys = [SIM_TESTS.get(ts, ts)] + ([f"L4b {ts[0]}"] if level == "L4b" else [])
        for k in keys:
            for m, v in r.items():
                cols[(k, corr)][m].append(v)
    return {k: {m: mean(v) for m, v in d.items()} for k, d in cols.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="data/fusion/runs")
    p.add_argument("--out", default="results/fusion/summary.md")
    a = p.parse_args()
    order = ["sim", "real", "real2sim", "mix"]
    rank = lambda n: (order.index(n.split("_")[0]) if n.split("_")[0] in order else len(order), n)
    runs = {}
    for f in sorted(Path(a.runs).glob("*/final.json"), key=lambda f: rank(f.parent.name)):
        runs[f.parent.name] = json.load(open(f))
    if not runs:
        print("no finished runs in", a.runs)
        return
    g = {n: group(r["test"]) for n, r in runs.items()}
    clean_cols = sorted({c for d in g.values() for (c, corr) in d if not corr},
                        key=lambda c: (not c.startswith("sim"), not c.startswith("L4b"), c))
    L = ["# Glass segmentation runs", "",
         "IoU (%) / loc (% of images whose most confident pixel is on the glass), best checkpoint, clean inputs. "
         "Sim columns average the levels the run was tested on; `L4b x` is L4b alone.", "",
         "| run | train data | modalities | fusion | " + " | ".join(clean_cols) + " |",
         "|---|---|---|---|" + "---|" * len(clean_cols)]
    for n, r in runs.items():
        args = r["args"]
        train = ", ".join(sorted({s.split(":")[0] + (":" + s.split(":")[1] if s.startswith("real") else "")
                                  for s in args["train"]}))
        init = " (init: " + Path(args["init"]).parent.name + ")" if args.get("init") else ""
        fusion = ("static" if args["gate_mode"] == "static" else f"learned/{args['gate_norm']}" + ("" if args.get("no_cues") else "+cues")) \
            + ("+sup" if args.get("gate_sup", 0) > 0 and args["gate_mode"] != "static" else "") \
            + (" ws" if args.get("fusion_mode") == "weighted_sum" else "")
        cells = [f"{fmt(g[n].get((c, ''), {}).get('iou'), True)} / {fmt(g[n].get((c, ''), {}).get('loc'), True)}"
                 for c in clean_cols]
        L.append(f"| {n} | {train}{init} | {''.join({'rgb': 'rgb', 'depth': 'd', 'thermal': 't'}[m] for m in args['modalities'])} | {fusion} | " + " | ".join(cells) + " |")

    L += ["", "## Robustness and gates on sim test set a (all levels)", "",
          "IoU (%) clean and under a corruption applied to every test image; mean gate of the corrupted modality "
          "(clean → corrupted). A reliability-aware model should lose little IoU and lower that modality's gate.", "",
          "| run | clean IoU | thermal crossover IoU | thermal gate | thermal blank IoU | thermal gate | depth holes IoU | depth gate |",
          "|---|---|---|---|---|---|---|---|"]
    col = SIM_TESTS["a_in_distribution"]
    for n in runs:
        d = g[n]
        if (col, "") not in d:
            continue
        c = d[(col, "")]
        row = [fmt(c.get("iou"), True)]
        for corr, mod in (("thermal_crossover", "thermal"), ("thermal_blank", "thermal"), ("depth_holes", "depth")):
            x = d.get((col, corr))
            row += [fmt(x.get("iou"), True), f"{fmt(c.get('gate_' + mod))} → {fmt(x.get('gate_' + mod))}"] if x else ["–", "–"]
        L.append(f"| {n} | " + " | ".join(row) + " |")

    L += ["", "## Thermal gate vs. real thermal crossover (sim test set f vs a)", "",
          "| run | thermal gate, set a (contrast) | thermal gate, set f (crossover) | IoU a | IoU f |", "|---|---|---|---|---|"]
    for n in runs:
        a_, f_ = g[n].get((col, "")), g[n].get((SIM_TESTS["f_crossover"], ""))
        if a_ and f_ and a_.get("gate_thermal") is not None:
            L.append(f"| {n} | {fmt(a_.get('gate_thermal'))} | {fmt(f_.get('gate_thermal'))} | "
                     f"{fmt(a_.get('iou'), True)} | {fmt(f_.get('iou'), True)} |")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nSaved", a.out)


if __name__ == "__main__":
    main()