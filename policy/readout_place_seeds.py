"""
Read-out of the seeds on the pick-and-place level L0_vis_place (results/hard_scenario/place_bench/README.md, step 4),
from the episodes.json files of the final evaluations (no new runs):

  python policy/readout_place_seeds.py > results/hard_scenario/place_bench/readout_seeds.txt

1. the macro average of every run per seed; 2. the table of the baselines (seeds pooled) and the best baseline of
each cell, built as Stage 6 builds it on the pick level (results/hard_scenario/grid_bench/README.md); 3. the method
(hcg2_cue) under rules 4 and 5 of Stage 6 with that table; 4. the pick and the place apart, per seed.
Runs are paired by seed on the same episodes and pooled; two-sided sign test on the episodes only one of the two
solves. Seeds without a finished evaluation of every run are left out and named; the tables are then not final.
"""
from pathlib import Path

import click
import numpy as np

from readout_place_method import BASELINES, SHOWS, load, ok, picked, sign_p

METHOD = "hcg2_cue"
RUNS = {**{w: f"L0_vis_place_{w}_700demos_imagenet" for w in BASELINES},     # ties for the best baseline: the first
        METHOD: f"L0_vis_place_rgbdt_700demos_{METHOD}"}
CELLS = range(11)


def folder(name, seed):
    return name if seed == 42 else f"{name}_seed{seed}"


def rate(runs, cells=CELLS, f=ok):
    return float(np.mean([f(e) for r in runs for e in r.values() if e["cell"] in cells]))


def pooled(A, B, cells=CELLS):
    a = b = 0
    for x, y in zip(A, B):
        keys = [k for k in x if x[k]["cell"] in cells]
        a += sum(ok(x[k]) and not ok(y[k]) for k in keys)
        b += sum(ok(y[k]) and not ok(x[k]) for k in keys)
    return rate(A, cells), rate(B, cells), a, b, sign_p(a, b)


def macro(runs):
    return float(np.mean([rate(runs, (c,)) for c in CELLS]))


def sd(v):
    return f"{np.std(v, ddof=1):.3f}" if len(v) > 1 else "-"


def fmt(p):
    return "< 1e-300" if p < 1e-300 else f"{p:.2g}"


@click.command()
@click.option("--root", default="results/hard_scenario/place_bench")
@click.option("--seeds", default="42,43,44", help="for testing the script on the seeds that exist")
def main(root, seeds):
    root, want = Path(root), [int(s) for s in seeds.split(",")]
    runs = {n: {s: load(root / folder(f, s)) for s in want} for n, f in RUNS.items()}
    seeds = [s for s in want if all(runs[n][s] is not None for n in RUNS)]
    final = len(seeds) == len(want)
    if not final:
        left = {n: [s for s in want if runs[n][s] is None] for n in RUNS}
        print(f"Not finished yet: { {n: v for n, v in left.items() if v} }; the tables below are on seeds {seeds} and "
              "not final.\n")
    if not seeds:
        return
    R = {n: [runs[n][s] for s in seeds] for n in RUNS}
    assert all(set(r) == set(R["rgbdt"][0]) for rs in R.values() for r in rs), "not the same episodes"

    print(f"## 1. Macro average per seed ({len(R['rgbdt'][0])} episodes per run, {len(R['rgbdt'][0]) // 11} per cell)\n")
    print("| run | " + " | ".join(f"seed {s}" for s in seeds) + " | mean | standard deviation |")
    print("|---|" + "---|" * (len(seeds) + 2))
    for n in RUNS:
        m = [macro([r]) for r in R[n]]
        print(f"| {n} | " + " | ".join(f"{x:.3f}" for x in m) + f" | {np.mean(m):.3f} | {sd(m)} |")

    print(f"\n## 2. The baselines per cell (seeds {seeds} pooled) and the best baseline of each cell\n")
    print("| cell | sensors that show the glass | " + " | ".join(BASELINES) + " | best baseline | which |")
    print("|---|---|" + "---|" * (len(BASELINES) + 2))
    which = []
    for c in CELLS:
        r = {w: rate(R[w], (c,)) for w in BASELINES}
        which.append(max(BASELINES, key=lambda w: r[w]))
        print(f"| {c} | {SHOWS[c]} | " + " | ".join(f"{r[w]:.3f}" for w in BASELINES)
              + f" | {r[which[-1]]:.3f} | {which[-1]} |")
    best = [rate(R[which[c]], (c,)) for c in CELLS]
    print("| macro average | | " + " | ".join(f"{macro(R[w]):.3f}" for w in BASELINES) + f" | {np.mean(best):.3f} | |")

    print(f"\n## 3. `{METHOD}` under rules 4 and 5 of Stage 6, with the table of item 2\n")
    print("| cell | sensors that show the glass | " + " | ".join(f"seed {s}" for s in seeds) + " | pooled | best "
          "baseline of the cell | only the method succeeds / only the baseline | p | |")
    print("|---|---|" + "---|" * (len(seeds) + 5))
    below, above, M = [], [], R[METHOD]
    for c in CELLS:
        sa, sb, a, b, p = pooled(M, R[which[c]], (c,))
        if p < 0.05:
            (below if sa < sb else above).append(c)
        print(f"| {c} | {SHOWS[c]} | " + " | ".join(f"{rate([r], (c,)):.3f}" for r in M) + f" | {sa:.3f} | {sb:.3f} "
              f"({which[c]}) | {a} / {b} | {fmt(p)} | {('above' if sa > sb else 'below') if p < 0.05 else ''} |")
    print("| macro average | | " + " | ".join(f"{macro([r]):.3f}" for r in M)
          + f" | {macro(M):.3f} | {np.mean(best):.3f} | | | |\n")
    print("| baseline | the method | the baseline, same seeds | only the method succeeds / only the baseline | p |")
    print("|---|---|---|---|---|")
    over = True
    for w in BASELINES:
        sa, sb, a, b, p = pooled(M, R[w])
        over &= macro(M) > macro(R[w])
        print(f"| {w} | {sa:.3f} | {sb:.3f} | {a} / {b} | {fmt(p)} |")
    print(f"\nCells below their best baseline with p < 0.05: {below or 'none'}; above: {above or 'none'}; neither: "
          f"{[c for c in CELLS if c not in below + above] or 'none'}.")
    print(f"Macro average {macro(M):.3f} against the highest baseline {max(macro(R[w]) for w in BASELINES):.3f}: "
          f"{'above' if over else 'not above'} every baseline.")
    print("\"Better in nearly every case\" " + ("holds" if not below and over else "does not hold")
          + (" on L0_vis_place." if final else f" on the seeds finished so far ({seeds}); not final."))

    print("\n## 4. The pick and the place apart, per seed\n")
    print("| run | glass picked, share of all episodes: " + " / ".join(f"seed {s}" for s in seeds)
          + " | pooled | placed, among the picked: " + " / ".join(f"seed {s}" for s in seeds) + " | pooled |")
    print("|---|---|---|---|---|")
    among = lambda rs: float(np.mean([ok(e) for r in rs for e in r.values() if picked(e)]))
    for n in ("rgbdt", METHOD):
        print(f"| {n} | " + " / ".join(f"{rate([r], f=picked):.3f}" for r in R[n]) + f" | {rate(R[n], f=picked):.3f} | "
              + " / ".join(f"{among([r]):.3f}" for r in R[n]) + f" | {among(R[n]):.3f} |")


if __name__ == "__main__":
    main()
