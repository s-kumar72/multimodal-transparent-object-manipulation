"""
Read-out of the seeds of hindsight cue gating (results/hard_scenario/cue_gate/README.md, item 5 of round 1) under the
rules of Stage 6 of results/hard_scenario/grid_bench/README.md, from the episodes.json files of the final evaluations
(no new runs):

  python policy/readout_cue_seeds.py [--arm full] > results/hard_scenario/cue_gate/readout_seeds.txt

The method's run of seed s is paired with the baseline's run of seed s on the same episodes (test set and episode
seed); the pairs of the seeds are pooled and the two-sided sign test is on the episodes only one of the two solves.
Printed: success per cell and seed; per cell against the best baseline of that cell (frozen table of Stage 6); over
all episodes against each of the four baselines; and the two conditions of "better in nearly every case" (no cell
below its best baseline with p < 0.05, macro average above every baseline); and, next to it and not a criterion,
the learned attention weighting P9 with the same seeds (extra supervision; --fusion_root, left out when its runs are
not found). Seeds without a finished evaluation are left out and named.
"""
from pathlib import Path

import click
import numpy as np

from readout_cue_gate import BEST, REF, SHOWS, WHICH, load, mark, ok, sign_p

SEEDS = (42, 43, 44)
BASELINES = ("rgbdt", "rgbd", "rgbt", "rgb")
FROZEN = {"rgbdt": 0.428, "rgbd": 0.397, "rgbt": 0.194, "rgb": 0.153}   # macro averages, three seeds, Stage 6


def fmt(p):
    return "< 1e-300" if p < 1e-300 else f"{p:.2g}"


def folder(name, seed):
    return name if seed == 42 else f"{name}_seed{seed}"


def rate(runs, cells):
    """Success over the episodes of these cells, the runs pooled."""
    return np.mean([ok(e) for r in runs for e in r.values() if e["cell"] in cells])


def pooled(A, B, cells):
    """Runs paired by seed: the two rates, the episodes only one of the two solves, sign test."""
    a = b = 0
    for x, y in zip(A, B):
        keys = [k for k in x if x[k]["cell"] in cells]
        a += sum(ok(x[k]) and not ok(y[k]) for k in keys)
        b += sum(ok(y[k]) and not ok(x[k]) for k in keys)
    return rate(A, cells), rate(B, cells), a, b, sign_p(a, b)


@click.command()
@click.option("--arm", default="full")
@click.option("--root", default="results/hard_scenario/cue_gate")
@click.option("--base_root", default="results/hard_scenario/grid_bench")
@click.option("--fusion_root", default="results/hard_scenario/fusion_vis")
def main(arm, root, base_root, fusion_root):
    name = f"L0_vis_rgbdt_700demos_hcg_{arm}"
    runs = {s: load(Path(root) / folder(name, s)) for s in SEEDS}
    seeds = [s for s in SEEDS if runs[s] is not None]
    if len(seeds) < len(SEEDS):
        print(f"Not finished yet, left out: seeds {[s for s in SEEDS if s not in seeds]}\n")
    meth = [runs[s] for s in seeds]
    base = {w: [load(Path(base_root) / folder(f"L0_vis_{w}_700demos_imagenet", s)) for s in seeds] for w in BASELINES}
    assert all(set(r) == set(meth[0]) for rs in [meth, *base.values()] for r in rs), "not the same episodes"
    n = len(meth[0]) // 11

    print(f"## Success per cell of `hcg_{arm}` ({n} episodes per cell and seed; seeds {seeds}). Marks: against the "
          "best baseline of the cell, runs paired by seed on the same episodes and pooled, sign test, p < 0.05\n")
    print("| cell | sensors that show the glass | " + " | ".join(f"seed {s}" for s in seeds) + " | pooled | best "
          "baseline of the cell, same seeds | the same, frozen | only the method succeeds / only the baseline | p | "
          "reference |")
    print("|---|---|" + "---|" * (len(seeds) + 6))
    below, above = [], []
    for c in range(11):
        sa, sb, a, b, p = pooled(meth, base[WHICH[c]], (c,))
        if p < 0.05:
            (below if sa < sb else above).append(c)
        print(f"| {c} | {SHOWS[c]} | " + " | ".join(f"{rate([r], (c,)):.3f}" for r in meth)
              + f" | {sa:.3f}{mark(p, sa - sb)} | {sb:.3f} ({WHICH[c]}) | {BEST[c]:.3f} | {a} / {b} | {fmt(p)} | "
              f"{REF[c]:.3f} |")
    macros = [np.mean([rate([r], (c,)) for c in range(11)]) for r in meth]
    macro = float(np.mean([rate(meth, (c,)) for c in range(11)]))
    best_same = np.mean([rate(base[WHICH[c]], (c,)) for c in range(11)])
    print("| macro average | | " + " | ".join(f"{m:.3f}" for m in macros)
          + f" | {macro:.3f} | {best_same:.3f} | {np.mean(BEST):.3f} | | | {np.mean(REF):.3f} |")
    print(f"\nMacro average over the seeds: mean {np.mean(macros):.3f}, lowest {min(macros):.3f}, highest "
          f"{max(macros):.3f}" + (f", standard deviation {np.std(macros, ddof=1):.3f}" if len(macros) > 1 else ""))

    print("\n## Over all episodes against each baseline (runs paired by seed, pooled)\n")
    print("| baseline | the method | the baseline, same seeds | frozen macro | only the method succeeds / only the "
          "baseline | p |")
    print("|---|---|---|---|---|---|")
    over = True
    for w in BASELINES:
        sa, sb, a, b, p = pooled(meth, base[w], range(11))
        over &= macro > FROZEN[w]
        print(f"| {w} ImageNet | {sa:.3f} | {sb:.3f} | {FROZEN[w]:.3f} | {a} / {b} | {fmt(p)} |")

    print("\n## The claim of Stage 6 (rule 5)\n")
    print(f"Cells below their best baseline with p < 0.05: {below or 'none'}; above: {above or 'none'}; neither: "
          f"{[c for c in range(11) if c not in below + above] or 'none'}.")
    print(f"Macro average {macro:.3f} against the highest baseline {max(FROZEN.values()):.3f}: "
          f"{'above' if over else 'not above'} every baseline.")
    verdict = "holds" if not below and over else "does not hold"
    print(f"\"Better in nearly every case\" {verdict}" + ("." if len(seeds) == len(SEEDS) else
                                                         f" on the seeds finished so far ({seeds}); not final."))

    p9 = [load(Path(fusion_root) / folder("L0_vis_rgbdt_700demos_fusion_kp", s)) for s in seeds]
    if all(r is not None and set(r) == set(meth[0]) for r in p9):
        print("\n## Next to it, not a criterion: the learned attention weighting P9 (extra supervision: segmentation "
              "pre-training on simulator masks), same seeds, same episodes\n")
        print("| cell | sensors that show the glass | the method | P9 | only the method succeeds / only P9 | p |")
        print("|---|---|---|---|---|---|")
        for c in [*range(11), None]:
            sa, sb, a, b, p = pooled(meth, p9, range(11) if c is None else (c,))
            print(f"| {'all episodes' if c is None else c} | {'' if c is None else SHOWS[c]} | {sa:.3f} | {sb:.3f} | "
                  f"{a} / {b} | {fmt(p)} |")


if __name__ == "__main__":
    main()
