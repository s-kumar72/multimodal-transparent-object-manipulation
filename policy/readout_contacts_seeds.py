"""
Read-out of round 2 of the contact check (results/hard_scenario/contact_check/README.md): the pick level of the
benchmark with the contact model of the place level (L0_vis_ell), three seeds, from the episodes.json files of the
evaluations (no new runs):

  python policy/readout_contacts_seeds.py > results/hard_scenario/contact_check/readout_seeds.txt

1. every run on L0_vis against the same checkpoint on L0_vis_ell, same episodes; 2. the table of the baselines on
L0_vis_ell, built as the frozen table of Stage 6 (results/hard_scenario/grid_bench/README.md): per cell the baseline
with the highest success over its three seeds; 3. the method (hcg_full) under rules 4 and 5 of Stage 6 with that
table; 4. the method against the rgbdt baseline, and the baselines against each other, on both levels.
Runs are paired by seed on the same episodes and pooled; two-sided sign test on the episodes only one of the two
solves. Seeds without a finished evaluation on L0_vis_ell of every run are left out and named.
"""
from pathlib import Path

import click
import numpy as np

from readout_cue_gate import BEST, SHOWS, WHICH, load, ok, sign_p

SEEDS = (42, 43, 44)
BASELINES = ("rgbdt", "rgbd", "rgbt", "rgb")          # ties for the best baseline of a cell go to the first of these
FROZEN = {"rgbdt": 0.428, "rgbd": 0.397, "rgbt": 0.194, "rgb": 0.153}      # macro averages on L0_vis, Stage 6
RUNS = {**{w: ("grid_bench", f"L0_vis_{w}_700demos_imagenet") for w in BASELINES},
        "hcg_full": ("cue_gate", "L0_vis_rgbdt_700demos_hcg_full")}
METHOD = "hcg_full"
CELLS = range(11)


def fmt(p):
    return "< 1e-300" if p < 1e-300 else f"{p:.2g}"


def folder(name, seed):
    return name if seed == 42 else f"{name}_seed{seed}"


def rate(runs, cells=CELLS):
    return float(np.mean([ok(e) for r in runs for e in r.values() if e["cell"] in cells]))


def pooled(A, B, cells=CELLS):
    """Runs paired one to one: the two rates, the episodes only one of the two solves, sign test."""
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


@click.command()
@click.option("--root", default="results/hard_scenario")
@click.option("--out_group", default="contact_check")
def main(root, out_group):
    root = Path(root)
    old = {n: {s: load(root / g / folder(f, s)) for s in SEEDS} for n, (g, f) in RUNS.items()}
    new = {n: {s: load(root / out_group / folder(f, s)) for s in SEEDS} for n, (g, f) in RUNS.items()}
    seeds = [s for s in SEEDS if all(new[n][s] is not None for n in RUNS)]
    final = len(seeds) == len(SEEDS)
    if not final:
        left = {n: [s for s in SEEDS if new[n][s] is None] for n in RUNS}
        print(f"Not evaluated on L0_vis_ell yet: { {n: v for n, v in left.items() if v} }; the tables below are on "
              f"seeds {seeds} and not final.\n")
    if not seeds:
        return
    O = {n: [old[n][s] for s in seeds] for n in RUNS}
    N = {n: [new[n][s] for s in seeds] for n in RUNS}
    assert all(set(r) == set(O["rgbdt"][0]) for rs in [*O.values(), *N.values()] for r in rs), "not the same episodes"
    n_cell = len(N["rgbdt"][0]) // 11

    print("## 1. Every run on L0_vis and on L0_vis_ell: the same checkpoint on the same episodes\n")
    print("| run | seed | L0_vis | L0_vis_ell | only L0_vis / only L0_vis_ell | p |\n|---|---|---|---|---|---|")
    for n in RUNS:
        for s in SEEDS:
            if new[n][s] is None:
                continue
            so, sn, a, b, p = pooled([old[n][s]], [new[n][s]])
            print(f"| {n} | {s} | {so:.3f} | {sn:.3f} | {a} / {b} | {fmt(p)} |")
    print("\nMacro average over the seeds, mean (standard deviation):\n")
    print("| run | L0_vis | frozen (Stage 6) | L0_vis_ell | difference | runs that rise with p < 0.05 |\n|---|---|---|---|---|---|")
    for n in RUNS:
        mo, mn = [macro([r]) for r in O[n]], [macro([r]) for r in N[n]]
        up = sum(p < 0.05 and b > a for _, _, a, b, p in (pooled([x], [y]) for x, y in zip(O[n], N[n])))
        frozen = f"{FROZEN[n]:.3f}" if n in FROZEN else "-"
        print(f"| {n} | {np.mean(mo):.3f} ({sd(mo)}) | {frozen} | {np.mean(mn):.3f} ({sd(mn)}) | "
              f"{np.mean(mn) - np.mean(mo):+.3f} | {up} of {len(mo)} |")

    print(f"\n## 2. The baselines on L0_vis_ell per cell ({n_cell} episodes per cell and seed, seeds {seeds} pooled) "
          "and the best baseline of each cell\n")
    print("| cell | sensors that show the glass | " + " | ".join(BASELINES) + " | best baseline on L0_vis_ell | which | "
          "best baseline on L0_vis (frozen) | which |")
    print("|---|---|" + "---|" * (len(BASELINES) + 4))
    which = []
    for c in CELLS:
        r = {w: rate(N[w], (c,)) for w in BASELINES}
        which.append(max(BASELINES, key=lambda w: r[w]))
        print(f"| {c} | {SHOWS[c]} | " + " | ".join(f"{r[w]:.3f}" for w in BASELINES)
              + f" | {r[which[-1]]:.3f} | {which[-1]} | {BEST[c]:.3f} | {WHICH[c]} |")
    best = [rate(N[which[c]], (c,)) for c in CELLS]
    print("| macro average | | " + " | ".join(f"{macro(N[w]):.3f}" for w in BASELINES)
          + f" | {np.mean(best):.3f} | | {np.mean(BEST):.3f} | |")
    changed = [c for c in CELLS if which[c] != WHICH[c]]
    print(f"\nCells whose best baseline is another one than on L0_vis: {changed or 'none'}.")

    print(f"\n## 3. `{METHOD}` on L0_vis_ell under rules 4 and 5 of Stage 6, with the table of item 2\n")
    print("| cell | sensors that show the glass | " + " | ".join(f"seed {s}" for s in seeds) + " | pooled | best "
          "baseline of the cell on L0_vis_ell, same seeds | only the method succeeds / only the baseline | p | |")
    print("|---|---|" + "---|" * (len(seeds) + 5))
    below, above = [], []
    M = N[METHOD]
    for c in CELLS:
        sa, sb, a, b, p = pooled(M, N[which[c]], (c,))
        if p < 0.05:
            (below if sa < sb else above).append(c)
        print(f"| {c} | {SHOWS[c]} | " + " | ".join(f"{rate([r], (c,)):.3f}" for r in M) + f" | {sa:.3f} | {sb:.3f} "
              f"({which[c]}) | {a} / {b} | {fmt(p)} | {('above' if sa > sb else 'below') if p < 0.05 else ''} |")
    mm = [macro([r]) for r in M]
    print("| macro average | | " + " | ".join(f"{m:.3f}" for m in mm) + f" | {macro(M):.3f} | {np.mean(best):.3f} | | | |")
    print(f"\nMacro average over the seeds: mean {np.mean(mm):.3f}, lowest {min(mm):.3f}, highest {max(mm):.3f}, "
          f"standard deviation {sd(mm)}\n")
    print("| baseline | the method | the baseline, same seeds | only the method succeeds / only the baseline | p |")
    print("|---|---|---|---|---|")
    over = True
    for w in BASELINES:
        sa, sb, a, b, p = pooled(M, N[w])
        over &= macro(M) > macro(N[w])
        print(f"| {w} | {sa:.3f} | {sb:.3f} | {a} / {b} | {fmt(p)} |")
    print(f"\nCells below their best baseline with p < 0.05: {below or 'none'}; above: {above or 'none'}; neither: "
          f"{[c for c in CELLS if c not in below + above] or 'none'}.")
    print(f"Macro average {macro(M):.3f} against the highest baseline {max(macro(N[w]) for w in BASELINES):.3f}: "
          f"{'above' if over else 'not above'} every baseline.")
    print("\"Better in nearly every case\" " + ("holds" if not below and over else "does not hold")
          + (" on L0_vis_ell." if final else f" on the seeds finished so far ({seeds}); not final."))

    print("\n## 4. The differences on both levels (runs paired by seed, pooled, all episodes)\n")
    print("| pair | level | first | second | difference | only the first succeeds / only the second | p |")
    print("|---|---|---|---|---|---|---|")
    for x, y in ((METHOD, "rgbdt"), ("rgbdt", "rgbd"), ("rgbd", "rgbt"), ("rgbt", "rgb")):
        for level, R in (("L0_vis", O), ("L0_vis_ell", N)):
            sa, sb, a, b, p = pooled(R[x], R[y])
            print(f"| {x} against {y} | {level} | {sa:.3f} | {sb:.3f} | {sa - sb:+.3f} | {a} / {b} | {fmt(p)} |")


if __name__ == "__main__":
    main()
