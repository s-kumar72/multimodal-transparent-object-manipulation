"""
Read-out of the frozen second version of the benchmark (results/hard_scenario/bench_v2/README.md, section 13), from the
episodes.json files of the final evaluations (no new runs):

  python policy/readout_v2_full.py > results/hard_scenario/v2/readout.txt

Per task (T1 V2b_place, T2 V2b_align, T3 V2b_two), seeds 42, 43 and 44:
  1. the macro average of every run per seed, its mean and standard deviation;
  2. success per cell (T3: per pair) over the seeds, and per cell the best baseline: the one with the highest success
     over its seeds among the plain baselines and the baselines from the literature (a tie goes to the first listed);
  3. `cue` against the best baseline of every cell, and the verdict "better in nearly every case": it holds when no
     cell is below its best baseline and the macro average is above that of every baseline;
  4. `cue` against plain rgbdt and against the one network, over all episodes and per cell;
  5. the generalisation sets (the 15 combinations of states that are in no cell; glass shapes that are in no training
     set): the macro average of every run, `cue` against plain rgbdt and against the one network;
  6. the layout reference (plain rgbdt trained and tested on the level's scenes with the glass in no image).
Runs are paired by seed on the same episodes and pooled; two-sided sign test on the episodes only one of the two
solves; "above" and "below" mean p < 0.05. Cell 9 (thermal faint alone) is the benchmark's open cell on T1: it is in
every table and average and is marked there. A run without a finished evaluation of every seed is left out and named.
"""
import json
from math import comb
from pathlib import Path

import click
import numpy as np

TASKS = {"T1": ("V2b_place", 700, ("j_cells11", "k_cells11_heldout")),
         "T2": ("V2b_align", 630, ("j_cells11", "k_cells11_heldout")),
         "T3": ("V2b_two", 700, ("j_pairs12", "k_pairs12_heldout"))}
GENERAL = {"new combinations": ("l_cells15_new", "m_cells15_new_heldout"), "unseen objects": ("u_cells11_unseen",)}
NOGLASS = {"T1": ("w_noglass", "x_noglass_heldout"), "T2": ("w_noglass", "x_noglass_heldout"),
           "T3": ("w_noglass_pairs", "x_noglass_pairs_heldout")}
PLAIN = {"plain rgbdt": "rgbdt_{n}demos_imagenet", "plain rgbd": "rgbd_{n}demos_imagenet",
         "plain rgbt": "rgbt_{n}demos_imagenet", "plain rgb": "rgb_{n}demos_imagenet"}
LIT = {"self-attention fusion (MulSA)": "rgbdt_{n}demos_mulsa", "sensor dropout": "rgbdt_{n}demos_dropout_imagenet",
       "stage-guided fusion (MS-Bot)": "rgbdt_{n}demos_msbot"}              # T1 only
CUE = {"T1": "rgbdt_{n}demos_hcg2_cue", "T2": "rgbdt_{n}demos_hcg2a_cue", "T3": "rgbdt_{n}demos_hcg2_cue"}
OTHER = {"second policy (ACT head)": "rgbdt_{n}demos_act", "second policy with the cue": "rgbdt_{n}demos_act_cue"}  # T1 only
OPEN_CELLS = {"T1": (9,), "T2": (), "T3": ()}        # T2: the pilot's `cue` run solved cell 9 (0.417)
P = 0.05


def load(folder, sets):
    eps = {}
    for ts in sets:
        f = Path(folder) / ts / "episodes.json"
        if not f.exists():
            return None
        eps.update({(ts, e["seed"]): e for e in json.loads(f.read_text())})
    return eps


def ok(e):
    return e["outcome"] == "success"


def sign_p(a, b):
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def cells_of(runs):
    return sorted({e["cell"] for r in runs for e in r.values()})


def rate(runs, cells=None):
    """Success over the seeds' episodes; with several cells the macro average over them."""
    cs = cells_of(runs) if cells is None else cells
    return float(np.mean([np.mean([ok(e) for r in runs for e in r.values() if e["cell"] == c]) for c in cs]))


def pooled(A, B, cells=None):
    """Episodes only A solves, only B solves, over the seeds both have, and the p of the sign test."""
    a = b = 0
    for ra, rb in zip(A, B):
        for k in ra:
            if cells is None or ra[k]["cell"] in cells:
                a += ok(ra[k]) and not ok(rb[k])
                b += ok(rb[k]) and not ok(ra[k])
    return a, b, sign_p(a, b)


def fmt(p):
    return "< 1e-300" if p < 1e-300 else f"{p:.2g}"


def against(name_a, A, name_b, B, cs):
    a, b, p = pooled(A, B)
    up = [c for c in cs if (lambda x: x[2] < P and x[0] > x[1])(pooled(A, B, (c,)))]
    dn = [c for c in cs if (lambda x: x[2] < P and x[0] < x[1])(pooled(A, B, (c,)))]
    print(f"{name_a} against {name_b}: {rate(A):.3f} against {rate(B):.3f}; only the first succeeds in {a} episodes, only "
          f"the second in {b}, p = {fmt(p)}; cells above: {up or 'none'}, below: {dn or 'none'}")


@click.command()
@click.option("--root", default="results/hard_scenario/v2")
@click.option("--seeds", default="42,43,44", help="for testing the script on the seeds that exist")
@click.option("--suffix", default="_s{seed}", help="what a run folder ends in after its tag (for testing: '')")
@click.option("--only", default=None, help="for testing: TASK:LEVEL:DEMOS, e.g. T1:V2b_place:700")
def main(root, seeds, suffix, only):
    root, seeds = Path(root), [int(s) for s in seeds.split(",")]
    tasks = dict(TASKS)
    if only:
        t, level, n = only.split(":")
        tasks = {t: (level, int(n), TASKS[t][2])}

    for task, (level, n, sets) in tasks.items():
        def get(named, test_sets=sets, lvl=level):
            out = {k: [load(root / f"{lvl}_{v.format(n=n)}{suffix.format(seed=s)}", test_sets) for s in seeds]
                   for k, v in named.items()}
            return ({k: v for k, v in out.items() if all(x is not None for x in v)},
                    [k for k, v in out.items() if any(x is None for x in v)])

        named = {**PLAIN, **(LIT if task == "T1" else {}), "`cue`": CUE[task], "one network": CUE[task] + "_early",
                 **(OTHER if task == "T1" else {})}
        runs, missing = get(named)
        print(f"# {task}: {level}, {n} demos, seeds {seeds}, test sets {', '.join(sets)}\n")
        if missing:
            print(f"Without a finished evaluation of every seed, left out: {missing}\n")
        if not runs:
            continue
        keys = set(next(iter(runs.values()))[0])
        assert all(set(r) == keys for rs in runs.values() for r in rs), "the runs do not hold the same episodes"
        cs = cells_of(next(iter(runs.values())))
        mark = {c: " (open)" for c in OPEN_CELLS[task]}

        print("## 1. Macro average per seed\n")
        print("| run | " + " | ".join(f"seed {s}" for s in seeds) + " | mean | standard deviation |")
        print("|---|" + "---|" * (len(seeds) + 2))
        for k, rs in runs.items():
            m = [rate([r]) for r in rs]
            sd = f"{np.std(m, ddof=1):.3f}" if len(m) > 1 else "-"
            print(f"| {k} | " + " | ".join(f"{x:.3f}" for x in m) + f" | {np.mean(m):.3f} | {sd} |")

        base = [k for k in runs if k in PLAIN or k in LIT]
        print("\n## 2. Success per cell over the seeds, and the best baseline of every cell\n")
        print("| cell | " + " | ".join(runs) + " | best baseline | which |\n|---|" + "---|" * (len(runs) + 2))
        best = {}
        for c in cs:
            r = {k: rate(runs[k], (c,)) for k in runs}
            if base:
                best[c] = max(base, key=lambda k: r[k])          # max keeps the first of equal ones
            b = f"{r[best[c]]:.3f} | {best[c]}" if base else "- | -"
            print(f"| {c}{mark.get(c, '')} | " + " | ".join(f"{r[k]:.3f}" for k in runs) + f" | {b} |")
        mb = f"{np.mean([rate(runs[best[c]], (c,)) for c in cs]):.3f}" if base else "-"
        print("| macro average | " + " | ".join(f"{rate(rs):.3f}" for rs in runs.values()) + f" | {mb} | |")
        low = [c for c in cs if max(rate(rs, (c,)) for rs in runs.values()) < 0.30]
        print(f"\nCells in which no run is at 0.30: {low or 'none'}. Open cells of the task: {list(OPEN_CELLS[task]) or 'none'}.")

        if "`cue`" in runs and base:
            M = runs["`cue`"]
            print("\n## 3. `cue` against the best baseline of every cell\n")
            print("| cell | `cue` | best baseline | only `cue` succeeds / only the baseline | p | |\n|---|---|---|---|---|---|")
            below, above = [], []
            for c in cs:
                a, b, p = pooled(M, runs[best[c]], (c,))
                v = "" if p >= P else ("above" if a > b else "below")
                (above if v == "above" else below if v == "below" else []).append(c)
                print(f"| {c}{mark.get(c, '')} | {rate(M, (c,)):.3f} | {rate(runs[best[c]], (c,)):.3f} ({best[c]}) | {a} / {b} | {fmt(p)} | {v} |")
            top = max(base, key=lambda k: rate(runs[k]))
            holds = not below and rate(M) > rate(runs[top])
            print(f"\nCells below their best baseline: {below or 'none'}; above: {above or 'none'}. Macro average {rate(M):.3f} "
                  f"against the highest baseline {rate(runs[top]):.3f} ({top}). \"Better in nearly every case\" "
                  f"{'holds' if holds else 'does not hold'} for `cue` on {task}.")

        print("\n## 4. `cue` against plain rgbdt and against the one network\n")
        for a, b in (("`cue`", "plain rgbdt"), ("`cue`", "one network"), ("one network", "plain rgbdt"),
                     ("second policy with the cue", "second policy (ACT head)")):
            if a in runs and b in runs:
                against(a, runs[a], b, runs[b], cs)

        if task != "T3":
            print("\n## 5. Generalisation sets\n")
            for what, gsets in GENERAL.items():
                g, gmiss = get(named, gsets)
                print(f"**{what}** ({', '.join(gsets)})" + (f"; not evaluated on every seed, left out: {gmiss}" if gmiss else ""))
                if not g:
                    print()
                    continue
                print("\n| run | " + " | ".join(f"seed {s}" for s in seeds) + " | mean |\n|---|" + "---|" * (len(seeds) + 1))
                for k, rs in g.items():
                    m = [rate([r]) for r in rs]
                    print(f"| {k} | " + " | ".join(f"{x:.3f}" for x in m) + f" | {np.mean(m):.3f} |")
                print()
                gc = cells_of(next(iter(g.values())))
                for a, b in (("`cue`", "plain rgbdt"), ("`cue`", "one network")):
                    if a in g and b in g:
                        against(a, g[a], b, g[b], gc)
                print()

        print("\n## 6. The layout reference (the glass in no image)\n")
        ref = load(root / f"{level}_noglass_rgbdt_{n}demos_imagenet{suffix.format(seed=42)}", NOGLASS[task])
        if ref is None:
            print("No finished evaluation.\n")
        else:
            print(f"Plain rgbdt trained and tested on {level}_noglass, seed 42, {len(ref)} episodes: success "
                  f"{np.mean([ok(e) for e in ref.values()]):.3f}. Next to it, the cells of the task in which one sensor "
                  "alone shows the glass faintly (7, 8, 9) are those in which a run's success can come from the layout.\n")


if __name__ == "__main__":
    main()
