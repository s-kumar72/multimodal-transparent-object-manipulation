"""
Read-out of the baselines from the literature with three seeds (results/hard_scenario/lit_base/README.md, "Seeds 43
and 44"), from the episodes.json files of the final evaluations (no new runs):

  PYTHONPATH=$PWD/policy python policy/readout_lit_seeds.py > results/hard_scenario/lit_base/readout_seeds.txt

On each level (L0_vis; L0_vis_ell, the evaluations in contact_check): 1. the macro average of every run per seed;
2. every baseline from the literature against the plain rgbdt baseline; 3. rules 4 and 5 of Stage 6
(results/hard_scenario/grid_bench/README.md) with the baselines from the literature among the baselines: per cell the
baseline with the highest success over its seeds, and the method against it. 4. On the 15 new combinations (L0_vis):
every baseline from the literature against hcg_cue.
Runs are paired by seed on the same episodes and pooled; two-sided sign test on the episodes only one of the two
solves. A run without a finished evaluation of every seed on a level is left out there and named.
"""
from pathlib import Path

import click
import numpy as np

from readout_contacts_seeds import BASELINES, CELLS, fmt, folder, macro, pooled, rate, sd
from readout_cue_gate import SHOWS, load
from readout_lit_base import LIT, NEW_SETS
from readout_lit_base import load as load_sets

PLAIN = {w: ("grid_bench", f"L0_vis_{w}_700demos_imagenet") for w in BASELINES}
LITR = {n: ("lit_base", r) for n, r in LIT.items()}
METHODS = {"hcg_full": ("cue_gate", "L0_vis_rgbdt_700demos_hcg_full"),
           "hcg_cue": ("cue_gate", "L0_vis_rgbdt_700demos_hcg_cue")}
LEVELS = {"L0_vis": None, "L0_vis_ell": "contact_check"}      # the group that holds a level's evaluations (None: the run's own)


@click.command()
@click.option("--root", default="results/hard_scenario")
@click.option("--seeds", default="42,43,44", help="for testing the script on the seeds that exist")
def main(root, seeds):
    root, seeds = Path(root), [int(s) for s in seeds.split(",")]

    for level, group in LEVELS.items():
        def get(runs):
            out = {n: [load(root / (group or g) / folder(r, s)) for s in seeds] for n, (g, r) in runs.items()}
            return ({n: v for n, v in out.items() if all(x is not None for x in v)},
                    [n for n, v in out.items() if any(x is None for x in v)])
        plain, miss = get(PLAIN)
        assert not miss, f"the plain baselines are needed: {miss}"
        lit, no_lit = get(LITR)
        meth, no_meth = get(METHODS)
        keys = set(plain["rgbdt"][0])
        assert all(set(r) == keys for rs in [*plain.values(), *lit.values(), *meth.values()] for r in rs)
        print(f"# {level} (seeds {seeds}, {len(keys)} episodes per run, {len(keys) // 11} per cell)\n")
        if no_lit or no_meth:
            print(f"Without a finished evaluation of every seed on this level, left out: {no_lit + no_meth}\n")

        print("## 1. Macro average per seed\n")
        print("| run | " + " | ".join(f"seed {s}" for s in seeds) + " | mean | standard deviation |")
        print("|---|" + "---|" * (len(seeds) + 2))
        for n, rs in [*plain.items(), *lit.items(), *meth.items()]:
            m = [macro([r]) for r in rs]
            print(f"| {n} | " + " | ".join(f"{x:.3f}" for x in m) + f" | {np.mean(m):.3f} | {sd(m)} |")

        print("\n## 2. The baselines from the literature against the plain rgbdt baseline (paired by seed, pooled)\n")
        print("| run | run | plain rgbdt | only the run succeeds / only plain rgbdt | p | cells above / below plain "
              "rgbdt (p < 0.05) |\n|---|---|---|---|---|---|")
        for n, rs in lit.items():
            sa, sb, a, b, p = pooled(rs, plain["rgbdt"])
            per = [(c, *pooled(rs, plain["rgbdt"], (c,))) for c in CELLS]
            up = [c for c, xa, xb, _, _, pp in per if pp < 0.05 and xa > xb]
            down = [c for c, xa, xb, _, _, pp in per if pp < 0.05 and xa < xb]
            print(f"| {n} | {sa:.3f} | {sb:.3f} | {a} / {b} | {fmt(p)} | {up or 'none'} / {down or 'none'} |")

        print("\n## 3. Rules 4 and 5 of Stage 6 with the baselines from the literature among the baselines\n")
        cand = {**plain, **lit}                                  # ties for the best baseline go to the plain ones
        which = [max(cand, key=lambda w: rate(cand[w], (c,))) for c in CELLS]
        best = [rate(cand[which[c]], (c,)) for c in CELLS]
        print("| cell | sensors that show the glass | " + " | ".join(cand) + " | best baseline | which |")
        print("|---|---|" + "---|" * (len(cand) + 2))
        for c in CELLS:
            print(f"| {c} | {SHOWS[c]} | " + " | ".join(f"{rate(rs, (c,)):.3f}" for rs in cand.values())
                  + f" | {best[c]:.3f} | {which[c]} |")
        print("| macro average | | " + " | ".join(f"{macro(rs):.3f}" for rs in cand.values())
              + f" | {np.mean(best):.3f} | |")
        taken = sorted({w for w in which if w in lit})
        print(f"\nCells whose best baseline is one from the literature: "
              f"{[c for c in CELLS if which[c] in lit] or 'none'}{' (' + ', '.join(taken) + ')' if taken else ''}.\n")
        for mname, M in meth.items():
            print(f"**`{mname}`**\n")
            print("| cell | the method | best baseline of the cell | only the method succeeds / only the baseline | p | |")
            print("|---|---|---|---|---|---|")
            below, above = [], []
            for c in CELLS:
                sa, sb, a, b, p = pooled(M, cand[which[c]], (c,))
                if p < 0.05:
                    (below if sa < sb else above).append(c)
                print(f"| {c} | {sa:.3f} | {sb:.3f} ({which[c]}) | {a} / {b} | {fmt(p)} | "
                      f"{('above' if sa > sb else 'below') if p < 0.05 else ''} |")
            over = all(macro(M) > macro(rs) for rs in cand.values())
            top = max(cand, key=lambda w: macro(cand[w]))
            print(f"\nCells below their best baseline with p < 0.05: {below or 'none'}; above: {above or 'none'}; "
                  f"neither: {[c for c in CELLS if c not in below + above] or 'none'}. Macro average "
                  f"{macro(M):.3f} against the highest baseline {macro(cand[top]):.3f} ({top}): "
                  f"{'above' if over else 'not above'} every baseline. \"Better in nearly every case\" "
                  f"{'holds' if not below and over else 'does not hold'} for `{mname}` on {level}.\n")

    print("# The 15 new combinations (L0_vis)\n")
    new = lambda g, r: [load_sets(root / g / folder(r, s), NEW_SETS) for s in seeds]
    cue = new(*METHODS["hcg_cue"])
    print("| run | seeds with both runs | run | hcg_cue | only the run succeeds / only hcg_cue | p |\n|---|---|---|---|---|---|")
    for n, (g, r) in LITR.items():
        runs = new(g, r)
        both = [(s, x, y) for s, x, y in zip(seeds, runs, cue) if x is not None and y is not None]
        if not both:
            print(f"| {n} | none | | | | |")
            continue
        sa, sb, a, b, p = pooled([x for _, x, _ in both], [y for _, _, y in both], range(64))
        print(f"| {n} | {[s for s, _, _ in both]} | {sa:.3f} | {sb:.3f} | {a} / {b} | {fmt(p)} |")


if __name__ == "__main__":
    main()
