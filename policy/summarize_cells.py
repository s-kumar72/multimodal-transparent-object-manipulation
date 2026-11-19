"""
Success per visibility cell for a level with cells (sim/config.py cells), from the episodes.json files policy/eval.py
writes.

  python policy/summarize_cells.py RUN_DIR [RUN_DIR ...] [--test_set h_cells --test_set i_cells_heldout] [--json FILE]
  python policy/summarize_cells.py RUN_DIR [RUN_DIR ...] --versus OTHER_RUN_DIR [--versus ...]

RUN_DIR holds <test set>/episodes.json. Several run folders (e.g. the seeds of one policy) are pooled into one row.
Printed per run group: success per cell (k/n) over the given test sets pooled, the macro average (mean of the seven
cell rates, the benchmark's headline number: every cell counts the same whatever its share of the episodes) with a
bootstrap 95% interval (episodes resampled within each cell), and the plain share of successes.

--versus: instead, the paired comparison with another policy on the same episodes (the i-th RUN_DIR against the i-th
--versus folder, same test set and episode seed): per cell and over all episodes, the two success rates, the episodes
only one of the two solves, and the two-sided exact sign test on those.
"""
import json
from math import comb
from pathlib import Path

import click
import numpy as np

CELLS = ["rgb+depth+thermal", "depth+thermal", "rgb+thermal", "rgb+depth", "thermal", "depth", "rgb"]


def load(run_dirs, test_sets):
    eps = []
    for run in run_dirs:
        for ts in test_sets:
            f = Path(run) / ts / "episodes.json"
            assert f.exists(), f
            eps += [dict(e, test_set=ts, run=str(run)) for e in json.loads(f.read_text())]
    assert all("cell" in e for e in eps), "episodes without a cell: not a level with visibility cells"
    return eps


def per_cell(eps):
    """{cell: (successes, episodes)}"""
    out = {}
    for c in sorted({e["cell"] for e in eps}):
        r = [e["outcome"] == "success" for e in eps if e["cell"] == c]
        out[c] = (int(sum(r)), len(r))
    return out


def macro(eps, n_boot=2000, seed=0):
    """Mean of the per-cell success rates and its bootstrap 95% interval (resampling episodes within each cell)."""
    by = {c: np.array([e["outcome"] == "success" for e in eps if e["cell"] == c], float)
          for c in sorted({e["cell"] for e in eps})}
    rng = np.random.default_rng(seed)
    boot = np.mean([[rng.choice(v, len(v)).mean() for v in by.values()] for _ in range(n_boot)], axis=1)
    return float(np.mean([v.mean() for v in by.values()])), tuple(np.quantile(boot, [0.025, 0.975]))


def sign_p(a, b):
    """Two-sided exact sign test for a wins against b."""
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def versus(run_dirs, others, test_sets):
    """Paired comparison on the same episodes; rows (cell or None for all, n, successes A, successes B, only A, only B)."""
    assert len(run_dirs) == len(others), "one --versus folder per RUN_DIR"
    pairs = []
    for a, b in zip(run_dirs, others):
        ea = {(e["test_set"], e["seed"]): e for e in load([a], test_sets)}
        eb = {(e["test_set"], e["seed"]): e for e in load([b], test_sets)}
        assert ea.keys() == eb.keys(), "the two policies were not evaluated on the same episodes"
        pairs += [(ea[k]["cell"], ea[k]["outcome"] == "success", eb[k]["outcome"] == "success") for k in sorted(ea)]
    rows = []
    for c in [*sorted({p[0] for p in pairs}), None]:
        g = [p for p in pairs if c is None or p[0] == c]
        rows.append((c, len(g), sum(p[1] for p in g), sum(p[2] for p in g),
                     sum(p[1] and not p[2] for p in g), sum(p[2] and not p[1] for p in g)))
    return rows


@click.command()
@click.argument("run_dirs", nargs=-1, required=True)
@click.option("--test_set", "test_sets", multiple=True, default=("h_cells", "i_cells_heldout"), show_default=True)
@click.option("--json", "json_out", default=None)
@click.option("--versus", "others", multiple=True, help="paired comparison with these run folders instead")
def main(run_dirs, test_sets, json_out, others):
    if others:
        rows = versus(run_dirs, others, test_sets)
        print(f"A = {', '.join(Path(r).name for r in run_dirs)}; B = {', '.join(Path(r).name for r in others)}; "
              f"test sets {', '.join(test_sets)}")
        print("| cell | sensors that show the glass | A | B | A - B | only A succeeds | only B succeeds | sign test p |")
        print("|---|---|---|---|---|---|---|---|")
        for c, n, sa, sb, oa, ob in rows:
            name = ("| | all episodes (= macro average with equal cells)" if c is None
                    else f"| {c} | {CELLS[c] if c < len(CELLS) else '?'}")
            print(f"{name} | {sa / n:.3f} | {sb / n:.3f} | {(sa - sb) / n:+.3f} | {oa} | {ob} | {sign_p(oa, ob):.2g} |")
        if json_out:
            json.dump(dict(a=list(run_dirs), b=list(others), test_sets=list(test_sets),
                           rows=[dict(cell=c, n=n, a=sa, b=sb, only_a=oa, only_b=ob, p=sign_p(oa, ob))
                                 for c, n, sa, sb, oa, ob in rows]), open(json_out, "w"), indent=1)
        return
    eps = load(run_dirs, test_sets)
    cells = per_cell(eps)
    m, (lo, hi) = macro(eps)
    k, n = sum(v[0] for v in cells.values()), sum(v[1] for v in cells.values())
    print(f"{len(run_dirs)} run folder(s), test sets {', '.join(test_sets)}: {n} episodes")
    print("| cell | sensors that show the glass | success |")
    print("|---|---|---|")
    for c, (s, t) in cells.items():
        print(f"| {c} | {CELLS[c] if c < len(CELLS) else '?'} | {s / t:.3f} ({s}/{t}) |")
    print(f"| | macro average (mean of the cells) | {m:.3f} (95% {lo:.3f}-{hi:.3f}) |")
    print(f"| | all episodes | {k / n:.3f} ({k}/{n}) |")
    if json_out:
        json.dump(dict(runs=list(run_dirs), test_sets=list(test_sets), macro=m, macro_ci=[lo, hi],
                       cells={str(c): dict(success=s, n=t) for c, (s, t) in cells.items()}, success=k, n=n),
                  open(json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
