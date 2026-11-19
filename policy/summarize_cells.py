"""
Success per visibility cell for a level with cells (sim/config.py cells), from the episodes.json files policy/eval.py
writes.

  python policy/summarize_cells.py RUN_DIR [RUN_DIR ...] [--test_set h_cells --test_set i_cells_heldout] [--json FILE]

RUN_DIR holds <test set>/episodes.json. Several run folders (e.g. the seeds of one policy) are pooled into one row.
Printed per run group: success per cell (k/n) over the given test sets pooled, the macro average (mean of the seven
cell rates, the benchmark's headline number: every cell counts the same whatever its share of the episodes) with a
bootstrap 95% interval (episodes resampled within each cell), and the plain share of successes.
"""
import json
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


@click.command()
@click.argument("run_dirs", nargs=-1, required=True)
@click.option("--test_set", "test_sets", multiple=True, default=("h_cells", "i_cells_heldout"), show_default=True)
@click.option("--json", "json_out", default=None)
def main(run_dirs, test_sets, json_out):
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
