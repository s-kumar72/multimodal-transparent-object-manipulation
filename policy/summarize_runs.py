"""
Summaries of evaluated runs (folders with <test set>/eval_log.json, <test set>/episodes.json, train_logs.json.txt),
as written by scripts/run_experiment.sh and scripts/aws/sweep.sh. Plain Python, no extra packages.

  python policy/summarize_runs.py curve <train_logs.json.txt> <curve.json>   # rollout scores per evaluated epoch
  python policy/summarize_runs.py table <run dir> [<run dir> ...]            # markdown tables for each run
  python policy/summarize_runs.py compare <run dir A> <run dir B> [--test_set a_in_distribution]
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
try:  # category names, in the order sim/env.py stores them as ids
    from env import BACKGROUNDS, LIGHTING_NAMES
except Exception:
    BACKGROUNDS = LIGHTING_NAMES = None

TEST_SETS = ["a_in_distribution", "g_heldout_all"]
OUTCOMES = ["success", "missed_grasp", "other", "dropped", "wrong_height", "wrong_object", "timeout"]

def wilson(k, n, z=1.96):
    """95% Wilson interval of a success rate."""
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

def fisher_two_sided(k1, n1, k2, n2):
    """Two-sided Fisher exact test p-value for k1/n1 vs k2/n2 successes."""
    total, n = k1 + k2, n1 + n2
    def p(x):
        return math.comb(n1, x) * math.comb(n2, total - x) / math.comb(n, total)
    p_obs = p(k1)
    return min(1.0, sum(p(x) for x in range(max(0, total - n2), min(total, n1) + 1) if p(x) <= p_obs * (1 + 1e-9)))

def read_logs(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows

def curve(logs_path, out_path):
    """Rollout scores per evaluated epoch (the last one wins if a resumed run repeated an epoch)."""
    rows = {}
    for d in read_logs(logs_path):
        if "test/mean_score" in d:
            rows[d["epoch"]] = {k: v for k, v in d.items() if k.startswith("test/") or
                                k in ("epoch", "global_step", "val_loss", "train_loss")}
    json.dump([rows[e] for e in sorted(rows)], open(out_path, "w"), indent=1)
    print("curve:", ", ".join(f"ep{e} {rows[e]['test/mean_score']:.2f}" for e in sorted(rows)))

def losses(logs_path):
    """Per-epoch validation loss and the last training loss of each epoch."""
    val, train = {}, {}
    for d in read_logs(logs_path):
        if "val_loss" in d:
            val[d["epoch"]] = d["val_loss"]
        if "train_loss" in d:
            train[d["epoch"]] = d["train_loss"]
    return val, train

def episodes(run, test_set):
    path = Path(run) / test_set / "episodes.json"
    return json.load(open(path)) if path.exists() else None

def name(names, i):
    return names[i] if names is not None and 0 <= i < len(names) else str(i)

def breakdown(eps, key):
    groups = defaultdict(list)
    for e in eps:
        groups[key(e)].append(e["reward"] > 0)
    return {g: (sum(v), len(v)) for g, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}

CONDITIONS = {
    "object": lambda e: e["obj_name"],
    "background": lambda e: name(BACKGROUNDS, e["background"]),
    "lighting": lambda e: name(LIGHTING_NAMES, e["lighting"]),
    "distractors": lambda e: e.get("n_distractors", 0),
    "look-alike": lambda e: e.get("lookalike", 0),
    "thermal crossover": lambda e: e["is_crossover"],
    "liquid": lambda e: "filled" if e["fill_level"] > 0 else "empty",
    "depth see-through": lambda e: e.get("glass_seethrough", 0),
}

def table(run):
    run = Path(run)
    print(f"\n## {run.name}\n")
    print("| test set | success | 95% CI | " + " | ".join(OUTCOMES[1:]) + " |")
    print("|---|---|---|" + "---|" * (len(OUTCOMES) - 1))
    for ts in TEST_SETS:
        eps = episodes(run, ts)
        if eps is None:
            continue
        k, n = sum(e["reward"] > 0 for e in eps), len(eps)
        lo, hi = wilson(k, n)
        shares = [sum(e["outcome"] == o for e in eps) / n for o in OUTCOMES[1:]]
        print(f"| {ts} | {k / n:.2f} ({k}/{n}) | {lo:.2f}-{hi:.2f} | " + " | ".join(f"{s:.2f}" for s in shares) + " |")
    logs = run / "train_logs.json.txt"
    if logs.exists():
        rows = [d for d in read_logs(logs) if "test/mean_score" in d]
        last = {d["epoch"]: d["test/mean_score"] for d in rows}
        val, train = losses(logs)
        print("\nrollouts on (a) during training: " + ", ".join(f"ep{e} {s:.2f}" for e, s in sorted(last.items())))
        if val:
            best = min(val, key=val.get)
            end = max(val)
            print(f"val loss: lowest {val[best]:.4f} at epoch {best}, last {val[end]:.4f} at epoch {end}; "
                  f"train loss last {train.get(end, float('nan')):.4f}")
    eps = episodes(run, TEST_SETS[0])
    if eps:
        print(f"\nsuccess by condition on {TEST_SETS[0]} (successes/episodes):")
        for cond, key in CONDITIONS.items():
            try:
                parts = breakdown(eps, key)
            except KeyError:
                continue
            print(f"- {cond}: " + ", ".join(f"{g} {k}/{n}" for g, (k, n) in parts.items()))

def compare(a, b, test_set):
    ea, eb = episodes(a, test_set), episodes(b, test_set)
    ka, na = sum(e["reward"] > 0 for e in ea), len(ea)
    kb, nb = sum(e["reward"] > 0 for e in eb), len(eb)
    print(f"{Path(a).name} {ka}/{na} vs {Path(b).name} {kb}/{nb} on {test_set}: "
          f"Fisher exact two-sided p = {fisher_two_sided(ka, na, kb, nb):.3g}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["curve", "table", "compare"])
    p.add_argument("paths", nargs="+")
    p.add_argument("--test_set", default="a_in_distribution")
    args = p.parse_args()
    if args.command == "curve":
        curve(*args.paths)
    elif args.command == "table":
        for r in args.paths:
            table(r)
    else:
        compare(*args.paths, args.test_set)
