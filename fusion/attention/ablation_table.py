"""
Results table for the fusion ablations on the policy benchmark (L0_bench). (Optional to organize ablations.)

Reads the per-episode results that policy/eval.py writes (<run>/<test set>/episodes.json), pools the test sets and
the training seeds of each row, and splits the episodes into the same groups as results/hard_scenario/fusion_bench:
    all          every episode
    normal       the depth sensor sees the glass        (glass_seethrough = 0)
    see-through  depth passes through the glass         (glass_seethrough = 1)
    crossover    glass within ±1 °C of the room          (is_crossover = 1; overlaps the two groups above)
Each cell is the success rate in %, with the change against the baseline row in percentage points and stars for a
two-sided Fisher exact test (* p < 0.05, ** p < 0.01, *** p < 0.001). The baseline is restricted to the seeds the row
was trained with, so a one-seed ablation is compared with the baseline's same seed.

  python -m fusion.attention.ablation_table --spec fusion/attention/ablations.yaml --out results/fusion_ablations/table

Writes <out>.md (markdown, for GitHub), <out>.tex (LaTeX booktabs), <out>.csv (every number, with
95% Wilson intervals) and <out>.png (success per group with intervals). Rows whose runs are not there yet are listed
as "not run" in the markdown and LaTeX tables (use --skip_missing to leave them out).
The spec format is described in fusion/attention/ablations.yaml.
"""
import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import yaml

GROUPS = ("all", "normal", "see-through", "crossover")


# ---------------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fisher(k1, n1, k2, n2):
    """Two-sided Fisher exact test on the 2x2 table [[k1, n1-k1], [k2, n2-k2]]."""
    if min(n1, n2) == 0:
        return float("nan")
    K, N = k1 + k2, n1 + n2
    lf = lambda x: math.lgamma(x + 1)
    logp = lambda a: lf(K) + lf(N - K) + lf(n1) + lf(n2) - lf(N) - lf(a) - lf(K - a) - lf(n1 - a) - lf(n2 - K + a)
    lo, hi = max(0, K - n2), min(K, n1)
    p_obs = logp(k1)
    total = sum(math.exp(logp(a)) for a in range(lo, hi + 1) if logp(a) <= p_obs + 1e-9)
    return min(1.0, total)


def auc(pos, neg):
    """P(score of a positive > score of a negative), ties count half (Mann-Whitney)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def stars(p):
    return "" if not p == p or p >= 0.05 else "***" if p < 0.001 else "**" if p < 0.01 else "*"


# ---------------------------------------------------------------------- loading
def seed_of(run_name, default=42):
    m = re.search(r"_seed(\d+)$", run_name)
    return int(m.group(1)) if m else default


def load_run(path, test_sets):
    """Episodes of one run (all test sets), or None if any test set is missing."""
    eps = []
    for ts in test_sets:
        f = Path(path) / ts / "episodes.json"
        if not f.exists():
            return None
        for e in json.load(open(f)):
            eps.append(e)
    return eps


def group_mask(eps, group):
    st = np.array([int(e.get("glass_seethrough", 0) or 0) for e in eps])
    xo = np.array([int(e.get("is_crossover", 0) or 0) for e in eps])
    return {"all": np.ones(len(eps), bool), "normal": st == 0, "see-through": st == 1, "crossover": xo == 1}[group]


def success(eps):
    return np.array([float(e.get("reward", 0.0)) > 0.5 for e in eps])


def miss_distance_cm(eps):
    """Median distance (cm) between the gripper and the glass at the last gripper close for failed episodes only."""
    d = [e["closes"][-1]["d_target"] * 100 for e in eps
         if float(e.get("reward", 0)) <= 0.5 and e.get("closes") and e["closes"][-1].get("d_target") is not None]
    return float(np.median(d)) if d else float("nan")


def gate_aucs(eps, mods=("rgb", "depth", "thermal")):
    """AUC of the mean depth weight (normal > see-through) and thermal weight (no crossover > crossover), from the
    per-step weights "w" that policy/eval.py --save_w stores; NaN when the run has none."""
    have = [e for e in eps if e.get("w")]
    if not have:
        return float("nan"), float("nan")
    W = np.array([np.mean(np.asarray(e["w"], float), axis=0) for e in have])
    if W.shape[1] != len(mods):
        return float("nan"), float("nan")
    st = group_mask(have, "see-through")
    xo = group_mask(have, "crossover")
    return auc(W[~st, 1], W[st, 1]), auc(W[~xo, 2], W[xo, 2])


# ---------------------------------------------------------------------- table
def run_path(entry, roots):
    """'<root key>:<run>' -> roots[key]/<run>; a plain name uses roots["default"]."""
    key, _, name = entry.rpartition(":")
    return Path(roots[key or "default"]) / name


def evaluate_row(row, roots, test_sets):
    runs = {}
    for r in row["runs"]:
        eps = load_run(run_path(r, roots), test_sets)
        if eps is not None:
            runs[seed_of(r.rpartition(":")[2])] = eps
    return runs


def summarize(runs_by_seed, baseline_by_seed):
    """Per group: k, n, rate, CI, delta and p against the baseline on the shared seeds."""
    seeds = sorted(runs_by_seed)
    eps = [e for s in seeds for e in runs_by_seed[s]]
    base_seeds = [s for s in seeds if s in baseline_by_seed] or sorted(baseline_by_seed)
    base = [e for s in base_seeds for e in baseline_by_seed.get(s, [])]
    out = {"seeds": seeds, "n_episodes": len(eps)}
    ok = success(eps)
    bok = success(base) if base else np.array([])
    for g in GROUPS:
        m = group_mask(eps, g)
        k, n = int(ok[m].sum()), int(m.sum())
        rate = k / n if n else float("nan")
        lo, hi = wilson(k, n)
        if len(base):
            bm = group_mask(base, g)
            bk, bn = int(bok[bm].sum()), int(bm.sum())
            delta = rate - bk / bn if bn else float("nan")
            p = fisher(k, n, bk, bn)
        else:
            delta, p = float("nan"), float("nan")
        out[g] = dict(k=k, n=n, rate=rate, lo=lo, hi=hi, delta=delta, p=p)
    per_seed = {s: float(success(runs_by_seed[s]).mean()) for s in seeds}
    out["per_seed"] = per_seed
    if len(seeds) >= 2:
        kk = {s: int(success(runs_by_seed[s]).sum()) for s in seeds}
        nn = {s: len(runs_by_seed[s]) for s in seeds}
        hi_s, lo_s = max(seeds, key=lambda s: kk[s] / nn[s]), min(seeds, key=lambda s: kk[s] / nn[s])
        out["seed_p"] = fisher(kk[hi_s], nn[hi_s], kk[lo_s], nn[lo_s])
    else:
        out["seed_p"] = float("nan")
    out["miss_cm"] = miss_distance_cm(eps)
    out["auc_depth"], out["auc_thermal"] = gate_aucs(eps)
    return out


def cell(g, baseline_row, pct_digits=1):
    if g["n"] == 0:
        return "–"
    s = f"{100 * g['rate']:.{pct_digits}f}"
    if not baseline_row and g["delta"] == g["delta"]:
        s += f" ({100 * g['delta']:+.1f}){stars(g['p'])}"
    return s


def fmt(x, d=2):
    return "–" if x is None or x != x else f"{x:.{d}f}"


def build(spec, root_override=None, skip_missing=False):
    roots = dict(spec.get("roots", {}))
    roots.setdefault("default", spec.get("results_root", "."))
    if root_override:
        roots["default"] = root_override
    test_sets = spec.get("test_sets", ["a_in_distribution", "g_heldout_all"])
    rows = spec["rows"]
    by_name = {r["name"]: r for r in rows}
    base_name = spec["baseline"]
    assert base_name in by_name, f"baseline {base_name} is not a row"
    baseline_runs = evaluate_row(by_name[base_name], roots, test_sets)
    assert baseline_runs, f"no results for the baseline {base_name} under {roots}"
    results = []
    for r in rows:
        runs = evaluate_row(r, roots, test_sets)
        if not runs:
            if not skip_missing:
                results.append((r, None))
            continue
        results.append((r, summarize(runs, baseline_runs if r["name"] != base_name else {})))
    return results, base_name, test_sets


def write_markdown(results, base_name, test_sets, path):
    L = [f"Success rate (%) on L0_bench, test sets {' + '.join(test_sets)}, pooled over the seeds listed. "
         f"In brackets: change against **{base_name}** on the same seeds (percentage points); "
         "two-sided Fisher exact test, * p < 0.05, ** p < 0.01, *** p < 0.001 (no correction for multiple comparisons). "
         "Miss = median distance between gripper and glass at the last close, failed episodes. "
         "Gate AUC: depth weight higher on normal than on see-through episodes / thermal weight higher without than with "
         "crossover (0.5 = no separation; needs `policy/eval.py --save_w`).", ""]
    head = "| method | seeds | episodes | all | normal | see-through | crossover | per seed | miss (cm) | gate AUC depth / thermal |"
    L += [head, "|---|---|---|---|---|---|---|---|---|---|"]
    section = None
    for r, s in results:
        if r.get("section") != section:
            section = r.get("section")
            if section:
                L.append(f"| **{section}** | | | | | | | | | |")
        name = r["name"] + (f" — {r['note']}" if r.get("note") else "")
        if s is None:
            L.append(f"| {name} | | not run | | | | | | | |")
            continue
        is_base = r["name"] == base_name
        cells = [cell(s[g], is_base) for g in GROUPS]
        seeds = ", ".join(str(x) for x in s["seeds"])
        per = " / ".join(f"{100 * v:.1f}" for v in s["per_seed"].values())
        if s["seed_p"] == s["seed_p"]:
            per += f" (p {s['seed_p']:.2g})"
        L.append(f"| {'**' + name + '**' if is_base else name} | {seeds} | {s['n_episodes']} | " + " | ".join(cells)
                 + f" | {per} | {fmt(s['miss_cm'], 1)} | {fmt(s['auc_depth'])} / {fmt(s['auc_thermal'])} |")
    n = next((s for _, s in results if s), None)
    if n:
        L += ["", "Episodes per group (baseline row): " + ", ".join(
            f"{g} {s[g]['n']}" for r, s in results if r['name'] == base_name for g in GROUPS)]
    Path(path).write_text("\n".join(L) + "\n")


def write_latex(results, base_name, test_sets, path):
    tex = lambda x: x.replace("_", r"\_").replace("%", r"\%").replace("—", "--").replace("±", r"$\pm$")
    def c(g, is_base):
        if g["n"] == 0:
            return "--"
        s = f"{100 * g['rate']:.1f}"
        if not is_base and g["delta"] == g["delta"]:
            st = stars(g["p"])
            s += f" \\scriptsize({100 * g['delta']:+.1f}){'$^{' + st + '}$' if st else ''}"
        return s
    L = [r"% generated by fusion/attention/ablation_table.py; needs \usepackage{booktabs}",
         r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{lccccc}", r"\toprule",
         r"Method & Seeds & All & Normal & See-through & Crossover \\", r"\midrule"]
    section = None
    for r, s in results:
        if r.get("section") != section:
            if section is not None:
                L.append(r"\midrule")
            section = r.get("section")
            if section:
                L.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{tex(section)}}}}} \\")
        if s is None:
            L.append(rf"{tex(r['name'])} & \multicolumn{{5}}{{c}}{{not run}} \\")
            continue
        is_base = r["name"] == base_name
        name = rf"\textbf{{{tex(r['name'])}}}" if is_base else tex(r["name"])
        L.append(f"{name} & {len(s['seeds'])} & " + " & ".join(c(s[g], is_base) for g in GROUPS) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{Success rate (\%) on L0\_bench ({tex(' + '.join(test_sets))}), pooled over seeds. In brackets: "
          rf"change against {tex(base_name)} on the same seeds; two-sided Fisher exact test, $^{{*}}p<0.05$, "
          r"$^{**}p<0.01$, $^{***}p<0.001$. Normal: depth sees the glass; see-through: depth reads the table behind "
          r"the glass; crossover: glass within $\pm$1\,$^\circ$C of the room.}",
          r"\label{tab:fusion_ablations}", r"\end{table}"]
    Path(path).write_text("\n".join(L) + "\n")


def write_csv(results, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "method", "seeds", "episodes", "group", "successes", "n", "rate", "ci_low", "ci_high",
                    "delta_vs_baseline", "p_vs_baseline", "per_seed", "seed_p", "miss_cm", "auc_depth", "auc_thermal"])
        for r, s in results:
            if s is None:
                continue
            for g in GROUPS:
                x = s[g]
                w.writerow([r.get("section", ""), r["name"], " ".join(map(str, s["seeds"])), s["n_episodes"], g,
                            x["k"], x["n"], round(x["rate"], 4), round(x["lo"], 4), round(x["hi"], 4),
                            round(x["delta"], 4) if x["delta"] == x["delta"] else "", f"{x['p']:.3g}" if x["p"] == x["p"] else "",
                            " ".join(f"{s_}:{v:.3f}" for s_, v in s["per_seed"].items()),
                            f"{s['seed_p']:.3g}" if s["seed_p"] == s["seed_p"] else "",
                            fmt(s["miss_cm"], 2), fmt(s["auc_depth"], 3), fmt(s["auc_thermal"], 3)])


def write_plot(results, base_name, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    done = [(r, s) for r, s in results if s]
    sections = list(dict.fromkeys(r.get("section", "") for r, _ in done))
    palette = ["#8a8984", "#2a78d6", "#eb6834", "#1baf7a", "#a35bd6", "#d6a23a", "#4b5d73"]
    color = {sec: palette[i % len(palette)] for i, sec in enumerate(sections)}
    names = [r["name"] for r, _ in done][::-1]
    fig, axes = plt.subplots(1, 4, figsize=(15, 0.42 * len(done) + 1.6), sharey=True)
    base = next(s for r, s in done if r["name"] == base_name)
    for ax, g in zip(axes, GROUPS):
        for i, (r, s) in enumerate(done[::-1]):
            x = s[g]
            if x["n"] == 0:
                continue
            ax.barh(i, 100 * x["rate"], height=0.62, color=color[r.get("section", "")], zorder=2)
            ax.errorbar(100 * x["rate"], i, xerr=[[100 * (x["rate"] - x["lo"])], [100 * (x["hi"] - x["rate"])]],
                        fmt="none", ecolor="#2a2a28", lw=1, capsize=2, zorder=3)
            if r["name"] != base_name and stars(x["p"]):
                ax.text(100 * x["hi"] + 1, i, stars(x["p"]), va="center", fontsize=8, color="#2a2a28")
        ax.axvline(100 * base[g]["rate"], color="#2a2a28", lw=1, ls="--", zorder=1)
        ax.set_title(f"{g} (n = {base[g]['n']} baseline episodes)", fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_xlabel("success (%)")
        ax.grid(axis="x", color="#e5e4df", zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color[sec]) for sec in sections]
    fig.legend(handles, sections, loc="lower center", ncol=min(len(sections), 4), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.text(0.99, 0.995, f"dashed: {base_name}; bars: 95% Wilson intervals; stars: Fisher test vs {base_name}",
             ha="right", va="top", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", default="fusion/attention/ablations.yaml")
    p.add_argument("--root", default=None, help="results folder for run names without a root key (overrides the spec)")
    p.add_argument("--out", default="results/fusion_ablations/table")
    p.add_argument("--skip_missing", action="store_true")
    p.add_argument("--no_plot", action="store_true")
    a = p.parse_args()
    spec = yaml.safe_load(open(a.spec))
    results, base_name, test_sets = build(spec, a.root, a.skip_missing)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(results, base_name, test_sets, out.with_suffix(".md"))
    write_latex(results, base_name, test_sets, out.with_suffix(".tex"))
    write_csv(results, out.with_suffix(".csv"))
    if not a.no_plot:
        write_plot(results, base_name, out.with_suffix(".png"))
    print(out.with_suffix(".md").read_text())
    print("wrote", ", ".join(str(out.with_suffix(x)) for x in (".md", ".tex", ".csv") + (() if a.no_plot else (".png",))))


if __name__ == "__main__":
    main()