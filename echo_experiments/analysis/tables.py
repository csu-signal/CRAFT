"""
Results tables in three renderings from one computed frame: CSV (raw
numbers, for anything downstream), Markdown (notes / PR descriptions), and
LaTeX booktabs (paste into the paper; needs \\usepackage{booktabs}).

Main table cell: mean with 95% CI half-width as a subscript; best per
column in bold (only for metrics with a direction); a dagger where the
paired difference vs the reference survives Holm correction at alpha=0.05.
"""
import numpy as np
import pandas as pd

from .load import structure_units
from .stats import ALPHA, ci, holm, paired_diff


def compute(df, conditions, metrics, reference=None):
    """Long frame, one row per (condition, metric): estimate + paired test vs reference."""
    rows = []
    for cond in conditions:
        tests = []
        for metric in metrics:
            units = structure_units(df, cond.label, metric.key)
            est = ci(units.to_numpy(), binary=metric.binary)
            row = {"condition": cond.display, "label": cond.label, "metric": metric.key,
                   "mean": est.mean, "ci_lo": est.lo, "ci_hi": est.hi, "n_structures": est.n,
                   "n_episodes": int((df["label"] == cond.label).sum())}
            if reference is not None and cond.label != reference.label:
                pdiff = paired_diff(units, structure_units(df, reference.label, metric.key))
                row.update(diff=pdiff.diff, diff_lo=pdiff.lo, diff_hi=pdiff.hi, p=pdiff.p, n_paired=pdiff.n)
                tests.append(len(rows))
            rows.append(row)
        # Holm across this condition's metrics (one family per comparison)
        if tests:
            adj = holm([rows[i]["p"] for i in tests])
            for i, a in zip(tests, adj):
                rows[i]["p_holm"] = a
    return pd.DataFrame(rows)


def _scale(metric, v):
    return v * 100 if metric.percent else v


def _cell_value(metric, v):
    if pd.isna(v):
        return "—"  # not applicable, e.g. oracle match when no candidates were shown
    return f"{_scale(metric, v):.{metric.decimals}f}"


def _ci_text(metric, r, latex=False):
    """Half-width when the CI is roughly symmetric, else the explicit
    interval -- Wilson CIs near 0%/100% are lopsided, and "0.0 +- 8.1"
    would imply negative rates."""
    lo, hi, mean = _scale(metric, r["ci_lo"]), _scale(metric, r["ci_hi"]), _scale(metric, r["mean"])
    if pd.isna(lo):
        return ""
    d = metric.decimals
    hw = (hi - lo) / 2
    if hw > 0 and abs((mean - lo) - (hi - mean)) > 0.25 * hw:
        return rf"_{{[{lo:.{d}f}, {hi:.{d}f}]}}" if latex else f" [{lo:.{d}f}, {hi:.{d}f}]"
    return rf"_{{\pm {hw:.{d}f}}}" if latex else f" ± {hw:.{d}f}"


def _p(p):
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def _best_labels(res, metrics):
    best = {}
    for m in metrics:
        if m.higher_is_better is None:
            continue
        sub = res[res["metric"] == m.key].dropna(subset=["mean"])
        if len(sub) < 2:
            continue
        target = sub["mean"].max() if m.higher_is_better else sub["mean"].min()
        best[m.key] = set(sub.loc[np.isclose(sub["mean"], target), "label"])
    return best


def _significant(row):
    return "p_holm" in row and pd.notna(row.get("p_holm")) and row["p_holm"] < ALPHA


def _header(metric):
    arrow = {True: " ↑", False: " ↓", None: ""}[metric.higher_is_better]
    unit = " (%)" if metric.percent else ""
    return f"{metric.display}{unit}{arrow}"


def main_markdown(res, conditions, metrics):
    best = _best_labels(res, metrics)
    lines = ["| Condition | " + " | ".join(_header(m) for m in metrics) + " |",
             "|---|" + "---:|" * len(metrics)]
    for cond in conditions:
        cells = []
        for m in metrics:
            r = res[(res["label"] == cond.label) & (res["metric"] == m.key)].iloc[0]
            v = _cell_value(m, r["mean"])
            cell = (f"**{v}**" if cond.label in best.get(m.key, ()) else v) + _ci_text(m, r)
            if _significant(r):
                cell += " †"
            cells.append(cell)
        lines.append(f"| {cond.display} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main_latex(res, conditions, metrics, reference=None, caption=None):
    best = _best_labels(res, metrics)
    tex_header = {True: r"$\uparrow$", False: r"$\downarrow$", None: ""}
    cols = " & ".join(
        f"{m.display}{' (\\%)' if m.percent else ''} {tex_header[m.higher_is_better]}".strip() for m in metrics)
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{l" + "r" * len(metrics) + "}", r"\toprule",
             f"Condition & {cols} \\\\", r"\midrule"]
    for i, cond in enumerate(conditions):
        if i and cond.method.kind != conditions[i - 1].method.kind:
            lines.append(r"\midrule")
        cells = []
        for m in metrics:
            r = res[(res["label"] == cond.label) & (res["metric"] == m.key)].iloc[0]
            v = _cell_value(m, r["mean"])
            v = rf"\mathbf{{{v}}}" if cond.label in best.get(m.key, ()) else v
            sub = _ci_text(m, r, latex=True)
            sup = r"^{\dagger}" if _significant(r) else ""
            cells.append(f"${v}{sub}{sup}$")
        lines.append(f"{cond.display} & " + " & ".join(cells) + r" \\")
    n = int(res["n_structures"].max())
    ref_note = (f" $^\\dagger$: paired sign-flip test vs {reference.display}, Holm-corrected, $p<{ALPHA}$."
                if reference is not None else "")
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{{caption or 'Full-game evaluation on the held-out benchmark.'} "
              rf"Mean with 95\% CI (half-width, or [low, high] where asymmetric) over {n} structures (bootstrap; Wilson for completion). "
              rf"Best per column in bold.{ref_note}}}",
              r"\label{tab:main_results}", r"\end{table}"]
    return "\n".join(lines)


def comparison_markdown(res, metrics, reference):
    """Paired differences vs the reference, with raw and Holm-adjusted p."""
    by_key = {m.key: m for m in metrics}
    sub = res.dropna(subset=["diff"]) if "diff" in res else res.iloc[0:0]
    if sub.empty:
        return ""
    lines = [f"Paired difference vs **{reference.display}** (same structures; Δ = condition − reference)", "",
             "| Condition | Metric | Δ | 95% CI | p | p (Holm) | n |", "|---|---|---:|---:|---:|---:|---:|"]
    for _, r in sub.iterrows():
        m = by_key[r["metric"]]
        f = lambda v: f"{_scale(m, v):+.{m.decimals}f}"
        lines.append(f"| {r['condition']} | {m.display}{' (pp)' if m.percent else ''} | {f(r['diff'])} | "
                     f"[{f(r['diff_lo'])}, {f(r['diff_hi'])}] | {_p(r['p'])} | {_p(r['p_holm'])} | {int(r['n_paired'])} |")
    return "\n".join(lines)
