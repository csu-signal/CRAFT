"""
Results tables in three renderings from one computed frame: CSV (raw
numbers, for anything downstream), Markdown (notes / PR descriptions), and
LaTeX booktabs (paste into the paper; needs \\usepackage{booktabs}).

Main table cell: mean +- SEM (over passes or structures, see compute); best per
column in bold (only for metrics with a direction); a dagger where the
paired difference vs the reference survives Holm correction at alpha=0.05.
"""
import numpy as np
import pandas as pd

from .load import SEM_UNITS, structure_units
from .stats import ALPHA, holm, paired_diff, sem


def compute(df, conditions, metrics, reference=None, sem_over="passes"):
    """Long frame, one row per (condition, metric): estimate + paired test vs reference.
    sem_over: "passes" (SEM across full passes over the eval set) or "structures"; paired tests
    always pair on structures."""
    rows = []
    for cond in conditions:
        tests = []
        for metric in metrics:
            units = structure_units(df, cond.label, metric.key)
            spread = sem(SEM_UNITS[sem_over](df, cond.label, metric.key).to_numpy())
            row = {"condition": cond.display, "label": cond.label, "metric": metric.key,
                   "mean": float(units.mean()) if len(units) else np.nan, "sem": spread.sem,
                   "sem_over": sem_over, "n_sem_units": spread.n, "n_structures": len(units),
                   "n_episodes": int((df["label"] == cond.label).sum())}
            if reference is not None and cond.label != reference.label:
                pdiff = paired_diff(units, structure_units(df, reference.label, metric.key))
                row.update(diff=pdiff.diff, diff_sem=pdiff.sem, p=pdiff.p, n_paired=pdiff.n)
                tests.append(len(rows))
            rows.append(row)
        # Holm across this condition's metrics (one family per comparison)
        if tests:
            adj = holm([rows[i]["p"] for i in tests])
            for i, a in zip(tests, adj):
                rows[i]["p_holm"] = a
    out = pd.DataFrame(rows)
    out.attrs["sem_over"] = sem_over
    return out


def sem_description(res):
    n_struct = int(res["n_structures"].max())
    if res.attrs.get("sem_over") == "passes":
        k = int(res["n_sem_units"].max())
        return f"SEM over {k} independent passes over the {n_struct}-structure eval set"
    return f"SEM over {n_struct} structures (each structure's episodes averaged first)"


def _scale(metric, v):
    return v * 100 if metric.percent else v


def _cell_value(metric, v):
    if pd.isna(v):
        return "—"  # not applicable, e.g. oracle match when no candidates were shown
    return f"{_scale(metric, v):.{metric.decimals}f}"


def _sem_text(metric, r, latex=False):
    if pd.isna(r["sem"]):
        return ""
    s = f"{_scale(metric, r['sem']):.{metric.decimals}f}"
    return rf"_{{\pm {s}}}" if latex else f" ± {s}"


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
            cell = (f"**{v}**" if cond.label in best.get(m.key, ()) else v) + _sem_text(m, r)
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
            sub = _sem_text(m, r, latex=True)
            sup = r"^{\dagger}" if _significant(r) else ""
            cells.append(f"${v}{sub}{sup}$")
        lines.append(f"{cond.display} & " + " & ".join(cells) + r" \\")
    ref_note = (f" $^\\dagger$: paired sign-flip test vs {reference.display}, Holm-corrected, $p<{ALPHA}$."
                if reference is not None else "")
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{{caption or 'Full-game evaluation on the held-out benchmark.'} "
              rf"Mean $\pm$ {sem_description(res)}. "
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
             "| Condition | Metric | Δ ± SEM (over structures) | p | p (Holm) | n structures |", "|---|---|---:|---:|---:|---:|"]
    for _, r in sub.iterrows():
        m = by_key[r["metric"]]
        f = lambda v: f"{_scale(m, v):+.{m.decimals}f}"
        diff_sem = f"{_scale(m, r['diff_sem']):.{m.decimals}f}"
        lines.append(f"| {r['condition']} | {m.display}{' (pp)' if m.percent else ''} | {f(r['diff'])} ± {diff_sem} | "
                     f"{_p(r['p'])} | {_p(r['p_holm'])} | {int(r['n_paired'])} |")
    return "\n".join(lines)


def main_figure_latex(panels, sem_desc, fig_path="", legend_file="legend_panels.png", ncols=2):
    """Figure* with one subfigure per (file, subcaption, label) panel and a shared legend.
    Needs \\usepackage{graphicx} and \\usepackage{subcaption}."""
    ncols = min(ncols, len(panels))
    width = f"{0.98 / ncols:.3f}"
    lines = [r"\begin{figure*}[t]", r"\centering"]
    for i, (fname, subcap, label) in enumerate(panels):
        lines += [rf"\begin{{subfigure}}[t]{{{width}\textwidth}}", r"\centering",
                  rf"\includegraphics[width=\linewidth]{{{fig_path}{fname}}}",
                  rf"\caption{{{subcap}}}", rf"\label{{fig:main_results:{label}}}", r"\end{subfigure}"]
        if i < len(panels) - 1:
            lines.append(r"\hfill" if (i + 1) % ncols else r"\par\medskip")
    lines += [r"\par\medskip",
              rf"\includegraphics[width=0.85\textwidth]{{{fig_path}{legend_file}}}",
              rf"\caption{{Full-game evaluation on the held-out benchmark. Bars: mean $\pm$ {sem_desc}, "
              r"sorted by value; Qwen2.5-7B + ECHO outlined. Curve: cumulative progress gained per turn, "
              r"shaded $\pm$ SEM; episodes that finish early hold their final value.}",
              r"\label{fig:main_results}", r"\end{figure*}"]
    return "\n".join(lines)
