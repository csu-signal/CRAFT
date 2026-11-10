"""
Oracle-on vs oracle-off comparison: the same conditions evaluated with and without the candidate moves
in the builder prompt (two eval_full_game.py result directories that differ only in --no_oracle).

    python compare_oracle.py --on eval_results_v2 --off eval_results_v2_no_oracle \
        --labels base_7b echo grpo --out analysis_out_oracle_on_vs_off
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.load import load_results, pass_units, structure_units
from analysis.plots import OURS_EDGE, _curves, _value_text, _y_label
from analysis.registry import METRICS_BY_KEY
from analysis.stats import holm, paired_diff, sem
from analysis.style import INK, save, use_paper_style
from analysis.tables import _header, _p

ALPHA = 0.05
DEFAULT_METRICS = ["final_progress", "completed", "progress_gain", "correct_move_rate",
                   "invalid_move_rate", "parse_failure_rate", "clarify_rate"]
PANEL_METRICS = ["final_progress", "correct_move_rate", "clarify_rate", "invalid_move_rate"]
SETTINGS = ("on", "off")
OFF_STYLE = dict(alpha=0.45, hatch="///")


def _delta(dfs, label, metric):
    """Per-structure off - on for one condition."""
    return structure_units(dfs["off"], label, metric).sub(structure_units(dfs["on"], label, metric)).dropna()


def compute(dfs, conditions, metrics, reference):
    """One row per (condition, metric): level in each setting, paired off - on, and off - on vs the reference."""
    rows = []
    for c in conditions:
        start = len(rows)
        for m in metrics:
            row = {"condition": c.display, "label": c.label, "metric": m.key}
            for s in SETTINGS:
                row[f"mean_{s}"] = structure_units(dfs[s], c.label, m.key).mean()
                row[f"sem_{s}"] = sem(pass_units(dfs[s], c.label, m.key)).sem
            d = paired_diff(structure_units(dfs["off"], c.label, m.key), structure_units(dfs["on"], c.label, m.key))
            row.update(delta=d.diff, delta_sem=d.sem, delta_p=d.p, n_structures=d.n)
            if reference is not None and c.label != reference.label:
                dd = paired_diff(_delta(dfs, c.label, m.key), _delta(dfs, reference.label, m.key))
                row.update(did=dd.diff, did_sem=dd.sem, did_p=dd.p)
            rows.append(row)
        # Holm across this condition's metrics, separately for each family of tests
        block = rows[start:]
        for key in ("delta", "did"):
            ps = [r.get(f"{key}_p", np.nan) for r in block]
            for r, a in zip(block, holm(ps)):
                r[f"{key}_p_holm"] = a
    return pd.DataFrame(rows)


def _fmt(m, v, signed=False):
    if pd.isna(v):
        return "—"
    v = v * 100 if m.percent else v
    return f"{v:+.{m.decimals}f}" if signed else f"{v:.{m.decimals}f}"


def _sig(r, key):
    return pd.notna(r.get(f"{key}_p_holm")) and r[f"{key}_p_holm"] < ALPHA


def markdown(res, conditions, metrics, reference, sem_desc):
    unit = lambda m: "pp" if m.percent else ""
    lines = ["## Oracle on vs off\n",
             "| Condition | Metric | Oracle on | Oracle off | Δ off − on |",
             "|---|---|---:|---:|---:|"]
    for c in conditions:
        for m in metrics:
            r = res[(res["label"] == c.label) & (res["metric"] == m.key)].iloc[0]
            delta = f"{_fmt(m, r['delta'], True)} ± {_fmt(m, r['delta_sem'])}" + (" †" if _sig(r, "delta") else "")
            lines.append(f"| {c.display} | {_header(m)} | {_fmt(m, r['mean_on'])} ± {_fmt(m, r['sem_on'])} | "
                         f"{_fmt(m, r['mean_off'])} ± {_fmt(m, r['sem_off'])} | {delta} |")
    lines.append(f"\nLevels: mean over structures ± {sem_desc}. Δ: paired over structures (± SEM over "
                 f"structures); †: sign-flip test, Holm-corrected across each condition's metrics, p < {ALPHA}.")
    lines += ["\n## Paired tests: Δ = off − on\n",
              "| Condition | Metric | Δ ± SEM | p | p (Holm) | n structures |", "|---|---|---:|---:|---:|---:|"]
    for c in conditions:
        for m in metrics:
            r = res[(res["label"] == c.label) & (res["metric"] == m.key)].iloc[0]
            lines.append(f"| {c.display} | {m.display} ({unit(m)}) | {_fmt(m, r['delta'], True)} ± "
                         f"{_fmt(m, r['delta_sem'])} | {_p(r['delta_p'])} | {_p(r['delta_p_holm'])} | "
                         f"{int(r['n_structures'])} |")
    if reference is not None and "did" in res:
        lines += [f"\n## Difference-in-differences vs {reference.display}: "
                  f"Δ[condition] − Δ[{reference.display}]\n",
                  "Negative on a ↑ metric: hiding the oracle hurts this condition more than the reference.\n",
                  "| Condition | Metric | ΔΔ ± SEM | p | p (Holm) |", "|---|---|---:|---:|---:|"]
        for c in conditions:
            if c.label == reference.label:
                continue
            for m in metrics:
                r = res[(res["label"] == c.label) & (res["metric"] == m.key)].iloc[0]
                lines.append(f"| {c.display} | {m.display} ({unit(m)}) | {_fmt(m, r['did'], True)} ± "
                             f"{_fmt(m, r['did_sem'])} | {_p(r['did_p'])} | {_p(r['did_p_holm'])} |")
    return "\n".join(lines)


def latex(res, conditions, metrics, sem_desc):
    cols = "l" + "r" * len(metrics)
    head = " & ".join(_header(m).replace("%", r"\%").replace("↑", r"$\uparrow$").replace("↓", r"$\downarrow$")
                      for m in metrics)
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             f"Condition & {head} \\\\", r"\midrule"]
    for c in conditions:
        for s, name in (("on", "oracle on"), ("off", "oracle off")):
            cells = []
            for m in metrics:
                r = res[(res["label"] == c.label) & (res["metric"] == m.key)].iloc[0]
                cell = f"${_fmt(m, r[f'mean_{s}'])}_{{\\pm {_fmt(m, r[f'sem_{s}'])}}}$"
                if s == "off" and _sig(r, "delta"):
                    cell += r"$^\dagger$"
                cells.append(cell)
            lines.append(f"{c.display} ({name}) & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule" if c is not conditions[-1] else r"\bottomrule")
    lines += [r"\end{tabular}",
              rf"\caption{{Oracle on vs off. Cells: mean $\pm$ {sem_desc}. $\dagger$: oracle-off differs from "
              rf"oracle-on for the same condition (paired sign-flip test over structures, Holm-corrected, "
              rf"$p<{ALPHA}$).}}",
              r"\label{tab:oracle_on_vs_off}", r"\end{table*}"]
    return "\n".join(lines)


def paired_bar(res, conditions, metric, ax=None):
    """Oracle-on (solid) and oracle-off (hatched) bars per condition, mean +- SEM."""
    rows = res[res["metric"] == metric.key].set_index("label")
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(3.6, 0.9 * len(conditions) + 1.4), 3.4))
    w = 0.38
    for i, c in enumerate(conditions):
        r = rows.loc[c.label]
        for j, s in enumerate(SETTINGS):
            x, scale = i + (j - 0.5) * w, 100 if metric.percent else 1
            v, e = r[f"mean_{s}"], r[f"sem_{s}"]
            if pd.isna(v):
                continue
            style = (OURS_EDGE if c.is_ours else {}) | (OFF_STYLE if s == "off" else {})
            ax.bar(x, v * scale, w * 0.95, color=c.color, **style)
            ax.errorbar(x, v * scale, yerr=0 if pd.isna(e) else e * scale, color=INK, capsize=2.5, lw=0.9)
            ax.text(x, v * scale + (0 if pd.isna(e) else e * scale), _value_text(metric, v),
                    ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(conditions)), [_tick(c) for c in conditions], fontsize=9)
    ax.set_ylabel(_y_label(metric))
    if metric.bounds and metric.percent:
        ax.set_ylim(0, 112)
        ax.set_yticks(range(0, 101, 20))
    return fig


def _tick(c):
    return c.model.removeprefix("Claude ").replace(" ", "\n", 1) if c.method.key == "api" and c.model else c.method.display


def _setting_legend(fig, conditions):
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#bbbbbb", label="oracle on"),
               Patch(facecolor="#bbbbbb", label="oracle off", **OFF_STYLE)]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))


def progress_curves(dfs, conditions):
    """Cumulative progress per turn, oracle on (left) vs off (right), +-SEM over passes."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, s in zip(axes, SETTINGS):
        for c in conditions:
            curves, turns = _curves(dfs[s], c.label, gain=True, group="rep")
            if curves is None:
                continue
            est = [sem(curves[:, t]) for t in range(curves.shape[1])]
            mean, lo, hi = (np.array([getattr(e, k) for e in est]) * 100 for k in ("mean", "lo", "hi"))
            ax.fill_between(turns, lo, hi, color=c.color, alpha=0.12, lw=0)
            ax.plot(turns, mean, color=c.color, lw=2.2 if c.is_ours else 1.5, marker=c.method.marker,
                    ms=3.5, markevery=max(1, len(turns) // 10), label=c.display)
        ax.set_title(f"Oracle {s}")
        ax.set_xlabel("Turn")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(axis="both")
    axes[0].set_ylabel("Cumulative progress (%) ↑")
    axes[1].legend(loc="upper right", fontsize=8)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--on", default="eval_results_v2", help="results dir with the oracle shown")
    parser.add_argument("--off", default="eval_results_v2_no_oracle", help="results dir with --no_oracle")
    parser.add_argument("--labels", nargs="*", default=["base_7b", "echo", "grpo"])
    parser.add_argument("--reference", default="base_7b", help="condition for the difference-in-differences")
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    parser.add_argument("--out", default="analysis_out_oracle_on_vs_off")
    parser.add_argument("--formats", nargs="*", default=["png"])
    args = parser.parse_args()

    use_paper_style()
    dfs = {}
    (dfs["on"], conditions), (dfs["off"], off_conditions) = (load_results(args.on, args.labels),
                                                            load_results(args.off, args.labels))
    missing = {c.label for c in conditions} ^ {c.label for c in off_conditions}
    if missing:
        raise SystemExit(f"[oracle] labels present in only one setting: {sorted(missing)}")
    for s in SETTINGS:
        oracle_flags = {str(v) for v in dfs[s].get("oracle_in_prompt", pd.Series(dtype=object)).dropna()}
        print(f"[oracle] {s}: {len(dfs[s])} episodes" + (f", oracle_in_prompt={oracle_flags}" if oracle_flags else ""))
    reference = next((c for c in conditions if c.label == args.reference), None)
    metrics = [METRICS_BY_KEY[k] for k in args.metrics]
    out = Path(args.out)

    res = compute(dfs, conditions, metrics, reference)
    k = int(pass_units(dfs["on"], conditions[0].label, metrics[0].key).size)
    sem_desc = f"SEM over {k} independent passes over the {int(res['n_structures'].max())}-structure eval set"

    for m in metrics:
        fig = paired_bar(res, conditions, m)
        _setting_legend(fig, conditions)
        for p in save(fig, out, f"bar_{m.key}", args.formats):
            print(f"  wrote {p}")
        plt.close(fig)
    panel = [METRICS_BY_KEY[k] for k in PANEL_METRICS if k in args.metrics]
    fig, axes = plt.subplots(1, len(panel), figsize=(3.0 * len(panel), 3.2))
    for ax, m in zip(np.atleast_1d(axes), panel):
        paired_bar(res, conditions, m, ax=ax)
    _setting_legend(fig, conditions)
    fig.tight_layout()
    for p in save(fig, out, "panel_on_vs_off", args.formats):
        print(f"  wrote {p}")
    plt.close(fig)
    fig = progress_curves(dfs, conditions)
    for p in save(fig, out, "progress_curve_on_vs_off", args.formats):
        print(f"  wrote {p}")
    plt.close(fig)

    res.to_csv(out / "results.csv", index=False)
    md = markdown(res, conditions, metrics, reference, sem_desc)
    (out / "results.md").write_text(md + "\n")
    (out / "results.tex").write_text(latex(res, conditions, metrics, sem_desc) + "\n")
    print(f"  wrote {out / 'results.csv'}, {out / 'results.md'}, {out / 'results.tex'}\n")
    print(md)


if __name__ == "__main__":
    main()
