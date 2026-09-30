"""
Paper figures, styled to match the paper's existing ones. Each figure
function takes (df, conditions, metric, reference, res) -- `res` is
tables.compute()'s frame, so figures and tables show the same numbers --
and returns a Figure, or None when the data can't support it.

  * PER_METRIC figures are drawn once per metric -> <name>_<metric>.png
  * SINGLE figures are drawn once -> <name>.png (metric is None)

Every figure identifies conditions through a legend (colour = model, ECHO
outlined as ours), and legend.png holds the same legend on its own for
assembling multi-panel figures.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from .stats import ci
from .style import INK

BAR_W = 0.78
OURS_EDGE = dict(edgecolor="black", linewidth=1.3)


def _arrow(metric):
    return {True: " ↑", False: " ↓", None: ""}[metric.higher_is_better]


def _value_text(metric, v):
    text = f"{v * 100:.{metric.decimals}f}" if metric.percent else f"{v:.{metric.decimals}f}"
    return text.replace("-", "−")


def _y_label(metric):
    unit = " (%)" if metric.percent else ""
    return f"{metric.display}{unit}{_arrow(metric)}"


def _legend_handles(conditions, kind="bar"):
    if kind == "bar":
        return [Patch(facecolor=c.color, label=c.display, **(OURS_EDGE if c.is_ours else {}))
                for c in conditions]
    return [Line2D([], [], color=c.color, lw=2.2 if c.is_ours else 1.5,
                   marker=c.method.marker, ms=4.5, label=c.display) for c in conditions]


def _legend_below(ax, conditions, kind="bar", ncol=2):
    ax.legend(handles=_legend_handles(conditions, kind), loc="upper center",
              bbox_to_anchor=(0.5, -0.08 if kind == "bar" else -0.2), ncol=ncol, handletextpad=0.5)


def metric_bar(df, conditions, metric, reference, res):
    """One colour-coded bar per condition: mean with 95% CI error bars and
    the value above each bar, like the paper's 'Task resolution' panel."""
    rows = res[res["metric"] == metric.key].set_index("label")
    conds = [c for c in conditions if c.label in rows.index and not np.isnan(rows.loc[c.label, "mean"])]
    if not conds:
        return None
    # tallest bar first (stable, so ties keep the registry order); the legend follows the bars
    conds.sort(key=lambda c: -rows.loc[c.label, "mean"])
    n = len(conds)
    fig, ax = plt.subplots(figsize=(max(3.6, 0.42 * n + 1.4), 3.4))
    xs = np.arange(n)
    scale = 100 if metric.percent else 1
    means = np.array([rows.loc[c.label, "mean"] for c in conds]) * scale
    los = np.array([rows.loc[c.label, "ci_lo"] for c in conds]) * scale
    his = np.array([rows.loc[c.label, "ci_hi"] for c in conds]) * scale

    for x, c, m in zip(xs, conds, means):
        ax.bar(x, m, width=BAR_W, color=c.color, zorder=2, **(OURS_EDGE if c.is_ours else {}))
    has_ci = ~np.isnan(los)
    # clip: with identical values the bootstrap bounds can sit a float epsilon past the mean
    yerr = np.clip([means[has_ci] - los[has_ci], his[has_ci] - means[has_ci]], 0, None)
    ax.errorbar(xs[has_ci], means[has_ci], yerr=yerr, fmt="none", ecolor="black",
                elinewidth=1.0, capsize=3, capthick=1.0, zorder=3)

    tops = np.where(has_ci, np.fmax(his, means), means)
    bottoms = np.where(has_ci, np.fmin(los, means), means)
    data_lo, data_hi = min(0.0, np.nanmin(bottoms)), max(0.0, np.nanmax(tops))
    span = (data_hi - data_lo) or 1.0
    for x, m, top, bottom in zip(xs, means, tops, bottoms):
        if m < 0:
            ax.text(x, min(bottom, 0) - 0.02 * span, _value_text(metric, m / scale),
                    ha="center", va="top", fontsize=9.5)
        else:
            ax.text(x, max(top, 0) + 0.02 * span, _value_text(metric, m / scale),
                    ha="center", va="bottom", fontsize=9.5)

    # like the paper, the axis follows the data (not the metric's full 0-100%) plus label headroom
    ax.set_ylim(data_lo - (0.14 * span if data_lo < 0 else 0), data_hi + 0.14 * span)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    if metric.bounds:
        hi_b = metric.bounds[1] * scale
        ax.set_yticks([t for t in ax.get_yticks() if data_lo - 1e-9 <= t <= hi_b + 1e-9])
    if data_lo < 0:
        ax.axhline(0, color=INK, lw=0.8, zorder=1)
    ax.set_xticks([])
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylabel(_y_label(metric))
    ax.set_title(metric.display)
    _legend_below(ax, conds, "bar")
    return fig


def _curves(df, label, gain):
    """(n_structures, T+1) per-structure mean progress curve for one
    condition. Episodes that finished early hold their last value."""
    sub = df[df["label"] == label]
    if "progress_curve" not in sub or sub["progress_curve"].isna().all():
        return None, None
    T = max(len(c) for c in sub["progress_curve"]) - 1
    per_structure = []
    for _, g in sub.groupby("structure_idx"):
        eps = []
        for curve in g["progress_curve"]:
            c = np.array([np.nan if v is None else v for v in curve], dtype=float)
            c = np.r_[c, np.full(T + 1 - len(c), c[-1])]
            eps.append(c - c[0] if gain else c)
        per_structure.append(np.nanmean(eps, axis=0))
    return np.array(per_structure), np.arange(T + 1)


def progress_curve(df, conditions, metric, reference, res, gain=True):
    """Cumulative progress vs turn, one line per condition with a 95% CI
    band over structures -- like the paper's per-turn line panels."""
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    drawn, last_turn = [], 0
    for c in conditions:
        curves, turns = _curves(df, c.label, gain)
        if curves is None:
            continue
        est = [ci(curves[:, t]) for t in range(curves.shape[1])]
        mean, lo, hi = (np.array([getattr(e, k) for e in est]) * 100 for k in ("mean", "lo", "hi"))
        ax.fill_between(turns, lo, hi, color=c.color, alpha=0.12, lw=0, zorder=1)
        ax.plot(turns, mean, color=c.color, lw=2.2 if c.is_ours else 1.5, marker=c.method.marker,
                ms=4 if c.is_ours else 3.2, markevery=max(1, len(turns) // 10), zorder=3 if c.is_ours else 2)
        drawn.append(c)
        last_turn = max(last_turn, turns[-1])
    if not drawn:
        return None
    ax.set_xlabel("Turn")
    ax.set_ylabel(("Cumulative progress (%)" if gain else "Structure progress (%)") + " ↑")
    ax.set_title("Cumulative progress")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.set_xlim(0, last_turn)
    if gain:
        ax.set_ylim(bottom=0)
    ax.grid(axis="both")
    _legend_below(ax, drawn, "line")
    return fig


def legend_only(conditions, kind="bar", ncol=4):
    fig = plt.figure(figsize=(7, 0.3 + 0.25 * np.ceil(len(conditions) / ncol)))
    fig.legend(handles=_legend_handles(conditions, kind), loc="center", ncol=ncol, handletextpad=0.5)
    return fig


PER_METRIC = {"bar": metric_bar}
SINGLE = {"progress_curve": progress_curve}
FIGURES = {**PER_METRIC, **SINGLE}
