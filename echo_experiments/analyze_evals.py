"""
Analyze full-game eval runs: paper-ready figures + results tables from
eval_results/*.json (written by eval_full_game.py / baseline_sanity_check.py).

Usage:
    python analyze_evals.py                                   # everything, vs the zero-shot base model
    python analyze_evals.py --reference base --metrics final_progress completed
    python analyze_evals.py --results eval_results_no_oracle        # the ablation, analyzed separately
    python analyze_evals.py --labels base_7b echo rloo grpo

Outputs (in --out):
    bar_<metric>.png             one figure per metric: colour-coded mean ± SEM per model, legend below
    progress_curve.png           cumulative progress vs turn, one line per model, ± SEM bands
    legend_{bar,line}.png        the legend alone, for assembling multi-panel figures
    results.csv                  every estimate, CI and paired test
    results.md / results.tex     formatted tables

Adding a baseline or metric: see analysis/registry.py. Adding a figure: see analysis/plots.py.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from analysis.load import load_results
from analysis.plots import FIGURES, PER_METRIC, legend_only
from analysis.registry import METRICS, METRICS_BY_KEY
from analysis.style import save, use_paper_style
from analysis import tables


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="eval_results", help="directory of per-run JSON results")
    parser.add_argument("--out", default="analysis_out")
    parser.add_argument("--labels", nargs="*", default=None, help="only these eval labels (default: all)")
    parser.add_argument("--reference", default="base_7b",
                        help="label every condition is paired against (vertical line, dagger, comparison table)")
    parser.add_argument("--metrics", nargs="*", default=[m.key for m in METRICS if not m.diagnostic],
                        help=f"one figure per metric; any of {[m.key for m in METRICS]}")
    parser.add_argument("--table_metrics", nargs="*", default=None,
                        help="table columns (default: all non-diagnostic metrics; pass e.g. director_failure_rate to add one back)")
    parser.add_argument("--figures", nargs="*", default=list(FIGURES), choices=list(FIGURES))
    parser.add_argument("--formats", nargs="*", default=["png"])
    args = parser.parse_args()

    use_paper_style()
    df, conditions = load_results(args.results, args.labels)
    reference = next((c for c in conditions if c.label == args.reference), None)
    if reference is None:
        print(f"[analysis] reference {args.reference!r} not found -- skipping paired tests")
    fig_metrics = [METRICS_BY_KEY[k] for k in args.metrics]
    table_metrics = ([METRICS_BY_KEY[k] for k in args.table_metrics] if args.table_metrics
                     else [m for m in METRICS if not m.diagnostic])
    out = Path(args.out)

    print(f"[analysis] {len(df)} episodes, {len(conditions)} conditions: "
          + ", ".join(f"{c.label} (n={int((df['label'] == c.label).sum())})" for c in conditions))

    all_metrics = list({m.key: m for m in fig_metrics + table_metrics}.values())
    res = tables.compute(df, conditions, all_metrics, reference)
    jobs = [(name, m) for name in args.figures for m in (fig_metrics if name in PER_METRIC else [None])]
    for name, metric in jobs:
        fname = f"{name}_{metric.key}" if metric else name
        fig = FIGURES[name](df, conditions, metric, reference, res)
        if fig is None:
            print(f"[analysis] skipped {fname} (no data -- e.g. runs predating per-turn progress logging)")
            continue
        for p in save(fig, out, fname, args.formats):
            print(f"  wrote {p}")
        plt.close(fig)
    for kind in ("bar", "line"):
        fig = legend_only(conditions, kind)
        save(fig, out, f"legend_{kind}", args.formats)
        plt.close(fig)

    res = res[res["metric"].isin([m.key for m in table_metrics])]
    res.to_csv(out / "results.csv", index=False)
    md = tables.main_markdown(res, conditions, table_metrics)
    n_struct = int(res["n_structures"].max())
    md += (f"\n\nCells: mean ± SEM over {n_struct} structures (each structure's episodes averaged first). "
           f"**Bold**: best in the column (metrics with a direction only).")
    if reference is not None:
        md += (f" †: differs from {reference.display} (paired sign-flip test on the same structures, "
               f"Holm-corrected across this row's metrics, p < 0.05).")
    if reference is not None:
        md += "\n\n" + tables.comparison_markdown(res, table_metrics, reference)
    (out / "results.md").write_text(md + "\n")
    (out / "results.tex").write_text(tables.main_latex(res, conditions, table_metrics, reference) + "\n")
    print(f"  wrote {out / 'results.csv'}, {out / 'results.md'}, {out / 'results.tex'}\n")
    print(md)


if __name__ == "__main__":
    main()
