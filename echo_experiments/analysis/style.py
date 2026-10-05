"""
Figure style for the paper: matches its existing matplotlib figures (see
use_paper_style) and writes 300-dpi PNGs.
"""
from pathlib import Path

import matplotlib as mpl

# Categorical slots in fixed order (validated CVD-safe as adjacent pairs; the
# first three also all-pairs). Never cycle or generate a ninth -- fold extra
# conditions into small multiples instead.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
NEUTRAL = "#8a8984"

INK = "#222222"
INK_2 = "#52514e"
INK_3 = "#8a8984"
GRID = "#e6e5e1"
AXIS = "#bdbcb6"
SURFACE = "#ffffff"

COL_W = 3.25    # single column
FULL_W = 6.75   # full text width


def use_paper_style():
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "center",
        "axes.titlepad": 8,
        "axes.labelsize": 11,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.facecolor": SURFACE,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "grid.linestyle": ":",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.color": INK,
        "ytick.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.handlelength": 1.6,
        "legend.handleheight": 0.9,
        "legend.columnspacing": 1.4,
        "figure.facecolor": SURFACE,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def save(fig, out_dir, name, formats=("png",)):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(path)
        paths.append(path)
    return paths
