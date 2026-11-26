"""Fig. 2b: RF error of VecME against FastME's, one point per alignment of the FastME comparison,
each program from its own start tree.

Inputs:  data/tables/fastme_cases.tsv                                   (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    one point per alignment (linear axes), on the dashed identity line if both are as accurate
           |
           v
         the JC and RNASim sets as two clusters, each labelled with its data and sizes
Outputs: img/fig2/fig2b_fastme.{pdf,svg,png}

    pixi run -e paper python -m plots.fig2b_fastme
"""

import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter, MultipleLocator

from plots._config import FIG2_PANEL, SPEC, STYLE
from plots.utils import OURS, parser, read_tsv, save

# The comparison's sets (fastme_cases.tsv) by data: the JC sets are simulated on the PASTA tree
FAMILIES = {
    "JC": "PASTA (JC69)",
    "JC subsets": "PASTA (JC69)",
    "RNASim": "RNASim",
    "RNASim subsets": "RNASim",
}
MARKERS = {"PASTA (JC69)": "o", "RNASim": "s"}  # as the datasets elsewhere in the paper
# Each cluster's label: position (RF error of FastME, of VecME, in %) and alignment, off the points
LABELS = {"PASTA (JC69)": (5.2, 3.3, "left"), "RNASim": (12.5, 24, "right")}
LIMITS = (0, 25)  # RF error (%), both axes, linear: the data span less than a decade
TICK_STEP = 5  # RF error (%)
POINT_ALPHA = 0.3  # the points overlap: hundreds per cluster


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    rows = read_tsv(args.tables_dir, "fastme_cases")
    if unknown := {r["dataset"] for r in rows} - set(FAMILIES):
        raise SystemExit(f"fastme_cases.tsv: sets {sorted(unknown)} not in FAMILIES")
    values = [float(r[k]) for r in rows for k in ("rf_fastme_pct", "rf_vecme_pct")]
    if not values or min(values) < LIMITS[0] or max(values) > LIMITS[1]:
        raise SystemExit(f"fastme_cases.tsv: RF errors outside {LIMITS} %, or none")
    fig, ax = plt.subplots(figsize=FIG2_PANEL)
    # left as Fig. 2d's, below it (its method names take the room); the rest as Fig. 2a
    fig.subplots_adjust(left=0.36, right=0.97, bottom=0.16, top=0.96)
    ax.plot(LIMITS, LIMITS, color="#999999", linewidth=0.8, linestyle="--", zorder=1)
    for family, marker in MARKERS.items():
        group = [r for r in rows if FAMILIES[r["dataset"]] == family]
        if not group:
            raise SystemExit(f"fastme_cases.tsv: no {family} alignments")
        ax.plot(
            [float(r["rf_fastme_pct"]) for r in group],
            [float(r["rf_vecme_pct"]) for r in group],
            linestyle="none",
            marker=marker,
            markersize=3.2,
            markerfacecolor=SPEC[OURS]["color"],  # each point is a VecME tree
            markeredgecolor="none",
            alpha=POINT_ALPHA,
            zorder=3,
        )
        sizes = [int(r["n"]) for r in group]
        x, y, ha = LABELS[family]
        ax.text(x, y, f"{family}\nN = {min(sizes):,}–{max(sizes):,}", ha=ha, va="top")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MultipleLocator(TICK_STEP))
        axis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.set_xlim(*LIMITS)
    ax.set_ylim(*LIMITS)
    ax.set_xlabel("RF error of FastME (%)")
    ax.set_ylabel("RF error of VecME (%)")
    ax.grid(color="#E6E6E6", linewidth=0.5)
    ax.set_axisbelow(True)
    save(fig, args.img_dir, "fig2b_fastme")


if __name__ == "__main__":
    main()
