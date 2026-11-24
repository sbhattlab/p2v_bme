"""Fig. 2b: RF error against the true tree, by the number of sequences,
on the three simulated families.

Inputs:  data/tables/scaling.tsv, random_tree.tsv                      (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    mean and sample SD of the RF error at each size, one series per dataset
           |
           v
         datasets above 10^5 sequences on the PASTA tree expand a 10^5-leaf alignment: their error
         on the inherited (backbone) splits, open markers, dotted from the 10^5-leaf alignment
Outputs: img/fig2/fig2b_scaling_rf_error.{pdf,svg,png}

    pixi run -e paper python -m plots.fig2b_scaling_rf_error
"""

import matplotlib.pyplot as plt

from plots._config import SCALING_SERIES, STYLE
from plots.utils import parser, save, scaling_axes, scaling_data, scaling_series

FIGSIZE = (52 / 25.4, 62 / 25.4)
INHERITED_ALPHA = 0.35  # the inherited splits are secondary: the lightest series


def backbone_series(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    """Draws the error on the inherited splits of the expanded datasets: (axes, rows) -> None."""
    for dataset, _label, colour, marker, _style, _filled, _alpha in SCALING_SERIES:
        group = sorted((r for r in rows if r["dataset"] == dataset), key=lambda r: int(r["n"]))
        expanded = [i for i, r in enumerate(group) if r.get("large_scale_design")]
        if not expanded:
            continue
        ks = [expanded[0] - 1, *expanded]  # from the expanded alignment's own size
        n = [int(group[k]["n"]) for k in ks]
        inherited = [float(group[k]["backbone_rf_pct_mean"]) for k in ks]
        ax.plot(
            n,
            inherited,
            color=colour,
            linestyle=":",
            linewidth=1.0,
            alpha=INHERITED_ALPHA,
            zorder=2,
        )
        ax.plot(
            n[1:],
            inherited[1:],
            linestyle="none",
            marker=marker,
            markersize=3.4,
            markerfacecolor="white",
            markeredgecolor=colour,
            markeredgewidth=0.9,
            alpha=INHERITED_ALPHA,
            zorder=3,
        )


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    rows = scaling_data(args.tables_dir)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.27, right=0.97, bottom=0.21, top=0.96)
    scaling_series(ax, rows, "rf_pct")
    backbone_series(ax, rows)
    scaling_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        plt.Line2D(
            [],
            [],
            color="#444444",
            linestyle=":",
            linewidth=1.0,
            marker="o",
            markersize=3.4,
            markerfacecolor="white",
            markeredgecolor="#444444",
            markeredgewidth=0.9,
            alpha=INHERITED_ALPHA,
        )
    )
    labels.append("inherited splits only")
    ax.set_ylim(0.004, 100)  # room for the legend below the curves; RF error ends at 100 %
    ax.legend(
        handles,
        labels,
        loc="lower left",
        frameon=False,
        handlelength=1.9,
        handletextpad=0.5,
        labelspacing=0.35,
        borderaxespad=0.15,
        borderpad=0.2,
    )
    ax.set_ylabel("Normalised RF error (%)")
    save(fig, args.img_dir, "fig2b_scaling_rf_error")


if __name__ == "__main__":
    main()
