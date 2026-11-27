"""Fig. 3a: time against normalised RF distance of VecME, MAPLE and CMAPLE on the simulated pandemic
alignments.

Inputs:  data/tables/maple.tsv                                          (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    per program and mean branch length: one line through the three sizes (time, RF distance)
           |
           v
         the three branch lengths as three bands, named on the right in substitutions per site
Outputs: img/fig3/fig3a_pandemic_speed_accuracy.{pdf,svg,png}

    pixi run -e paper python -m plots.fig3a_pandemic_speed_accuracy
"""

import matplotlib.pyplot as plt
import numpy as np

from plots._config import FIG3_AXES_Y, FIG3_HEIGHT, PANDEMIC, STYLE
from plots.utils import (
    LABEL_GREY,
    OURS,
    PANDEMIC_BRANCH_LENGTHS,
    data_label_above,
    method_style,
    pandemic_rows,
    parser,
    save,
)

FIGSIZE = (70 / 25.4, FIG3_HEIGHT)  # in the row of Figs 3a-c (70 + 50 + 60 mm) across the page
LEFT, RIGHT = 0.166, 0.71  # 11.6 mm for the y label; 20.3 mm on the right for the branch lengths
BRANCH_X = 1.04  # the branch lengths and their header, in axes widths: 1.5 mm right of the axes
BRANCH_FONTSIZE = 8  # header and branch lengths, quieter than the axis labels
HEADROOM = (
    1.5  # y axis up to this multiple of the largest error: room for the legend above the data
)
LEGEND_FONTSIZE = 7


def power_of_ten(value: float) -> str:
    """Writes a value as a multiple of a power of ten, in the text font: 1.5e-4 -> "1.5 x 10^-4"."""
    exponent = int(np.floor(np.log10(value)))
    return rf"$\mathdefault{{{value / 10**exponent:g}\times10^{{{exponent}}}}}$"


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    rows = pandemic_rows(args.tables_dir)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bottom, top = FIG3_AXES_Y
    fig.subplots_adjust(left=LEFT, right=RIGHT, bottom=bottom, top=top)
    bands = []  # (mean branch length, mean RF distance of every run at that branch length)
    for length in PANDEMIC_BRANCH_LENGTHS:
        group = [r for r in rows if float(r["mean_branch_length"]) == length]
        bands.append((length, float(np.mean([float(r["rf_pct"]) for r in group]))))
        for method, key in PANDEMIC:
            runs = sorted((r for r in group if r["method"] == method), key=lambda r: int(r["n"]))
            ax.plot(
                [float(r["time_s"]) for r in runs],
                [float(r["rf_pct"]) for r in runs],
                label=method if length == PANDEMIC_BRANCH_LENGTHS[0] else None,
                zorder=5 if key == OURS else 3,
                **method_style(key),
            )
    times = [float(r["time_s"]) for r in rows]
    ax.set_xscale("log")
    ax.set_xlim(10 ** np.floor(np.log10(min(times))), 10 ** np.ceil(np.log10(max(times))))
    ax.set_ylim(0, HEADROOM * max(float(r["rf_pct"]) for r in rows))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalised RF distance (%)")
    data_label_above(ax, "simulations")
    # the legend boxed in the upper left corner, above the top band
    ax.legend(
        loc="upper left",
        fontsize=LEGEND_FONTSIZE,
        handlelength=1.3,
        handletextpad=0.3,
        borderaxespad=0.4,
        borderpad=0.4,
        frameon=True,
        fancybox=False,
        edgecolor="#BBBBBB",
        framealpha=1.0,
    )
    # the bands' branch lengths right of the axes, each at its band's height, under a header
    ax.text(
        BRANCH_X,
        1.0,
        "Branch length\n(subs per site)",
        transform=ax.transAxes,
        va="top",
        color=LABEL_GREY,
        fontsize=BRANCH_FONTSIZE,
    )
    at_band = ax.get_yaxis_transform()  # x in axes widths, y in RF distance
    for length, rf in bands:
        ax.text(
            BRANCH_X,
            rf,
            power_of_ten(length),
            transform=at_band,
            va="center",
            color=LABEL_GREY,
            fontsize=BRANCH_FONTSIZE,
        )
    save(fig, args.img_dir, "fig3a_pandemic_speed_accuracy", bbox_inches=None)


if __name__ == "__main__":
    main()
