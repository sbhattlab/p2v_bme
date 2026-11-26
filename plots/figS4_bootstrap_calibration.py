"""Fig. S4: bootstrap calibration: the share of branches in the true tree against their exact bootstrap support,
on RNASim, the SARS-CoV-2-like simulation and the random tree.

Inputs:  data/tables/calibration.tsv                                    (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    each dataset's support bins with at least MIN_BRANCHES branches
           |
           v
         mean support of the bin (x) against its share of branches in the true tree (y), with the diagonal
         of perfect calibration
Outputs: img/fig2/figS4_bootstrap_calibration.{pdf,svg,png}

    pixi run -e paper python -m plots.figS4_bootstrap_calibration
"""

import matplotlib.pyplot as plt

from plots._config import SCALING_SERIES, STYLE
from plots.utils import parser, read_tsv, save

FIGSIZE = (58 / 25.4, 74 / 25.4)  # a third of the 180 mm page, with the legend above
MIN_BRANCHES = 50  # bins with fewer branches are left out: their share is noise
# The datasets as Fig. 2 draws them (RNASim and the random tree); the pandemic simulation in the MAPLE purple
STYLES = {  # dataset in the table -> legend name, colour, marker
    "RNASim": ("RNASim", SCALING_SERIES[0][2], "s"),
    "SARS-CoV-2-like simulation": ("SARS-CoV-2-like", "#7570B3", "o"),
    "Random tree": ("Random tree", SCALING_SERIES[2][2], "^"),
}


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    rows = read_tsv(args.tables_dir, "calibration")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.25, right=0.95, bottom=0.16, top=0.72)
    ax.plot([0, 100], [0, 100], linestyle="--", color="#999999", linewidth=0.8, zorder=1)
    for dataset, (name, colour, marker) in STYLES.items():
        points = [r for r in rows if r["dataset"] == dataset and int(r["branches"]) >= MIN_BRANCHES]
        if not points:
            raise SystemExit(
                f"calibration.tsv: no bin of {dataset} with {MIN_BRANCHES} branches or more"
            )
        ax.plot(
            [float(r["mean_support_pct"]) for r in points],
            [float(r["correct_pct"]) for r in points],
            color=colour,
            marker=marker,
            markersize=3.4,
            linewidth=1.1,
            label=f"{name}, {int(points[0]['n']):,}",
            zorder=3,
        )
    ax.set_xlim(0, 101)
    ax.set_ylim(0, 101)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Bootstrap support (%)")
    ax.set_ylabel("Branches in the true tree (%)")
    ax.grid(color="#E5E5E5", linewidth=0.5)
    ax.set_axisbelow(True)
    # above the axes: inside, the names would cross the curves at this width
    ax.legend(loc="lower left", bbox_to_anchor=(-0.28, 1.02), handlelength=1.6, borderaxespad=0.0)
    save(fig, args.img_dir, "figS4_bootstrap_calibration")


if __name__ == "__main__":
    main()
