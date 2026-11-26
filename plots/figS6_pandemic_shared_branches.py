"""Fig. S6: the branches of our tree of a simulated pandemic that are also in the true tree, in CMAPLE's tree
and in a VecME run from another random start, by the substitutions on the branch.

Inputs:  data/tables/pandemic_branches.tsv                              (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    per bin of substitutions on the branch: the share of our branches in each other tree,
         as three nested lines (widest underneath), so that each stays visible where they coincide;
         the number of branches of each bin under its tick
Outputs: img/fig3/figS6_pandemic_shared_branches.{pdf,svg,png}

    pixi run -e paper python -m plots.figS6_pandemic_shared_branches
"""

# pylint: disable=invalid-name  # figS<N>: the paper's numbering of its supplementary figures

import matplotlib.pyplot as plt

from plots._config import SPEC, STYLE
from plots.utils import LABEL_GREY, data_label, parser, read_tsv, save

FIGSIZE = (88 / 25.4, 74 / 25.4)
BINS = {  # (from, to) substitutions on the branch, as in the data -> tick label
    ("0.0", "0.5"): "0",
    ("0.5", "1.5"): "1",
    ("1.5", "2.5"): "2",
    ("2.5", "3.5"): "3",
    ("3.5", "5.5"): "4–5",
    ("5.5", ""): "≥6",
}
# column, legend entry (after "Also found in"), line keywords; drawn in this order, the widest first
SERIES: list[tuple[str, str, dict]] = [
    (
        "in_repeat_pct",
        "a VecME run from another random start",
        {
            "color": SPEC["P2V"]["color"],
            "alpha": 0.45,
            "lw": 5.0,
            "marker": "o",
            "ms": 7,
            "zorder": 3,
        },
    ),
    (
        "in_cmaple_pct",
        "the CMAPLE tree",
        {
            "color": SPEC["CMAPLE"]["color"],
            "alpha": 0.8,
            "lw": 2.4,
            "marker": "h",
            "ms": 5.5,
            "zorder": 4,
        },
    ),
    (
        "in_true_pct",
        "the true tree",
        {"color": "#000000", "alpha": 0.85, "lw": 0.8, "marker": "s", "ms": 2.6, "zorder": 5},
    ),
]
COUNT_Y = -0.115  # the branch counts under the tick labels, in axes heights


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    rows = read_tsv(args.tables_dir, "pandemic_branches")
    if [(r["substitutions_from"], r["substitutions_to"]) for r in rows] != list(BINS):
        raise SystemExit("the substitution bins of the data are not the ones this figure labels")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.20, top=0.96)
    x = range(len(rows))
    for column, label, style in SERIES:
        ax.plot(x, [float(r[column]) for r in rows], label=label, **style)
    ax.set_xticks(x, list(BINS.values()))
    below = ax.get_xaxis_transform()  # x in bins, y in axes heights
    for xi, r in zip(x, rows):
        ax.text(
            xi,
            COUNT_Y,
            f"{int(r['branches']):,}",
            transform=below,
            ha="center",
            va="top",
            fontsize=7,
            color=LABEL_GREY,
        )
    ax.text(
        -0.02,
        COUNT_Y,
        "Branches",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color=LABEL_GREY,
    )
    ax.set_xlabel("Substitutions on the branch", labelpad=12)
    ax.set_ylabel("Branches of the VecME tree (%)")
    ax.set_ylim(0, 118)  # headroom for the data label
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower right",
        title="Also found in",
        title_fontsize=8,
        fontsize=8,
        handlelength=2.2,
        alignment="left",
    )
    data_label(ax, "simulated pandemic, 100,000 sequences")
    save(fig, args.img_dir, "figS6_pandemic_shared_branches", bbox_inches=None)


if __name__ == "__main__":
    main()
