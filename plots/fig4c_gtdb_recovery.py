"""Fig. 4c: how well the SILVA tree recovers the Genome Taxonomy Database's taxa, by rank: the mean F1 of the GTDB
taxa of the labelled sequences, scored within the full tree and in trees of those sequences alone (VecME,
FastTree's distance stage and FastTree with likelihood).

Inputs:  data/tables/silva_gtdb.tsv                                     (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    one line per tree over the ranks, phylum to genus (the number of taxa under each rank)
Outputs: img/fig4/fig4c_gtdb_recovery.{pdf,svg,png}

    pixi run -e paper python -m plots.fig4c_gtdb_recovery
"""

import matplotlib.pyplot as plt

from plots._config import SPEC, STYLE
from plots.utils import parser, read_tsv, save

FIGSIZE = (2.7, 2.8)  # Fig. 4b's width: the two stack in one column
RANKS = ["phylum", "class", "order", "family", "genus"]
# tree in the table -> legend label, colour, marker, line width (VecME's full tree drawn strongest)
TREES = {
    "full": ("VecME, full tree", SPEC["P2V"]["color"], "o", 2.0),
    "ours": ("VecME, labelled only", "#F4A582", "o", 1.1),
    "ft_me": (
        "FastTree-ME, labelled only",
        SPEC["FastTree-ME"]["color"],
        SPEC["FastTree-ME"]["marker"],
        1.0,
    ),
    "ft_ml": (
        "FastTree-ML, labelled only",
        SPEC["FastTree-ML"]["color"],
        SPEC["FastTree-ML"]["marker"],
        1.0,
    ),
}


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    rows = read_tsv(args.tables_dir, "silva_gtdb")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.24, right=0.93, bottom=0.2, top=0.97)
    x = range(len(RANKS))
    for tree, (label, colour, marker, width) in TREES.items():
        f1 = {r["rank"]: float(r["mean_f1"]) for r in rows if r["tree"] == tree}
        if set(f1) != set(RANKS):
            raise SystemExit(f"silva_gtdb.tsv: {tree} has ranks {sorted(f1)}, not {RANKS}")
        ax.plot(
            x,
            [f1[rank] for rank in RANKS],
            color=colour,
            marker=marker,
            markersize=4 if tree == "full" else 3,
            linewidth=width,
            label=label,
            zorder=3 if tree == "full" else 2,
        )
    taxa = {r["rank"]: int(r["taxa"]) for r in rows if r["tree"] == "full"}
    ax.set_xticks(list(x), [f"{rank.capitalize()}\n({taxa[rank]:,})" for rank in RANKS])
    ax.tick_params(axis="x", labelsize=plt.rcParams["xtick.labelsize"] * 0.85)
    ax.set_ylim(0.5, 0.95)  # room for the legend under the lines
    ax.set_ylabel("Mean F$_1$ of GTDB taxa")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", handlelength=1.6, borderaxespad=0.3)
    save(fig, args.img_dir, "fig4c_gtdb_recovery")


if __name__ == "__main__":
    main()
