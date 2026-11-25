"""Fig. 3d: support of the branch of each named SARS-CoV-2 variant in our tree of
the Viridian genomes.

Inputs:  data/tables/viridian_variants.tsv                              (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    one row per variant, largest first:
         transfer support (filled), exact support (hollow), substitutions
Outputs: img/fig3/fig3d_sarscov2_variants.{pdf,svg,png}

    pixi run -e paper python -m plots.fig3d_sarscov2_variants
"""

import matplotlib.pyplot as plt

from plots._config import SPEC, STYLE
from plots.utils import parser, read_tsv, save

FIGSIZE = (84 / 25.4, 66 / 25.4)  # panel a of the 150 mm wide figure
# The two label columns left of the axes, in axes widths:
# names left-aligned, genome counts right-aligned
NAME_X, COUNT_X = -0.57, -0.03


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    rows = sorted(
        read_tsv(args.tables_dir, "viridian_variants"),
        key=lambda v: -float(v["genomes"]),
    )
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.36, right=0.97, top=0.84, bottom=0.17)
    for y, v in enumerate(rows):
        exact, tbe = 100 * float(v["ours_exact"]), 100 * float(v["ours_tbe"])
        ax.plot([exact, tbe], [y, y], color="#CCCCCC", linewidth=1.0, zorder=1)
        ax.plot(
            exact,
            y,
            marker="o",
            markersize=4.4,
            markerfacecolor="white",
            markeredgecolor=SPEC["P2V"]["color"],
            markeredgewidth=1.1,
            linestyle="none",
            zorder=3,
        )
        ax.plot(
            tbe,
            y,
            marker="o",
            markersize=4.4,
            color=SPEC["P2V"]["color"],
            linestyle="none",
            zorder=4,
        )
        ax.text(
            116,
            y,
            f"{float(v['ours_sub']):.0f}",
            ha="center",
            va="center",
        )
    ax.text(116, -0.95, "subst.", ha="center", va="center")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([])
    columns = ax.get_yaxis_transform()  # x in axes widths, y in rows
    for y, v in enumerate(rows):
        ax.text(NAME_X, y, v["name"], transform=columns, ha="left", va="center")
        count = f"{float(v['genomes']):,.0f}"
        ax.text(COUNT_X, y, count, transform=columns, ha="right", va="center")
    ax.text(COUNT_X, -0.95, "genomes", transform=columns, ha="right", va="center")
    ax.set_ylim(len(rows) - 0.5, -1.4)
    ax.set_xlim(-4, 124)
    ax.spines["bottom"].set_bounds(-4, 104)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Support of the variant's branch (%)")
    ax.tick_params(axis="y", length=0)
    handles = [
        plt.Line2D(
            [],
            [],
            color=SPEC["P2V"]["color"],
            marker="o",
            markersize=4.6,
            linestyle="none",
            label="Bootstrap, transfer (TBE)",
        ),
        plt.Line2D(
            [],
            [],
            color=SPEC["P2V"]["color"],
            marker="o",
            markersize=4.6,
            linestyle="none",
            markerfacecolor="white",
            markeredgewidth=1.1,
            label="Bootstrap, exact",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=1,
        handletextpad=0.2,
    )
    save(fig, args.img_dir, "fig3d_sarscov2_variants", bbox_inches=None)


if __name__ == "__main__":
    main()
