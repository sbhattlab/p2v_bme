"""Fig. S5a: support of the branch of each named SARS-CoV-2 variant in our tree of the Viridian genomes.

Inputs:  data/tables/viridian_variants.tsv                              (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    one row per variant, largest first: transfer support (filled), exact support (hollow), substitutions
Outputs: img/figS5a_sarscov2_variants.{pdf,svg}

    pixi run -e paper python -m plots.figS5a_sarscov2_variants
"""

import matplotlib.pyplot as plt

from plots._config import OURS_COLOUR, STYLE
from plots.utils import parser, read_tsv, save

FIGSIZE = (84 / 25.4, 66 / 25.4)  # panel a of the 150 mm wide figure
RC = {
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2,
}


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    plt.rcParams.update(RC)
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
            markeredgecolor=OURS_COLOUR,
            markeredgewidth=1.1,
            linestyle="none",
            zorder=3,
        )
        ax.plot(
            tbe,
            y,
            marker="o",
            markersize=4.4,
            color=OURS_COLOUR,
            linestyle="none",
            zorder=4,
        )
        ax.text(
            116,
            y,
            f"{float(v['ours_sub']):.0f}",
            ha="center",
            va="center",
            fontsize=7.5,
        )
    ax.text(116, -0.95, "subst.", ha="center", va="center", fontsize=7.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{v['name']}  {float(v['genomes']):,.0f}" for v in rows])
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
            color=OURS_COLOUR,
            marker="o",
            markersize=4.6,
            linestyle="none",
            label="Bootstrap, transfer (TBE)",
        ),
        plt.Line2D(
            [],
            [],
            color=OURS_COLOUR,
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
        loc="upper left",
        bbox_to_anchor=(0.02, 1.0),
        ncol=2,
        columnspacing=1.2,
        handletextpad=0.2,
    )
    save(fig, args.img_dir, "figS5a_sarscov2_variants", bbox_inches=None)


if __name__ == "__main__":
    main()
