"""Fig. 3c: support of the branch of each named SARS-CoV-2 variant in our tree of the Viridian genomes,
with the variant's number of genomes beside it.

Inputs:  data/tables/viridian_variants.tsv                              (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    one row per variant, largest first:
         transfer support (filled) and exact support (hollow), joined by a line
           |
           v
         on the right, the variant's genomes as a bar on a log scale, coloured as in Fig. 3d
Outputs: img/fig3/fig3c_sarscov2_variants.{pdf,svg,png}

    pixi run -e paper python -m plots.fig3c_sarscov2_variants
"""

import matplotlib.pyplot as plt

from plots._config import FIG3_AXES_Y, FIG3_HEIGHT, SPEC, STYLE, VARIANT_GROUPS
from plots.utils import parser, read_tsv, save

FIGSIZE = (60 / 25.4, FIG3_HEIGHT)  # in the row of Figs 3a-c across the 180 mm page
SUPPORT_X, SUPPORT_W = 0.27, 0.39  # the support axes, in panel widths; the names to their left
GENOMES_X, GENOMES_W = 0.74, 0.23  # the axes of the genome counts, on the right
GENOMES_LIM = (1e3, 2e6)  # every variant has between 10^3 and 10^6 genomes
XLABEL_Y = -0.17  # both x labels at this height, in axes heights: the log ticks are taller
MARKER_SIZE = 4.4
# variant -> colour, as in Fig. 3d
COLOURS = {name: colour for _key, name, colour in VARIANT_GROUPS}
OTHER = VARIANT_GROUPS[0][2]


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    rows = sorted(
        read_tsv(args.tables_dir, "viridian_variants"),
        key=lambda v: -float(v["genomes"]),
    )
    fig = plt.figure(figsize=FIGSIZE)
    bottom, top = FIG3_AXES_Y
    ax = fig.add_axes((SUPPORT_X, bottom, SUPPORT_W, top - bottom))
    genomes = fig.add_axes((GENOMES_X, bottom, GENOMES_W, top - bottom), sharey=ax)
    colour = SPEC["P2V"]["color"]
    for y, v in enumerate(rows):
        exact, tbe = 100 * float(v["ours_exact"]), 100 * float(v["ours_tbe"])
        ax.plot([exact, tbe], [y, y], color="#CCCCCC", linewidth=1.0, zorder=1)
        ax.plot(
            exact,
            y,
            marker="o",
            markersize=MARKER_SIZE,
            markerfacecolor="white",
            markeredgecolor=colour,
            markeredgewidth=1.1,
            linestyle="none",
            zorder=3,
        )
        ax.plot(
            tbe, y, marker="o", markersize=MARKER_SIZE, color=colour, linestyle="none", zorder=4
        )
        genomes.barh(y, float(v["genomes"]), height=0.62, color=COLOURS.get(v["name"], OTHER))
    ax.set_yticks(range(len(rows)), [v["name"] for v in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.set_xlim(-4, 104)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Bootstrap support (%)")
    genomes.set_xscale("log")
    genomes.set_xlim(*GENOMES_LIM)
    genomes.set_xticks([1e3, 1e6])
    genomes.minorticks_off()
    genomes.tick_params(axis="y", left=False, labelleft=False)
    genomes.spines["left"].set_visible(False)
    genomes.set_xlabel("Genomes")
    for a in (ax, genomes):
        a.xaxis.set_label_coords(0.5, XLABEL_Y)
    handles = [
        plt.Line2D(
            [],
            [],
            color=colour,
            marker="o",
            markersize=4.6,
            linestyle="none",
            label="Transfer (TBE)",
        ),
        plt.Line2D(
            [],
            [],
            color=colour,
            marker="o",
            markersize=4.6,
            linestyle="none",
            markerfacecolor="white",
            markeredgewidth=1.1,
            label="Exact",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=1,
        labelspacing=0.2,
        borderaxespad=0.1,
        handletextpad=0.2,
    )
    save(fig, args.img_dir, "fig3c_sarscov2_variants", bbox_inches=None)


if __name__ == "__main__":
    main()
