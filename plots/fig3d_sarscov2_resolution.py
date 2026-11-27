"""Fig. 3d: the Viridian SARS-CoV-2 genomes sampled each month by variant, and how finely our tree places
them, by sampling month.

Top: the genomes sampled each month, stacked by variant group. Bottom: for each genome, the smallest clade
around it whose branch has at least 70% transfer support and half a substitution; the line is the monthly
median of the genomes of that clade sampled within 15 days of the genome, with the interquartile range
shaded. Dots take the colour of the month's most common variant.

Inputs:  data/tables/viridian_resolution.tsv                            (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    the months at the 70% threshold, checked for gaps
           |
           v
         genomes per month by variant (stacked areas) above the resolution curve, on one month axis
Outputs: img/fig3/fig3d_sarscov2_resolution.{pdf,svg,png}

    pixi run -e paper python -m plots.fig3d_sarscov2_resolution
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator, NullFormatter

from plots._config import STYLE, VARIANT_GROUPS
from plots.utils import parser, read_tsv, save

# The panel from the bottom, in mm: margin, resolution axes, gap, genomes-per-month axes, margin;
# the resolution axes are the taller, room for their y label on two lines
BOTTOM, CLADES_HEIGHT, GAP, COUNTS_HEIGHT, TOP = 14.8, 39.0, 1.5, 26.6, 3.0
HEIGHT = BOTTOM + CLADES_HEIGHT + GAP + COUNTS_HEIGHT + TOP
FIGSIZE = (180 / 25.4, HEIGHT / 25.4)  # the full page width, below Figs 3a-c
LEFT, RIGHT = 0.16, 0.99  # of both axes
COUNTS_AXES = ((BOTTOM + CLADES_HEIGHT + GAP) / HEIGHT, COUNTS_HEIGHT / HEIGHT)  # bottom, height
CLADES_AXES = (BOTTOM / HEIGHT, CLADES_HEIGHT / HEIGHT)
YLABEL_X = -0.10  # both y labels at this distance from the axes, in axes widths, so that they align
THRESHOLD = 0.70  # transfer support that a branch needs
QUARTER = {"01": "Jan", "04": "Apr", "07": "Jul", "10": "Oct"}
COUNT_STEP = 100_000  # genomes per month between ticks
LABEL_SHARE = 0.5  # a variant is named on the areas where it holds more than this share of a month
CLADE_LIM = (1, 2e5)  # genomes in the smallest supported clade
NAME_FONTSIZE = 8  # the variant names on the areas
GRID_COLOUR, GRID_WIDTH = "#EEEEEE", 0.6


def months_at_threshold(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Selects the monthly rows at THRESHOLD and checks that no month is missing: rows -> rows."""
    rows = [r for r in rows if float(r["threshold"]) == THRESHOLD]
    if not rows:
        raise SystemExit(f"no rows at the threshold {THRESHOLD}")
    index = [int(r["month"][:4]) * 12 + int(r["month"][5:7]) - 1 for r in rows]
    if index != list(range(index[0], index[0] + len(index))):
        raise SystemExit("the months are not consecutive")
    return rows


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    rows = months_at_threshold(read_tsv(args.tables_dir, "viridian_resolution"))
    months = [r["month"] for r in rows]

    def column(name: str) -> np.ndarray:
        return np.array([float(r[name]) for r in rows])

    fig = plt.figure(figsize=FIGSIZE)
    counts = fig.add_axes((LEFT, COUNTS_AXES[0], RIGHT - LEFT, COUNTS_AXES[1]))
    ax = fig.add_axes((LEFT, CLADES_AXES[0], RIGHT - LEFT, CLADES_AXES[1]), sharex=counts)
    x = np.arange(len(rows))
    with_commas = FuncFormatter(lambda v, _pos: f"{int(v):,}")

    # genomes per month by variant, stacked; each variant named at its peak month
    shares = np.array([column(f"share_{key}") for key, _name, _colour in VARIANT_GROUPS])
    per_month = shares * column("genomes")
    counts.stackplot(x, per_month, colors=[c for _, _, c in VARIANT_GROUPS], linewidth=0)
    below = np.vstack([np.zeros(len(rows)), np.cumsum(per_month, axis=0)[:-1]])
    for (_key, name, _colour), share, count, base in zip(VARIANT_GROUPS, shares, per_month, below):
        if share.max() > LABEL_SHARE:
            peak = int(np.argmax(count))
            counts.text(
                peak,
                base[peak] + count[peak] / 2,
                name,
                color="white",
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=NAME_FONTSIZE,
            )
    counts.set_ylim(0, None)
    counts.yaxis.set_major_locator(MultipleLocator(COUNT_STEP))
    counts.yaxis.set_major_formatter(with_commas)
    counts.set_ylabel("Genomes sampled\nper month")
    counts.tick_params(axis="x", labelbottom=False)
    counts.grid(axis="y", color=GRID_COLOUR, linewidth=GRID_WIDTH)
    counts.set_axisbelow(True)

    # the resolution: median and interquartile range, dots in the colour of the month's main variant
    lead = shares.argmax(axis=0)
    med = column("window_median")
    ax.fill_between(
        x, column("window_q25"), column("window_q75"), color="#E3E3E3", linewidth=0, zorder=1
    )
    ax.plot(x, med, color="#444444", linewidth=2.0, zorder=2)
    ax.scatter(
        x,
        med,
        s=40,
        c=[VARIANT_GROUPS[i][2] for i in lead],
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    ax.set_yscale("log")
    ax.set_ylim(*CLADE_LIM)
    ax.yaxis.set_major_locator(FixedLocator([10, 1e3, 1e5]))
    ax.yaxis.set_major_formatter(with_commas)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("Genomes in the smallest\nsupported clade")
    ticks = [i for i, m in enumerate(months) if m[5:7] in QUARTER]
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [
            QUARTER[months[i][5:7]] + ("\n" + months[i][:4] if months[i][5:7] == "01" else "")
            for i in ticks
        ]
    )
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.set_xlabel("Sampling month")
    ax.grid(axis="y", color=GRID_COLOUR, linewidth=GRID_WIDTH, zorder=0)
    ax.set_axisbelow(True)
    for a in (counts, ax):
        a.yaxis.set_label_coords(YLABEL_X, 0.5)
    save(fig, args.img_dir, "fig3d_sarscov2_resolution", bbox_inches=None)


if __name__ == "__main__":
    main()
