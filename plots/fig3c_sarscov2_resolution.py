"""Fig. 3c: how finely the SARS-CoV-2 tree places genomes, by sampling month (Viridian genomes).

For each genome, the smallest clade around it whose branch has at least 70% transfer support and
half a substitution: the line is the monthly median of the genomes of that clade sampled within 15
days of the genome, with the interquartile range shaded; the dashed line counts all genomes sampled
in those 31 days, so the axis counts genomes sampled in the same 31 days for both.
Dots take the colour of the month's most common variant; the strip above shows each variant's share.

Inputs:  data/tables/viridian_resolution.tsv                            (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    the months at the 70% threshold, checked for gaps
           |
           v
         variant strip and resolution curve on a shared month axis
Outputs: img/fig3/fig3c_sarscov2_resolution.{pdf,svg}

    pixi run -e paper python -m plots.fig3c_sarscov2_resolution
"""

import matplotlib.pyplot as plt

from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from numpy import arange, array, average, zeros

from plots._config import STYLE
from plots.utils import parser, read_tsv, save

SCALE = 1.15  # line and marker scale
GROUPS = (
    ("other", "Other", "#9A9A9A"),
    ("alpha", "Alpha", "#3B7FB6"),
    ("delta", "Delta", "#3A9E7E"),
    ("omicron", "Omicron", "#D9713A"),
    ("recombinant", "Recombinant", "#B5559B"),
)
THRESHOLD = 0.70  # transfer support that a branch needs
QUARTER = {"01": "Jan", "04": "Apr", "07": "Jul", "10": "Oct"}


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
    column = lambda name: array([float(r[name]) for r in rows])  # noqa: E731
    fig = plt.figure(figsize=(10.2, 3.9))
    strip = fig.add_axes((0.15, 0.87, 0.835, 0.07))
    ax = fig.add_axes((0.15, 0.2, 0.835, 0.65), sharex=strip)

    x = arange(len(rows))
    shares = array([column(f"share_{key}") for key, _name, _colour in GROUPS])
    bottom = zeros(len(rows))
    for (_key, name, colour), share in zip(GROUPS, shares):
        strip.bar(x, share, bottom=bottom, width=1.0, color=colour, linewidth=0)
        bottom += share
        if share.max() > 0.5:  # name a variant on the strip where it holds most of the genomes
            strip.text(
                float(average(x, weights=share)),
                0.5,
                name,
                color="white",
                ha="center",
                va="center",
                fontweight="bold",
            )
    strip.set_ylim(0, 1)
    strip.set_yticks([])
    strip.tick_params(axis="x", length=0, labelbottom=False)
    for spine in strip.spines.values():
        spine.set_visible(False)

    lead = shares.argmax(axis=0)
    med = column("window_median")
    ax.fill_between(
        x,
        column("window_q25"),
        column("window_q75"),
        color="#E3E3E3",
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        x,
        med,
        color="#444444",
        linewidth=2.4 * SCALE,
        zorder=2,
        label="In the smallest supported clade (median, interquartile range)",
    )
    ax.plot(
        x,
        column("window_total_median"),
        color="#888888",
        linewidth=1.6 * SCALE,
        linestyle=(0, (3, 2)),
        zorder=1,
        label="All",
    )
    ax.legend(
        frameon=False,
        loc="upper left",
        ncol=2,
        handlelength=1.8,
        columnspacing=1.6,
    )
    ax.scatter(
        x,
        med,
        s=58 * SCALE**2,
        c=[GROUPS[i][2] for i in lead],
        edgecolors="white",
        linewidths=0.9 * SCALE,
        zorder=3,
    )
    ax.set_yscale("log")
    ax.set_ylim(1, 3e6)
    ax.yaxis.set_major_locator(FixedLocator([10, 100, 1e3, 1e4, 1e5, 1e6]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{int(v):,}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel(
        "Genomes sampled in\nthe same 31 days",
        fontweight="bold",
    )
    ticks = [i for i, m in enumerate(months) if m[5:7] in QUARTER]
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [
            QUARTER[months[i][5:7]] + ("\n" + months[i][:4] if months[i][5:7] == "01" else "")
            for i in ticks
        ]
    )
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.set_xlabel("Sampling month", fontweight="bold")
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save(fig, args.img_dir, "fig3c_sarscov2_resolution")


if __name__ == "__main__":
    main()
