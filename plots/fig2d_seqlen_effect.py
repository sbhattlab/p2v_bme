"""Fig. 2d: normalised RF distance at 10,000 sequences on JC alignments of 1,000 and 10,000 sites,
for every method.

Inputs:  data/tables/scaling_methods.tsv                                (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    per method with runs at both lengths:
         mean and range of the normalised RF distance (hollow: 1,000 sites, filled: 10,000)
Outputs: img/fig2/fig2d_seqlen_effect.{pdf,svg,png}

    pixi run -e paper python -m plots.fig2d_seqlen_effect
"""

import statistics

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

from plots._config import FIG2_PANEL, METHODS, SPEC, STYLE
from plots.utils import OURS, data_label, parser, read_tsv, save

N = 10000
SHORT, LONG = "jc1k", "jc10k"
# Normalised RF distance (%), linear, as Fig. 2b: the data span little more than a decade
XLIM = (0, 25)
TICK_STEP = 5  # normalised RF distance (%)


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    data = read_tsv(args.tables_dir, "scaling_methods")

    def values(dataset: str, key: str) -> list[float]:
        return [
            float(r["rf_pct"])
            for r in data
            if (r["dataset"], r["method"], int(r["n"])) == (dataset, key, N)
        ]

    rows = []  # bottom to top, VecME at the bottom: top to bottom as Fig. 2a's legend
    for key, *_ in reversed(METHODS):
        a, b = values(SHORT, key), values(LONG, key)
        if a and b:
            rows.append(
                (key, (statistics.mean(a), min(a), max(a)), (statistics.mean(b), min(b), max(b)))
            )
        else:
            have = [tag for tag, runs in ((SHORT, a), (LONG, b)) if runs]
            raise SystemExit(f"{key} at N = {N}: results for {have}, not for {SHORT} and {LONG}")
    if (top := max(max(r[1][2], r[2][2]) for r in rows)) > XLIM[1]:
        raise SystemExit(f"RF distance up to {top:.1f} %: beyond the axis ({XLIM[1]} %)")

    fig, ax = plt.subplots(figsize=FIG2_PANEL)
    fig.subplots_adjust(left=0.36, right=0.97, bottom=0.16, top=0.88)  # right, bottom as in Fig. 2c
    for y, (key, (m1, lo1, hi1), (m2, lo2, hi2)) in enumerate(rows):
        s = SPEC[key]
        ours = key == OURS  # VecME drawn stronger; the markers are the data, so none is faded
        size = 7 if ours else 6
        ax.plot([m1, m2], [y, y], color=s["color"], lw=2.6 if ours else 1.6, alpha=0.55, zorder=2)
        ax.errorbar(
            [m1],
            [y],
            xerr=[[m1 - lo1], [hi1 - m1]],
            fmt="o",  # the rows name the methods: no shapes needed
            ms=size,
            mfc="white",
            mec=s["color"],
            mew=1.3,
            ecolor=s["color"],
            elinewidth=0.9,
            capsize=0,  # caps stuck out of the markers as second marks
            zorder=4,
        )
        ax.errorbar(
            [m2],
            [y],
            xerr=[[m2 - lo2], [hi2 - m2]],
            fmt="o",
            ms=size,
            mfc=s["color"],
            mec=s["color"],
            ecolor=s["color"],
            elinewidth=0.9,
            capsize=0,
            zorder=5,
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([SPEC[k]["label"] for k, *_ in rows])
    ax.set_ylim(-0.6, len(rows) + 0.2)  # room above the top row for the data label
    ax.set_xlim(*XLIM)
    ax.xaxis.set_major_locator(MultipleLocator(TICK_STEP))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.set_xlabel("Normalised RF distance (%)")
    ax.grid(axis="x", color="#E6E6E6", lw=0.6)
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                ms=6,
                mfc="white",
                mec="#333333",
                label="1,000 sites",
            ),
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                ms=6,
                mfc="#333333",
                mec="#333333",
                label="10,000 sites",
            ),
        ],
        # above the axes, in one row: inside, it covered the best row's markers
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        handletextpad=0.3,
        columnspacing=1.5,
        borderaxespad=0.3,
    )
    data_label(ax, "PASTA (JC69), N = 10,000")
    save(fig, args.img_dir, "fig2d_seqlen_effect")


if __name__ == "__main__":
    main()
