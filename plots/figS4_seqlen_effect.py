"""Fig. S4: RF error at 10,000 sequences on JC alignments of 1,000 and 10,000 sites,
for every method.

Inputs:  data/tables/scaling_methods.tsv                                (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    per method with runs at both lengths:
         mean and range of the RF error (hollow: 1,000 sites, filled: 10,000)
Outputs: img/fig2/figS4_seqlen_effect.{pdf,svg,png}

    pixi run -e paper python -m plots.figS4_seqlen_effect
"""

# pylint: disable=invalid-name  # figS<N>: the paper's numbering of its supplementary figures

import statistics

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from plots._config import METHODS, SPEC, STYLE
from plots.utils import OURS, parser, read_tsv, save

N = 10000
SHORT, LONG = "jc1k", "jc10k"


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

    rows = []
    for key, *_ in METHODS:
        a, b = values(SHORT, key), values(LONG, key)
        if a and b:
            rows.append(
                (key, (statistics.mean(a), min(a), max(a)), (statistics.mean(b), min(b), max(b)))
            )
        elif a or b:
            print(f"skipping {key}: results for only one alignment length at N = {N}")
    if not rows:
        raise SystemExit(f"no method has both {SHORT} and {LONG} results at N = {N}")
    rows.sort(key=lambda r: -r[2][0])  # worst at the bottom of the reading order -> best on top

    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    fig.subplots_adjust(left=0.35, right=0.94, bottom=0.19, top=0.95)
    for y, (key, (m1, lo1, hi1), (m2, lo2, hi2)) in enumerate(rows):
        s = SPEC[key]
        ours = key == OURS  # Phylo2VecME drawn stronger; the markers are the data, so none is faded
        size = 7 if ours else 6
        ax.plot([m1, m2], [y, y], color=s["color"], lw=2.6 if ours else 1.6, alpha=0.55, zorder=2)
        ax.errorbar(
            [m1],
            [y],
            xerr=[[m1 - lo1], [hi1 - m1]],
            fmt=s["marker"],
            ms=size,
            mfc="white",
            mec=s["color"],
            mew=1.3,
            ecolor=s["color"],
            elinewidth=0.9,
            capsize=1.8,
            zorder=4,
        )
        ax.errorbar(
            [m2],
            [y],
            xerr=[[m2 - lo2], [hi2 - m2]],
            fmt=s["marker"],
            ms=size,
            mfc=s["color"],
            mec=s["color"],
            ecolor=s["color"],
            elinewidth=0.9,
            capsize=1.8,
            zorder=5,
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([SPEC[k]["label"] for k, *_ in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xscale("log")
    ax.set_xlim(1.3, 50)
    ax.xaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("RF error at N = 10,000 (%)")
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
        loc="upper right",
        handletextpad=0.3,
        borderaxespad=0.2,
    )
    save(fig, args.img_dir, "figS4_seqlen_effect")


if __name__ == "__main__":
    main()
