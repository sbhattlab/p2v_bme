"""Fig. 2c: running time of Phylo2VecME, MAPLE and CMAPLE on the simulated pandemic alignments.

Inputs:  data/tables/maple.tsv                                          (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    per program and size: mean over the three branch-length scales, the bar spans their range
Outputs: img/fig2c_pandemic_time.{pdf,svg}

    pixi run -e paper python -m plots.fig2c_pandemic_time
"""

from statistics import mean

import matplotlib.pyplot as plt

from matplotlib.ticker import NullLocator
from numpy import ceil, floor, log10

from plots._config import MAIN_RC, PANDEMIC
from plots.utils import parser, read_tsv, save

FIGSIZE = (52 / 25.4, 62 / 25.4)
SIZES = [100000, 200000, 500000]


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.rcParams.update(MAIN_RC)
    rows = read_tsv(args.tables_dir, "maple")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.27, right=0.97, bottom=0.21, top=0.86)
    for method, colour, marker, style in PANDEMIC:
        means, los, his = [], [], []
        for n in SIZES:
            values = [
                float(r["time_s"])
                for r in rows
                if r["method"] == method and int(r["n"]) == n
            ]
            means.append(mean(values))
            los.append(means[-1] - min(values))
            his.append(max(values) - means[-1])
        ax.errorbar(
            SIZES,
            means,
            yerr=[los, his],
            color=colour,
            marker=marker,
            markersize=3.6,
            linestyle=style,
            linewidth=1.1,
            elinewidth=0.75,
            capsize=2,
            capthick=0.7,
            label=method,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(8e4, 6.2e5)
    times = [float(r["time_s"]) for r in rows]
    ax.set_ylim(
        10 ** (floor(log10(min(times)) - 0.3)), 10 ** (ceil(log10(max(times)) + 0.8))
    )
    ax.set_xticks(SIZES)
    ax.set_xticklabels(["100k", "200k", "500k"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_xlabel("Number of sequences")
    ax.set_ylabel("Time (s)")
    ax.set_title("Pandemic runtime", loc="left", fontsize=9, pad=8)
    ax.grid(axis="y", which="major", color="#E5E5E5", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=6.7, frameon=False, handlelength=1.5)
    save(fig, args.img_dir, "fig2c_pandemic_time")


if __name__ == "__main__":
    main()
