"""Fig. 2d: RF error of Phylo2VecME against that of MAPLE and CMAPLE,
on the same simulated pandemic alignments.

Inputs:  data/tables/maple.tsv                                          (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    one point per alignment (size x branch-length scale) and program;
         the diagonal marks equal error
Outputs: img/fig2/fig2d_pandemic_rf_error.{pdf,svg,png}

    pixi run -e paper python -m plots.fig2d_pandemic_rf_error
"""

import matplotlib.pyplot as plt

from plots._config import PANDEMIC, STYLE
from plots.utils import parser, read_tsv, save

FIGSIZE = (52 / 25.4, 62 / 25.4)
LIMIT = 28  # per cent, both axes


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    paired: dict[tuple[int, float], dict[str, float]] = {}
    for r in read_tsv(args.tables_dir, "maple"):
        paired.setdefault((int(r["n"]), float(r["mean_branch_length"])), {})[r["method"]] = float(
            r["rf_pct"]
        )
    colours = {method: colour for method, colour, _marker, _style in PANDEMIC}
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.27, right=0.97, bottom=0.21, top=0.86)
    gap = 0.0
    for method, marker in (("MAPLE", "o"), ("CMAPLE", "x")):
        x = [g[method] for g in paired.values()]
        y = [g["Phylo2VecME"] for g in paired.values()]
        gap = max([gap, *(abs(a - b) for a, b in zip(x, y))])
        if marker == "o":
            ax.scatter(
                x,
                y,
                label=method,
                marker=marker,
                s=26,
                linewidths=0.85,
                facecolors="none",
                edgecolors=colours[method],
                zorder=3,
            )
        else:
            ax.scatter(
                x,
                y,
                label=method,
                marker=marker,
                s=16,
                linewidths=0.85,
                color=colours[method],
                zorder=4,
            )
    ax.plot([0, LIMIT], [0, LIMIT], color="#AAAAAA", linewidth=0.8, linestyle="--", zorder=1)
    ax.set(xlim=(0, LIMIT), ylim=(0, LIMIT), xticks=[0, 10, 20], yticks=[0, 10, 20])
    ax.set_xlabel("Baseline RF error (%)")
    ax.set_ylabel("Phylo2VecME RF error (%)")
    ax.set_title("Pandemic RF error", loc="left", pad=8)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.3)
    save(fig, args.img_dir, "fig2d_pandemic_rf_error")
    print(f"largest RF gap to a baseline: {gap:.6f} percentage points")


if __name__ == "__main__":
    main()
