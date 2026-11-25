"""Fig. 2d: RF error of VecME, MAPLE and CMAPLE on the simulated pandemic alignments,
by mean branch length in substitutions per branch (as Fig. 2c draws their times).

Inputs:  data/tables/maple.tsv                                          (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    per program and branch-length scale: the mean over the three sizes, the whisker spans them;
         the programs side by side at each scale, as their errors nearly coincide
Outputs: img/fig2/fig2d_pandemic_rf_error.{pdf,svg,png}

    pixi run -e paper python -m plots.fig2d_pandemic_rf_error
"""

import statistics

import matplotlib.pyplot as plt

from plots._config import PANDEMIC, STYLE
from plots.utils import OURS, method_style, parser, read_tsv, save

FIGSIZE = (52 / 25.4, 62 / 25.4)
SITES = 30000  # every pandemic alignment: x = mean branch length x SITES = substitutions per branch
DODGE = {"MAPLE": -0.21, "CMAPLE": 0.0, "VecME": 0.21}  # side by side, in x units
TOP = 40  # per cent: headroom above the data, so that the legend clears them (as Fig. 2c)


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    rows = read_tsv(args.tables_dir, "maple")
    scales = sorted({float(r["mean_branch_length"]) for r in rows})
    rf = {
        (r["method"], int(r["n"]), float(r["mean_branch_length"])): float(r["rf_pct"]) for r in rows
    }
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.27, right=0.97, bottom=0.21, top=0.96)
    for method, key in PANDEMIC:
        mu, lo, hi = [], [], []
        for b in scales:
            values = [v for (m, _n, s), v in rf.items() if m == method and s == b]
            mu.append(statistics.mean(values))
            lo.append(mu[-1] - min(values))
            hi.append(max(values) - mu[-1])
        ax.errorbar(
            [b * SITES + DODGE[method] for b in scales],
            mu,
            yerr=[lo, hi],
            elinewidth=0.75,
            capsize=2,
            capthick=0.7,
            label=method,
            zorder=5 if key == OURS else 3,
            **method_style(key),
        )
    ax.set_xticks([b * SITES for b in scales])
    ax.set_xticklabels([f"{b * SITES:g}" for b in scales])
    ax.set_xlim(0.75, 5.25)
    ax.set_ylim(0, TOP)
    ax.set_xlabel("Substitutions per branch")
    ax.set_ylabel("RF error (%)")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, handletextpad=0.4)
    save(fig, args.img_dir, "fig2d_pandemic_rf_error")
    gap = max(
        abs(rf[("VecME", n, s)] - v)
        for (m, n, s), v in rf.items()
        if m in ("MAPLE", "CMAPLE") and ("VecME", n, s) in rf
    )
    print(f"largest RF gap to MAPLE or CMAPLE on one alignment: {gap:.6f} percentage points")


if __name__ == "__main__":
    main()
