"""Fig. 3a: time against RF error of VecME, MAPLE and CMAPLE on the simulated pandemic alignments.

Inputs:  data/tables/maple.tsv                                          (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    per program and mean branch length: one line through the three sizes (time, RF error)
           |
           v
         the three branch lengths as three bands, named on the right in substitutions per branch
Outputs: img/fig3/fig3a_pandemic_speed_accuracy.{pdf,svg,png}

    pixi run -e paper python -m plots.fig3a_pandemic_speed_accuracy
"""

import matplotlib.pyplot as plt
import numpy as np

from plots._config import FIG3_AXES_Y, FIG3_HEIGHT, PANDEMIC, STYLE
from plots.utils import (
    LABEL_GREY,
    OURS,
    PANDEMIC_BRANCH_LENGTHS,
    data_label,
    method_style,
    pandemic_rows,
    parser,
    save,
)

FIGSIZE = (96 / 25.4, FIG3_HEIGHT)  # beside Fig. 3b in the 180 mm page
LEFT, RIGHT = 0.11, 0.85  # room on the right for the branch-length axis
SITES = 30000  # every pandemic alignment: mean branch length x SITES = substitutions per branch
HEADROOM = 1.3  # y axis up to this multiple of the largest error, for the data label


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    rows = pandemic_rows(args.tables_dir)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bottom, top = FIG3_AXES_Y["ab"]
    fig.subplots_adjust(left=LEFT, right=RIGHT, bottom=bottom, top=top)
    bands = []  # (substitutions per branch, mean RF error of every run at that branch length)
    for length in PANDEMIC_BRANCH_LENGTHS:
        group = [r for r in rows if float(r["mean_branch_length"]) == length]
        bands.append((length * SITES, float(np.mean([float(r["rf_pct"]) for r in group]))))
        for method, key in PANDEMIC:
            runs = sorted((r for r in group if r["method"] == method), key=lambda r: int(r["n"]))
            ax.plot(
                [float(r["time_s"]) for r in runs],
                [float(r["rf_pct"]) for r in runs],
                label=method if length == PANDEMIC_BRANCH_LENGTHS[0] else None,
                zorder=5 if key == OURS else 3,
                **method_style(key),
            )
    times = [float(r["time_s"]) for r in rows]
    ax.set_xscale("log")
    ax.set_xlim(10 ** np.floor(np.log10(min(times))), 10 ** np.ceil(np.log10(max(times))))
    ax.set_ylim(0, HEADROOM * max(float(r["rf_pct"]) for r in rows))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RF error (%)")
    ax.grid(color="#E5E5E5", linewidth=0.5)
    ax.set_axisbelow(True)
    data_label(ax, "simulated pandemic")
    # the legend between the two upper bands, at the right
    upper = sorted(rf for _, rf in bands)[-2:]
    ax.legend(
        loc="center right",
        bbox_to_anchor=(1.0, np.mean(upper) / ax.get_ylim()[1]),
        handlelength=1.8,
    )
    # the bands' branch lengths, on a right-hand axis at each band's height
    right = ax.twinx()
    right.set_ylim(ax.get_ylim())
    right.set_yticks([rf for _, rf in bands], [f"{subs:g}" for subs, _ in bands])
    right.set_ylabel("Substitutions per branch", color=LABEL_GREY)
    right.tick_params(axis="y", length=0, colors=LABEL_GREY)
    right.spines[["top", "right"]].set_visible(False)
    save(fig, args.img_dir, "fig3a_pandemic_speed_accuracy", bbox_inches=None)


if __name__ == "__main__":
    main()
