"""Fig. 2c: RF error of every method against the number of sequences, on the JC alignments simulated on
random trees (no sequences added by expansion).

Inputs:  data/tables/scaling_methods.tsv                                (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    the random-tree rows (yule10k) of every method
           |
           v
         one line per method over the sizes (log-log axes): the mean at each size; the whisker spans
         the replicates (as Fig. S3b, with its own legend)
Outputs: img/fig2/fig2c_random_trees.{pdf,svg,png}

    pixi run -e paper python -m plots.fig2c_random_trees
"""

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from plots._config import FIG2_PANEL, SPEC, STYLE
from plots.utils import (
    PANELS,
    PLOTTED,
    method_lines,
    method_stats,
    method_style,
    parser,
    read_tsv,
    save,
)

DATASET = "yule10k"


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    field, scale, ylabel = PANELS["rf"]
    stats = method_stats(read_tsv(args.tables_dir, "scaling_methods"), DATASET, field, scale)
    if not stats:
        raise SystemExit(f"no {field} data for {DATASET}")
    fig, ax = plt.subplots(figsize=FIG2_PANEL)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.16, top=0.76)
    method_lines(ax, stats, "rf")
    ax.set_ylabel(ylabel)
    present = [k for k in PLOTTED if any(key == k for _, key in stats)]
    handles = [Line2D([], [], label=SPEC[k]["label"], **method_style(k)) for k in reversed(present)]
    # above the axes, in two columns: six names do not fit inside a half-page panel
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.45, 1.02),
        ncol=2,
        handlelength=1.8,
        handletextpad=0.4,
        columnspacing=1.2,
        borderaxespad=0.0,
    )
    save(fig, args.img_dir, "fig2c_random_trees")


if __name__ == "__main__":
    main()
