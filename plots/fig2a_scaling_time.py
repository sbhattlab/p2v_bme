"""Fig. 2a: running time of every method against the number of sequences, on the RNASim alignments
(as Fig. S1a).

Inputs:  data/tables/scaling_methods.tsv                                (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    the RNASim rows of every method
           |
           v
         one line per method over the sizes (log-log axes): the mean at each size; the whisker spans
         the replicates
Outputs: img/fig2/fig2a_scaling_time.{pdf,svg,png}

    pixi run -e paper python -m plots.fig2a_scaling_time
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

DATASET = "rnasim"


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    field, scale, ylabel = PANELS["time"]
    stats = method_stats(read_tsv(args.tables_dir, "scaling_methods"), DATASET, field, scale)
    if not stats:
        raise SystemExit(f"no {field} data for {DATASET}")
    fig, ax = plt.subplots(figsize=FIG2_PANEL)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.16, top=0.96)  # left, right as in Fig. 2c
    method_lines(ax, stats, "time")
    ax.set_ylabel(ylabel)
    present = [k for k in PLOTTED if any(key == k for _, key in stats)]
    handles = [Line2D([], [], label=SPEC[k]["label"], **method_style(k)) for k in reversed(present)]
    # in the empty lower right: below VecME's line, right of the CPU programs' last size
    ax.legend(
        handles=handles,
        loc="lower right",
        handlelength=1.8,
        handletextpad=0.4,
        labelspacing=0.3,
        borderaxespad=0.2,
    )
    save(fig, args.img_dir, "fig2a_scaling_time")


if __name__ == "__main__":
    main()
