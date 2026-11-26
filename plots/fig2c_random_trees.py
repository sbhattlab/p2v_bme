"""Fig. 2c: RF error of every method against the number of sequences, on the JC alignments simulated on
random trees (no sequences added by expansion).

Inputs:  data/tables/scaling_methods.tsv                                (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    the random-tree rows (yule10k) of every method
           |
           v
         one line per method over the sizes (log-log axes): the mean at each size; the whisker spans
         the replicates (as Fig. S3b)
Outputs: img/fig2/fig2c_random_trees.{pdf,svg,png}

    pixi run -e paper python -m plots.fig2c_random_trees
"""

import matplotlib.pyplot as plt

from plots._config import FIG2_PANEL, STYLE
from plots.utils import (
    PANELS,
    data_label,
    method_legend_inside,
    method_lines,
    method_stats,
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
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.16, top=0.96)  # as in Fig. 2a
    method_lines(ax, stats, "rf")
    ax.set_ylabel(ylabel)
    method_legend_inside(ax, stats)  # as in Fig. 2a
    data_label(ax, "random trees (JC69)")
    save(fig, args.img_dir, "fig2c_random_trees")


if __name__ == "__main__":
    main()
