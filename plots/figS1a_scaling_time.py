"""Figs S1a, S2a, S3a: running time of every method at each size,
on RNASim, JC10k and random-tree JC.

Inputs:  data/tables/scaling_methods.tsv                                (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    each dataset (rnasim -> S1, jc10k -> S2, yule10k -> S3)
           |
           v
         one bar per method and size:
         the mean, from the panel's floor (log axis); the whisker spans the replicates
Outputs: img/fig2/figS1a_scaling_time.{pdf,svg}, figS2a_..., figS3a_...

    pixi run -e paper python -m plots.figS1a_scaling_time
"""

import matplotlib.pyplot as plt

from plots._config import SCALING_FIGURES, STYLE
from plots.utils import method_bars, method_legend, method_stats, parser, read_tsv, save

FIGSIZE = (7.2, 2.6)  # one row of the full-width four-panel page


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    rows = read_tsv(args.tables_dir, "scaling_methods")
    for dataset, figure in SCALING_FIGURES.items():
        stats = method_stats(rows, dataset, "time_s", 1.0)
        if not stats:
            print(f"no time_s data for {dataset}: {figure}a not drawn")
            continue
        fig, ax = plt.subplots(figsize=FIGSIZE)
        fig.subplots_adjust(left=0.09, right=0.985, top=0.80, bottom=0.2)
        method_bars(ax, stats, "time")
        ax.set_ylabel("Runtime (s)")
        method_legend(fig, stats)
        save(fig, args.img_dir, f"{figure}a_scaling_time")


if __name__ == "__main__":
    main()
