"""Figs S1c, S2c, S3c: peak host memory of every method at each size,
on RNASim, JC10k and random-tree JC.

Inputs:  data/tables/scaling_methods.tsv                                (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    each dataset (rnasim -> S1, jc10k -> S2, yule10k -> S3)
           |
           v
         one bar per method and size:
         the mean, from the panel's floor (log axis); the whisker spans the replicates
Outputs: img/fig2/figS1c_scaling_host_memory.{pdf,svg}, figS2c_..., figS3c_...

    pixi run -e paper python -m plots.figS1c_scaling_host_memory
"""

# pylint: disable=invalid-name  # figS<N>: the paper's numbering of its supplementary figures

import matplotlib.pyplot as plt

from plots._config import SCALING_FIGURES, STYLE
from plots.utils import MIB_PER_GIB, method_bars, method_stats, parser, read_tsv, save

FIGSIZE = (7.2, 2.1)  # one row of the full-width four-panel page


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    rows = read_tsv(args.tables_dir, "scaling_methods")
    for dataset, figure in SCALING_FIGURES.items():
        stats = method_stats(rows, dataset, "host_memory_mb", 1 / MIB_PER_GIB)
        if not stats:
            print(f"no host_memory_mb data for {dataset}: {figure}c not drawn")
            continue
        fig, ax = plt.subplots(figsize=FIGSIZE)
        fig.subplots_adjust(left=0.09, right=0.985, top=0.95, bottom=0.2)
        method_bars(ax, stats, "host")
        ax.set_ylabel("Host memory (GiB)")
        save(fig, args.img_dir, f"{figure}c_scaling_host_memory")


if __name__ == "__main__":
    main()
