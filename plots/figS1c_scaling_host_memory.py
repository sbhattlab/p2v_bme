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

from plots._config import STYLE
from plots.utils import method_figures, parser


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    method_figures(args, "host", "c_scaling_host_memory")


if __name__ == "__main__":
    main()
