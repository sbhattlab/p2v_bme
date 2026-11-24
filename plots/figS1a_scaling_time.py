"""Figs S1a, S2a, S3a: running time of every method at each size,
on RNASim, JC10k and random-tree JC.

Inputs:  data/tables/scaling_methods.tsv                                (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    each dataset (rnasim -> S1, jc10k -> S2, yule10k -> S3)
           |
           v
         one bar per method and size:
         the mean, from the panel's floor (log axis); the whisker spans the replicates
Outputs: img/fig2/figS1a_scaling_time.{pdf,svg,png}, figS2a_..., figS3a_...

    pixi run -e paper python -m plots.figS1a_scaling_time
"""

# pylint: disable=invalid-name  # figS<N>: the paper's numbering of its supplementary figures

import matplotlib.pyplot as plt

from plots._config import STYLE
from plots.utils import method_figures, parser


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    method_figures(args, "time", "a_scaling_time", legend=True)


if __name__ == "__main__":
    main()
