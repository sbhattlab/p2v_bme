"""Figs S1, S2, S3: running time, RF error, host and GPU memory of every method at each size,
on RNASim, JC10k and random-tree JC.

Inputs:  data/tables/scaling_methods.tsv                                (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    each dataset (rnasim -> S1, jc10k -> S2, yule10k -> S3)
           |
           v
         four panels (a runtime, b RF error, c host memory, d GPU memory), one line per method over
         the sizes (log-log axes): the mean at each size; the whisker spans the replicates
Outputs: img/fig2/figS1_scaling.{pdf,svg,png}, figS2_scaling..., figS3_scaling...

    pixi run -e paper python -m plots.figS1_scaling
"""

# pylint: disable=invalid-name  # figS<N>: the paper's numbering of its supplementary figures

import matplotlib.pyplot as plt

from plots._config import STYLE
from plots.utils import method_pages, parser


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    method_pages(args)


if __name__ == "__main__":
    main()
