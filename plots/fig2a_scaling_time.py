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

from plots._config import STYLE
from plots.utils import method_panel, parser

DATASET = "rnasim"


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    method_panel(args, DATASET, "time", "RNASim", "fig2a_scaling_time")


if __name__ == "__main__":
    main()
