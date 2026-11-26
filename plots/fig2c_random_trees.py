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

from plots._config import STYLE
from plots.utils import method_panel, parser

DATASET = "yule10k"


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    method_panel(args, DATASET, "rf", "random trees (JC69)", "fig2c_random_trees")


if __name__ == "__main__":
    main()
