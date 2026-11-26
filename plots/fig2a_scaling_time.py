"""Fig. 2a: timed inference against the number of sequences, on the three simulated families.

Inputs:  data/tables/scaling.tsv, random_tree.tsv                      (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    mean and sample SD of the time at each size, one series per dataset
Outputs: img/fig2/fig2a_scaling_time.{pdf,svg,png}

    pixi run -e paper python -m plots.fig2a_scaling_time
"""

import matplotlib.pyplot as plt

from plots._config import FIG2_PANEL, STYLE
from plots.utils import parser, save, scaling_axes, scaling_data, scaling_series


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=FIG2_PANEL)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.16, top=0.96)  # left, right as in Fig. 2c
    scaling_series(ax, scaling_data(args.tables_dir), "time_s")
    scaling_axes(ax)
    ax.set_ylabel("Time (s)")
    save(fig, args.img_dir, "fig2a_scaling_time")


if __name__ == "__main__":
    main()
