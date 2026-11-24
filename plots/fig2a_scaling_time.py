"""Fig. 2a: timed inference against the number of sequences, on the three simulated families.

Inputs:  data/tables/scaling.tsv, random_tree.tsv                      (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    mean and sample SD of the time at each size, one series per dataset
Outputs: img/fig2/fig2a_scaling_time.{pdf,svg}

    pixi run -e paper python -m plots.fig2a_scaling_time
"""

import matplotlib.pyplot as plt

from plots._config import STYLE
from plots.utils import parser, save, scaling_axes, scaling_data, scaling_series

FIGSIZE = (
    52 / 25.4,
    62 / 25.4,
)  # a quarter of the 180 mm page, as in the paper's four-panel row


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.27, right=0.97, bottom=0.21, top=0.86)
    scaling_series(ax, scaling_data(args.tables_dir), "time_s")
    scaling_axes(ax)
    ax.set_title("Timed inference", loc="left", pad=8)
    ax.set_ylabel("Time (s)")
    save(fig, args.img_dir, "fig2a_scaling_time")


if __name__ == "__main__":
    main()
