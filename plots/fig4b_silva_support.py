"""Fig. 4b: transfer bootstrap support of the internal branches of the SILVA 144 tree,
by the size of their split.

Inputs:  data/tables/silva_support.tsv                                  (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    per bin of split size (sequences on the smaller side): the share of its branches with at
         least 90%, 70-90% and under 70% transfer bootstrap support, as one stacked bar;
         the key above is a small stacked bar in the same shades
Outputs: img/fig4/fig4b_silva_support.{pdf,svg,png}

    pixi run -e paper python -m plots.fig4b_silva_support
"""

import matplotlib.pyplot as plt
import numpy as np

from plots._config import STYLE
from plots.utils import parser, read_tsv, save

# Fig. 4a's height; 64 mm wide: with 4a (112 mm) and a 4 mm gap, the 180 mm page
FIGSIZE = (64 / 25.4, 117 / 25.4)
LABELS = {
    "10001-2106464": ">10,000",
    ">10k": ">10,000",
    "1k-10k": "1,001–10,000",
    "1001-10000": "1,001–10,000",
    "101-1k": "101–1,000",
    "101-1000": "101–1,000",
    "3-10": "3–10",
    "11-100": "11–100",
    "2": "2",
}
# Support classes, dark to light, so that the well-supported share is read from the left;
# 70% is the threshold of Fig. 3d. Shades of Bacteria's colour in Fig. 4a.
CLASSES = (("≥ 90%", "#1f6f8b"), ("70–90%", "#79AFC4"), ("< 70%", "#DCE9EF"))
EDGE_COLOUR, EDGE_WIDTH = "white", 0.8  # gaps between the segments


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    bins = read_tsv(args.tables_dir, "silva_support")
    ge90 = np.array([float(b["frac_ge90_pct"]) for b in bins])
    ge70 = np.array([float(b["frac_ge70_pct"]) for b in bins])
    shares = (ge90, ge70 - ge90, 100 - ge70)
    y = np.arange(len(bins))  # one row per bin, smallest splits at the bottom

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes((0.40, 0.11, 0.55, 0.70))
    left = np.zeros(len(bins))
    for share, (_label, colour) in zip(shares, CLASSES):
        ax.barh(
            y,
            share,
            left=left,
            color=colour,
            height=0.72,
            edgecolor=EDGE_COLOUR,
            linewidth=EDGE_WIDTH,
        )
        left += share
    ax.set_yticks(y, [LABELS[b["split_size_bin"]] for b in bins])
    ax.set_ylim(-0.5, len(bins) - 0.5)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Branches (%)")
    ax.set_ylabel("Split size (sequences)")
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # the key: the three classes as a small stacked bar, labelled underneath
    key = fig.add_axes((0.40, 0.88, 0.55, 0.035))
    for i, (label, colour) in enumerate(CLASSES):
        key.barh(0, 1, left=i, color=colour, height=1, edgecolor=EDGE_COLOUR, linewidth=EDGE_WIDTH)
        key.text(i + 0.5, -0.9, label, ha="center", va="top")
    key.set_xlim(0, len(CLASSES))
    key.set_ylim(-0.5, 0.5)
    key.set_axis_off()
    key.text(len(CLASSES) / 2, 0.9, "Transfer bootstrap support", ha="center", va="bottom")
    save(fig, args.img_dir, "fig4b_silva_support")


if __name__ == "__main__":
    main()
