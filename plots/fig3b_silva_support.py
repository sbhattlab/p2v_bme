"""Fig. 3b: transfer support of every internal branch of the SILVA 144 tree,
by the size of its split.

Inputs:  data/tables/silva_support.tsv                                  (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    per bin of split size (smaller side):
         median TBE and interquartile range, with the bin's branch count
Outputs: img/fig3/fig3b_silva_support.{pdf,svg,png}

    pixi run -e paper python -m plots.fig3b_silva_support
"""

import matplotlib.pyplot as plt

from plots._config import STYLE
from plots.utils import parser, read_tsv, save

COLOUR = "#1f6f8b"  # Bacteria's colour in Fig. 3a
LABELS = {
    "10001-2106464": ">10k",
    ">10k": ">10k",
    "1k-10k": "1,001–10k",
    "1001-10000": "1,001–10k",
    "101-1k": "101–1k",
    "101-1000": "101–1k",
    "3-10": "3–10",
    "11-100": "11–100",
}


def compact(n: int) -> str:
    """Writes a branch count to three significant figures: 1290000 -> '1.29M', 826000 -> '826k'."""
    return f"{n / 1e6:.3g}M" if n >= 1e6 else f"{n / 1e3:.3g}k"


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    bins = read_tsv(args.tables_dir, "silva_support")
    y = range(len(bins))  # one row per split-size bin, smallest splits at the top
    # 3a's height and the rest of the page width, so that the row 3a + 3b is as wide as 3c
    fig, ax = plt.subplots(figsize=(2.7, 4.6))
    fig.subplots_adjust(left=0.38, right=0.95, top=0.9, bottom=0.11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    q25 = [float(b["q25_tbe_pct"]) for b in bins]
    q75 = [float(b["q75_tbe_pct"]) for b in bins]
    ax.fill_betweenx(y, q25, q75, color=COLOUR, alpha=0.18, label="Interquartile range")
    ax.plot(
        [float(b["median_tbe_pct"]) for b in bins],
        y,
        "-o",
        color=COLOUR,
        lw=2.0,
        ms=5.5,
        label="Median",
    )
    labels = [
        f"{LABELS.get(b['split_size_bin'], b['split_size_bin'])}\n({compact(int(b['n_branches']))})"
        for b in bins
    ]
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_ylim(len(bins) - 0.6, -0.4)
    ax.set_xlim(0, 104)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("TBE (%)")
    ax.set_ylabel("Split size (branches)")
    ax.grid(True, ls=":", alpha=0.45, color="#cbd5e0")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False)
    save(fig, args.img_dir, "fig3b_silva_support")


if __name__ == "__main__":
    main()
