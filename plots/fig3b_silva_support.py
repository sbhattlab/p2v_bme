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
    x = range(len(bins))
    fig, ax = plt.subplots(figsize=(4.3, 6.3))
    fig.subplots_adjust(left=0.2, right=0.97, top=0.95, bottom=0.17)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    q25 = [float(b["q25_tbe_pct"]) for b in bins]
    q75 = [float(b["q75_tbe_pct"]) for b in bins]
    ax.fill_between(x, q25, q75, color=COLOUR, alpha=0.18, label="Interquartile range")
    ax.plot(
        x,
        [float(b["median_tbe_pct"]) for b in bins],
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
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.25, len(bins) - 0.65)
    ax.set_ylim(0, 104)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Split size (branches in brackets)", fontweight="bold")
    ax.set_ylabel("Transfer bootstrap support (%)", fontweight="bold")
    ax.grid(True, ls=":", alpha=0.45, color="#cbd5e0")
    ax.legend(
        loc="upper left",
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#cbd5e0",
    )
    save(fig, args.img_dir, "fig3b_silva_support")


if __name__ == "__main__":
    main()
