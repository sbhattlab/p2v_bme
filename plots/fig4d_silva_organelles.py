"""Fig. 4d: where the SILVA 144 tree places the sequences SILVA files under Bacteria as organelles:
the share of mitochondria, chloroplasts, the other bacteria and the archaea within the tree's
Eukaryota clade.

Inputs:  data/tables/silva_organelles.tsv                                (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    one bar per group of sequences by SILVA's label: the share placed within Eukaryota,
         with the counts beside it; the organelles in Eukaryota's colour of Fig. 4a;
         a note gives the share of those organelle sequences that sit inside Chloroplastida
Outputs: img/fig4/fig4d_silva_organelles.{pdf,svg,png}

    pixi run -e paper python -m plots.fig4d_silva_organelles
"""

import matplotlib.pyplot as plt

from plots._config import STYLE
from plots.utils import parser, read_tsv, save

FIGSIZE = (88 / 25.4, 71 / 25.4)  # Fig. 4c's size: the two share a row
# group in the table -> label, top to bottom
GROUPS = {
    "mitochondria": "Mitochondria",
    "chloroplasts": "Chloroplasts",
    "other_bacteria": "Other Bacteria",
    "archaea": "Archaea",
}
ORGANELLES = ("mitochondria", "chloroplasts")
EUKARYOTA, OTHER = "#8c0800", "#BBBBBB"  # Eukaryota's colour in Fig. 4a; the other groups
NOTE, NOTE_SIZE = "#4a5568", 7  # the grey of Fig. 4a's branches; points, as Fig. 4a's supports
COUNT_GAP = 1.5  # percentage points between a bar and its counts


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    rows = {r["group"]: r for r in read_tsv(args.tables_dir, "silva_organelles")}
    if set(rows) != set(GROUPS):
        raise SystemExit(f"silva_organelles.tsv: groups {sorted(rows)}, not {sorted(GROUPS)}")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.3, right=0.96, bottom=0.2, top=0.88)
    for y, group in enumerate(reversed(GROUPS)):
        k, n = int(rows[group]["in_eukaryota"]), int(rows[group]["total"])
        share = 100 * k / n
        ax.barh(y, share, height=0.6, color=EUKARYOTA if group in ORGANELLES else OTHER)
        ax.text(share + COUNT_GAP, y, f"{k:,} of {n:,}", va="center", fontsize=NOTE_SIZE)
    ax.set_yticks(range(len(GROUPS)), list(reversed(GROUPS.values())))
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Placed within Eukaryota (%)")
    ax.text(
        0,
        len(GROUPS) - 0.3,
        "SILVA label (organelles are filed under Bacteria)",
        fontsize=NOTE_SIZE,
        color=NOTE,
        va="bottom",
    )
    inside = sum(int(rows[g]["in_chloroplastida"]) for g in ORGANELLES)
    placed = sum(int(rows[g]["in_eukaryota"]) for g in ORGANELLES)
    ax.text(
        99,
        0.9,
        f"{100 * inside / placed:.1f}% of the organelle\nsequences in Eukaryota\n"
        "sit inside Chloroplastida",
        fontsize=NOTE_SIZE,
        color=NOTE,
        ha="right",
        va="center",
    )
    save(fig, args.img_dir, "fig4d_silva_organelles")


if __name__ == "__main__":
    main()
