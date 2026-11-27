"""Fig. 4d: where the SILVA 144 tree places the sequences SILVA files under Bacteria as organelles:
for mitochondria, chloroplasts, the other bacteria and the archaea, the share placed within
the tree's Eukaryota clade against the number of sequences in the group.

Inputs:  data/tables/silva_organelles.tsv                                (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    one point per group of sequences by SILVA's label: x the share placed within Eukaryota,
         y the group's distinct sequences (log scale); each point labelled with its group;
         the organelles in Eukaryota's colour of Fig. 4a
Outputs: img/fig4/fig4d_silva_organelles.{pdf,svg,png}

    pixi run -e paper python -m plots.fig4d_silva_organelles
"""

import matplotlib.pyplot as plt

from plots._config import STYLE
from plots.utils import parser, read_tsv, save

FIGSIZE = (88 / 25.4, 71 / 25.4)  # Fig. 4c's size: the two share a row
# group in the table -> label
GROUPS = {
    "mitochondria": "Mitochondria",
    "chloroplasts": "Chloroplasts",
    "other_bacteria": "Other Bacteria",
    "archaea": "Archaea",
}
ORGANELLES = ("mitochondria", "chloroplasts")
EUKARYOTA, OTHER = "#8c0800", "#9A9A9A"  # Eukaryota's colour in Fig. 4a; the other groups
MARKER_SIZE = 7  # points
LABEL_OFFSET = (7, 0)  # points: each group's name, right of its point
Y_RANGE = (2e4, 1e7)  # distinct sequences: the groups span 37,084 to 3,361,885


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    rows = {r["group"]: r for r in read_tsv(args.tables_dir, "silva_organelles")}
    if set(rows) != set(GROUPS):
        raise SystemExit(f"silva_organelles.tsv: groups {sorted(rows)}, not {sorted(GROUPS)}")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.19, right=0.96, bottom=0.2, top=0.96)
    for group, label in GROUPS.items():
        k, n = int(rows[group]["in_eukaryota"]), int(rows[group]["total"])
        if not Y_RANGE[0] <= n <= Y_RANGE[1]:
            raise SystemExit(f"silva_organelles.tsv: {group}: {n:,} sequences, off the y-axis")
        share = 100 * k / n
        colour = EUKARYOTA if group in ORGANELLES else OTHER
        # groups with next to no sequences in Eukaryota sit on the y-axis: drawn over it
        ax.plot(share, n, "o", ms=MARKER_SIZE, color=colour, clip_on=False, zorder=3)
        ax.annotate(label, (share, n), xytext=LABEL_OFFSET, textcoords="offset points", va="center")
    ax.set_yscale("log")
    ax.set_ylim(*Y_RANGE)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Placed within Eukaryota (%)")
    ax.set_ylabel("Distinct sequences")
    ax.grid(color="#E5E5E5", linewidth=0.5)
    ax.set_axisbelow(True)
    save(fig, args.img_dir, "fig4d_silva_organelles")


if __name__ == "__main__":
    main()
