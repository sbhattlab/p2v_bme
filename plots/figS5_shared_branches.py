"""Fig. S5: the branches of the VecME tree also found in another tree, by the substitutions on the
branch: a, SARS-CoV-2 genomes, in the MAPLE tree of the same genomes;
b, a simulated pandemic, in the true tree and in the MAPLE tree.

Inputs:  data/tables/viridian_substitutions.tsv, pandemic_branches.tsv  (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    per bin of substitutions on the branch (as the VecME tree estimates them):
         the share of the VecME tree's branches also in each other tree, one bar per tree;
         one key above both panels
Outputs: img/fig3/figS5_shared_branches.{pdf,svg,png}

    pixi run -e paper python -m plots.figS5_shared_branches
"""

# pylint: disable=invalid-name  # figS<N>: the paper's numbering of its supplementary figures

import matplotlib.pyplot as plt
import numpy as np

from plots._config import FIG2_PANEL, SPEC, STYLE
from plots.utils import data_label_above, parser, read_tsv, save

# Two of Fig. 2's panels side by side, and a row for the key above them
FIGSIZE = (2 * FIG2_PANEL[0], FIG2_PANEL[1] + 12 / 25.4)
BINS = {  # (from, to) substitutions on the branch, as in the data (to: None = no end) -> tick label
    (0.0, 0.5): "0",
    (0.5, 1.5): "1",
    (1.5, 2.5): "2",
    (2.5, 3.5): "3",
    (3.5, 5.5): "4–5",
    (5.5, None): "≥6",
}
# The other trees, in the key's order: legend entry (after "Also found in") and colour
OTHER_TREES = {
    "true": ("the true tree", "#000000"),
    "maple": ("the MAPLE tree", SPEC["MAPLE"]["color"]),
}
# Each panel: its data, the column of each other tree, and the data it names
PANELS = [
    ("viridian_substitutions", {"maple": "ours_in_maple_pct"}, "SARS-CoV-2, 2,072,111 genomes"),
    (
        "pandemic_branches",
        {"true": "in_true_pct", "maple": "in_maple_pct"},
        "simulated pandemic, 100,000 sequences",
    ),
]
GROUP_WIDTH = 0.8  # the share of a bin's width that its bars take
BAR_FILL = 0.92  # the share of a bar's slot that the bar fills, leaving a gap between neighbours


def bins_of(rows: list[dict[str, str]]) -> list[tuple[float, float | None]]:
    """Reads the bins of a figure-data file: rows -> [(from, to)], to None for the last bin."""
    return [
        (
            float(r["substitutions_from"]),
            float(r["substitutions_to"]) if r["substitutions_to"] else None,
        )
        for r in rows
    ]


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    fig.subplots_adjust(
        left=0.09, right=0.99, bottom=0.14, top=0.78, wspace=0.25
    )  # left as Figs S1-S3
    x = np.arange(len(BINS))
    for ax, letter, (name, columns, data) in zip(axes, "ab", PANELS):
        rows = read_tsv(args.tables_dir, name)
        if bins_of(rows) != list(BINS):
            raise SystemExit(
                f"{name}.tsv: the substitution bins are not the ones this figure labels"
            )
        width = GROUP_WIDTH / len(columns)
        for i, (tree, column) in enumerate(columns.items()):
            share = [float(r[column]) for r in rows]
            if not all(0.0 <= v <= 100.0 for v in share):
                raise SystemExit(f"{name}.tsv: {column} outside 0-100 %")
            label, colour = OTHER_TREES[tree]
            left = x - GROUP_WIDTH / 2 + width * (i + 0.5)
            ax.bar(left, share, width=width * BAR_FILL, color=colour, label=label)
        ax.set_xticks(x, list(BINS.values()))
        ax.set_xlabel("Substitutions on the branch")
        ax.set_ylabel("Branches of the VecME tree (%)")
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.text(-0.2, 1.0, letter, transform=ax.transAxes, fontweight="bold", va="bottom")
        data_label_above(ax, data)
    # One key for both panels, above them: panel b draws every other tree
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Also found in",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(labels),
        title_fontsize=9,
    )
    save(fig, args.img_dir, "figS5_shared_branches", bbox_inches=None)


if __name__ == "__main__":
    main()
