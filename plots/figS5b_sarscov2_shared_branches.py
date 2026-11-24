"""Fig. S5b: share of the branches of our SARS-CoV-2 tree that are in MAPLE's,
and of MAPLE's in ours.

Inputs:  data/tables/viridian_substitutions.tsv                         (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    per bin of substitutions on the branch:
         the share of branches also in the other tree, for each tree
Outputs: img/fig3/figS5b_sarscov2_shared_branches.{pdf,svg,png}

    pixi run -e paper python -m plots.figS5b_sarscov2_shared_branches
"""

# pylint: disable=invalid-name  # figS<N>: the paper's numbering of its supplementary figures

import matplotlib.pyplot as plt

from plots._config import SPEC, STYLE
from plots.utils import parser, read_tsv, save

FIGSIZE = (66 / 25.4, 66 / 25.4)  # panel b of the 150 mm wide figure
BINS = {
    (0.5, 1.5): "1",
    (1.5, 2.5): "2",
    (2.5, 3.5): "3",
    (3.5, 5.5): "4–5",
    (5.5, None): "≥6",
}


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    rows = read_tsv(args.tables_dir, "viridian_substitutions")
    keys = [
        (
            float(r["substitutions_from"]),
            float(r["substitutions_to"]) if r["substitutions_to"] else None,
        )
        for r in rows
    ]
    if keys != list(BINS):
        raise SystemExit("the substitution bins of the data are not the ones this figure labels")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.84, bottom=0.17)
    x = range(len(BINS))
    for field, label, colour, marker in (
        ("ours_in_maple_pct", "Ours, in MAPLE's tree", SPEC["P2V"]["color"], SPEC["P2V"]["marker"]),
        ("maple_in_ours_pct", "MAPLE's, in ours", SPEC["MAPLE"]["color"], SPEC["MAPLE"]["marker"]),
    ):
        ax.plot(
            x,
            [float(r[field]) for r in rows],
            color=colour,
            marker=marker,
            markersize=4.4,
            linestyle="-",
            linewidth=1.3,
            label=label,
            markeredgecolor=colour,
            markeredgewidth=1.1,
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(BINS.values()))
    ax.set_xlabel("Substitutions on the branch")
    ax.set_ylabel("Branches in the other tree (%)")
    ax.set_ylim(75, 100)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    ax.legend(loc="lower right", handletextpad=0.4, borderaxespad=0.2)
    save(fig, args.img_dir, "figS5b_sarscov2_shared_branches", bbox_inches=None)


if __name__ == "__main__":
    main()
