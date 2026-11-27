"""Fig. S4: bootstrap support against the true tree, on RNASim, the SARS-CoV-2-like simulation and
the random tree: (a) the ROC of exact bootstrap support, a branch being correct when its split is in
the true tree; (b) the share of branches in the true tree against their support.

Inputs:  data/tables/roc.tsv, data/tables/calibration.tsv               (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    a: each dataset's true against false positive rate as the support threshold falls, its AUC
           |  in the key, and the points of the THRESHOLDS
           v
         b: each dataset's support bins with at least MIN_BRANCHES branches, mean support of the bin
            (x) against its share of branches in the true tree (y), with the diagonal of perfect
            calibration
Outputs: img/fig2/figS4_bootstrap_calibration.{pdf,svg,png}

    pixi run -e paper python -m plots.figS4_bootstrap_calibration
"""

import matplotlib.pyplot as plt
import numpy as np

from plots._config import FIG2_PANEL, SCALING_SERIES, STYLE
from plots.utils import parser, read_tsv, save

FIGSIZE = (2 * FIG2_PANEL[0], FIG2_PANEL[1] + 12 / 25.4)  # as Fig. S5: two panels, the key above
MIN_BRANCHES = 50  # bins with fewer branches are left out: their share is noise
# Support thresholds marked on the ROC (%), as in the text: filled, then open marker
THRESHOLDS = {95: True, 70: False}
# The datasets as Fig. 2 draws them (RNASim and the random tree); the pandemic simulation in the MAPLE purple
STYLES = {  # dataset in the table -> legend name, colour, marker
    "RNASim": ("RNASim", SCALING_SERIES[0][2], "s"),
    "SARS-CoV-2-like simulation": ("SARS-CoV-2-like", "#7570B3", "o"),
    "Random tree": ("Random tree", SCALING_SERIES[2][2], "^"),
}
TICKS = [0, 25, 50, 75, 100]


def draw_roc(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    """Draws panel a, one ROC per dataset with its AUC and thresholds: (axes, rows) -> None."""
    ax.plot([0, 100], [0, 100], linestyle="--", color="#999999", linewidth=0.8, zorder=1)
    for dataset, (name, colour, marker) in STYLES.items():
        points = sorted(
            (r for r in rows if r["dataset"] == dataset),
            key=lambda r: float(r["support_pct"]),
            reverse=True,
        )
        if not points:
            raise SystemExit(f"roc.tsv: no point of {dataset}")
        support = np.array([float(r["support_pct"]) for r in points])
        # from (0, 0), nothing kept, to (100, 100), every branch kept
        tpr = np.r_[0.0, [float(r["tpr_pct"]) for r in points]]
        fpr = np.r_[0.0, [float(r["fpr_pct"]) for r in points]]
        if np.any(np.diff(tpr) < 0) or np.any(np.diff(fpr) < 0) or min(tpr[-1], fpr[-1]) < 99.99:
            raise SystemExit(f"roc.tsv: the points of {dataset} are not a ROC from 0 to 100 %")
        auc = np.trapezoid(tpr, fpr) / 100**2
        ax.plot(
            fpr,
            tpr,
            color=colour,
            linewidth=1.1,
            label=f"{name}, {int(points[0]['n']):,} (AUC {auc:.3f})",
            zorder=3,
        )
        for threshold, filled in THRESHOLDS.items():
            k = int(np.sum(support >= threshold))  # the branches kept: support >= threshold
            ax.plot(
                fpr[k],
                tpr[k],
                marker=marker,
                markersize=4.2 if filled else 3.4,
                color=colour,
                markerfacecolor=colour if filled else "white",
                linestyle="none",
                zorder=4,
            )
    ax.set_xlabel("False positive rate (%)\n(wrong branches kept)")
    ax.set_ylabel("True positive rate (%)\n(correct branches kept)")


def draw_calibration(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    """Draws panel b, the share of branches in the true tree per bin: (axes, rows) -> None."""
    ax.plot([0, 100], [0, 100], linestyle="--", color="#999999", linewidth=0.8, zorder=1)
    for dataset, (_, colour, marker) in STYLES.items():
        points = [r for r in rows if r["dataset"] == dataset and int(r["branches"]) >= MIN_BRANCHES]
        if not points:
            raise SystemExit(
                f"calibration.tsv: no bin of {dataset} with {MIN_BRANCHES} branches or more"
            )
        ax.plot(
            [float(r["mean_support_pct"]) for r in points],
            [float(r["correct_pct"]) for r in points],
            color=colour,
            marker=marker,
            markersize=3.4,
            linewidth=1.1,
            zorder=3,
        )
    ax.set_xlabel("Bootstrap support (%)")
    ax.set_ylabel("Branches in the true tree (%)")


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    fig.subplots_adjust(left=0.1, right=0.97, bottom=0.2, top=0.76, wspace=0.3)
    draw_roc(axes[0], read_tsv(args.tables_dir, "roc"))
    draw_calibration(axes[1], read_tsv(args.tables_dir, "calibration"))
    for ax, letter in zip(axes, "ab"):
        ax.set_xlim(0, 101)
        ax.set_ylim(0, 101)
        ax.set_xticks(TICKS)
        ax.set_yticks(TICKS)
        ax.grid(color="#E5E5E5", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.text(-0.2, 1.0, letter, transform=ax.transAxes, fontweight="bold", va="bottom")
    # One key above both panels: the datasets with their AUCs, then the threshold markers
    for threshold, filled in THRESHOLDS.items():
        axes[0].plot(
            [],
            [],
            marker="o",
            markersize=3.8,
            color="black",
            markerfacecolor="black" if filled else "white",
            linestyle="none",
            label=f"kept at support ≥ {threshold}%",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper left", bbox_to_anchor=(0.03, 1.0), ncol=2, handlelength=1.6
    )
    save(fig, args.img_dir, "figS4_bootstrap_calibration", bbox_inches=None)


if __name__ == "__main__":
    main()
