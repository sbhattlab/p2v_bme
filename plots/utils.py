"""Helpers several plot scripts share:
reading figure data, saving figures, the line plots of Figs 2a, 2c and S1-S3."""

import re
import statistics

from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from pathlib import Path

import matplotlib
import pandas as pd

# The backend is set before pyplot is imported, hence the imports below it
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

from matplotlib.lines import Line2D  # pylint: disable=wrong-import-position
from matplotlib.ticker import (  # pylint: disable=wrong-import-position
    FixedLocator,
    FuncFormatter,
    LogFormatterMathtext,
    LogLocator,
    MultipleLocator,
    NullFormatter,
    NullLocator,
)

from plots._config import (  # pylint: disable=wrong-import-position
    DEFAULT_IMG_DIR,
    DEFAULT_TABLES_DIR,
    FIG2_PANEL,
    FIGURE_FOLDERS,
    FIGURE_FORMATS,
    PANDEMIC,
    SAVE_METADATA,
    SCALING_FIGURES,
    SPEC,
)


def parser(doc: str) -> ArgumentParser:
    """Builds a plot script's command line: docstring -> parser with --tables-dir and --img-dir."""
    p = ArgumentParser(description=doc, formatter_class=RawDescriptionHelpFormatter)
    p.add_argument(
        "--tables-dir",
        type=Path,
        default=Path(DEFAULT_TABLES_DIR),
        help="figure data (make_tables.py)",
    )
    p.add_argument("--img-dir", type=Path, default=Path(DEFAULT_IMG_DIR), help="output folder")
    return p


def read_tsv(tables_dir: Path, name: str) -> list[dict[str, str]]:
    """Reads a figure-data file: (folder, name) -> rows;
    stops if make_tables.py has not written it."""
    path = tables_dir / f"{name}.tsv"
    if not path.exists():
        raise SystemExit(f"{path} is missing: run python -m paper.make_tables first")
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return [{str(k): v for k, v in row.items()} for row in table.to_dict("records")]


def save(fig: plt.Figure, img_dir: Path, name: str, **kwargs) -> None:
    """Saves a figure once per format in its figure's folder, without timestamps:
    (figure, folder, name) -> img_dir/<FIGURE_FOLDERS>/name.{pdf,svg,png}."""
    figure = re.match(r"figS?\d+", name)
    if figure is None or figure[0] not in FIGURE_FOLDERS:
        raise SystemExit(f"{name}: no folder for this figure in FIGURE_FOLDERS (plots/_config.py)")
    img_dir = img_dir / FIGURE_FOLDERS[figure[0]]
    img_dir.mkdir(parents=True, exist_ok=True)
    for ext in FIGURE_FORMATS:
        fig.savefig(
            img_dir / f"{name}.{ext}",
            dpi=300,
            metadata=SAVE_METADATA.get(ext),
            **kwargs,
        )
    plt.close(fig)
    print(f"wrote {img_dir / name}.{{{','.join(FIGURE_FORMATS)}}}")


# ---- Fig. 3a: the pandemic simulations ----

PANDEMIC_SIZES = (100_000, 200_000, 500_000)
PANDEMIC_BRANCH_LENGTHS = (5e-5, 1e-4, 1.5e-4)  # mean, in substitutions per site


def pandemic_rows(tables_dir: Path) -> list[dict[str, str]]:
    """Reads Fig. 3a's data, one row per program, size and mean branch length;
    stops unless every program has exactly one of each: folder -> rows of maple.tsv."""
    rows = read_tsv(tables_dir, "maple")
    cells = [(r["method"], int(r["n"]), float(r["mean_branch_length"])) for r in rows]
    expected = [
        (m, n, b) for m, _ in PANDEMIC for n in PANDEMIC_SIZES for b in PANDEMIC_BRANCH_LENGTHS
    ]
    if sorted(cells) != sorted(expected):
        missing = sorted(set(expected) - set(cells))
        extra = sorted(c for c in set(cells) if cells.count(c) > 1 or c not in expected)
        raise SystemExit(f"maple.tsv: missing {missing}; duplicated or unexpected {extra}")
    return rows


# ---- Figs S1-S3: one line per method over the number of sequences ----

PLOTTED = [
    "FastTree-ME",
    "FastTree-ML",
    "VFT-ME",
    "VFT-ML",
    "DIPPER",
    "P2V",
]  # drawing order; the legend lists them the other way round, VecME first
FLOORS = {
    "time": 1e-2,
    "rf": 0.01,
    "host": 2e-2,
    "gpu": 0.1,
}  # lower axis limits
TOPS = {"time": 3e4, "rf": 100, "host": 1000, "gpu": 100}  # RF error ends at 100 %
# The new method is drawn stronger than the methods it is compared with
OURS = "P2V"
COMPARED_ALPHA = 0.6
# From 10^6 sequences, the PASTA-tree datasets expand the 100,000-sequence alignment
# with small clades, so their RF error is not on a par with the smaller sizes.
# The RF line steps there with a dashed segment.
EXPANDED_FROM = 1_000_000
# The expanded datasets, keyed as in scaling_methods.tsv, and their name in scaling.tsv,
# which holds VecME's error on the splits they inherit from the 100,000-sequence alignment
EXPANDED_DATASETS = {"rnasim": "RNASim", "jc10k": "JC"}
INHERITED_ALPHA = 0.35  # the inherited splits are secondary: lighter than every method
LABEL_GREY = "#555555"  # text that names a panel's data, quieter than the axis labels
MIB_PER_GIB = 1024.0
# Each panel's column in scaling_methods.tsv, its scale and its y label
PANELS = {
    "time": ("time_s", 1.0, "Runtime (s)"),
    "rf": ("rf_pct", 1.0, "RF error (%)"),
    "host": ("host_memory_mb", 1 / MIB_PER_GIB, "Host memory (GiB)"),
    "gpu": ("gpu_memory_peak_mib", 1 / MIB_PER_GIB, "GPU memory (GiB)"),
}
GPU_PROGRAMS = ("P2V", "DIPPER")  # the methods with a GPU memory; the others run on CPUs only
# One full-width page per dataset: the four panels in two rows, the methods' legend above
SCALING_PAGE = (7.2, 5.4)


def method_stats(rows: list[dict[str, str]], dataset: str, field: str, scale: float) -> dict:
    """Summarises one field per method and size:
    (rows, dataset, field) -> {(n, method): (mean, min, max)}.

    Every run must have the field, except the CPU programs' runs for GPU memory (left out).
    """
    values: dict[tuple[int, str], list[float]] = {}
    blank: dict[str, list[str]] = {}
    for r in rows:
        if r["dataset"] != dataset or r["method"] not in PLOTTED:
            continue
        if field == PANELS["gpu"][0] and r["method"] not in GPU_PROGRAMS:
            continue
        if r[field] == "":
            blank.setdefault(r["method"], []).append(r["n"])
        else:
            values.setdefault((int(r["n"]), r["method"]), []).append(float(r[field]) * scale)
    if blank:
        raise SystemExit(f"{dataset}: runs without {field}, by method (sizes): {blank}")
    return {k: (statistics.mean(v), min(v), max(v)) for k, v in values.items()}


def method_style(key: str) -> dict:
    """Gives a method's line in the line plots (Figs 2a, 2c, S1-S3, S5):
    key -> line keywords, VecME's stronger."""
    ours = key == OURS
    return {
        "color": SPEC[key]["color"],
        "marker": SPEC[key]["marker"],
        "linestyle": SPEC[key]["ls"],
        "markersize": 4.6 if ours else 3.0,
        "linewidth": 2.2 if ours else 0.9,
        "alpha": 1.0 if ours else COMPARED_ALPHA,
    }


def method_lines(ax: plt.Axes, stats: dict, panel: str, step_at: int | None = None) -> None:
    """Draws each method's mean at each size as a line on log-log axes, with the range as whiskers:
    (axes, stats, panel, step) -> None.

    step_at: the size from which the dataset is built differently; the line reaches it dashed.
    """
    # the panel's fixed range, so that S1-S3 compare;
    # widened (never clipped) if data fall outside it
    lows = [lo for _mu, lo, _hi in stats.values()]
    highs = [hi for _mu, _lo, hi in stats.values()]
    bottom, top = min(FLOORS[panel], min(lows) / 2), max(TOPS[panel], max(highs) * 2)
    for key in PLOTTED:
        points = sorted((n, v) for (n, k), v in stats.items() if k == key)
        if not points:
            continue
        n = [p[0] for p in points]
        mu = [p[1][0] for p in points]
        whiskers = [[p[1][0] - p[1][1] for p in points], [p[1][2] - p[1][0] for p in points]]
        style = method_style(key)
        zorder = 5 if key == OURS else 3
        ax.errorbar(
            n,
            mu,
            yerr=whiskers,
            color=style["color"],
            marker=style["marker"],
            markersize=style["markersize"],
            alpha=style["alpha"],
            linestyle="none",  # the line is drawn below, in pieces
            elinewidth=0.9 if key == OURS else 0.6,
            capsize=1.8,
            zorder=zorder,
        )
        # the line: solid within each design, dashed across the step between them
        cut = next((i for i, x in enumerate(n) if step_at is not None and x >= step_at), len(n))
        line = {k: style[k] for k in ("color", "linewidth", "alpha")}
        for xs, ys, ls in (
            (n[:cut], mu[:cut], style["linestyle"]),
            (n[cut - 1 : cut + 1] if 0 < cut < len(n) else [], mu[cut - 1 : cut + 1], "--"),
            (n[cut:], mu[cut:], style["linestyle"]),
        ):
            if len(xs) > 1:
                ax.plot(xs, ys, linestyle=ls, zorder=zorder, **line)
    ax.set_xscale("log")
    ax.set_xlim(650, 1.5e7)
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Number of sequences")
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(bottom, top)
    if panel != "time":
        ax.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 100, 1000]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.grid(axis="y", which="major", color="#E5E5E5", linewidth=0.5)
    ax.set_axisbelow(True)


def method_legend(fig: plt.Figure, stats: dict) -> None:
    """Adds the methods drawn, as a figure legend above the panel: (figure, stats) -> None."""
    present = [k for k in PLOTTED if any(key == k for _, key in stats)]
    handles = [Line2D([], [], label=SPEC[k]["label"], **method_style(k)) for k in reversed(present)]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.53, 0.998),
        handletextpad=0.4,
        columnspacing=2.0,
        frameon=False,
    )


def inherited_splits(ax: plt.Axes, tables_dir: Path, name: str) -> None:
    """Draws VecME's error on the splits that an expanded dataset inherits, with its key:
    (axes, folder, name in scaling.tsv) -> None.

    The line starts at the 100,000-sequence alignment that the larger sizes expand.
    """
    rows = read_tsv(tables_dir, "scaling")
    group = sorted((r for r in rows if r["dataset"] == name), key=lambda r: int(r["n"]))
    if len({r["n"] for r in group}) != len(group):
        raise SystemExit(f"scaling.tsv: {name} has more than one row per size")
    expanded = [i for i, r in enumerate(group) if r["large_scale_design"]]
    if not expanded or expanded[0] == 0:
        raise SystemExit(f"scaling.tsv: {name} has no expanded sizes (large_scale_design)")
    ks = [expanded[0] - 1, *expanded]  # from the expanded alignment's own size
    n = [int(group[k]["n"]) for k in ks]
    inherited = [float(group[k]["backbone_rf_pct_mean"]) for k in ks]
    colour, marker = SPEC[OURS]["color"], SPEC[OURS]["marker"]
    ax.plot(
        n,
        inherited,
        color=colour,
        linestyle=":",
        linewidth=1.0,
        alpha=INHERITED_ALPHA,
        zorder=4,
    )
    ax.plot(
        n[1:],
        inherited[1:],
        linestyle="none",
        marker=marker,
        markersize=3.4,
        markerfacecolor="white",
        markeredgecolor=colour,
        markeredgewidth=0.9,
        alpha=INHERITED_ALPHA,
        zorder=4,
    )
    key = Line2D(
        [],
        [],
        color=colour,
        linestyle=":",
        linewidth=1.0,
        marker=marker,
        markersize=3.4,
        markerfacecolor="white",
        markeredgecolor=colour,
        markeredgewidth=0.9,
        alpha=INHERITED_ALPHA,
        label="VecME, inherited splits only",
    )
    ax.legend(handles=[key], loc="lower left", handlelength=1.9, borderaxespad=0.3)


def method_legend_inside(ax: plt.Axes, stats: dict) -> None:
    """Adds the methods drawn as one column in the lower right of a panel (Figs 2a, 2c),
    below the lines, where the log axes leave room: (axes, stats) -> None."""
    present = [k for k in PLOTTED if any(key == k for _, key in stats)]
    handles = [Line2D([], [], label=SPEC[k]["label"], **method_style(k)) for k in reversed(present)]
    ax.legend(
        handles=handles,
        loc="lower right",
        handlelength=1.8,
        handletextpad=0.4,
        labelspacing=0.3,
        borderaxespad=0.2,
    )


def data_label(ax: plt.Axes, data: str) -> None:
    """Names a panel's data in its upper left corner, as "Data: <data>": (axes, data) -> None."""
    ax.text(
        0.03, 0.97, f"Data: {data}", transform=ax.transAxes, ha="left", va="top", color=LABEL_GREY
    )


def data_label_above(ax: plt.Axes, data: str) -> None:
    """Names a panel's data just above its axes, at the left, as "Data: <data>": (axes, data) -> None."""
    ax.text(
        0.0, 1.03, f"Data: {data}", transform=ax.transAxes, ha="left", va="bottom", color=LABEL_GREY
    )


def method_panel(  # pylint: disable=too-many-arguments
    args: Namespace,
    dataset: str,
    panel: str,
    data: str,
    name: str,
    *,
    top: float,
    legend: bool = True,
    linear_y: tuple[float, float] | None = None,
) -> None:
    """Draws a Fig. 2 panel of every method over the number of sequences (2a, 2c), with the data's
    name inside: (args, dataset, panel, data, name) -> img_dir/fig2/name.*.

    top: the axes' top edge, as the other panel of the row's.
    legend: the methods' legend inside; without it, the panel relies on another panel's legend.
    linear_y: (axis top, tick step) for a linear y axis from 0 instead of the log one."""
    field, scale, ylabel = PANELS[panel]
    stats = method_stats(read_tsv(args.tables_dir, "scaling_methods"), dataset, field, scale)
    if not stats:
        raise SystemExit(f"no {field} data for {dataset}")
    fig, ax = plt.subplots(figsize=FIG2_PANEL)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.16, top=top)
    method_lines(ax, stats, panel)
    if linear_y is not None:
        y_top, step = linear_y
        if (highest := max(hi for _mean, _lo, hi in stats.values())) > y_top:
            raise SystemExit(f"{dataset}: {field} up to {highest:g}, beyond the axis ({y_top:g})")
        ax.set_yscale("linear")
        ax.set_ylim(0, y_top)
        ax.yaxis.set_major_locator(MultipleLocator(step))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.set_ylabel(ylabel)
    if legend:
        method_legend_inside(ax, stats)
    data_label(ax, data)
    save(fig, args.img_dir, name)


def method_pages(args: Namespace) -> None:
    """Draws Figs S1-S3, one page per dataset:
    runtime, RF error, host and GPU memory of every method over the number of sequences;
    on the expanded datasets (S1, S2), VecME's error on the inherited splits too:
    args -> img_dir/<folder>/figS<N>_scaling.{pdf,svg,png}."""
    rows = read_tsv(args.tables_dir, "scaling_methods")
    for dataset, figure in SCALING_FIGURES.items():
        fig, axes = plt.subplots(2, 2, figsize=SCALING_PAGE, sharex=True)
        fig.subplots_adjust(left=0.09, right=0.985, top=0.87, bottom=0.09, hspace=0.12, wspace=0.27)
        drawn: dict = {}
        for ax, letter, panel in zip(axes.flat, "abcd", PANELS):
            field, scale, ylabel = PANELS[panel]
            ax.text(-0.2, 1.0, letter, transform=ax.transAxes, fontweight="bold", va="bottom")
            stats = method_stats(rows, dataset, field, scale)
            if not stats:
                raise SystemExit(f"no {field} data for {dataset}: {figure}{letter} would be empty")
            step = EXPANDED_FROM if panel == "rf" and dataset in EXPANDED_DATASETS else None
            method_lines(ax, stats, panel, step)
            if step is not None:
                inherited_splits(ax, args.tables_dir, EXPANDED_DATASETS[dataset])
            ax.set_ylabel(ylabel)
            drawn.update(stats)
        for ax in axes[0]:
            ax.set_xlabel("")  # the x axis is shared: labelled once, under the bottom row
        method_legend(fig, drawn)
        save(fig, args.img_dir, f"{figure}_scaling")
