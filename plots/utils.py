"""Helpers several plot scripts share:
reading figure data, saving figures, the panels of Figs 2a,b and S1-S3."""

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
    NullFormatter,
)

from plots._config import (  # pylint: disable=wrong-import-position
    DEFAULT_IMG_DIR,
    DEFAULT_TABLES_DIR,
    FIGURE_FOLDERS,
    FIGURE_FORMATS,
    SAVE_METADATA,
    SCALING_FIGURES,
    SCALING_SERIES,
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


# ---- Figs 2a,b: one series per dataset over the number of sequences ----


def scaling_series(ax: plt.Axes, rows: list[dict[str, str]], field: str) -> None:
    """Draws a field's mean and SD per size for every dataset of SCALING_SERIES:
    (axes, rows, field) -> None."""
    for dataset, label, colour, marker, linestyle, filled in SCALING_SERIES:
        group = sorted((r for r in rows if r["dataset"] == dataset), key=lambda r: int(r["n"]))
        if not group:
            print(f"no data for {dataset}: its series is not drawn")
            continue
        n = [int(r["n"]) for r in group]
        mu = [float(r[f"{field}_mean"]) for r in group]
        sd = [float(r[f"{field}_sd"] or 0) for r in group]
        ax.errorbar(
            n,
            mu,
            yerr=sd,
            color=colour,
            marker=marker,
            markersize=3.4,
            linestyle=linestyle,
            markerfacecolor=colour if filled else "white",
            markeredgewidth=0.9,
            linewidth=1.1,
            elinewidth=0.75,
            capsize=2,
            capthick=0.7,
            label=label,
        )


def scaling_axes(ax: plt.Axes) -> None:
    """Sets the log-log axes of Figs 2a,b: axes -> None."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1e3, 1e5, 1e7])
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.set_xlabel("Number of sequences")
    ax.grid(axis="y", which="major", color="#E5E5E5", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(which="minor", length=2)
    ax.set_xlim(650, 1.5e7)


def scaling_data(tables_dir: Path) -> list[dict[str, str]]:
    """Reads the data of Figs 2a,b:
    folder -> sample-and-place rows of scaling.tsv and random_tree.tsv."""
    rows = read_tsv(tables_dir, "scaling") + read_tsv(tables_dir, "random_tree")
    rows = [r for r in rows if r["initialisation"] == "sample"]
    if len({(r["gpu_model"], r["gpu_count"]) for r in rows}) != 1:
        raise SystemExit("mixed hardware in the scaling data: update the caption")
    return rows


# ---- Figs S1-S3: one line per method over the number of sequences ----

PLOTTED = [
    "FastTree-ME",
    "FastTree-ML",
    "VFT-ME",
    "VFT-ML",
    "DIPPER",
    "P2V",
]  # drawing order; the legend lists them the other way round, Phylo2VecME first
FLOORS = {
    "time": 1e-2,
    "rf": 0.01,
    "host": 2e-2,
    "gpu": 0.1,
}  # lower axis limits
TOPS = {"time": 3e4, "rf": 130, "host": 1000, "gpu": 100}
# The new method is drawn stronger than the methods it is compared with
OURS = "P2V"
COMPARED_ALPHA = 0.6
# From 10^6 sequences, the PASTA-tree datasets expand the 100,000-sequence alignment
# with small clades, so their RF error is not on a par with the smaller sizes.
# The RF line steps there with a dashed segment.
EXPANDED_FROM = 1_000_000
EXPANDED_DATASETS = ("rnasim", "jc10k")
MIB_PER_GIB = 1024.0
# Each panel's column in scaling_methods.tsv, its scale and its y label
PANELS = {
    "time": ("time_s", 1.0, "Runtime (s)"),
    "rf": ("rf_pct", 1.0, "RF error (%)"),
    "host": ("host_memory_mb", 1 / MIB_PER_GIB, "Host memory (GiB)"),
    "gpu": ("gpu_memory_peak_mib", 1 / MIB_PER_GIB, "GPU memory (GiB)"),
}
# One row of the full-width four-panel page; the panel with the methods' legend is taller
METHOD_FIGSIZE = (7.2, 2.1)
METHOD_FIGSIZE_LEGEND = (7.2, 2.6)


def method_stats(rows: list[dict[str, str]], dataset: str, field: str, scale: float) -> dict:
    """Summarises one field per method and size:
    (rows, dataset, field) -> {(n, method): (mean, min, max)}."""
    values: dict[tuple[int, str], list[float]] = {}
    for r in rows:
        if r["dataset"] == dataset and r["method"] in PLOTTED and r[field] != "":
            values.setdefault((int(r["n"]), r["method"]), []).append(float(r[field]) * scale)
    return {k: (statistics.mean(v), min(v), max(v)) for k, v in values.items()}


def method_style(key: str) -> dict:
    """Gives a method's line in the line plots (Figs 2c, S1-S3, S5b):
    key -> line keywords, Phylo2VecME's stronger."""
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
    for (n, key), (mu, lo, hi) in stats.items():
        if lo <= FLOORS[panel] or hi >= TOPS[panel]:
            raise SystemExit(
                f"{key} N={n}: {lo}..{hi} outside the {panel} axis {FLOORS[panel]}..{TOPS[panel]}"
            )
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
    ax.set_ylim(FLOORS[panel], TOPS[panel])
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


def method_figures(args: Namespace, panel: str, name: str, legend: bool = False) -> None:
    """Draws one panel of Figs S1-S3, every method's line over the sizes:
    (args, panel, name) -> img_dir/<folder>/figS<N><name>.{pdf,svg,png} for each dataset.

    name starts with the panel's letter (a_scaling_time); legend adds the methods' legend above.
    """
    field, scale, ylabel = PANELS[panel]
    rows = read_tsv(args.tables_dir, "scaling_methods")
    for dataset, figure in SCALING_FIGURES.items():
        stats = method_stats(rows, dataset, field, scale)
        if not stats:
            print(f"no {field} data for {dataset}: {figure}{name[0]} not drawn")
            continue
        fig, ax = plt.subplots(figsize=METHOD_FIGSIZE_LEGEND if legend else METHOD_FIGSIZE)
        fig.subplots_adjust(left=0.09, right=0.985, top=0.80 if legend else 0.95, bottom=0.2)
        step = EXPANDED_FROM if panel == "rf" and dataset in EXPANDED_DATASETS else None
        method_lines(ax, stats, panel, step)
        ax.set_ylabel(ylabel)
        if legend:
            method_legend(fig, stats)
        save(fig, args.img_dir, f"{figure}{name}")
