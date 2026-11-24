"""Helpers several plot scripts share: reading figure data, saving figures, the panels of Figs 2a,b and S1-S3."""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from csv import DictReader
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import (  # noqa: E402
    FixedLocator,
    FuncFormatter,
    LogFormatterMathtext,
    LogLocator,
    NullFormatter,
)

from plots._config import (  # noqa: E402
    DEFAULT_IMG_DIR,
    DEFAULT_TABLES_DIR,
    FIGURE_FORMATS,
    SAVE_METADATA,
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
    p.add_argument(
        "--img-dir", type=Path, default=Path(DEFAULT_IMG_DIR), help="output folder"
    )
    return p


def read_tsv(tables_dir: Path, name: str) -> list[dict[str, str]]:
    """Reads a figure-data file: (folder, name) -> rows; stops if make_tables.py has not written it."""
    path = tables_dir / f"{name}.tsv"
    if not path.exists():
        raise SystemExit(f"{path} is missing: run python -m paper.make_tables first")
    with path.open(newline="") as fh:
        return list(DictReader(fh, delimiter="\t"))


def save(fig: plt.Figure, img_dir: Path, name: str, **kwargs) -> None:
    """Saves a figure once per format, without timestamps: (figure, folder, name) -> img_dir/name.{pdf,svg}."""
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
    """Draws a field's mean and SD per size for every dataset of SCALING_SERIES: (axes, rows, field) -> None."""
    for dataset, label, colour, marker, linestyle, filled in SCALING_SERIES:
        group = sorted(
            (r for r in rows if r["dataset"] == dataset), key=lambda r: int(r["n"])
        )
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
    """Reads the data of Figs 2a,b: folder -> sample-and-place rows of scaling.tsv and random_tree.tsv."""
    rows = read_tsv(tables_dir, "scaling") + read_tsv(tables_dir, "random_tree")
    rows = [r for r in rows if r["initialisation"] == "sample"]
    if len({(r["gpu_model"], r["gpu_count"]) for r in rows}) != 1:
        raise SystemExit("mixed hardware in the scaling data: update the caption")
    return rows


# ---- Figs S1-S3: bars per method at each size ----

N_LIST = [1000, 5000, 10000, 50000, 100000, 1000000, 10000000]
N_LABELS = ["1k", "5k", "10k", "50k", "100k", "1M", "10M"]
PLOTTED = [
    "FastTree-ME",
    "VFT-ME",
    "FastTree-ML",
    "VFT-ML",
    "DIPPER",
    "P2V",
]  # left to right
GROUP = 0.86  # width of one size group, in category units
BAND = "#F3F3F3"
FLOORS = {
    "time": 1e-2,
    "rf": 0.01,
    "host": 2e-2,
    "gpu": 0.1,
}  # bar baselines = lower axis limits
TOPS = {"time": 3e4, "rf": 130, "host": 1000, "gpu": 100}
MIB_PER_GIB = 1024.0


def method_stats(
    rows: list[dict[str, str]], dataset: str, field: str, scale: float
) -> dict:
    """Summarises one field per method and size: (rows, dataset, field) -> {(n, method): (mean, min, max)}."""
    values: dict[tuple[int, str], list[float]] = {}
    for r in rows:
        if r["dataset"] == dataset and r["method"] in PLOTTED and r[field] != "":
            values.setdefault((int(r["n"]), r["method"]), []).append(
                float(r[field]) * scale
            )
    return {k: (mean(v), min(v), max(v)) for k, v in values.items()}


def method_bars(ax: plt.Axes, stats: dict, panel: str) -> None:
    """Draws each method's bar at each size, from the panel floor to the mean, with the range as a whisker."""
    for (n, key), (mu, lo, hi) in stats.items():
        if lo <= FLOORS[panel] or hi >= TOPS[panel]:
            raise SystemExit(
                f"{key} N={n}: {lo}..{hi} outside the {panel} axis {FLOORS[panel]}..{TOPS[panel]}"
            )
    slot = GROUP / len(PLOTTED)
    for i, n in enumerate(N_LIST):
        keys = [k for k in PLOTTED if (n, k) in stats]
        for j, key in enumerate(keys):
            x = i + (j - (len(keys) - 1) / 2) * slot
            mu, lo, hi = stats[(n, key)]
            ax.bar(
                [x],
                [mu - FLOORS[panel]],
                bottom=FLOORS[panel],
                width=0.9 * slot,
                color=SPEC[key]["color"],
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
            if hi > lo:
                ax.errorbar(
                    [x],
                    [mu],
                    yerr=[[mu - lo], [hi - mu]],
                    fmt="none",
                    ecolor="#222222",
                    elinewidth=0.7,
                    capsize=1.4,
                    capthick=0.7,
                    zorder=4,
                )
    ax.set_xlim(-0.55, len(N_LIST) - 0.45)
    ax.set_xticks(range(len(N_LIST)))
    ax.set_xticklabels(N_LABELS)
    ax.tick_params(axis="x", length=0)
    for i in range(0, len(N_LIST), 2):
        ax.axvspan(i - 0.5, i + 0.5, color=BAND, lw=0, zorder=0)
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Number of taxa, N")
    ax.set_ylim(FLOORS[panel], TOPS[panel])
    if panel != "time":
        ax.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 100, 1000]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))


def method_legend(fig: plt.Figure, stats: dict) -> None:
    """Adds the methods drawn, as a figure legend above the panel: (figure, stats) -> None."""
    present = [k for k in PLOTTED if any(key == k for _, key in stats)]
    handles = [
        Patch(facecolor=SPEC[k]["color"], edgecolor="none", label=SPEC[k]["label"])
        for k in reversed(present)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.53, 0.998),
        handletextpad=0.4,
        columnspacing=2.0,
        frameon=False,
    )
