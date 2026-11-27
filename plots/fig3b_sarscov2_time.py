"""Fig. 3b: time to a tree of the Viridian SARS-CoV-2 genomes, for VecME on each recorded hardware,
CMAPLE and MAPLE.

Inputs:  data/tables/viridian_time.tsv                                  (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    one bar per run, by program (VecME fastest first), on a log time axis with the time
         written above; the program and its hardware under each bar, at 45 degrees;
         a reported time (MAPLE's, from its authors) as an open, hatched bar
Outputs: img/fig3/fig3b_sarscov2_time.{pdf,svg,png}

    pixi run -e paper python -m plots.fig3b_sarscov2_time
"""

import matplotlib.pyplot as plt
import numpy as np

from plots._config import FIG3_AXES_Y, FIG3_HEIGHT, SPEC, STYLE
from plots.utils import data_label_above, parser, read_tsv, save

FIGSIZE = (50 / 25.4, FIG3_HEIGHT)  # in the row of Figs 3a-c (70 + 50 + 60 mm) across the page
LEFT, RIGHT = 0.29, 0.945  # room for the rotated names under the first bars and the last time
PROGRAMS = {"VecME": "P2V", "CMAPLE": "CMAPLE", "MAPLE": "MAPLE"}  # in order; -> key in SPEC
MINUTE, HOUR, DAY = 60, 3600, 86400
HEADROOM = 4  # the axis ends 4 times above the slowest run, room for its label
TIME_FONTSIZE = 7  # the times above the bars, smaller to fit over the narrow bars on one line
BAR_WIDTH = 0.62
LABEL_ROTATION = 45  # degrees


def duration(seconds: float) -> str:
    """Writes a time as the bar's label: seconds -> '27 min', '3.4 h' or '10 days'."""
    if seconds < HOUR:
        return f"{seconds / MINUTE:.0f} min"
    if seconds < DAY:
        return f"{seconds / HOUR:.1f} h"
    return f"{seconds / DAY:.0f} days"


def run_label(row: dict[str, str]) -> str:
    """Names a run under its bar: row -> 'VecME, 8× B200', 'VecME, 1× A6000' or 'MAPLE, 14 cores'."""
    units = int(row["units"])
    if row["unit"] == "GPU":
        return f"{row['program']}, {units}× {row['model'].removeprefix('RTX ')}"
    return f"{row['program']}, {units} core" + ("s" if units > 1 else "")


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    rows = read_tsv(args.tables_dir, "viridian_time")
    if sorted({r["program"] for r in rows}) != sorted(PROGRAMS):
        raise SystemExit(f"viridian_time.tsv: expected runs of {', '.join(PROGRAMS)}")
    order = list(PROGRAMS)
    rows.sort(key=lambda r: (order.index(r["program"]), float(r["time_s"])))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bottom, top = FIG3_AXES_Y
    fig.subplots_adjust(left=LEFT, right=RIGHT, bottom=bottom, top=top)
    times = [float(r["time_s"]) for r in rows]
    base = 10 ** np.floor(np.log10(min(times)))  # bars start at the decade below the fastest run
    for x, r in enumerate(rows):
        colour = SPEC[PROGRAMS[r["program"]]]["color"]
        reported = r["source"] == "reported"
        seconds = float(r["time_s"])
        ax.bar(
            x,
            seconds - base,
            bottom=base,
            width=BAR_WIDTH,
            color="white" if reported else colour,
            edgecolor=colour,
            hatch="////" if reported else None,
            linewidth=1.0,
        )
        label = ("~" if reported else "") + duration(seconds)
        ax.text(x, seconds * 1.25, label, ha="center", fontsize=TIME_FONTSIZE)
    ax.set_yscale("log")
    ax.set_ylim(base, HEADROOM * max(times))
    ax.set_xticks(
        range(len(rows)),
        [run_label(r) for r in rows],
        rotation=LABEL_ROTATION,
        ha="right",
        rotation_mode="anchor",
    )
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, len(rows) - 0.45)
    ax.set_ylabel("Time (s)")
    data_label_above(ax, "Viridian")
    save(fig, args.img_dir, "fig3b_sarscov2_time", bbox_inches=None)


if __name__ == "__main__":
    main()
