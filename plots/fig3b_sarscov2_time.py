"""Fig. 3b: time to a tree of the Viridian SARS-CoV-2 genomes, for VecME on each recorded hardware,
CMAPLE and MAPLE.

Inputs:  data/tables/viridian_time.tsv                                  (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    one bar per run, grouped by program (VecME fastest first), on a log time axis with the time
         written above; the hardware under each bar and the program under its group;
         a reported time (MAPLE's, from its authors) as an open, hatched bar
Outputs: img/fig3/fig3b_sarscov2_time.{pdf,svg,png}

    pixi run -e paper python -m plots.fig3b_sarscov2_time
"""

import matplotlib.pyplot as plt
import numpy as np

from plots._config import FIG3_AXES_Y, FIG3_HEIGHT, SPEC, STYLE
from plots.utils import data_label, parser, read_tsv, save

FIGSIZE = (80 / 25.4, FIG3_HEIGHT)  # beside Fig. 3a in the 180 mm page
LEFT, RIGHT = 0.17, 0.98
PROGRAMS = {"VecME": "P2V", "CMAPLE": "CMAPLE", "MAPLE": "MAPLE"}  # in order; -> key in SPEC
MINUTE, HOUR, DAY = 60, 3600, 86400
HEADROOM = 6  # the axis ends 6 times above the slowest run, room for its label
BAR_WIDTH = 0.62
GROUP_Y = -0.12  # the program names under the hardware, in axes heights


def duration(seconds: float) -> str:
    """Writes a time as the bar's label: seconds -> '27 min', '3.4 h' or '10 days'."""
    if seconds < HOUR:
        return f"{seconds / MINUTE:.0f} min"
    if seconds < DAY:
        return f"{seconds / HOUR:.1f} h"
    return f"{seconds / DAY:.0f} days"


def hardware(row: dict[str, str]) -> str:
    """Names a run's hardware under its bar: row -> '8× B200', 'A6000', '1 core' or '14 cores'."""
    units = int(row["units"])
    if row["unit"] == "GPU":
        model = row["model"].removeprefix("RTX ")
        return f"{units}× {model}" if units > 1 else model
    return f"{units} core" + ("s" if units > 1 else "")


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    rows = read_tsv(args.tables_dir, "viridian_time")
    if sorted({r["program"] for r in rows}) != sorted(PROGRAMS):
        raise SystemExit(f"viridian_time.tsv: expected runs of {', '.join(PROGRAMS)}")
    order = list(PROGRAMS)
    rows.sort(key=lambda r: (order.index(r["program"]), float(r["time_s"])))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bottom, top = FIG3_AXES_Y["ab"]
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
        ax.text(x, seconds * 1.25, ("~" if reported else "") + duration(seconds), ha="center")
    # the program under its group of bars, with a line under the hardware when the group has several
    below = ax.get_xaxis_transform()  # x in bars, y in axes heights
    for program in PROGRAMS:
        xs = [x for x, r in enumerate(rows) if r["program"] == program]
        ax.text(np.mean(xs), GROUP_Y, program, transform=below, ha="center", va="top")
        if len(xs) > 1:
            ax.plot(
                [xs[0] - 0.35, xs[-1] + 0.35],
                [GROUP_Y + 0.01] * 2,
                transform=below,
                color="#333333",
                linewidth=0.6,
                clip_on=False,
            )
    ax.set_yscale("log")
    ax.set_ylim(base, HEADROOM * max(times))
    ax.set_xticks(range(len(rows)), [hardware(r) for r in rows])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, len(rows) - 0.45)
    ax.set_ylabel("Time (s)")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    ax.set_axisbelow(True)
    data_label(ax, "SARS-CoV-2")
    save(fig, args.img_dir, "fig3b_sarscov2_time", bbox_inches=None)


if __name__ == "__main__":
    main()
