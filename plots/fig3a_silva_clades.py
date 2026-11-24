"""Fig. 3a: the SILVA 144 tree reduced to the three domains and 20 major clades,
with counts, F1 and support.

Inputs:  data/tables/silva_clades.tsv, silva_clade_paths.tsv            (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    the paths from each selected clade's best node to the root
           |
           v
         the tree they induce, rooted on the branch above the smallest clade holding
         Archaea and Eukaryota
           |
           v
         one row per clade: triangle, name, unique sequences,
         best F1, transfer support (TBE) of its stem; backbone nodes with TBE >= 75% marked
Outputs: img/fig3/fig3a_silva_clades.{pdf,svg}

    pixi run -e paper python -m plots.fig3a_silva_clades
"""

import math

import matplotlib.pyplot as plt

from matplotlib.patches import Polygon, Rectangle

from plots._config import STYLE
from plots.utils import parser, read_tsv, save

COLOURS = {"Archaea": "#6a4c93", "Eukaryota": "#c85a32", "Bacteria": "#1f6f8b"}
TBE_MARKS = (
    (95.0, "#1b7837", "≥ 95%"),
    (85.0, "#2b6cb0", "85–95%"),
    (75.0, "#d99b26", "75–85%"),
)
SELECTED_CLADES = [
    ("Archaea;Halobacteriota", "Halobacteriota"),
    ("Archaea;Thermoproteota", "Thermoproteota"),
    ("Eukaryota;Alveolata", "Alveolata"),
    ("Eukaryota;Obazoa", "Obazoa"),
    ("Eukaryota;Chloroplastida", "Chloroplastida"),
    ("Bacteria;Planctomycetota", "Planctomycetota"),
    ("Bacteria;Bacteroidota", "Bacteroidota"),
    ("Bacteria;Chloroflexota--other", "Chloroflexota"),
    ("Bacteria;Spirochaetota", "Spirochaetota"),
    ("Bacteria;Actinomycetota", "Actinomycetota"),
    ("Bacteria;Clostridiota", "Clostridiota"),
    ("Bacteria;Bacillota", "Bacillota"),
    ("Bacteria;Mycoplasmatota", "Mycoplasmatota"),
    ("Bacteria;Verrucomicrobiota", "Verrucomicrobiota"),
    ("Bacteria;Acidobacteriota", "Acidobacteriota"),
    ("Bacteria;Desulfobacteria", "Desulfobacteria"),
    ("Bacteria;Epsilonproteobacteria", "Epsilonproteobacteria"),
    ("Bacteria;Myxococcota", "Myxococcota"),
    ("Bacteria;Alphaproteobacteria", "Alphaproteobacteria"),
    ("Bacteria;Gammaproteobacteria", "Gammaproteobacteria"),
]
# Rows (y) covered by each domain's shaded band and by its tag on the left
DOMAIN_BANDS = {
    "Archaea": (17.55, 19.60),
    "Eukaryota": (14.55, 17.45),
    "Bacteria": (-0.60, 14.45),
}
DOMAIN_TAGS = {
    "Archaea": (17.55, 19.55),
    "Eukaryota": (14.55, 17.45),
    "Bacteria": (-0.55, 14.45),
}


def domain_colour(name: str) -> str:
    """Gives a clade its domain's colour: 'Archaea;...' -> colour."""
    return COLOURS[name.split(";")[0]]


def read_clades(tables_dir) -> tuple[dict, dict]:
    """Reads the clades and their paths to the root: folder -> (clades by name, paths by name)."""
    clades = {r["name"]: r for r in read_tsv(tables_dir, "silva_clades")}
    paths: dict[str, list[dict]] = {}
    for r in read_tsv(tables_dir, "silva_clade_paths"):
        paths.setdefault(r["name"], []).append(r)
    for rows in paths.values():
        rows.sort(key=lambda r: int(r["step"]))
    return clades, paths


def induced_tree(clades: dict, paths: dict) -> dict:
    """Builds the tree induced by the selected clades and the domains:
    (clades, paths) -> nested dict, rooted."""
    adj: dict[int, dict[int, tuple[float, float]]] = {}
    above: dict[int, tuple[float, float]] = (
        {}
    )  # node -> (branch length, TBE) of the branch above it
    for name in [k for k, _ in SELECTED_CLADES] + list(COLOURS):
        for r in paths[name]:
            if r["parent"] == "":
                continue
            u, p = int(r["node"]), int(r["parent"])
            edge = (max(0.0, float(r["branch_length"])), float(r["tbe_pct"]))
            adj.setdefault(u, {})[p] = edge
            adj.setdefault(p, {})[u] = edge
            above[u] = edge
    euk = [int(r["node"]) for r in paths["Eukaryota"]]
    arc = {int(r["node"]) for r in paths["Archaea"]}
    euk_node, arc_node = int(clades["Eukaryota"]["best_node"]), int(clades["Archaea"]["best_node"])
    # The smallest clade holding both Archaea and Eukaryota:
    # the figure is rooted on the branch above it
    ae = next(x for x in euk if x in arc)
    if ae == euk[-1]:
        raise SystemExit(
            "Archaea and Eukaryota meet only at the root; the layout assumes they form a clade"
        )
    ae_parent = euk[euk.index(ae) + 1]
    targets = {int(clades[k]["best_node"]): (k, label) for k, label in SELECTED_CLADES}
    order = {k: i for i, (k, _) in enumerate(SELECTED_CLADES)}

    def build(u: int, prev: int, length: float, tbe: float) -> dict | None:
        if u in targets:
            key, label = targets[u]
            c = clades[key]
            return {
                "leaf": True,
                "node": u,
                "key": key,
                "label": label,
                "length": length,
                "tbe": float(c["tbe_pct"]),
                "f1": float(c["f1"]),
                "leaves": int(c["unique_leaves"]),
                "rank": order[key],
            }
        children = [
            s
            for v, (bl, t) in adj.get(u, {}).items()
            if v != prev
            if (s := build(v, u, bl, t)) is not None
        ]
        if not children:
            return None
        if len(children) == 1 and u not in (ae, euk_node, arc_node):
            children[0]["length"] += length
            return children[0]
        children.sort(key=lambda c: c["rank"])
        return {
            "leaf": False,
            "node": u,
            "length": length,
            "tbe": tbe,
            "children": children,
            "rank": children[0]["rank"],
        }

    ae_length, ae_tbe = above[ae]
    root = {
        "leaf": False,
        "node": -1,
        "length": 0.0,
        "tbe": 100.0,
        "rank": 0,
        "children": [
            build(ae, ae_parent, ae_length / 2, ae_tbe),
            build(ae_parent, ae, ae_length / 2, ae_tbe),
        ],
    }
    return root


def place(node: dict, x0: float, counter: list[int]) -> float:
    """Sets the x (square-root branch lengths) and y (row) of every node:
    -> the largest x below the node."""
    step = 0.0 if node["node"] == -1 else 0.016 + 0.36 * math.sqrt(max(0.0, node["length"]))
    node["x0"], node["x"] = x0, x0 + step
    if node["leaf"]:
        node["y"] = len(SELECTED_CLADES) - 1 - counter[0]
        counter[0] += 1
        return node["x"]
    right = max(place(c, node["x"], counter) for c in node["children"])
    node["y"] = 0.5 * (node["children"][0]["y"] + node["children"][-1]["y"])
    return right


def draw(ax: plt.Axes, node: dict, columns: dict) -> None:
    """Draws a node, its row if it is a selected clade, and its subtree:
    (axes, node, column x) -> None."""
    x0, x1, y = node["x0"], node["x"], node["y"]
    tips = columns["tips"]
    if node["node"] != -1:
        ax.plot([x0, x1], [y, y], color="#2d3748", lw=1.35, solid_capstyle="round", zorder=2)
    if node["leaf"]:
        colour = domain_colour(node["key"])
        if x1 < tips - 0.010:
            ax.plot(
                [x1, tips - 0.008],
                [y, y],
                color=colour,
                lw=0.75,
                ls=":",
                alpha=0.75,
                zorder=1,
            )
        h = 0.22 + 0.05 * (math.log10(max(1000, node["leaves"])) - 4.0)
        ax.add_patch(
            Polygon(
                [[x1, y], [tips, y + h], [tips, y - h]],
                closed=True,
                facecolor=colour,
                edgecolor=colour,
                alpha=0.85,
                lw=0.8,
                zorder=3,
            )
        )
        ax.text(
            columns["name"],
            y,
            node["label"],
            color="#1a202c",
            fontweight="bold" if node["leaves"] >= 200000 else "normal",
            va="center",
            ha="left",
        )
        ax.text(
            columns["k"],
            y,
            f"{node['leaves']:,}",
            color="#2d3748",
            va="center",
            ha="right",
        )
        f1 = node["f1"]
        f1_colour = "#1b7837" if f1 >= 0.95 else ("#d95f02" if f1 >= 0.85 else "#7570b3")
        ax.text(
            columns["f1"],
            y,
            f"{f1:.3f}",
            color=f1_colour,
            fontweight="bold",
            va="center",
            ha="right",
        )
        bar, width = columns["bar"], columns["bar_width"]
        ax.add_patch(
            Rectangle(
                (bar, y - 0.22),
                width,
                0.44,
                facecolor="#edf2f7",
                edgecolor="none",
                zorder=1,
            )
        )
        ax.add_patch(
            Rectangle(
                (bar, y - 0.22),
                width * node["tbe"] / 100.0,
                0.44,
                facecolor=colour,
                edgecolor="none",
                alpha=0.85,
                zorder=2,
            )
        )
        ax.text(
            bar + width + 0.014,
            y,
            f"{node['tbe']:.1f}%",
            color="#1a202c",
            fontweight="bold",
            va="center",
            ha="left",
        )
        return
    ys = [c["y"] for c in node["children"]]
    ax.plot(
        [x1, x1],
        [min(ys), max(ys)],
        color="#2d3748",
        lw=1.35,
        solid_capstyle="round",
        zorder=2,
    )
    if node["node"] != -1:
        for threshold, colour, _label in TBE_MARKS:
            if node["tbe"] >= threshold:
                ax.scatter(
                    [x1],
                    [y],
                    s=20,
                    facecolor=colour,
                    edgecolor="white",
                    lw=0.6,
                    zorder=4,
                )
                break
    for c in node["children"]:
        draw(ax, c, columns)


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    clades, paths = read_clades(args.tables_dir)
    tree = induced_tree(clades, paths)
    tips = place(tree, 0.0, [0]) + 0.025
    columns = {
        "tips": tips,
        "name": tips + 0.015,
        "k": tips + 1.565,
        "f1": tips + 1.875,
        "bar": tips + 1.930,
        "bar_width": 0.145,
    }

    fig = plt.figure(figsize=(6.0, 6.3))
    ax = fig.add_axes((0.02, 0.02, 0.96, 0.93))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for name, (low, high) in DOMAIN_TAGS.items():
        ax.axhspan(
            *DOMAIN_BANDS[name],
            color=COLOURS[name],
            alpha=0.045 if name == "Bacteria" else 0.06,
            zorder=0,
        )
        mid, c = 0.5 * (low + high), clades[name]
        ax.plot(
            [-0.030, -0.030],
            [low + 0.15, high - 0.15],
            color=COLOURS[name],
            lw=2.5,
            solid_capstyle="round",
        )
        ax.text(
            -0.046,
            mid,
            f"{name}\n{int(c['unique_leaves']):,}\nTBE {float(c['tbe_pct']):.0f}%",
            color=COLOURS[name],
            fontweight="bold",
            ha="right",
            va="center",
            linespacing=1.15,
        )
    draw(ax, tree, columns)
    header = 19.82
    for x, text, ha in (
        (columns["name"], "Clade", "left"),
        (columns["k"], "Sequences", "right"),
        (columns["f1"], "F1", "right"),
        (columns["bar"], "TBE (%)", "left"),
    ):
        ax.text(
            x,
            header,
            text,
            fontweight="bold",
            color="#1a202c",
            ha=ha,
            va="bottom",
        )
    ax.plot(
        [0.0, columns["bar"] + columns["bar_width"] + 0.35],
        [19.62, 19.62],
        color="#cbd5e0",
        lw=0.8,
    )
    for i, (_threshold, colour, label) in enumerate(TBE_MARKS):
        ax.scatter(
            [1.01 * i],
            [-1.05],
            s=24,
            facecolor=colour,
            edgecolor="white",
            lw=0.6,
            clip_on=False,
        )
        ax.text(
            1.01 * i + 0.020,
            -1.05,
            f"Backbone TBE {label}",
            va="center",
            color="#2d3748",
        )
    ax.set_xlim(-0.50, columns["bar"] + columns["bar_width"] + 0.36)
    ax.set_ylim(-1.35, 20.45)
    save(fig, args.img_dir, "fig3a_silva_clades")


if __name__ == "__main__":
    main()
