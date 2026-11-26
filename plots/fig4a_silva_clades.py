"""Fig. 4a: the SILVA 144 tree reduced to the three domains and 20 major clades, as a cladogram,
with the transfer support of every branch written on the tree and each clade's depth.

Inputs:  data/tables/silva_clades.tsv, silva_clade_paths.tsv            (make_tables.py)
Options: --tables-dir, --img-dir
Flow:    the paths from each selected clade's best node to the root
           |
           v
         the tree they induce, rooted on the branch above the smallest clade holding
         Archaea and Eukaryota
           |
           v
         a cladogram: one step per branch, the clades lined up on the right;
         one row per clade: triangle (length: its median depth, against a scale bar), name, distinct sequences;
         each branch's transfer support (TBE) at its end; one colour per domain
Outputs: img/fig4/fig4a_silva_clades.{pdf,svg,png}

    pixi run -e paper python -m plots.fig4a_silva_clades
"""

import math

import matplotlib.pyplot as plt

from matplotlib import patheffects
from matplotlib.patches import Polygon

from plots._config import STYLE
from plots.utils import parser, read_tsv, save

COLOURS = {"Archaea": "#6a4c93", "Eukaryota": "#c85a32", "Bacteria": "#1f6f8b"}
# 112 mm wide: with Fig. 4b (64 mm) and a 4 mm gap, the 180 mm page
FIGSIZE = (112 / 25.4, 117 / 25.4)
BRANCH, TEXT = "#4a5568", "#1a202c"
SUPPORT_SIZE = 7  # points: the numbers on the branches, smaller than the labels
# A white outline, so that a number stays legible where it crosses a branch
SUPPORT_HALO = [patheffects.withStroke(linewidth=2, foreground="white")]
TREE_WIDTH = 1.25  # axis units from the root to the clades
TRIANGLE = 0.20  # axis units: the triangle of the deepest clade
NAME_GAP, COUNT_X = (
    0.03,
    1.65,
)  # x of the clade names and of the right edge of the counts, past the triangles
SCALE_BAR, SCALE_Y = 0.1, -1.05  # substitutions per site; its row, below the clades
SELECTED_CLADES = [
    ("Archaea;Halobacteriota", "Halobacteriota"),
    ("Archaea;Thermoproteota", "Thermoproteota"),
    ("Eukaryota;Alveolata", "Alveolata"),
    ("Eukaryota;Obazoa", "Obazoa"),
    ("Eukaryota;Chloroplastida", "Chloroplastida"),
    ("Bacteria;Planctomycetota", "Planctomycetota"),
    ("Bacteria;Bacteroidota", "Bacteroidota"),
    # SILVA's Chloroflexota without Chloroflexia, the class of the phylum's type: pro parte
    ("Bacteria;Chloroflexota--other", "Chloroflexota p.p."),
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
            if c["depth_median"] == "":
                raise SystemExit(
                    f"{key}: no depth in silva_clades.tsv (the other side of its best split)"
                )
            return {
                "leaf": True,
                "node": u,
                "key": key,
                "label": label,
                "length": length,
                "tbe": float(c["tbe_pct"]),
                "f1": float(c["f1"]),
                "leaves": int(c["unique_leaves"]),
                "depth": float(c["depth_median"]),
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


def height(node: dict) -> int:
    """Counts the steps from a node to its farthest clade, for it and the nodes below: node -> steps."""
    node["h"] = 0 if node["leaf"] else 1 + max(height(c) for c in node["children"])
    return node["h"]


def clades_of(node: dict) -> list[dict]:
    """Lists the clades below a node: node -> its leaves."""
    return [node] if node["leaf"] else [c for child in node["children"] for c in clades_of(child)]


def place(node: dict, x0: float, step: float, counter: list[int]) -> None:
    """Sets the x (one step per branch, the clades at TREE_WIDTH) and y (row) of every node."""
    node["x0"], node["x"] = x0, TREE_WIDTH - step * node["h"]
    if node["leaf"]:
        node["y"] = len(SELECTED_CLADES) - 1 - counter[0]
        counter[0] += 1
        return
    for c in node["children"]:
        place(c, node["x"], step, counter)
    node["y"] = 0.5 * (node["children"][0]["y"] + node["children"][-1]["y"])


def support(ax: plt.Axes, x: float, y: float, tbe: float) -> None:
    """Writes a branch's transfer support at its end, just above it: (axes, x, y, TBE in %) -> None."""
    ax.text(
        x - 0.006,
        y + 0.15,
        f"{tbe:.0f}",
        fontsize=SUPPORT_SIZE,
        color="#2d3748",
        ha="right",
        va="bottom",
        path_effects=SUPPORT_HALO,
    )


def draw(ax: plt.Axes, node: dict, tips: float, scale: float, label: bool = True) -> None:
    """Draws a node, its row if it is a selected clade, and its subtree:
    (axes, node, x past the longest triangle, axis units per substitution per site,
    whether to number the node's branch) -> None."""
    x0, x1, y = node["x0"], node["x"], node["y"]
    if node["node"] != -1:
        ax.plot([x0, x1], [y, y], color=BRANCH, lw=1.2, solid_capstyle="round", zorder=2)
    if node["leaf"]:
        colour = domain_colour(node["key"])
        end = x1 + scale * node["depth"]
        h = 0.22 + 0.05 * (math.log10(max(1000, node["leaves"])) - 4.0)
        ax.add_patch(
            Polygon(
                [[x1, y], [end, y + h], [end, y - h]],
                closed=True,
                facecolor=colour,
                edgecolor=colour,
                alpha=0.85,
                lw=0.8,
                zorder=3,
            )
        )
        if end < tips - 0.01:
            ax.plot(
                [end + 0.015, tips + NAME_GAP - 0.015],
                [y, y],
                color=colour,
                lw=0.75,
                ls=":",
                alpha=0.75,
            )
        if label:
            support(ax, x1, y, node["tbe"])
        ax.text(tips + NAME_GAP, y, node["label"], color=TEXT, va="center", ha="left")
        ax.text(tips + COUNT_X, y, f"{node['leaves']:,}", color=TEXT, va="center", ha="right")
        return
    ys = [c["y"] for c in node["children"]]
    ax.plot([x1, x1], [min(ys), max(ys)], color=BRANCH, lw=1.2, solid_capstyle="round", zorder=2)
    # a node with one child continues its branch: the branch is numbered once, at its end
    if node["node"] != -1 and label and len(node["children"]) > 1:
        support(ax, x1, y, node["tbe"])
    # the root splits one branch in two halves: it is numbered once
    for i, c in enumerate(node["children"]):
        draw(ax, c, tips, scale, label=not (node["node"] == -1 and i == 1))


def main() -> None:
    args = parser(__doc__).parse_args()
    plt.style.use(STYLE)
    clades, paths = read_clades(args.tables_dir)
    tree = induced_tree(clades, paths)
    place(tree, 0.0, TREE_WIDTH / height(tree), [0])
    scale = TRIANGLE / max(c["depth"] for c in clades_of(tree))
    tips = TREE_WIDTH + TRIANGLE

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes((0.02, 0.02, 0.96, 0.95))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for name, (low, high) in DOMAIN_TAGS.items():
        ax.axhspan(*DOMAIN_BANDS[name], color=COLOURS[name], alpha=0.05, zorder=0)
        ax.plot(
            [-0.030, -0.030],
            [low + 0.15, high - 0.15],
            color=COLOURS[name],
            lw=2.5,
            solid_capstyle="round",
        )
        ax.text(
            -0.046,
            0.5 * (low + high),
            f"{name}\n{int(clades[name]['unique_leaves']):,}",
            color=COLOURS[name],
            fontweight="bold",
            ha="right",
            va="center",
            linespacing=1.15,
        )
    draw(ax, tree, tips, scale)
    ax.text(tips + COUNT_X, 19.82, "Sequences", color=TEXT, ha="right", va="bottom")
    # the scale of the triangles, under them
    ax.plot(
        [TREE_WIDTH, TREE_WIDTH + scale * SCALE_BAR],
        [SCALE_Y, SCALE_Y],
        color=TEXT,
        lw=1.2,
        solid_capstyle="butt",
    )
    ax.text(
        TREE_WIDTH + scale * SCALE_BAR + 0.03, SCALE_Y, f"{SCALE_BAR:g}", color=TEXT, va="center"
    )
    ax.set_xlim(-0.56, tips + COUNT_X + 0.04)
    ax.set_ylim(-1.5, 20.45)
    save(fig, args.img_dir, "fig4a_silva_clades")


if __name__ == "__main__":
    main()
