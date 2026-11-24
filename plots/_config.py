"""Plot constants: folders, output formats, the method colours shared by every figure,
and the figure styles."""

from typing import Final, TypedDict

DEFAULT_TABLES_DIR: Final = "data/tables"  # figure data written by paper/make_tables.py
DEFAULT_IMG_DIR: Final = "img"  # figure outputs (git-ignored)
FIGURE_FORMATS: Final = ("pdf", "svg", "png")  # every figure is saved once per format
STYLE: Final = "plots/paper.mplstyle"  # every figure

# Matplotlib metadata without timestamps, so that an unchanged figure is an unchanged file
SAVE_METADATA: Final = {"pdf": {"CreationDate": None}, "svg": {"Date": None}}

# The folder of each figure in img/: one per main figure, holding its supplementary figures
FIGURE_FOLDERS: Final = {
    "fig2": "fig2",
    "figS1": "fig2",  # S1-S3: every method by size
    "figS2": "fig2",
    "figS3": "fig2",
    "figS4": "fig2",  # alignment length
    "fig3": "fig3",
    "figS5": "fig3",  # SARS-CoV-2 variants and shared branches
}


# One entry per method, used identically in every figure:
# key, legend label, colour, marker, filled, line style.
# Colour-blind safe: Okabe and Ito's colours plus two greys; each method has its own marker too.
# Vermilion = Phylo2VecME, greys = ME searches, blues = ML searches, reddish purple = DIPPER,
# bluish green = MAPLE, black = CMAPLE.
# METHODS: the methods of Figs S1-S4, in drawing order.
# PANDEMIC_METHODS: the other programs of Figs 2c,d and S5b.
METHODS: Final = [
    ("FastTree-ME", "FastTree-ME", "#8A8A8A", "s", True, "-"),
    ("VFT-ME", "VeryFastTree-ME", "#4D4D4D", "^", True, "-"),
    ("FastTree-ML", "FastTree-ML", "#56B4E9", "v", True, "-"),
    ("VFT-ML", "VeryFastTree-ML", "#0072B2", "D", True, "-"),
    ("DIPPER", "DIPPER", "#CC79A7", "X", True, "-"),
    ("P2V", "Phylo2VecME", "#D55E00", "o", True, "-"),
]
PANDEMIC_METHODS: Final = [
    ("MAPLE", "MAPLE", "#009E73", "P", True, "--"),
    ("CMAPLE", "CMAPLE", "#000000", "h", True, "-."),
]


class Method(TypedDict):
    """How a method is drawn: legend label, colour, marker, filled marker, line style."""

    label: str
    color: str
    marker: str
    filled: bool
    ls: str


SPEC: Final = {
    m[0]: Method(label=m[1], color=m[2], marker=m[3], filled=m[4], ls=m[5])
    for m in [*METHODS, *PANDEMIC_METHODS]
}

# The datasets of the scaling figures (S1-S3) and their figure numbers
SCALING_FIGURES: Final = {"rnasim": "figS1", "jc10k": "figS2", "yule10k": "figS3"}

# Fig. 2a,b: dataset in the data, legend label, colour, marker, line style, filled marker.
# Every series is Phylo2VecME, so all take its colour and differ by line style and marker fill.
# The random-tree series is drawn last, dash-dotted and with squares,
# so that it stays visible where it runs along the other JC series.
SCALING_SERIES: Final = [
    ("JC", "JC (PASTA)", SPEC["P2V"]["color"], "o", "-", True),
    ("RNASim", "RNASim (PASTA)", SPEC["P2V"]["color"], "o", "--", False),
    ("JC random tree", "JC random tree", SPEC["P2V"]["color"], "s", "-.", True),
]

# Fig. 2c,d: the pandemic programs, as named in maple.tsv, and their key in SPEC
PANDEMIC: Final = [("MAPLE", "MAPLE"), ("CMAPLE", "CMAPLE"), ("Phylo2VecME", "P2V")]
