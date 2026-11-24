"""Simulated datasets of the paper:
RNASim subsets, JC69 alignments on their trees and on Yule trees, expansions.

Every simulated input of the scaling, comparator, sequence-length, representative, FastME and
calibration tables and figures, with the paper's generators and seeds,
so that the alignments equal the paper's.

Inputs:  data/datasets/rnasim/{true_topo.tree, aln_dna.fa.xz}
           (download_data.py; RNASim's 200,000 sequences)
Options: --sets, --sizes, --replicates, --datasets-dir, --sim-jc69, --expansion-base, --fastme-base,
         --overwrite
Flow:    each set x size x replicate r
           |
           v
         tree   rnasim, jc*: N leaves of the RNASim tree drawn by random.Random(42 + r) and the tree
                they induce (JC: branch lengths / 3,200);
                yule10k: a Yule tree (seed 770000 + r, height 0.2 at 10^6 leaves);
                rnasim, jc10k at 10^6 and 10^7: replicate 0 at 10^5 with each leaf a ten-leaf clade,
                once or twice
           |
           v
         sites  rnasim: RNASim's own (1,596 sites); jc*: sim_jc69_fast (seed 20261104 + 1000 r);
                yule10k: sim_jc69_fast (seed 20261114 + 1000 r);
                expansions: on each new branch, each site redrawn from the four bases
                with probability 2^-7 (numpy seed 20261104)
           |
           v
         the random start tree the comparison programs started from
         (uniform; seeds as the paper's runs)

         fastme: the FastME comparison's sets, n sequences kept in order with their true tree
         restricted to them, and a uniform random start tree (seed r + 1):
         jc10k, rnasim from the 10^4-sequence set of replicate r, drawn by numpy default_rng(r + 1);
         jc1m, rna1m from the 10^6-sequence jc10k or rnasim set,
         subset r of n drawn by default_rng(1000 n + r)
Outputs: data/datasets/<set>/<n>/rep<r>.fa              leaves T0 ... T<n-1>, one line per sequence
         data/datasets/<set>/<n>/rep<r>.true.nwk        unrooted; branch lengths
                                                        (substitutions per site) where the sites
                                                        were simulated on them (jc*, yule10k)
         data/datasets/<set>/<n>/rep<r>.random_start.nwk  unrooted, every branch 0.05
         data/datasets/fastme/<set>/<n>/rep<r>.{fa,true.nwk,random_start.nwk}
                                                        the same, without branch lengths;
                                                        the random start rooted,
                                                        as the paper gave it to both programs
         Sets: rnasim, jc10k (10^3-10^7; replicate 0 only above 10^5), jc1k (10^4),
         yule10k (10^3-10^7), jc100k and jc1000k (10^5 and 10^6 sites on the JC trees,
         at 10^4 and 10^5), fastme (jc10k and rnasim at 1,000, 2,000, 5,000 and 10,000;
         jc1m and rna1m: 100 subsets at 1,000 and 2,000, 20 at 5,000).
         Expanded leaf i descends from leaf i // 10 (10^6) or i // 100 (10^7) of the 10^5 replicate.

    pixi run -e paper python -m paper.simulate_data --sets rnasim jc10k --sizes 1000 5000
"""

from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from lzma import open as open_xz
from pathlib import Path
from random import Random
from subprocess import run
from tempfile import TemporaryDirectory

from numpy import (
    arange,
    array,
    cumsum,
    empty,
    float32,
    float64,
    frombuffer,
    full,
    int32,
    int64,
    ndarray,
    packbits,
    repeat,
    sort,
    uint8,
    uint64,
    unpackbits,
    zeros,
)
from numpy.random import Generator, default_rng

from paper._config import DEFAULT_DATASETS_DIR, SIM_JC69

PAPER_SIZES = (1000, 5000, 10000, 50000, 100000, 1000000, 10000000)
SEQLEN_SIZES = (10000, 100000)
REPLICATES = (0, 1, 2, 3, 4)
# set -> sites (0: RNASim's own), 64-bit words per sequence in the paper's packed alignments
# (the expansion's random stream depends on it),
# default sizes, the simulator's seed base (0: not simulated)
SETS = {
    "rnasim": (0, 25, PAPER_SIZES, 0),
    "jc10k": (10000, 160, PAPER_SIZES, 20261104),
    "jc1k": (1000, 20, (10000,), 20261104),
    "yule10k": (10000, 160, PAPER_SIZES, 20261114),
    "jc100k": (100000, 1565, SEQLEN_SIZES, 20261104),
    "jc1000k": (1000000, 15625, SEQLEN_SIZES, 20261104),
}
EXPANDED_SETS = (
    "rnasim",
    "jc10k",
)  # above the expansion base these sets expand replicate 0
EXPANSION_BASE = 100000
# The FastME comparison's sets (folder fastme/<set>/<n>/):
# subsets of the 10^4-sequence sets, as the paper's
FASTME = "fastme"
FASTME_SETS = ("jc10k", "rnasim")
FASTME_SIZES = (1000, 2000, 5000, 10000)
FASTME_BASE = 10000  # the size of the sets they are drawn from
FASTME_SEED_STEP = 1  # subset and random start seed: replicate + 1
# jc1m, rna1m: subsets of the 10^6-sequence set (replicate 0) of jc10k and rnasim;
# subset k of n drawn with seed 1000 n + k
FASTME_1M_SETS = {"jc1m": "jc10k", "rna1m": "rnasim"}
FASTME_1M_SUBSETS = {1000: 100, 2000: 100, 5000: 20}  # subsets per size
FASTME_1M_SEED_STEP = 1000
JC_SCALE = 3200.0  # RNASim branch lengths / JC_SCALE = substitutions per site of the JC sets
START_LENGTH = 0.05  # branch length written on the random start trees
# Seeds of the paper's runs
SUBSET_SEED = 42
SIM_SEED_STEP = 1000
YULE_SEED = 770000
START_SEED = 0x5177A144
START_SEED_STEP = 99991
YULE_START_SEED = 9000
EXPANSION_SEED = 20261104
# Yule trees: one birth rate at every size, that of a tree of 10^6 lineages with expected height 0.2
YULE_ANCHOR_N = 1_000_000
YULE_ANCHOR_HEIGHT = 0.2
# Expansion: the ten-leaf clade ((((0,1),(2,3)),((4,5),(6,7))),(8,9));
# one mutation draw per branch of it (the leaves below the branch share it);
# a site is redrawn where 7 random words all have its bit set, so with probability 2^-7
CLADE = 10
CLADE_GROUPS = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [8, 9],
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
] + [[j] for j in range(CLADE)]
MUTATION_ANDS = 6
EXPANSION_CHUNK = {
    10: 5000,
    100: 1000,
}  # base sequences per chunk of the random stream, by expansion factor
UNPACK_ROWS = 10000  # sequences unpacked to text at a time
BASES = frombuffer(b"ACGT", dtype=uint8)
GAP = ord("-")


# ---- Trees ---------------------------------------------------------------------------------------


def parse_newick(text: str) -> tuple[list[list[int]], list[str], list[float]]:
    """Parses a Newick tree as the paper's scripts did:
    text -> (children, name, length) per node, root 0."""
    children: list[list[int]] = [[]]
    name, length, parent = [""], [0.0], [-1]
    curr, i, n = 0, 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "(" or ch == ",":
            if ch == ",":
                curr = parent[curr]
            children.append([])
            name.append("")
            length.append(0.0)
            parent.append(curr)
            children[curr].append(len(children) - 1)
            curr = len(children) - 1
            i += 1
        elif ch == ")":
            curr = parent[curr]
            i += 1
        elif ch == ";":
            break
        elif ch.isspace():
            i += 1
        else:
            j = i
            while j < n and text[j] not in "(),;":
                j += 1
            token = text[i:j].strip()
            if ":" in token:
                nm, ln = token.split(":", 1)
                name[curr] = nm.strip()
                try:
                    length[curr] = float(ln.strip())
                except ValueError:
                    length[curr] = 0.0
            else:
                name[curr] = token
            i = j
    return children, name, length


def induce_subtree(
    children: list[list[int]], name: list[str], length: list[float], keep: set[str]
) -> tuple[list[list[int]], list[str], list[float], int]:
    """Restricts a tree to some leaves:
    (tree, leaf names) -> (children, name, length, root) of the induced tree.

    Nodes left with one child are merged into it (lengths added);
    multifurcations become binary with zero lengths.
    """
    order, stack = [], [0]
    while stack:
        u = stack.pop()
        order.append(u)
        stack.extend(children[u])
    mapped: dict[int, int] = {}
    new_children: list[list[int]] = []
    new_name: list[str] = []
    new_len: list[float] = []
    for u in reversed(order):
        if not children[u]:
            if name[u] in keep:
                mapped[u] = len(new_children)
                new_children.append([])
                new_name.append(name[u])
                new_len.append(length[u])
            continue
        kept = [mapped[c] for c in children[u] if c in mapped]
        if len(kept) == 1:
            new_len[kept[0]] += length[u]
            mapped[u] = kept[0]
        elif len(kept) >= 2:
            curr = kept[0]
            for nxt in kept[1:-1]:
                new_children.append([curr, nxt])
                new_name.append("")
                new_len.append(0.0)
                curr = len(new_children) - 1
            new_children.append([curr, kept[-1]])
            new_name.append("")
            new_len.append(length[u])
            mapped[u] = len(new_children) - 1
    return new_children, new_name, new_len, mapped[0]


def rooted_arrays(
    children: list[list[int]],
    name: list[str],
    length: list[float],
    root: int,
    leaf_id: dict[str, int],
    scale: float,
) -> tuple[ndarray, ndarray, ndarray]:
    """Numbers a tree as the paper's bundles:
    (tree, leaf ids) -> (c1, c2, branch length / scale) per node.

    Leaves 0..N-1, internal nodes N..2N-2 in post-order (children first, root last);
    leaves' lengths at least 1e-6, the root's 0.
    """
    n = len(leaf_id)
    order, stack = [], [root]
    while stack:
        u = stack.pop()
        order.append(u)
        stack.extend(children[u])
    c1_list, c2_list = [], []
    blen = zeros(2 * n - 1, dtype=float32)
    mapped: dict[int, int] = {}
    for u in reversed(order):
        if not children[u]:
            mapped[u] = leaf_id[name[u]]
            blen[mapped[u]] = max(1e-6, float(length[u]) / scale)
            continue
        kids = [mapped[c] for c in children[u]]
        curr = kids[0]
        for k, nxt in enumerate(kids[1:]):
            node = n + len(c1_list)
            c1_list.append(curr)
            c2_list.append(nxt)
            last = k == len(kids) - 2
            blen[node] = max(0.0, float(length[u]) / scale) if last and u != root else 0.0
            curr = node
        mapped[u] = curr
    if len(c1_list) != n - 1:
        raise ValueError(f"expected {n - 1} internal nodes, got {len(c1_list)}")
    c1 = zeros(2 * n - 1, dtype=int32)
    c2 = zeros(2 * n - 1, dtype=int32)
    c1[n:] = c1_list
    c2[n:] = c2_list
    return c1, c2, blen


def uniform_random_tree(n: int, seed: int) -> tuple[ndarray, ndarray]:
    """Draws the comparison programs' random start tree:
    (n, seed) -> (c1, c2), as the paper's bundles."""
    rng = default_rng(seed)
    v = rng.integers(0, arange(1, n, dtype=int64), dtype=int32).tolist()
    perm = rng.permutation(n).astype(int32)
    uf = list(range(n))
    c1_int, c2_int = [0] * (n - 1), [0] * (n - 1)
    for k in range(n - 1):
        i = (n - 2) - k
        c1_int[k] = uf[i + 1]
        c2_int[k] = uf[v[i]]
        uf[v[i]] = n + k
    c1 = zeros(2 * n - 1, dtype=int32)
    c2 = zeros(2 * n - 1, dtype=int32)
    c1[n:] = c1_int
    c2[n:] = c2_int
    for c in (c1, c2):
        sub = c[n:]
        leaf = sub < n
        sub[leaf] = perm[sub[leaf]]
    return c1, c2


def yule_birth_rate() -> float:
    """Computes the Yule trees' birth rate:
    () -> the rate at which 10^6 lineages have expected height 0.2."""
    j = arange(YULE_ANCHOR_N, 1, -1, dtype=float64)
    return float((1.0 / j).sum() / YULE_ANCHOR_HEIGHT)


def yule_tree(n: int, rep: int) -> tuple[ndarray, ndarray, ndarray]:
    """Draws a Yule tree: (n, replicate) -> (c1, c2, branch length),
    internal nodes in order of creation.

    Going back in time, two of the j lineages alive join after an exponential time of
    rate j x birth rate.
    """
    rng = default_rng(YULE_SEED + rep)
    j = arange(n, 1, -1, dtype=float64)
    h_join = cumsum(rng.exponential(1.0 / (j * yule_birth_rate())))
    c1 = full(2 * n - 1, -1, int32)
    c2 = full(2 * n - 1, -1, int32)
    height = zeros(2 * n - 1)
    height[n:] = h_join
    alive = list(range(n))
    picks = rng.random((n - 1, 2))
    for k in range(n - 1):
        m = len(alive)
        a = int(picks[k, 0] * m)
        x = alive[a]
        alive[a] = alive[-1]
        alive.pop()
        b = int(picks[k, 1] * (m - 1))
        c1[n + k] = x
        c2[n + k] = alive[b]
        alive[b] = n + k
    parent = zeros(2 * n - 1, int64)
    parent[c1[n:]] = arange(n, 2 * n - 1)
    parent[c2[n:]] = arange(n, 2 * n - 1)
    blen = zeros(2 * n - 1, float32)
    blen[: 2 * n - 2] = (height[parent[: 2 * n - 2]] - height[: 2 * n - 2]).astype(float32)
    return c1, c2, blen


def expand_tree(c1: ndarray, c2: ndarray, n_base: int) -> tuple[ndarray, ndarray]:
    """Replaces each leaf by the ten-leaf clade:
    (c1, c2) on n_base leaves -> (c1, c2) on 10 n_base leaves.

    Leaf x becomes leaves 10x..10x+9;
    the clade's internal nodes come first, then the old internal nodes in order.
    """
    n = n_base * CLADE
    out1 = zeros(2 * n - 1, dtype=int32)
    out2 = zeros(2 * n - 1, dtype=int32)
    x = arange(n_base, dtype=int32)
    leaf = [CLADE * x + j for j in range(CLADE)]
    m = [n + (CLADE - 1) * x + j for j in range(CLADE - 1)]
    pairs = [
        (leaf[0], leaf[1]),
        (leaf[2], leaf[3]),
        (leaf[4], leaf[5]),
        (leaf[6], leaf[7]),
        (leaf[8], leaf[9]),
    ]
    pairs += [(m[0], m[1]), (m[2], m[3]), (m[5], m[6]), (m[7], m[4])]
    for k, (a, b) in enumerate(pairs):
        out1[m[k]] = a
        out2[m[k]] = b
    map_node = empty(2 * n_base - 1, dtype=int32)
    map_node[:n_base] = m[8]
    map_node[n_base:] = n + (CLADE - 1) * n_base + arange(n_base - 1, dtype=int32)
    backbone = slice(n + (CLADE - 1) * n_base, 2 * n - 1)
    out1[backbone] = map_node[c1[n_base:]]
    out2[backbone] = map_node[c2[n_base:]]
    return out1, out2


def newick(c1: ndarray, c2: ndarray, names: list[str], lengths: ndarray | None) -> str:
    """Writes a rooted tree as unrooted Newick: (c1, c2, names, length per node or None) -> text.

    The root's two branches become one, on the child that stays (its length plus the other's).
    """
    n = len(names)
    left, right = c1.tolist(), c2.tolist()
    blen = None if lengths is None else lengths.astype(float64).tolist()
    a, b = left[2 * n - 2], right[2 * n - 2]
    if b >= n:
        top, kept, gone = [a, left[b], right[b]], a, b
    elif a >= n:
        top, kept, gone = [left[a], right[a], b], b, a
    else:
        top, kept, gone = [a, b], a, -1
    if blen is not None and gone >= 0:
        blen[kept] += blen[gone]
    parts = ["("]
    stack: list[tuple[int, bool]] = []
    for k, u in enumerate(reversed(top)):
        stack.append((u, False))
        if k < len(top) - 1:
            stack.append((-1, False))
    while stack:
        u, closing = stack.pop()
        if u < 0:
            parts.append(",")
            continue
        if u >= n and not closing:
            parts.append("(")
            stack.extend([(u, True), (right[u], False), (-1, False), (left[u], False)])
            continue
        parts.append(names[u] if u < n else ")")
        if blen is not None:
            parts.append(f":{blen[u]:.9g}")
    parts.append(");\n")
    return "".join(parts)


def write_tree(path: Path, c1: ndarray, c2: ndarray, lengths: ndarray | None) -> None:
    """Writes a tree on leaves T0..T<n-1> as unrooted Newick: (path, c1, c2, lengths) -> None."""
    n = (len(c1) + 1) // 2
    path.write_text(newick(c1, c2, [f"T{i}" for i in range(n)], lengths))


def write_random_start(path: Path, c1: ndarray, c2: ndarray) -> None:
    """Writes a random start tree as the paper's runs gave it to the comparison programs:
    every branch 0.05."""
    n, root = (len(c1) + 1) // 2, len(c1) - 1
    lengths = full(len(c1), START_LENGTH)
    gone = c2[root] if c2[root] >= n else c1[root]
    if gone >= n:
        lengths[gone] = 0.0  # the root's two branches are written as one of 0.05
    write_tree(path, c1, c2, lengths)


def write_rooted_tree(path: Path, c1: ndarray, c2: ndarray) -> None:
    """Writes a tree on leaves T0..T<n-1> as rooted Newick without lengths: (path, c1, c2) -> None.

    From the root 2n - 2, children in c1, c2 order, as the paper's FastME comparison gave its random
    start to both programs: p2v-bme's search depends on how its start tree is rooted.
    Internal nodes must be in post-order (children first), as uniform_random_tree draws them.
    """
    n = (len(c1) + 1) // 2
    text: dict[int, str] = {}
    for u in range(n, 2 * n - 1):
        a, b = int(c1[u]), int(c2[u])
        left = text.pop(a) if a >= n else f"T{a}"
        right = text.pop(b) if b >= n else f"T{b}"
        text[u] = f"({left},{right})"
    path.write_text(text[2 * n - 2] + ";\n")


# ---- Sequences -----------------------------------------------------------------------------------


def pack(seqs: ndarray, words: int) -> ndarray:
    """Packs sequences into the paper's bit planes: (rows x sites) ASCII -> rows x 5 x words uint64.

    Planes A, C, G, T (or U), valid; site j is bit j % 64 of word j // 64.
    """
    rows, sites = seqs.shape
    planes = zeros((rows, 5, words * 64), dtype=bool)
    upper = seqs & 0xDF  # ASCII upper case
    for k, base in enumerate(b"ACGT"):
        planes[:, k, :sites] = upper == base
    planes[:, 3, :sites] |= upper == ord("U")
    planes[:, 4, :sites] = planes[:, :4, :sites].any(axis=1)
    return packbits(planes, axis=2, bitorder="little").view("<u8")


def unpack(planes: ndarray, sites: int) -> ndarray:
    """Unpacks bit planes: rows x 5 x words uint64
    -> rows x sites ASCII, '-' where no base is set."""
    bits = unpackbits(planes.view(uint8).reshape(planes.shape[0], 5, -1), axis=2, bitorder="little")
    bits = bits[:, :, :sites].astype(bool)
    out = full((planes.shape[0], sites), GAP, dtype=uint8)
    for k in range(4):
        out[bits[:, k]] = BASES[k]
    return out


def write_fasta_rows(handle, seqs: ndarray, first: int) -> None:
    """Appends sequences to an open FASTA file:
    (handle, rows x sites ASCII, number of the first) -> None."""
    for i, row in enumerate(seqs):
        handle.write(b">T%d\n" % (first + i))
        handle.write(row.tobytes())
        handle.write(b"\n")


def read_fasta(path: Path) -> tuple[list[str], list[str]]:
    """Reads a FASTA file (.xz too):
    path -> (names, sequences); a name is the header's first word."""
    names: list[str] = []
    seqs: list[str] = []
    chunks: list[str] = []
    opener = open_xz if path.suffix == ".xz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                if names:
                    seqs.append("".join(chunks))
                names.append(line[1:].strip().split()[0])
                chunks = []
            else:
                chunks.append(line.strip())
    if names:
        seqs.append("".join(chunks))
    return names, seqs


def mutate_level(rng: Generator, parent: ndarray, words: int) -> ndarray:
    """Makes ten children of each sequence:
    rows x 5 x words planes -> 10 rows x 5 x words, children of a row together.

    For each branch of the ten-leaf clade, each valid site is redrawn uniformly from the four bases
    with probability 2^-7, in every leaf below the branch.
    """
    rows = parent.shape[0]
    valid = parent[:, 4, :]
    level = repeat(parent[:, None, :, :], CLADE, axis=1)
    word_range = 1 << 64
    for group in CLADE_GROUPS:
        mut = rng.integers(0, word_range, size=(rows, words), dtype=uint64)
        for _ in range(MUTATION_ANDS):
            mut &= rng.integers(0, word_range, size=(rows, words), dtype=uint64)
        mut &= valid
        b0 = rng.integers(0, word_range, size=(rows, words), dtype=uint64)
        b1 = rng.integers(0, word_range, size=(rows, words), dtype=uint64)
        new = [~b0 & ~b1 & mut, b0 & ~b1 & mut, ~b0 & b1 & mut, b0 & b1 & mut]
        keep = ~mut
        for j in group:
            for k in range(4):
                level[:, j, k, :] = (level[:, j, k, :] & keep) | new[k]
    return level.reshape(rows * CLADE, 5, words)


def simulate_jc(
    sim_jc69: str,
    c1: ndarray,
    c2: ndarray,
    blen: ndarray,
    sites: int,
    seed: int,
    fasta: Path,
) -> None:
    """Simulates JC69 sites along a tree with sim_jc69_fast:
    (tree, sites, seed) -> FASTA T0..T<n-1>.

    Each 64-site word has a random stream of its own (from the seed and the word),
    so a longer alignment repeats the sites of a shorter one on the same tree and seed.
    """
    n = (len(c1) + 1) // 2
    with TemporaryDirectory(dir=fasta.parent) as tmp:
        tree_bin = Path(tmp) / "tree.bin"
        with tree_bin.open("wb") as handle:
            array([n], dtype="<i4").tofile(handle)
            c1.astype("<i4").tofile(handle)
            c2.astype("<i4").tofile(handle)
            blen.astype("<f4").tofile(handle)
        words = (sites + 63) // 64
        run(
            [
                sim_jc69,
                "--sim",
                str(tree_bin),
                str(sites),
                str(words),
                str(seed),
                "/dev/null",
                str(fasta),
                "none",
            ],
            check=True,
            capture_output=True,
        )


# ---- Datasets ------------------------------------------------------------------------------------


class Rnasim:
    """The RNASim reference tree and alignment, read once when a set needs them."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.tree: tuple[list[list[int]], list[str], list[float]] | None = None
        self.leaves: list[str] = []
        self.seqs: dict[str, str] = {}

    def subset(self, n: int, rep: int, scale: float) -> tuple[ndarray, ndarray, ndarray, list[str]]:
        """Draws a replicate's leaves:
        (n, replicate, scale) -> (c1, c2, lengths / scale, RNASim names of T0...)."""
        if self.tree is None:
            self.tree = parse_newick((self.folder / "true_topo.tree").read_text().strip())
            self.leaves = [nm for kids, nm in zip(self.tree[0], self.tree[1]) if not kids]
        chosen = Random(SUBSET_SEED + rep).sample(self.leaves, n)
        children, name, length, root = induce_subtree(*self.tree, set(chosen))
        c1, c2, blen = rooted_arrays(
            children, name, length, root, {nm: i for i, nm in enumerate(chosen)}, scale
        )
        return c1, c2, blen, chosen

    def sequences(self, names: list[str]) -> ndarray:
        """Looks up RNASim's sequences: names -> rows x sites ASCII, in the order given."""
        if not self.seqs:
            plain = self.folder / "aln_dna.fa"
            self.seqs = dict(
                zip(*read_fasta(plain if plain.exists() else self.folder / "aln_dna.fa.xz"))
            )
        return frombuffer(
            "".join(self.seqs[nm] for nm in names).encode("ascii"), dtype=uint8
        ).reshape(len(names), -1)


def start_seed(name: str, n: int, rep: int, expanded: bool) -> int:
    """Gives the random start tree's seed as the paper's runs:
    (set, n, replicate, expanded?) -> seed."""
    if name == "yule10k":
        return YULE_START_SEED + rep
    return START_SEED + n if expanded else START_SEED + START_SEED_STEP * rep + n


def make_dataset(name: str, n: int, rep: int, out: Path, rnasim: Rnasim, args: Namespace) -> None:
    """Writes one dataset: (set, n, replicate) -> <out>/rep<r>.{fa,true.nwk,random_start.nwk}."""
    sites, words, _, sim_seed = SETS[name]
    out.mkdir(parents=True, exist_ok=True)
    fasta = out / f"rep{rep}.fa"
    expanded = name in EXPANDED_SETS and n > args.expansion_base
    if expanded:
        factor = n // args.expansion_base
        if rep != 0 or factor not in EXPANSION_CHUNK or factor * args.expansion_base != n:
            raise SystemExit(f"{name}: {n} is not 10x or 100x {args.expansion_base}, replicate 0")
        base = out.parent / str(args.expansion_base) / "rep0.fa"
        if not base.exists():
            make_dataset(name, args.expansion_base, 0, base.parent, rnasim, args)
        c1, c2, _, _ = rnasim.subset(args.expansion_base, 0, JC_SCALE)
        base_seqs = read_fasta(base)[1]
        text = frombuffer("".join(base_seqs).encode("ascii"), dtype=uint8).reshape(
            len(base_seqs), -1
        )
        sites = text.shape[1]
        planes = pack(text, words)
        del base_seqs, text
        for _ in range(1 if factor == CLADE else 2):
            c1, c2 = expand_tree(c1, c2, (len(c1) + 1) // 2)
        write_tree(out / f"rep{rep}.true.nwk", c1, c2, None)
        rng = default_rng(EXPANSION_SEED)
        chunk = EXPANSION_CHUNK[factor]
        with fasta.open("wb") as handle:
            for s0 in range(0, planes.shape[0], chunk):
                seqs = mutate_level(rng, planes[s0 : s0 + chunk], words)
                if factor > CLADE:
                    seqs = mutate_level(rng, seqs, words)
                for r0 in range(0, seqs.shape[0], UNPACK_ROWS):
                    write_fasta_rows(
                        handle,
                        unpack(seqs[r0 : r0 + UNPACK_ROWS], sites),
                        s0 * factor + r0,
                    )
    elif name == "yule10k":
        c1, c2, blen = yule_tree(n, rep)
        simulate_jc(args.sim_jc69, c1, c2, blen, sites, sim_seed + SIM_SEED_STEP * rep, fasta)
        write_tree(out / f"rep{rep}.true.nwk", c1, c2, blen)
    elif name == "rnasim":
        c1, c2, _, chosen = rnasim.subset(n, rep, 1.0)
        with fasta.open("wb") as handle:
            write_fasta_rows(handle, rnasim.sequences(chosen), 0)
        write_tree(out / f"rep{rep}.true.nwk", c1, c2, None)
    else:
        c1, c2, blen, _ = rnasim.subset(n, rep, JC_SCALE)
        simulate_jc(args.sim_jc69, c1, c2, blen, sites, sim_seed + SIM_SEED_STEP * rep, fasta)
        write_tree(out / f"rep{rep}.true.nwk", c1, c2, blen)
    write_random_start(
        out / f"rep{rep}.random_start.nwk",
        *uniform_random_tree(n, start_seed(name, n, rep, expanded)),
    )


def make_fastme_set(
    name: str, n: int, rep: int, out: Path, rnasim: Rnasim, args: Namespace
) -> None:
    """Writes one set of the FastME comparison:
    (set, n, replicate) -> <out>/rep<r>.{fa,true.nwk,random_start.nwk}.

    As the paper's: the sequences sorted(default_rng(r + 1).choice(N, n)) of the N-sequence set of
    replicate r (N = --fastme-base, 10^4; made first if missing), renamed T0..T<n-1> in that order;
    that set's true tree restricted to them (topology only);
    a uniform random start tree (seed r + 1), written rooted.
    """
    base = args.datasets_dir / name / str(args.fastme_base)
    if not (base / f"rep{rep}.random_start.nwk").exists():
        make_dataset(name, args.fastme_base, rep, base, rnasim, args)
    names, seqs = read_fasta(base / f"rep{rep}.fa")
    if names != [f"T{i}" for i in range(len(names))]:
        raise ValueError(f"{base / f'rep{rep}.fa'}: the names are not T0..T<N-1> in order")
    seed = rep + FASTME_SEED_STEP
    chosen = sort(default_rng(seed).choice(len(seqs), n, replace=False))
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"rep{rep}.fa").open("w") as handle:
        handle.writelines(f">T{k}\n{seqs[i]}\n" for k, i in enumerate(chosen))
    leaf_id = {f"T{i}": k for k, i in enumerate(chosen)}
    tree = parse_newick((base / f"rep{rep}.true.nwk").read_text())
    children, label, length, root = induce_subtree(*tree, set(leaf_id))
    c1, c2, _ = rooted_arrays(children, label, length, root, leaf_id, 1.0)
    write_tree(out / f"rep{rep}.true.nwk", c1, c2, None)
    write_rooted_tree(out / f"rep{rep}.random_start.nwk", *uniform_random_tree(n, seed))


def make_fastme_subsets(
    name: str,
    source_set: str,
    todo: list[tuple[int, int]],
    rnasim: Rnasim,
    args: Namespace,
) -> None:
    """Writes subsets of a 10^6-sequence set for the FastME comparison:
    (set, [(n, k)]) -> fastme/<set>/<n>/rep<k>.*.

    As the paper's (jc1m, rna1m):
    subset k of n = the sequences sorted(default_rng(1000 n + k).choice(N, n)) of the N-sequence set
    of replicate 0 (N = 10 x --expansion-base, 10^6; made first if missing),
    renamed T0..T<n-1> in that order; that set's true tree restricted to them (topology only);
    a uniform random start tree (seed k + 1), written rooted.
    One pass over the large alignment writes the sequences of every subset.
    """
    size = CLADE * args.expansion_base
    source = args.datasets_dir / source_set / str(size)
    if not (source / "rep0.random_start.nwk").exists():
        make_dataset(source_set, size, 0, source, rnasim, args)
    chosen = {
        (n, k): sort(default_rng(FASTME_1M_SEED_STEP * n + k).choice(size, n, replace=False))
        for n, k in todo
    }
    takers: dict[int, list[tuple[int, int]]] = {}  # sequence -> the subsets with it
    for key, sel in chosen.items():
        for i in sel.tolist():
            takers.setdefault(i, []).append(key)
    folders = {(n, k): args.datasets_dir / FASTME / name / str(n) for n, k in todo}
    for folder in set(folders.values()):
        folder.mkdir(parents=True, exist_ok=True)
    handles = {key: (folders[key] / f"rep{key[1]}.fa").open("w") for key in todo}
    written = dict.fromkeys(todo, 0)
    try:
        with (source / "rep0.fa").open() as fasta:
            for i, (header, seq) in enumerate(zip(fasta, fasta)):
                for key in takers.get(i, ()):
                    if header.strip() != f">T{i}":
                        raise ValueError(f"{source / 'rep0.fa'}: sequence {i} is {header.strip()}")
                    handles[key].write(f">T{written[key]}\n{seq.strip()}\n")
                    written[key] += 1
    finally:
        for handle in handles.values():
            handle.close()
    if any(written[(n, k)] != n for n, k in todo):
        raise ValueError(f"{source / 'rep0.fa'}: fewer than {size} sequences")
    tree = parse_newick((source / "rep0.true.nwk").read_text())
    for (n, k), sel in chosen.items():
        leaf_id = {f"T{i}": j for j, i in enumerate(sel.tolist())}
        children, label, length, root = induce_subtree(*tree, set(leaf_id))
        c1, c2, _ = rooted_arrays(children, label, length, root, leaf_id, 1.0)
        folder = folders[(n, k)]
        write_tree(folder / f"rep{k}.true.nwk", c1, c2, None)
        start = uniform_random_tree(n, k + FASTME_SEED_STEP)
        write_rooted_tree(folder / f"rep{k}.random_start.nwk", *start)
        print(f"simulate_data: {FASTME} {name} {n} rep{k}", flush=True)


def make_fastme(args: Namespace, rnasim: Rnasim) -> None:
    """Makes the FastME comparison's sets that are missing:
    args -> data/datasets/fastme/<set>/<n>/rep<r>.*."""
    replicates = args.replicates if args.replicates is not None else REPLICATES
    for name in FASTME_SETS:
        for n in sorted(args.sizes or FASTME_SIZES):
            for rep in replicates:
                out = args.datasets_dir / FASTME / name / str(n)
                if args.overwrite or not (out / f"rep{rep}.random_start.nwk").exists():
                    make_fastme_set(name, n, rep, out, rnasim, args)
                    print(f"simulate_data: {FASTME} {name} {n} rep{rep}", flush=True)
    for name, source_set in FASTME_1M_SETS.items():
        todo = []
        for n in sorted(args.sizes or FASTME_1M_SUBSETS):
            subsets = (
                args.replicates
                if args.replicates is not None
                else range(FASTME_1M_SUBSETS.get(n, 0))
            )
            folder = args.datasets_dir / FASTME / name / str(n)
            for k in subsets:
                if args.overwrite or not (folder / f"rep{k}.random_start.nwk").exists():
                    todo.append((n, k))
        if todo:
            make_fastme_subsets(name, source_set, todo, rnasim, args)


def main() -> None:
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument(
        "--sets",
        nargs="+",
        choices=[*SETS, FASTME],
        default=[*SETS, FASTME],
        help=f"datasets to make ({FASTME}: the FastME comparison's subsets of jc10k and rnasim, jc1m and rna1m)",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        help="numbers of sequences (default: each set's own)",
    )
    parser.add_argument(
        "--replicates",
        nargs="+",
        type=int,
        help="replicates to make (default: 0-4; fastme's jc1m, rna1m: the paper's subsets per size)",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path(DEFAULT_DATASETS_DIR),
        help="input/output folder",
    )
    parser.add_argument("--sim-jc69", default=SIM_JC69, help="the sim_jc69_fast program")
    parser.add_argument(
        "--expansion-base",
        type=int,
        default=EXPANSION_BASE,
        help="size whose replicate 0 the larger rnasim/jc10k sets expand (smaller only for tests)",
    )
    parser.add_argument(
        "--fastme-base",
        type=int,
        default=FASTME_BASE,
        help="size of the sets the FastME comparison's subsets are drawn from (smaller only for tests)",
    )
    parser.add_argument("--overwrite", action="store_true", help="make datasets that exist again")
    args = parser.parse_args()

    rnasim = Rnasim(args.datasets_dir / "rnasim")
    replicates = args.replicates if args.replicates is not None else REPLICATES
    for name in args.sets:
        if name == FASTME:
            make_fastme(args, rnasim)
            continue
        for n in sorted(args.sizes or SETS[name][2]):
            expanded = name in EXPANDED_SETS and n > args.expansion_base
            for rep in [0] if expanded else replicates:
                out = args.datasets_dir / name / str(n)
                if not args.overwrite and (out / f"rep{rep}.random_start.nwk").exists():
                    continue
                make_dataset(name, n, rep, out, rnasim, args)
                print(f"simulate_data: {name} {n} rep{rep}", flush=True)


if __name__ == "__main__":
    main()
