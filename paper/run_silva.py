"""SILVA Release 144: tree, bootstrap, supports, biology, domains, leave-out series, GTDB (Fig. 3a,b; tables
silva_support, silva_gtdb).

Inputs:  data/datasets/silva144/SILVA_144_SSURef_tax_silva_full_align_trunc.fasta[.gz]      (download_data.py)
         --arb-dump: the SILVA 144 ARB database as text, made with ARB's tools (only for the gtdb step)
Options: --steps, --replicates, --bootstrap, --max-sites, --min-sites, --seed, --accelerator, --devices,
         --leaveout-runs, --arb-dump, --fasttree, --workers, --overwrite
Flow:    distinct     the alignment p2v-bme searches: the 1,600 best-occupied columns (--max-sites), the sequences
           |          with at least 1,200 valid sites there (--min-sites; the others are left out); one sequence
           |          per group of identical ones; every sequence's taxonomy
           v
         tree         p2v-bme -s searched.fa -m GTR+GC --seed 1; reattach_left_out puts each sequence left out
           |          next to its nearest searched sequence (tree.all.nwk)
           v
         replicates   p2v-bme -s searched.fa ... -b r-r      (one run per replicate; spread them
           |          with --replicates)
           v
         supports     p2v-bme -t tree.all.nwk --supports boot.nwk
           |
           v
         records      the tree of the distinct sequences: support by split size, taxonomic F1 and organelles,
           |          the domains in every bootstrap tree, the run records
           v
         leaveout     repeat searches, and searches without groups of sequences, compared with the tree
           |
           v
         gtdb         the sequences with a GTDB classification: their own trees (p2v-bme, FastTree with --fasttree)
                      and the tree, scored against GTDB's ranks
Outputs: data/results/silva/
           taxonomy.tsv, searched.fa, left_out.fa, distinct.fa, distinct.json
           tree.{nwk,json} (the searched sequences), tree.all.{nwk,json} (every sequence)
           boot/rep<r>.{boottrees,json}, boot.nwk, tree.boot.{support.nwk,json}
           runs.jsonl, silva144_dedup_tbe_summary.json, silva144_dedup_tbe_B100_quantiles.json,
           silva144_real_biology_B100.json, domain_arrangements.json        (the paper's record names and fields)
           leaveout/runs.json, leaveout/<run>/{tree.{nwk,json}, cmp.{support.nwk,json}, summary.json, branches.npz}
           gtdb/{gtdb.fa, gtdb_labels.tsv, ours.{nwk,json}, ft_me.nwk, ft_ml.nwk}, gtdb_labelled_scores.json

    pixi run -e paper python -m paper.run_silva --steps tree replicates --replicates 1-25
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections.abc import Iterator
from gzip import open as gzip_open
from hashlib import blake2b
from json import dumps
from multiprocessing import Pool
from pathlib import Path
from re import compile as regex
from re import match, search, sub
from shutil import copyfile
from subprocess import run

from numpy import (
    arange,
    argmax,
    argmin,
    argsort,
    array,
    bincount,
    concatenate,
    count_nonzero,
    cumsum,
    empty,
    flatnonzero,
    fmax,
    frombuffer,
    full,
    int8,
    int16,
    int64,
    isin,
    isnan,
    lexsort,
    logical_or,
    maximum,
    median,
    minimum,
    nan,
    nan_to_num,
    nanmean,
    nansum,
    ndarray,
    percentile,
    r_,
    savez_compressed,
    sort,
    uint8,
    unique,
    where,
    zeros,
)

from paper._config import (
    DEFAULT_DATASETS_DIR,
    DEFAULT_RESULTS_DIR,
    FASTTREE,
    P2VBME,
    REATTACH_LEFT_OUT,
)
from paper.utils import (
    concatenate_trees,
    read_json,
    read_tree,
    run_p2vbme,
    span,
    supports,
    write_json,
)

ALIGNMENT = (
    f"{DEFAULT_DATASETS_DIR}/silva144/SILVA_144_SSURef_tax_silva_full_align_trunc.fasta"
)
STEPS = ("distinct", "tree", "replicates", "supports", "records", "leaveout", "gtdb")
SIZE_BINS = (("2", 2, 2), ("3-10", 3, 10), ("11-100", 11, 100), ("101-1k", 101, 1000))
SIZE_BINS += (("1k-10k", 1001, 10000), (">10k", 10001, 2000000000))
TOP = 30  # largest phyla and classes in the biology summary
# Domain arrangements: share of misplaced sequences with which a group still counts as one
TOLERANCES = (0.01, 0.02, 0.05)
TOLERANCE = 0.02
# Names of the archaeal kingdoms in SILVA 144's paths, and the names in common use
KINGDOMS = {
    "Asgard": "Promethearchaeati",
    "TACK": "Thermoproteati",
    "DPANN": "Nanobdellati",
    "Euryarchaeota": "Methanobacteriati",
}
DPANN_NAMES = (
    "Nanobdell|Aenigmat|Micrarchae|Altarchae|Iainarchae|Woesearchae|Nanohalo|"
    "Huberarchae|Undinarchae"
)
ORGANISMS = ("metagenome", "uncultured", "unnamed_isolate", "named_species")
MIN_SEQUENCES = 4  # the fewest p2v-bme searches
# Leave-out runs: one per phylum of at least this many distinct sequences
BIG_PHYLUM = 10000
SUPPORT_BINS = ((0, 10), (10, 30), (30, 50), (50, 70), (70, 90), (90, 95), (95, 99))
SUPPORT_BINS += ((99, 100.01),)
RANKS = ("domain", "phylum", "class", "order", "family", "genus", "species")
MIN_TAXON = 5  # GTDB taxa with fewer labelled sequences are not scored
# Columns of the GTDB-classified sequences kept: at least this share of A/C/G/T
GTDB_MIN_OCCUPANCY = 0.10
VALID = zeros(256, bool)  # what p2v-bme counts as a valid site
CANONICAL = full(256, ord("-"), uint8)  # the base p2v-bme reads, '-' for a missing site
for _c, _b in zip(b"ACGTUacgtu", b"ACGTTACGTT"):
    VALID[_c], CANONICAL[_c] = True, _b


# ---- The alignment p2v-bme searches ----


def fasta_records(path: Path) -> Iterator[tuple[bytes, bytes]]:
    """Reads a FASTA file (plain or .gz) one record at a time: path -> (header without '>', sequence)."""
    with gzip_open(path, "rb") if path.suffix == ".gz" else open(path, "rb") as fh:
        header, parts = None, []
        for line in fh:
            if line.startswith(b">"):
                if header is not None:
                    yield header, b"".join(parts)
                header, parts = line[1:].rstrip(), []
            else:
                parts.append(line.rstrip())
        if header is not None:
            yield header, b"".join(parts)


def source(alignment: Path) -> Path:
    """Finds the alignment to read: path -> the path, or its .gz if only that exists."""
    gz = Path(f"{alignment}.gz")
    return gz if not alignment.exists() and gz.exists() else alignment


def distinct(
    alignment: Path, out: Path, max_sites: int, min_sites: int, overwrite: bool
) -> None:
    """Writes the alignment p2v-bme searches: alignment -> searched.fa, left_out.fa, distinct.fa, distinct.json,
    taxonomy.tsv.

    Columns: the max_sites with the most valid bases (ties: the earlier column), in order, with characters other than
    A, C, G, T/U as '-' (missing). Sequences with fewer than min_sites valid sites there go to left_out.fa, the others
    to searched.fa. distinct.fa: the first of each group of identical searched sequences (the same bases and missing
    sites), as the paper counts distinct sequences (p2v-bme searches two of each group).
    """
    if not overwrite and (out / "distinct.json").exists():
        return
    alignment = source(alignment)
    out.mkdir(parents=True, exist_ok=True)
    counts = None
    with open(out / "taxonomy.tsv", "w", encoding="utf-8") as tax:
        for header, seq in fasta_records(alignment):
            name, _, path = header.decode().partition(" ")
            tax.write(f"{name}\t{path.strip()}\n")
            valid = VALID[frombuffer(seq, uint8)]
            counts = valid.astype(int64) if counts is None else counts + valid
    if counts is None:
        raise SystemExit(f"{alignment}: no sequences")
    n_columns = counts.size
    columns = arange(n_columns)
    if 0 < max_sites < n_columns:
        columns = sort(lexsort((arange(n_columns), -counts))[:max_sites])
    groups: dict[bytes, int] = {}
    n_sequences = n_left_out = 0
    parts = [out / f"{name}.fa.part" for name in ("searched", "left_out", "distinct")]
    with (
        open(parts[0], "wb") as searched,
        open(parts[1], "wb") as left_out,
        open(parts[2], "wb") as fo,
    ):
        for header, seq in fasta_records(alignment):
            n_sequences += 1
            row = CANONICAL[frombuffer(seq, uint8)[columns]]
            record = b">" + header.split()[0] + b"\n" + row.tobytes() + b"\n"
            if count_nonzero(row != ord("-")) < min_sites:
                n_left_out += 1
                left_out.write(record)
                continue
            searched.write(record)
            key = blake2b(row.tobytes(), digest_size=16).digest()
            groups[key] = groups.get(key, 0) + 1
            if groups[key] == 1:
                fo.write(record)
    for part in parts:
        part.rename(part.with_suffix(""))
    write_json(
        out / "distinct.json",
        {
            "sequences": n_sequences,
            "columns": n_columns,
            "max_sites": max_sites,
            "min_sites": min_sites,
            "left_out": n_left_out,
            "searched": n_sequences - n_left_out,
            "distinct": len(groups),
            # p2v-bme keeps two of each group in the search and sets the others aside
            "identical_set_aside": sum(max(0, c - 2) for c in groups.values()),
            "columns_kept": columns.tolist(),
        },
    )


def check_distinct(out: Path) -> dict:
    """Checks that the distinct sequences are those p2v-bme searched: results -> distinct.json's record.

    p2v-bme's report counts the sequences it searched and the identical ones it set aside; both must match.
    """
    d, report = (
        read_json(out / "distinct.json"),
        read_json(out / "tree.json")["details"]["input"],
    )
    ours = (d["searched"], d["identical_set_aside"])
    theirs = (report["sequences"], report["identical_set_aside"])
    if ours != theirs:
        raise SystemExit(
            f"distinct.json (searched, set aside) {ours} != p2v-bme's {theirs}"
        )
    return d


def reattach(out: Path, tool: str, overwrite: bool) -> None:
    """Puts the sequences left out of the search back into its tree: results -> tree.all.nwk, tree.all.json.

    Each joins its nearest searched sequence by zero-length branches (paper/tools/reattach_left_out.cpp, whose
    counts go to tree.all.json).
    """
    if not overwrite and (out / "tree.all.json").exists():
        return
    d = read_json(out / "distinct.json")
    if d["left_out"] == 0:
        copyfile(out / "tree.nwk", out / "tree.all.nwk")
        write_json(
            out / "tree.all.json",
            {"searched": d["searched"], "left_out": 0, "no_shared_site": 0},
        )
        return
    files = ("tree.nwk", "searched.fa", "left_out.fa", "tree.all.nwk")
    with open(out / "tree.all.json.part", "w") as fo:
        run([tool, *(str(out / f) for f in files)], stdout=fo, check=True)
    (out / "tree.all.json.part").rename(out / "tree.all.json")


def read_taxonomy(out: Path) -> dict[str, str]:
    """Reads every sequence's SILVA taxonomy: results -> {name: path}."""
    lines = (out / "taxonomy.tsv").read_text(encoding="utf-8").splitlines()
    return dict(line.split("\t", 1) for line in lines)


def fasta_names(path: Path) -> list[str]:
    """Reads the names of a FASTA file's sequences: path -> names, in order."""
    with open(path, "rb") as fh:
        return [line[1:].split()[0].decode() for line in fh if line.startswith(b">")]


def write_subset(source: Path, keep: set[str], out: Path) -> None:
    """Copies the named sequences of a FASTA file: (source, names) -> out."""
    with open(out, "wb") as fo:
        for header, seq in fasta_records(source):
            if header.split()[0].decode() in keep:
                fo.write(b">" + header + b"\n" + seq + b"\n")


# ---- Trees as arrays (paper.utils.read_tree) ----


def prune(tree: dict, keep: ndarray) -> dict:
    """Restricts a tree to some of its leaves: (tree, keep mask over the leaves) -> the induced tree, same arrays.

    A branch of the induced tree that merges a chain of branches gets their summed length and their largest TBE and
    exact support (none for leaves). Leaves keep their order; internal nodes keep their order, the root last.
    """
    parent, first, size = tree["parent"], tree["first"], tree["size"]
    cum = concatenate([[0], cumsum(keep)])
    m = int(cum[-1])
    before, below = cum[first], cum[first + size] - cum[first]
    alive = flatnonzero(below > 0)
    # A chain of branches merged into one has one set of kept leaves: (kept before it, kept below it).
    keys, group = unique(before[alive] * (m + 1) + below[alive], return_inverse=True)
    n_groups = keys.size
    top = full(n_groups, -1, int64)
    maximum.at(top, group, alive)
    g_before, g_below = keys // (m + 1), keys % (m + 1)
    leaf = g_below == 1
    order = concatenate(
        [
            flatnonzero(leaf)[argsort(g_before[leaf])],
            flatnonzero(~leaf)[argsort(top[~leaf])],
        ]
    )
    new_id = empty(n_groups, int64)
    new_id[order] = arange(n_groups)
    group_of = full(parent.size, -1, int64)
    group_of[alive] = group
    above = parent[top]
    new_parent = where(above >= 0, new_id[group_of[above]], -1)
    length = bincount(
        group, weights=nan_to_num(tree["length"][alive]), minlength=n_groups
    )
    has_length = zeros(n_groups, bool)
    logical_or.at(has_length, group, ~isnan(tree["length"][alive]))
    length[~has_length | (g_below == m)] = nan
    tbe, exact = full(n_groups, nan), full(n_groups, nan)
    fmax.at(tbe, group, tree["tbe"][alive])
    fmax.at(exact, group, tree["exact"][alive])
    tbe[leaf], exact[leaf] = nan, nan
    names = tree["names"]
    return {
        "names": [names[i] for i in flatnonzero(keep)],
        "parent": new_parent[order],
        "first": g_before[order],
        "size": g_below[order],
        "length": length[order],
        "tbe": tbe[order],
        "exact": exact[order],
    }


def branch_rows(tree: dict) -> dict[str, ndarray]:
    """Lists the internal branches of the unrooted tree once each: tree -> {node, p, length, tbe, exact}.

    p: leaves on the smaller side; branches with p < 2 are not internal. When the root has two children, their
    branches are one: it is listed once, with the larger supports of the two.
    """
    n, parent, size = len(tree["names"]), tree["parent"], tree["size"]
    root = parent.size - 1
    node = arange(n, root)
    p = minimum(size[node], n - size[node])
    tbe, exact, length = (
        tree["tbe"][node].copy(),
        tree["exact"][node].copy(),
        tree["length"][node].copy(),
    )
    keep = p >= 2
    children = flatnonzero(parent == root)
    if children.size == 2 and (children >= n).all():
        a, b = children - n
        tbe[a], exact[a] = fmax(tbe[a], tbe[b]), fmax(exact[a], exact[b])
        length[a] = nansum([length[a], length[b]])
        keep[b] = False
    return {
        "node": node[keep],
        "p": p[keep],
        "length": length[keep],
        "tbe": tbe[keep],
        "exact": exact[keep],
    }


def distinct_tree(out: Path) -> dict:
    """Reads the tree of the distinct sequences, with the bootstrap supports: results -> tree arrays."""
    full_tree = read_tree(out / "tree.boot.support.nwk")
    names = set(fasta_names(out / "distinct.fa"))
    return prune(full_tree, array([x in names for x in full_tree["names"]]))


# ---- Records ----


def support_records(
    tree: dict, out: Path, n_all: int, n_boot: int, version: str
) -> None:
    """Writes the supports by split size: tree of the distinct sequences -> the summary and quantile records."""
    b = branch_rows(tree)
    p, tbe, exact, length = b["p"], b["tbe"], b["exact"], b["length"]
    if isnan(tbe).any():
        raise SystemExit(
            f"{int(isnan(tbe).sum())} branches of the distinct sequences' tree have no support"
        )
    k = len(tree["names"])
    summary_bins, quantile_bins = [], []
    for name, lo, hi in SIZE_BINS:
        m = (p >= lo) & (p <= hi)
        x = tbe[m]
        if not m.any():
            continue
        q10, q25, q50, q75, q90 = percentile(x, [10, 25, 50, 75, 90])
        summary_bins.append(
            {
                "bin": name,
                "p_min": lo,
                "p_max": hi,
                "n_branches": int(m.sum()),
                "mean_tbe_pct": float(x.mean()),
                "median_tbe_pct": float(q50),
                "mean_exact_fbp_pct": float(exact[m].mean()),
                "mean_branch_length": float(nanmean(length[m])),
            }
        )
        positive = length[m][length[m] > 0]
        rec = {
            "bin": name,
            "p_min": lo,
            "p_max": hi,
            "n_branches": int(m.sum()),
            "mean_tbe_pct": float(x.mean()),
            "median_tbe_pct": float(q50),
            "q25_tbe_pct": float(q25),
            "q75_tbe_pct": float(q75),
            "q10_tbe_pct": float(q10),
            "q90_tbe_pct": float(q90),
            "mean_exact_fbp_pct": float(exact[m].mean()),
            "mean_pos_branch_length": float(positive.mean()) if positive.size else nan,
        }
        for c in (70, 80, 90):
            rec[f"n_ge{c}"] = int((x >= c).sum())
            rec[f"frac_ge{c}_pct"] = 100.0 * rec[f"n_ge{c}"] / rec["n_branches"]
        quantile_bins.append(rec)
    head = {
        "K_unique_leaves": k,
        "B_replicates": n_boot,
        "total_internal_branches": int(p.size),
        "overall_mean_tbe_pct": float(tbe.mean()),
        "overall_mean_exact_fbp_pct": float(exact.mean()),
    }
    write_json(out / "silva144_dedup_tbe_summary.json", head | {"bins": summary_bins})
    write_json(
        out / "silva144_dedup_tbe_B100_quantiles.json",
        {"K_unique_leaves": k, "N_expanded_leaves": n_all}
        | head
        | {
            "version": version,
            "source": "tree.boot.support.nwk, restricted to distinct.fa",
            "bins": quantile_bins,
        },
    )


def best_split(tree: dict, leaves: ndarray, name: str) -> dict:
    """Finds the branch whose split matches a set of leaves best, by F1: (tree, leaves) -> its record."""
    k, size, first, parent = (
        len(tree["names"]),
        tree["size"],
        tree["first"],
        tree["parent"],
    )
    root = parent.size - 1
    total = leaves.size
    below = counts(tree, leaves)[:root]
    size = size[:root]
    f1_clade = 2.0 * below / (size + total)
    f1_comp = 2.0 * (total - below) / ((k - size) + total)
    u, v = int(argmax(f1_clade)), int(argmax(f1_comp))
    comp = bool(f1_comp[v] > f1_clade[u])
    node = v if comp else u
    tp = int(total - below[node]) if comp else int(below[node])
    clade = int(k - size[node]) if comp else int(size[node])
    chain = [node]
    while chain[-1] != root:
        chain.append(int(parent[chain[-1]]))

    def val(arr: ndarray, x: int) -> float:
        return 0.0 if x == root or isnan(arr[x]) else float(arr[x])

    return {
        "name": name,
        "unique_leaves": int(total),
        "best_node": node,
        "is_complement": comp,
        "clade_size": clade,
        "tp_leaves": tp,
        "purity_pct": round(100.0 * tp / clade, 4),
        "recall_pct": round(100.0 * tp / total, 4),
        "f1": round(2.0 * tp / (clade + total), 5),
        "tbe_pct": round(val(tree["tbe"], node), 4),
        "fbp_pct": round(val(tree["exact"], node), 4),
        "bl": val(tree["length"], node),
        "ancestor_nodes": chain,
        "ancestor_sz": [int(tree["size"][x]) for x in chain],
        "ancestor_tbe": [round(val(tree["tbe"], x), 4) for x in chain],
        "ancestor_fbp": [round(val(tree["exact"], x), 4) for x in chain],
        "ancestor_bl": [val(tree["length"], x) for x in chain],
    }


def counts(tree: dict, leaves: ndarray) -> ndarray:
    """Counts the given leaves below every node: (tree, leaf ids) -> one count per node."""
    mark = zeros(len(tree["names"]) + 1, int64)
    mark[leaves + 1] = 1
    cum = cumsum(mark)
    return cum[tree["first"] + tree["size"]] - cum[tree["first"]]


def inside(tree: dict, entry: dict, leaves: ndarray) -> int:
    """Counts the given leaves on a taxon's side of its best split: (tree, its record, leaves) -> count."""
    below = int(counts(tree, leaves)[entry["best_node"]])
    return int(leaves.size - below) if entry["is_complement"] else below


def groups_of(labels: list[str]) -> dict[str, ndarray]:
    """Groups leaves by label, largest group first: labels -> {label: leaf ids}."""
    arr = array(labels)
    names, inverse, count = unique(arr, return_inverse=True, return_counts=True)
    order = argsort(inverse, kind="stable")
    bounds = concatenate([[0], cumsum(count)])
    return {
        str(names[i]): order[bounds[i] : bounds[i + 1]]
        for i in argsort(-count, kind="stable")
    }


def biology(tree: dict, taxonomy: dict[str, str], alpha: float) -> dict:
    """Scores the tree against SILVA's taxonomy: (tree of the distinct sequences, taxonomy) -> the biology record.

    For every domain and the TOP largest phyla and classes: the best-matching branch, its purity, recall, F1 and
    support, and its ancestors; and where the organelle sequences filed under Bacteria are placed.
    """
    paths = [taxonomy[x].split(";") for x in tree["names"]]

    def field(p: list[str], i: int) -> str:
        return p[i] if len(p) > i else ""

    domains = groups_of([p[0] for p in paths])
    phyla = groups_of([f"{p[0]};{field(p, 2)}" for p in paths])
    classes = groups_of([f"{p[0]};{field(p, 3)}" for p in paths])
    root = tree["parent"].size - 1
    record = {
        "K_unique": len(paths),
        "root_node": root,
        "total_tree_length": round(float(nansum(tree["length"][:root])), 2),
        "gamma_alpha": alpha,
        "domains": [best_split(tree, v, n) for n, v in domains.items()],
        "phyla": [best_split(tree, v, n) for n, v in list(phyla.items())[:TOP]],
        "classes": [best_split(tree, v, n) for n, v in list(classes.items())[:TOP]],
    }
    is_bac = array([p[0] == "Bacteria" for p in paths])
    chl = flatnonzero(is_bac & array(["Chloroplast" in p for p in paths]))
    mit = flatnonzero(is_bac & array(["Mitochondria" in p for p in paths]))
    organelle = zeros(len(paths), bool)
    organelle[chl] = organelle[mit] = True
    other = flatnonzero(is_bac & ~organelle)
    euk = next(e for e in record["domains"] if e["name"] == "Eukaryota")
    plant = next(e for e in record["phyla"] if e["name"] == "Eukaryota;Chloroplastida")
    record["eukaryota_clade_breakdown"] = {
        "node_id": euk["best_node"],
        "clade_size": euk["clade_size"],
        "eukaryota_total": euk["unique_leaves"],
        "bacteria_chloroplast_in_clade": inside(tree, euk, chl),
        "bacteria_chloroplast_total": int(chl.size),
        "bacteria_mitochondria_in_clade": inside(tree, euk, mit),
        "bacteria_mitochondria_total": int(mit.size),
        "other_bacteria_in_clade": inside(tree, euk, other),
        "archaea_in_clade": inside(tree, euk, domains.get("Archaea", array([], int64))),
        "chloroplastida_node_id": plant["best_node"],
        "chloroplastida_clade_size": plant["clade_size"],
        "bacteria_chloroplast_in_chloroplastida": inside(tree, plant, chl),
        "bacteria_mitochondria_in_chloroplastida": inside(tree, plant, mit),
        "other_bacteria_in_chloroplastida": inside(tree, plant, other),
    }
    return record


# Domain arrangements: masks of each distinct sequence, set before the worker processes start (they inherit them)
CODES: dict[str, int] = {}


def domain_codes(taxonomy: dict[str, str], names: list[str]) -> dict[str, int]:
    """Encodes each sequence's domain and archaeal kingdom as bits: (taxonomy, names) -> {name: bits of DOMAIN_BITS}."""
    codes = {}
    for x in names:
        path = taxonomy[x]
        domain = path.split(";")[0]
        code = 1 if domain == "Eukaryota" else 0
        if domain == "Archaea":
            code |= 2
            for i, key in enumerate(KINGDOMS.values()):
                code |= (key in path) << (2 + i)
        codes[x] = code
    return codes


def nearest_branch(tree: dict, mask: ndarray) -> tuple[int, ndarray]:
    """Finds the branch nearest to a set of leaves: (tree, leaf mask) -> (misplaced leaves, the leaves on its side)."""
    k, first, size = len(tree["names"]), tree["first"], tree["size"]
    cum = r_[0, cumsum(mask)]
    total = int(mask.sum())
    below = cum[first + size] - cum[first]
    as_clade = (total - below) + (size - below)  # missed members plus strangers inside
    as_rest = below + (
        (k - size) - (total - below)
    )  # the same for the other side of the branch
    node = int(argmin(minimum(as_clade, as_rest)))
    side = zeros(k, bool)
    side[first[node] : first[node] + size[node]] = True
    return int(min(as_clade[node], as_rest[node])), (
        side if as_clade[node] <= as_rest[node] else ~side
    )


def domain_distances(tree: dict) -> dict[str, float]:
    """Measures how the domains are arranged in one tree: tree of the distinct sequences -> distances as fractions.

    archaea_alone: the Archaea's transfer distance to the nearest branch, per archaeon; eukaryota_with_<X>: that of
    the eukaryotes' best side together with the archaea X, per archaeon of X outside that side.
    """
    code = array([CODES[x] for x in tree["names"]])
    eukaryota = (code & 1) > 0
    groups = {name: (code & (1 << (2 + i))) > 0 for i, name in enumerate(KINGDOMS)}
    groups = groups | {
        "Asgard+TACK": groups["Asgard"] | groups["TACK"],
        "Archaea": (code & 2) > 0,
    }
    archaea = groups["Archaea"]
    out = {
        "archaea_alone": nearest_branch(tree, archaea)[0] / max(1, int(archaea.sum()))
    }
    _, estar = nearest_branch(tree, eukaryota)
    for name, group in groups.items():
        distance = nearest_branch(tree, estar | group)[0]
        out[f"eukaryota_with_{name}"] = distance / max(1, int((group & ~estar).sum()))
    return out


def replicate_distances(path: Path) -> dict[str, float]:
    """Measures the domains in one bootstrap tree: tree file -> domain_distances of its distinct sequences."""
    tree = read_tree(path)
    return domain_distances(prune(tree, array([x in CODES for x in tree["names"]])))


def domains(reference: dict, boot: list[Path], workers: int) -> dict:
    """Records the domains' arrangement in the tree and each bootstrap tree: (tree, bootstrap trees) -> record."""
    with Pool(workers) as pool:
        bootstrap = pool.map(replicate_distances, boot)
    ref = domain_distances(reference)
    counted = []
    for tolerance in TOLERANCES:
        split = [t for t in bootstrap if t["archaea_alone"] >= tolerance]
        with_asgard_tack = sum(
            t["eukaryota_with_Asgard+TACK"] < tolerance for t in split
        )
        counted.append(
            {
                "tolerance": tolerance,
                "archaea_one_group": len(bootstrap) - len(split),
                "archaea_not_one_group": len(split),
                "of_those_eukaryota_with_asgard_tack": with_asgard_tack,
            }
        )
    code = array(list(CODES.values()))
    sizes = {"Eukaryota": int((code & 1).sum())}
    sizes |= {
        name: int(((code >> (2 + i)) & 1).sum()) for i, name in enumerate(KINGDOMS)
    }
    sizes["Asgard+TACK"] = int(
        ((code & ((1 << 2) | (1 << 3))) > 0).sum()
    )  # Asgard is bit 2, TACK bit 3
    sizes["Archaea"] = int(((code & 2) > 0).sum())
    return {
        "bootstrap_trees": len(bootstrap),
        "tolerance_used": TOLERANCE,
        "sequences": sizes,
        "reference_archaea_one_group": bool(ref["archaea_alone"] < TOLERANCE),
        "counts": counted,
        "reference": ref,
        "bootstrap": bootstrap,
    }


def records(out: Path, n_boot: int, workers: int) -> None:
    """Writes the records of the tree and its bootstrap: results -> runs.jsonl and the four records."""
    global CODES
    d = check_distinct(out)
    report = read_json(out / "tree.json")
    tree = distinct_tree(out)
    if len(tree["names"]) != d["distinct"]:
        raise SystemExit(
            f"the tree holds {len(tree['names'])} distinct sequences, distinct.fa {d['distinct']}"
        )
    support_records(tree, out, d["sequences"], n_boot, report["version"])
    taxonomy = read_taxonomy(out)
    alpha = report["substitution_model"].get("alpha", nan)
    write_json(out / "silva144_real_biology_B100.json", biology(tree, taxonomy, alpha))
    CODES = domain_codes(taxonomy, tree["names"])
    boot = [out / f"boot/rep{r:03d}.boottrees" for r in range(1, n_boot + 1)]
    write_json(out / "domain_arrangements.json", domains(tree, boot, workers))
    lines = [dumps({"run": "rep000"} | report)]
    lines += [dumps({"run": p.stem} | read_json(p.with_suffix(".json"))) for p in boot]
    (out / "runs.jsonl").write_text("\n".join(lines) + "\n")
    print(f"records: {len(tree['names'])} distinct sequences, {n_boot} bootstrap trees")


# ---- Leave-out series ----


def leaveout_runs(
    names: list[str], taxonomy: dict[str, str]
) -> tuple[dict[str, ndarray], list[dict]]:
    """Lists the leave-out runs: (distinct sequences, taxonomy) -> (categories per sequence, runs).

    Runs: three repeat searches on all sequences, then each group left out or kept alone, and each phylum of at
    least BIG_PHYLUM sequences left out.
    """
    k = len(names)
    phylum, org = zeros(k, int16), zeros(k, int8)
    wgs, length = zeros(k, int8), zeros(k, int8)
    organelle, cpr, dpann = zeros(k, bool), zeros(k, bool), zeros(k, bool)
    ids: dict[str, int] = {}
    for i, name in enumerate(names):
        p = taxonomy[name]
        f = p.split(";")
        o = f[-1].lower()
        phylum[i] = ids.setdefault(";".join(f[:3]), len(ids))
        if "metagenome" in o:
            org[i] = 0
        elif o.startswith("uncultured") or "unidentified" in o or "environmental" in o:
            org[i] = 1
        elif search(
            r"\bsp\.|\bbacterium\b|\barchaeon\b|candidatus|\bclone\b|endosymbiont|symbiont",
            o,
        ):
            org[i] = 2
        else:
            org[i] = 3
        accession, start, stop = (name.split(".") + ["0", "0"])[:3]
        m = match(r"^([A-Z]+)\d+$", accession)
        wgs[i] = 1 if (m and len(m.group(1)) >= 4) else 0
        span = int(stop) - int(start) + 1
        length[i] = 0 if span < 1400 else 1 if span < 1600 else 2
        organelle[i] = f[0] == "Bacteria" and (
            "Chloroplast" in p or "Mitochondria" in p
        )
        cpr[i] = "Minisyncoccota" in p
        dpann[i] = f[0] == "Archaea" and bool(search(DPANN_NAMES, p))
    cats = {
        "phylum": phylum,
        "org": org,
        "wgs": wgs,
        "length": length,
        "organelle": organelle,
        "cpr": cpr,
        "dpann": dpann,
    }
    runs: list[dict] = [{"name": f"seed{s}", "seed": s, "all": True} for s in (2, 3, 4)]
    for j, o in enumerate(ORGANISMS):
        runs += [
            {"name": f"drop_org_{o}", "drop": ["org", j]},
            {"name": f"only_org_{o}", "only": ["org", j]},
        ]
    runs += [
        {"name": "drop_organelles", "drop": ["organelle", 1]},
        {"name": "drop_dpann", "drop": ["dpann", 1]},
        {
            "name": "drop_fast_lineages",
            "drop_any": [["organelle", 1], ["cpr", 1], ["dpann", 1]],
        },
        {"name": "drop_genome_derived", "drop": ["wgs", 1]},
        {"name": "drop_gene_records", "drop": ["wgs", 0]},
        {"name": "drop_len_1200_1399", "drop": ["length", 0]},
        {"name": "drop_len_1400_1599", "drop": ["length", 1]},
        {"name": "drop_len_1600plus", "drop": ["length", 2]},
    ]
    count = bincount(phylum)
    label = {v: key for key, v in ids.items()}
    for x in (int(x) for x in argsort(-count, kind="stable") if count[x] >= BIG_PHYLUM):
        tag = sub(r"[^A-Za-z0-9]+", "_", label[x].split(";", 1)[1]).strip("_")
        runs.append(
            {
                "name": f"drop_phylum_{tag}",
                "drop": ["phylum", x],
                "label": label[x],
                "size": int(count[x]),
            }
        )
    return cats, runs


def keep_mask(run_spec: dict, cats: dict[str, ndarray]) -> ndarray:
    """Selects the sequences of a leave-out run: (run, categories) -> mask over the distinct sequences."""
    k = cats["phylum"].size
    if run_spec.get("all"):
        return full(k, True)
    if "drop" in run_spec:
        return cats[run_spec["drop"][0]] != run_spec["drop"][1]
    if "only" in run_spec:
        return cats[run_spec["only"][0]] == run_spec["only"][1]
    keep = full(k, True)
    for f, v in run_spec["drop_any"]:
        keep &= cats[f] != v
    return keep


def leaveout_summary(name: str, n_leaves: int, b: dict[str, ndarray]) -> dict:
    """Summarises a leave-out comparison: (run, branches) -> the share of the tree's branches found again.

    b: per branch of the tree restricted to the run's sequences: p, sup (its bootstrap TBE), exact (the identical
    split is in the new tree) and agree (transfer agreement with the new tree, in %).
    """
    p, sup, exact, agree = b["p"], b["sup"], b["exact"], b["agree"]
    out: dict = {
        "name": name,
        "leaves": n_leaves,
        "branches": int(p.size),
        "identical_pct": float(100 * exact.mean()),
        "by_support": {},
    }
    for a, c in SUPPORT_BINS:
        m = (sup >= a) & (sup < c)
        out["by_support"][f"{a}-{c}"] = [
            int(m.sum()),
            float(100 * exact[m].mean()),
            float(agree[m].mean()),
        ]
    for label, m in (
        ("deep_ge95", (p > 10000) & (sup >= 95)),
        ("mid_ge95", (p > 100) & (p <= 10000) & (sup >= 95)),
        ("all_ge95", sup >= 95),
        ("all_ge70", sup >= 70),
    ):
        if m.any():
            out[label] = {
                "n": int(m.sum()),
                "identical_pct": float(100 * exact[m].mean()),
                "mean_agreement": float(agree[m].mean()),
                "median_agreement": float(median(agree[m])),
                "p05_agreement": float(percentile(agree[m], 5)),
                "below95": int((agree[m] < 95).sum()),
                "below90": int((agree[m] < 90).sum()),
            }
    return out


def leaveout(
    out: Path,
    exe: str,
    p2v: list[str],
    seed: int,
    selected: list[str] | None,
    overwrite: bool,
) -> None:
    """Runs the leave-out series: (results, p2v-bme options) -> leaveout/<run>/summary.json per run.

    Each run searches the distinct sequences it keeps (GTR+GC fitted to them; repeat searches: another seed), and
    every branch of the tree, restricted to those sequences, is looked up in the new tree (p2v-bme --supports with the
    new tree as the one replicate).
    """
    check_distinct(out)
    names = fasta_names(out / "distinct.fa")
    taxonomy = read_taxonomy(out)
    cats, runs = leaveout_runs(names, taxonomy)
    work = out / "leaveout"
    if overwrite or not (work / "runs.json").exists():
        write_json(work / "runs.json", {"runs": runs})
    reference = read_tree(out / "tree.boot.support.nwk")
    for spec in runs:
        if selected and spec["name"] not in selected:
            continue
        d = work / spec["name"]
        if not overwrite and (d / "summary.json").exists():
            continue
        keep = keep_mask(spec, cats)
        subset = set(x for x, k in zip(names, keep) if k)
        if len(subset) < MIN_SEQUENCES:
            print(f"leave-out {spec['name']}: {len(subset)} sequences, not run")
            continue
        d.mkdir(parents=True, exist_ok=True)
        fasta = out / "distinct.fa"
        if not spec.get("all"):
            fasta = d / "subset.fa"
            write_subset(out / "distinct.fa", subset, fasta)
        options = [
            "-s",
            str(fasta),
            "-m",
            "GTR+GC",
            "--seed",
            str(spec.get("seed", seed)),
            *p2v,
        ]
        run_p2vbme(exe, options, d / "tree", overwrite)
        if not spec.get("all"):
            fasta.unlink()
        supports(exe, out / "tree.all.nwk", d / "tree.nwk", d / "cmp", overwrite)
        compared = read_tree(d / "cmp.support.nwk")
        if (
            compared["names"] != reference["names"]
            or (compared["size"] != reference["size"]).any()
        ):
            raise SystemExit(
                f"{d}/cmp.support.nwk is not the tree of tree.boot.support.nwk"
            )
        mask = array([x in subset for x in reference["names"]])
        ref_b = branch_rows(prune(reference, mask))
        cmp_b = branch_rows(prune(compared, mask))
        branch = {
            "p": ref_b["p"],
            "sup": nan_to_num(ref_b["tbe"]),
            "exact": cmp_b["exact"] == 100,
            "agree": cmp_b["tbe"],
        }
        branch["delta"] = (1 - branch["agree"] / 100) * (branch["p"] - 1)
        write_json(
            d / "summary.json", leaveout_summary(spec["name"], len(subset), branch)
        )
        savez_compressed(
            d / "branches.npz",
            p=branch["p"].astype("int32"),
            support=branch["sup"].astype("float32"),
            delta=branch["delta"].astype("float32"),
            exact=branch["exact"],
        )
        print(
            f"leave-out {spec['name']}: {len(subset)} sequences, {100 * branch['exact'].mean():.1f}% identical"
        )


# ---- GTDB ----

ARB_FIELD = regex(
    r'^\t\t(name|acc|start|stop|tax_gtdb|tax_slv|tax_ltp)\t+(?::\d+\t)?(?:%i (\d+)|"(.*)")'
)
ARB_DATA = regex(r'^\t\t\tdata\t+(?::\d+\t)?"(.*)"')


def gtdb_sequences(arb_dump: Path, names: set[str], out: Path) -> None:
    """Extracts the distinct sequences with a GTDB classification: (ARB text dump, names) -> gtdb.fa, gtdb_labels.tsv.

    Columns kept: those with at least GTDB_MIN_OCCUPANCY A/C/G/T among these sequences.
    """
    rows: list[tuple[str, str, str, str, str]] = []
    seqs: list[bytes] = []
    cur: dict[str, str] = {}

    def flush() -> None:
        g = cur.get("tax_gtdb", "Unclassified;")
        name = f'{cur.get("acc")}.{cur.get("start")}.{cur.get("stop")}'
        if (
            "name" in cur
            and not g.startswith("Unclassified")
            and "data" in cur
            and name in names
        ):
            rows.append(
                (name, cur["name"], g, cur.get("tax_slv", ""), cur.get("tax_ltp", ""))
            )
            seqs.append(
                cur["data"].upper().encode().translate(bytes.maketrans(b"U.", b"T-"))
            )

    with open(arb_dump, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("\tspecies\t"):
                flush()
                cur = {}
            elif line.startswith("\t\t\tdata") and "data" not in cur:
                m = ARB_DATA.match(line)
                if m:
                    cur["data"] = m.group(1)
            elif line.startswith("\t\t") and line[2] != "\t":
                m = ARB_FIELD.match(line)
                if m:
                    cur[m.group(1)] = m.group(2) or m.group(3)
            elif line.startswith("\t%)"):
                flush()
                cur = {}
    flush()
    if not rows:
        raise SystemExit(f"{arb_dump}: no GTDB-classified sequence of the tree")
    a = frombuffer(b"".join(seqs), dtype=uint8).reshape(len(seqs), -1)
    acgt = isin(a, frombuffer(b"ACGT", dtype=uint8))
    columns = flatnonzero(acgt.mean(axis=0) >= GTDB_MIN_OCCUPANCY)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "gtdb.fa", "w") as fa, open(out / "gtdb_labels.tsv", "w") as tl:
        for (name, arb_name, g, slv, ltp), row in zip(rows, a[:, columns]):
            fa.write(f">{name}\n{row.tobytes().decode()}\n")
            tl.write(f"{name}\t{name}\t{arb_name}\t{g}\t{slv}\t{ltp}\n")
    print(f"GTDB: {len(rows)} sequences of the tree, {columns.size} columns")


def read_labels(path: Path) -> tuple[list[str], ndarray, list[ndarray], ndarray]:
    """Reads the GTDB labels: path -> (names, taxon id per rank and sequence, taxon sizes, Archaea mask)."""
    lab = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        f = line.split("\t")
        lab[f[0]] = [x for x in f[3].split(";") if x]
    names = sorted(lab)
    tax = zeros((len(RANKS), len(names)), dtype=int64)
    for r in range(len(RANKS)):
        ids: dict[str, int] = {}
        for i, x in enumerate(names):
            p = lab[x]
            # an empty rank (for example "g__") is a taxon of its own
            key = ";".join(p[: r + 1]) if len(p) > r and len(p[r]) > 3 else f"?{i}"
            tax[r, i] = ids.setdefault(key, len(ids))
    sizes = [bincount(tax[r]) for r in range(len(RANKS))]
    archaea = array([lab[x][0] == "d__Archaea" for x in names])
    return names, tax, sizes, archaea


def intervals(tree: dict, index: dict[str, int]) -> tuple[ndarray, ndarray, ndarray]:
    """Restricts a tree to the labelled sequences: (tree, label index) -> (their order, first and size of each split)."""
    labelled = array([x in index for x in tree["names"]])
    cum = concatenate([[0], cumsum(labelled)])
    order = array([index[x] for x, k in zip(tree["names"], labelled) if k])
    n = len(tree["names"])
    lo, size = (
        cum[tree["first"][n:]],
        cum[tree["first"][n:] + tree["size"][n:]] - cum[tree["first"][n:]],
    )
    keep = size >= 2
    _, first = unique(
        lo[keep] * (order.size + 1) + size[keep], return_index=True
    )  # several nodes, one split
    return order, lo[keep][first], size[keep][first]


def score(
    order: ndarray,
    lo: ndarray,
    size: ndarray,
    tax: ndarray,
    sizes: list[ndarray],
    archaea: ndarray,
) -> dict:
    """Scores one tree against GTDB: (its splits, labels) -> mean best F1 per rank and consistent branches."""
    n = order.size
    keep = (size >= 2) & (size <= n - 2)
    lo, size = lo[keep], size[keep]
    t = tax[:, order]
    res: dict = {"splits": int(lo.size)}
    for r in range(1, 6):
        best = []
        for taxon in flatnonzero(sizes[r] >= MIN_TAXON):
            cum = r_[0, cumsum(t[r] == taxon)]
            a = sizes[r][taxon]
            ins = cum[lo + size] - cum[lo]
            best.append(
                float(
                    maximum(2 * ins / (size + a), 2 * (a - ins) / (n - size + a)).max()
                )
            )
        best_f1 = array(best)
        res[RANKS[r]] = {
            "taxa": int(best_f1.size),
            "mean_f1": float(best_f1.mean()) if best_f1.size else nan,
            "median_f1": float(median(best_f1)) if best_f1.size else nan,
            "exact_pct": float(100 * (best_f1 == 1).mean()) if best_f1.size else nan,
        }
    ca = r_[0, cumsum(archaea[order])]
    na = int(archaea.sum())
    pos = arange(n)
    ok = zeros(lo.size, dtype=bool)
    for j in range(lo.size):
        a_in = ca[lo[j] + size[j]] - ca[lo[j]]
        inner = 2 * a_in < na or (
            2 * a_in == na and 2 * size[j] <= n
        )  # clades read with the root between domains
        sl = (
            pos[lo[j] : lo[j] + size[j]]
            if inner
            else r_[pos[: lo[j]], pos[lo[j] + size[j] :]]
        )
        r = len(RANKS) - 1
        while r >= 0 and t[r, sl].min() != t[r, sl].max():
            r -= 1
        if r == len(RANKS) - 1:
            ok[j] = True
            continue
        u, c = unique(t[r + 1, sl], return_counts=True)
        ok[j] = bool((c == sizes[r + 1][u]).all())
    res["branches_consistent_pct"] = float(100 * ok.mean())
    return res


def gtdb(
    out: Path,
    exe: str,
    p2v: list[str],
    seed: int,
    arb_dump: Path | None,
    fasttree: str | None,
    ow: bool,
) -> None:
    """Scores the trees of the GTDB-classified sequences: results -> gtdb/ and gtdb_labelled_scores.json.

    Trees: the tree of the distinct sequences restricted to them ("full"), p2v-bme on them alone ("ours"), and
    FastTree's distance and likelihood trees ("ft_me", "ft_ml") when --fasttree is given or the files exist.
    """
    g = out / "gtdb"
    if ow or not (g / "gtdb.fa").exists():
        if arb_dump is None:
            raise SystemExit(
                "the gtdb step needs --arb-dump (SILVA 144's ARB database as text, from ARB's tools)"
            )
        gtdb_sequences(arb_dump, set(fasta_names(out / "distinct.fa")), g)
    run_p2vbme(
        exe,
        ["-s", str(g / "gtdb.fa"), "-m", "GTR+GC", "--seed", str(seed), *p2v],
        g / "ours",
        ow,
    )
    if fasttree:
        for tag, options in (("ft_me", ["-nt", "-noml"]), ("ft_ml", ["-nt", "-gtr"])):
            if ow or not (g / f"{tag}.nwk").exists():
                with open(g / f"{tag}.nwk", "w") as fo:
                    run([fasttree, *options, str(g / "gtdb.fa")], stdout=fo, check=True)
    names, tax, sizes, archaea = read_labels(g / "gtdb_labels.tsv")
    index = {x: i for i, x in enumerate(names)}
    full_tree = distinct_tree(out)
    root = full_tree["parent"].size - 1
    trees = {"full": score(*intervals(full_tree, index), tax, sizes, archaea)}
    for tag in ("ours", "ft_me", "ft_ml"):
        if (g / f"{tag}.nwk").exists():
            trees[tag] = score(
                *intervals(read_tree(g / f"{tag}.nwk"), index), tax, sizes, archaea
            )
    write_json(
        out / "gtdb_labelled_scores.json",
        {
            "n_sequences": len(names),
            "min_taxon": MIN_TAXON,
            "full_tree_length": round(float(nansum(full_tree["length"][:root])), 2),
            "trees": trees,
        },
    )


# ---- Runs ----


def main() -> None:
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=STEPS,
        default=list(STEPS[:-1]),
        help="steps to run, in order",
    )
    parser.add_argument(
        "--alignment",
        type=Path,
        default=Path(ALIGNMENT),
        help="SILVA's aligned FASTA (or .gz)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(DEFAULT_RESULTS_DIR),
        help="results folder; this script writes into its silva/",
    )
    parser.add_argument("--p2vbme", default=P2VBME)
    parser.add_argument("--reattach-left-out", default=REATTACH_LEFT_OUT)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=100,
        help="bootstrap trees B used by the supports",
    )
    parser.add_argument(
        "--replicates", default=None, help="replicates to run, e.g. 1-25 (default: 1-B)"
    )
    parser.add_argument(
        "--max-sites", type=int, default=1600, help="columns kept: the best occupied"
    )
    parser.add_argument(
        "--min-sites",
        type=int,
        default=1200,
        help="sequences with fewer valid sites left out",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="seed of the tree and of the bootstrap"
    )
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument(
        "--leaveout-runs",
        nargs="+",
        default=None,
        help="leave-out runs to make (default: all)",
    )
    parser.add_argument(
        "--arb-dump",
        type=Path,
        default=None,
        help="SILVA 144's ARB database as text (gtdb step)",
    )
    parser.add_argument(
        "--fasttree",
        nargs="?",
        const=FASTTREE,
        default=None,
        help="also build FastTree's trees",
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="processes reading the bootstrap trees"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="run steps whose outputs exist again"
    )
    args = parser.parse_args()
    args.results_dir = args.results_dir / "silva"
    out, exe, ow = args.results_dir, args.p2vbme, args.overwrite
    p2v = ["-a", args.accelerator, "-d", args.devices, "--report-full"]
    run_args = [
        "-s",
        str(out / "searched.fa"),
        "-m",
        "GTR+GC",
        "--seed",
        str(args.seed),
        *p2v,
    ]
    for step in args.steps:
        if step == "distinct":
            distinct(args.alignment, out, args.max_sites, args.min_sites, ow)
        elif step == "tree":
            run_p2vbme(exe, run_args, out / "tree", ow)
            reattach(out, args.reattach_left_out, ow)
        elif step == "replicates":
            for r in span(args.replicates or f"1-{args.bootstrap}"):
                run_p2vbme(
                    exe,
                    [*run_args, "-b", f"{r}-{r}"],
                    out / f"boot/rep{r:03d}",
                    ow,
                )
        elif step == "supports":
            boot = [
                out / f"boot/rep{r:03d}.boottrees" for r in range(1, args.bootstrap + 1)
            ]
            supports(
                exe,
                out / "tree.all.nwk",
                concatenate_trees(boot, out / "boot.nwk", ow),
                out / "tree.boot",
                ow,
            )
        elif step == "records":
            records(out, args.bootstrap, args.workers)
        elif step == "leaveout":
            leaveout(out, exe, p2v, args.seed, args.leaveout_runs, ow)
        elif step == "gtdb":
            gtdb(out, exe, p2v, args.seed, args.arb_dump, args.fasttree, ow)


if __name__ == "__main__":
    main()
