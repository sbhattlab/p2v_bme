"""SARS-CoV-2 genomes (Viridian): tree, bootstrap, repeat searches and MAPLE's tree (Fig. 3c; tables viridian_support,
viridian_variants, viridian_thresholds; Supplementary Fig. viridian-support).

Inputs:  data/datasets/viridian/12733488/extracted/Viridian_2Malignment_filtered_masked_noShortDeletions.maple
         data/datasets/viridian/15546527/Viridian_2M_noShortDel_deeper_SPRTA0611_tree_combined.jsonl.zip
           (MAPLE 0.6.11's tree with each genome's Pango lineage and date)
         data/datasets/viridian/alias_key.json      Pango alias key (cov-lineages/pango-designation)  (download_data.py)
Options: --steps, --replicates, --repeats, --seed, --accelerator, --devices, --min-month-genomes,
         --min-lineage-genomes, --overwrite
Flow:    maple        MAPLE's tree with the genomes' names, and each genome's lineage and date
           |
           v
         tree         p2v-bme -s ALIGNMENT -m JC69 --no-dedup --seed 1   (every genome searched, as in the paper)
           |
           v
         replicates   p2v-bme ... -b r-r      (one run per replicate; spread them with --replicates)
           |
           v
         repeats      p2v-bme ... --seed s, s = 2..9          (repeat searches on the unchanged alignment)
           |
           v
         supports     p2v-bme -t TREE --supports FILE: our tree and MAPLE's over the bootstrap trees, the repeat
           |          searches and each other
           v
         records      agreement and support by branch length, named variants and lineages, local resolution by
                      sampling month
Outputs: data/results/viridian/
           maple.nwk, tips.tsv                          MAPLE's tree (rooted, as published) and the genomes
           tree.{nwk,json}, boot/rep<r>.{boottrees,json}, repeat/seed<s>.{nwk,json}, boot.nwk, repeats.nwk
           supports/{tree,maple}.{boot,repeats,maple,ours}.{support.nwk,json}
           summary.json, lineages.tsv, resolution_by_month.tsv, runs.jsonl   (the paper's record names and fields)

    pixi run -e paper python -m paper.run_viridian --steps tree replicates --replicates 1-25
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from io import TextIOWrapper
from json import dumps, loads
from pathlib import Path
from zipfile import ZipFile

from numpy import (
    arange,
    argsort,
    array,
    bincount,
    concatenate,
    cumsum,
    flatnonzero,
    full,
    inf,
    int64,
    isinf,
    isnan,
    lexsort,
    maximum,
    median,
    minimum,
    nan,
    ndarray,
    percentile,
    searchsorted,
    unique,
    where,
    zeros,
)
from pandas import DataFrame, Timestamp, read_csv, to_datetime

from paper._config import DEFAULT_DATASETS_DIR, DEFAULT_RESULTS_DIR, P2VBME
from paper.utils import (
    auc,
    concatenate_trees,
    pct,
    read_json,
    read_tree,
    run_p2vbme,
    span,
    supports,
    write_json,
)

DATA = f"{DEFAULT_DATASETS_DIR}/viridian"
ALIGNMENT = f"{DATA}/12733488/extracted/Viridian_2Malignment_filtered_masked_noShortDeletions.maple"
MAPLE_JSONL = "Viridian_2M_noShortDel_deeper_SPRTA0611_tree_combined.jsonl"
STEPS = ("maple", "tree", "replicates", "repeats", "supports", "records")
# MAPLE's JSONL stores distances on a display scale; one branch of the published Newick tree fixes it.
KNOWN = (
    "ERR7458074",
    4.40086731633712e-05,
)  # leaf and its branch length (substitutions per site)
PUBLISHED_GENOMES = 2072111  # the published tree: 1,605,004 zero-length and 458,518 (+- 5) >= 0.5-substitution
PUBLISHED_COUNTS = (1605004, 458518)  # internal branches
ROOT_STRAIN = (
    "ERR7458074"  # lineage A: the stored tree is read as rooted at this genome
)
MIN_SUBSTITUTIONS = 0.5  # shorter branches resolve genomes the data do not separate
THRESHOLDS = (0.70, 0.75, 0.80, 0.90, 0.95)  # transfer support of a supported clade
HALF_WINDOW_DAYS = (
    15  # a genome's contemporaries: sampled at most this many days before or after
)
FIRST_DAY = "2019-12-01"  # earlier sampling dates are errors of the metadata
TOLERANCE = 10  # genomes: a lineage's branch is the longest within this many misplaced genomes of the best
VARIANT_GROUPS = (
    ("B.1.1.7", "alpha"),
    ("B.1.617.2", "delta"),
    ("B.1.1.529", "omicron"),
)
GROUPS = ("other", "alpha", "delta", "omicron", "recombinant")
WHO = {
    "B.1.1.7": "Alpha",
    "B.1.351": "Beta",
    "B.1.1.28.1": "Gamma",
    "B.1.617.2": "Delta",
    "B.1.1.529": "Omicron",
    "B.1.621": "Mu",
    "B.1.1.1.37": "Lambda",
    "B.1.617.1": "Kappa",
    "B.1.525": "Eta",
    "B.1.526": "Iota",
    "B.1.429": "Epsilon",
    "B.1.427": "Epsilon (B.1.427)",
    "B.1.1.28.2": "Zeta",
    "B.1.1.28.3": "Theta",
}
VARIANTS = (
    "B.1.1.7",
    "B.1.351",
    "B.1.1.28.1",
    "B.1.617.2",
    "B.1.1.529",
    "B.1.621",
    "B.1.429",
    "B.1.526",
)
SUBSTITUTIONS = ((0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5.5), (5.5, inf))
SUPPORT = (
    (0.0, 0.05),
    (0.05, 0.3),
    (0.3, 0.5),
    (0.5, 0.7),
    (0.7, 0.9),
    (0.9, 0.95),
    (0.95, 0.99),
    (0.99, 1.01),
)
SIZES = ((2, 2), (3, 10), (11, 100), (101, 1000), (1001, 10000), (10001, 10**9))
TRANSFER = (0.70, 0.75, 0.80, 0.90)  # thresholds of transfer support in the summary
LINEAGE_COLUMNS = ["lineage", "full", "who", "genomes"] + [
    f"{tag}_{c}"
    for tag, last in (
        ("ours", ("repeat", "to_maple")),
        ("maple", ("to_ours", "repeat")),
    )
    for c in (
        "node",
        "size",
        "misplaced",
        "least_misplaced",
        "sub",
        "exact",
        "tbe",
        "moved_per_tree",
        *last,
    )
]


# ---- MAPLE's tree ----


def alignment_names(path: Path) -> tuple[list[str], int]:
    """Reads a .maple alignment's genome names and length: path -> (names without the reference, sites)."""
    names, sites, in_reference = [], 0, False
    with open(path, "rb") as fh:
        for line in fh:
            if line.startswith(b">"):
                in_reference = (
                    not names and not sites and line[1:].strip() == b"reference"
                )
                if not in_reference:
                    names.append(line[1:].strip().decode())
            elif in_reference:
                sites += len(line.strip())
    if not sites:
        raise SystemExit(f"{path}: the first sequence is not the reference")
    return names, sites


def maple_tree(jsonl_zip: Path, alignment: Path, out: Path, overwrite: bool) -> None:
    """Writes MAPLE's tree with the genomes' names and a table of the genomes: (JSONL, alignment) -> maple.nwk, tips.tsv.

    Branch lengths in substitutions per site; the tree stays rooted as published.
    """
    if not overwrite and (out / "maple.nwk").exists() and (out / "tips.tsv").exists():
        return
    names, sites = alignment_names(alignment)
    with ZipFile(jsonl_zip) as z, TextIOWrapper(
        z.open(MAPLE_JSONL), encoding="utf-8"
    ) as fh:
        total = loads(fh.readline())["total_nodes"]
        if total != 2 * len(names) - 1:
            raise SystemExit(
                f"{total} nodes for {len(names)} genomes: the tree is not binary"
            )
        parent, x, is_tip = full(total, -1, int64), zeros(total), zeros(total, bool)
        label, meta = [""] * total, {}
        for line in fh:
            d = loads(line)
            i = d["node_id"]
            parent[i], x[i], label[i], is_tip[i] = (
                d["parent_id"],
                d["x_dist"],
                d["name"],
                d["is_tip"],
            )
            if d["is_tip"]:
                meta[i] = (
                    d.get("meta_lineage", ""),
                    d.get("meta_date", ""),
                    d.get("meta_country", ""),
                )
    if sorted(label[i] for i in flatnonzero(is_tip)) != sorted(names):
        raise SystemExit(
            "MAPLE's tree does not hold every genome of the alignment once"
        )
    roots = flatnonzero(parent == arange(total))
    if roots.size != 1:
        raise SystemExit(f"{roots.size} roots")
    root = int(roots[0])
    children = [[] for _ in range(total)]
    for i in range(total):
        if i != root:
            children[parent[i]].append(i)
    if any(len(children[i]) != 2 for i in flatnonzero(~is_tip)):
        raise SystemExit("an internal node does not have two children")
    length = x - x[parent]  # x_dist: distance from the root on the display scale
    length[root] = 0.0
    if (length < -1e-6).any():
        raise SystemExit("negative branch length")
    known = label.index(KNOWN[0])
    length = maximum(length, 0.0) / (length[known] / KNOWN[1])  # substitutions per site
    parts, stack = [], [(root, 0)]
    while stack:
        u, state = stack.pop()
        if is_tip[u]:
            parts.append(f"{label[u]}:{length[u]:.7g}")
        elif state == 0:
            parts.append("(")
            stack += [(u, 1), (children[u][0], 0)]
        elif state == 1:
            parts.append(",")
            stack += [(u, 2), (children[u][1], 0)]
        else:
            parts.append(")" if u == root else f"):{length[u]:.7g}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "maple.nwk").write_text("".join(parts) + ";\n")
    tips = flatnonzero(is_tip)
    DataFrame(
        {
            "strain": [label[u] for u in tips],
            "lineage": [meta[u][0] for u in tips],
            "date": [meta[u][1] for u in tips],
            "country": [meta[u][2] for u in tips],
            "sub": length[tips] * sites,
        }
    ).to_csv(out / "tips.tsv", sep="\t", index=False, float_format="%.5f")
    inner = ~is_tip & (arange(total) != root)
    counts = (
        int((length[inner] == 0).sum()),
        int((length[inner] * sites >= MIN_SUBSTITUTIONS).sum()),
    )
    if len(names) == PUBLISHED_GENOMES and (
        counts[0] != PUBLISHED_COUNTS[0] or abs(counts[1] - PUBLISHED_COUNTS[1]) > 5
    ):
        raise SystemExit(
            f"branch lengths do not match the published tree: zero and >= 0.5 subst are {counts}"
        )
    print(
        f"MAPLE's tree: {len(names)} genomes; internal branches: {counts[0]} zero, {counts[1]} >= 0.5 subst"
    )


# ---- p2v-bme runs ----


def search_time(report: dict) -> float:
    """Sums the timed stages of a run as the paper times them: report -> seconds (start tree, search, branch lengths)."""
    t = report["details"]["time"]
    return t["start_tree"] + t["search"] + t["branch_lengths"]


# ---- Records ----


def bins(d: DataFrame, column: str, edges: tuple, fields: dict[str, str]) -> list[dict]:
    """Averages the named columns over [lo, hi) bins of one column: (table, column, bins, fields) -> rows."""
    rows = []
    for lo, hi in edges:
        e = d[(d[column] >= lo) & (d[column] < hi)]
        row = {"from": lo, "to": None if isinf(hi) else hi, "branches": len(e)}
        rows.append(row | {name: pct(e[col]) for name, col in fields.items()})
    return rows


def unalias(name: str, key: dict) -> str:
    """Expands a Pango alias: name -> full name (AY.4 -> B.1.617.2.4)."""
    head, _, rest = name.partition(".")
    full_name = key.get(head, "")
    if isinstance(full_name, str) and full_name:
        return full_name + ("." + rest if rest else "")
    return name


def branches(tree: dict, sites: int, labels: dict[str, dict]) -> DataFrame:
    """Tabulates the internal branches of a tree: (tree, sites, {column: tree with labels}) -> one row per branch.

    Columns: node, p (genomes on the smaller side), sub (substitutions), and for each labelled tree its exact and
    transfer support as fractions (<column>_exact, <column>_tbe).
    """
    n = len(tree["names"])
    node = arange(n, tree["parent"].size - 1)  # internal nodes but the root
    size = tree["size"][node]
    table = {
        "node": node,
        "p": minimum(size, n - size),
        "sub": tree["length"][node] * sites,
    }
    for column, other in labels.items():
        if other["names"] != tree["names"] or (other["size"] != tree["size"]).any():
            raise SystemExit(f"{column}: the support file is not the same tree")
        table[f"{column}_exact"] = other["exact"][node] / 100.0
        table[f"{column}_tbe"] = other["tbe"][node] / 100.0
    return DataFrame(table).set_index("node")


def transfer_distance(member: ndarray, tree: dict) -> int:
    """Finds the fewest genomes to move to obtain a split of the tree: (genome set in the tree's leaf order, tree) -> d."""
    n = member.size
    prefix = concatenate([[0], cumsum(member)])
    inside = prefix[tree["first"] + tree["size"]] - prefix[tree["first"]]
    d = prefix[-1] + tree["size"] - 2 * inside
    return int(minimum(d, n - d).min())


def best(member: ndarray, tree: dict, length: ndarray) -> tuple[int, int, int, int]:
    """Finds a lineage's branch: (genomes in leaf order, tree, sub per node) -> (node, misplaced, size, least misplaced).

    The fewest misplaced genomes of any branch is the transfer distance; among the branches within TOLERANCE of it,
    the longest is taken. length is negative where a node has no branch (leaves, the root).
    """
    n = member.size
    prefix = concatenate([[0], cumsum(member)])
    first, size = tree["first"], tree["size"]
    d = int(member.sum()) + size - 2 * (prefix[first + size] - prefix[first])
    d = minimum(d, n - d)
    d[length < 0] = n
    least = int(d.min())
    near = flatnonzero(d <= least + TOLERANCE)
    k = int(near[lexsort((d[near], -length[near]))[0]])  # the longest, then the closest
    return k, int(d[k]), int(size[k]), least


def lineage_rows(
    ours: dict,
    maple: dict,
    o: DataFrame,
    m: DataFrame,
    tips: DataFrame,
    alias: dict,
    least: int,
):
    """Matches every Pango lineage of >= least genomes in both trees: (trees, branch tables, tips) -> rows."""
    n = len(tips)
    full_names = {lab: unalias(lab, alias) for lab in tips.lineage.unique()}
    groups: dict[str, list[int]] = {}
    for g, name in enumerate(tips.lineage.map(full_names)):
        parts = name.split(".")
        for j in range(1, len(parts) + 1):
            groups.setdefault(".".join(parts[:j]), []).append(g)
    short = {v: k for k, v in full_names.items()}
    keep = sorted(
        (g for g, s in groups.items() if least <= len(s) <= n - least),
        key=lambda g: -len(groups[g]),
    )
    index = {name: g for g, name in enumerate(tips.strain)}
    leaf_genome = {
        tag: array([index[x] for x in t["names"]])
        for tag, t in (("ours", ours), ("maple", maple))
    }
    trees = {"ours": (ours, o), "maple": (maple, m)}
    rows = []
    for g in keep:
        member = zeros(n, int64)
        member[groups[g]] = 1
        row = {
            "lineage": short.get(g, g),
            "full": g,
            "who": WHO.get(g, ""),
            "genomes": len(groups[g]),
        }
        clades = {}
        for tag, (tree, table) in trees.items():
            length = full(tree["parent"].size, -1.0)
            length[table.index.to_numpy()] = table["sub"].to_numpy()
            k, d, size, least_d = best(member[leaf_genome[tag]], tree, length)
            r = table.loc[k]
            first = tree["first"][k]
            clades[tag] = leaf_genome[tag][first : first + tree["size"][k]]
            row |= {
                f"{tag}_node": k,
                f"{tag}_size": size,
                f"{tag}_misplaced": d,
                f"{tag}_least_misplaced": least_d,
                f"{tag}_sub": r["sub"],
                f"{tag}_exact": r.boot_exact,
                f"{tag}_tbe": r.boot_tbe,
                f"{tag}_moved_per_tree": (1 - r.boot_tbe) * (r.p - 1),
            }
        for tag, other in (("ours", "maple"), ("maple", "ours")):
            clade = zeros(n, int64)
            clade[clades[tag]] = 1
            row[f"{tag}_to_{other}"] = transfer_distance(
                clade[leaf_genome[other]], trees[other][0]
            )
        row["ours_repeat"] = o.loc[row["ours_node"]].repeats_exact
        row["maple_repeat"] = m.loc[row["maple_node"]].repeats_exact
        rows.append(row)
    return DataFrame(rows, columns=LINEAGE_COLUMNS)


def smallest_supported_clade(
    parent: ndarray, n: int, supported: ndarray, root_leaf: int
) -> tuple[ndarray, ndarray]:
    """Finds the smallest supported clade around every genome: (tree, supported internal nodes) -> (node, far).

    node is -1 when no branch qualifies; far is True when the clade is all genomes not under the node, which is the
    case on the path from the stored root to root_leaf (the tree is read as rooted at that genome).
    """
    total = parent.size
    chain = []  # internal nodes from root_leaf up to the stored root
    node = parent[root_leaf]
    while node >= 0:
        chain.append(int(node))
        node = parent[node]
    chain = chain[::-1]
    on_chain = zeros(total, bool)
    on_chain[chain] = True
    depth = full(total, -1, int64)
    depth[chain] = arange(len(chain))
    near = full(
        total, -1, int64
    )  # lowest supported node at or above a node, off the chain
    for node in range(total - 1, n - 1, -1):  # parents before children
        above = parent[node]
        if depth[node] < 0 and above >= 0:
            depth[node] = depth[above]
        inherited = near[above] if above >= 0 else -1
        near[node] = node if (supported[node - n] and not on_chain[node]) else inherited
    below = full(
        len(chain) + 1, -1, int64
    )  # first supported chain node further down the chain
    for j in range(len(chain) - 1, -1, -1):
        below[j] = chain[j] if supported[chain[j] - n] else below[j + 1]
    leaf_parent = parent[:n]
    inner = near[leaf_parent]
    outer = below[depth[leaf_parent] + 1]
    return where(inner >= 0, inner, outer), inner < 0


def resolution(
    ours: dict, o: DataFrame, tips: DataFrame, alias: dict, min_genomes: int
) -> DataFrame:
    """Tabulates how finely genomes are placed, by sampling month: (tree, branch table, tips) -> rows per threshold."""
    n = len(tips)
    index = {name: g for g, name in enumerate(tips.strain)}
    genome = array([index[x] for x in ours["names"]])  # genome of each leaf
    leaf = full(n, -1, int64)
    leaf[genome] = arange(n)
    root_leaf = int(leaf[index[ROOT_STRAIN]])
    internal = ours["parent"].size - n
    tbe, substitutions = full(internal, -1.0), zeros(internal)
    tbe[o.index.to_numpy() - n] = o.boot_tbe.fillna(-1.0).to_numpy()
    substitutions[o.index.to_numpy() - n] = o["sub"].to_numpy()
    lo, hi = ours["first"][n:], ours["first"][n:] + ours["size"][n:]

    def group_of(label: str) -> str:
        name = unalias(label, alias)
        for prefix, group in VARIANT_GROUPS:
            if name == prefix or name.startswith(prefix + "."):
                return group
        return "recombinant" if name.startswith("X") else "other"

    by_leaf = tips.iloc[genome].reset_index(
        drop=True
    )  # one row per leaf, in leaf order
    group = by_leaf.lineage.map(
        {x: group_of(x) for x in by_leaf.lineage.unique()}
    ).to_numpy()
    date = to_datetime(
        by_leaf.date.where(by_leaf.date.str.fullmatch(r"20\d\d-\d\d-\d\d")),
        errors="coerce",
    )
    day = (date - Timestamp(FIRST_DAY)).dt.days.fillna(-1).astype(int64).to_numpy()
    day = where(day < 0, -1, day)  # no full date, or a date before the pandemic
    month = where(day >= 0, date.dt.strftime("%Y-%m").to_numpy(dtype=object), "")
    names, counts = unique(month[day >= 0], return_counts=True)
    kept = [x for x, c in zip(names, counts) if c >= min_genomes]

    def members(node: ndarray, far: ndarray, prefix: ndarray) -> ndarray:
        k = maximum(node - n, 0)
        inside = where(node >= 0, prefix[hi[k]] - prefix[lo[k]], 0)
        return where((node < 0) | far, prefix[-1] - inside, inside)

    clades = [
        smallest_supported_clade(
            ours["parent"],
            n,
            (tbe >= t) & (substitutions >= MIN_SUBSTITUTIONS),
            root_leaf,
        )
        for t in THRESHOLDS
    ]
    everyone = arange(n + 1)
    size = [members(node, far, everyone) for node, far in clades]
    window = zeros(
        (len(THRESHOLDS), n), int64
    )  # contemporaries of each genome in its clade
    total = zeros(n, int64)  # all contemporaries of each genome
    by_day = argsort(day, kind="stable")
    starts = searchsorted(day[by_day], arange(day.max() + 2))
    for d in range(day.max() + 1):
        asked = by_day[starts[d] : starts[d + 1]]
        if not len(asked):
            continue
        near = (day >= max(d - HALF_WINDOW_DAYS, 0)) & (day <= d + HALF_WINDOW_DAYS)
        prefix = concatenate([[0], cumsum(near)])
        total[asked] = prefix[-1]
        for i, (node, far) in enumerate(clades):
            window[i, asked] = members(node[asked], far[asked], prefix)
    rows = []
    for i, threshold in enumerate(THRESHOLDS):
        for name in kept:
            here = month == name
            row = {"threshold": threshold, "month": name, "genomes": int(here.sum())}
            row |= {f"share_{g}": float((group[here] == g).mean()) for g in GROUPS}
            for key, values in (("clade", size[i][here]),):
                q25, q50, q75 = percentile(values, [25, 50, 75])
                row |= {f"{key}_q25": q25, f"{key}_median": q50, f"{key}_q75": q75}
            row["clade_share_at_most_100"] = float((size[i][here] <= 100).mean())
            q25, q50, q75 = percentile(window[i, here], [25, 50, 75])
            row |= {"window_q25": q25, "window_median": q50, "window_q75": q75}
            row["window_total_median"] = float(median(total[here]))
            row["window_share_median"] = float(median(window[i, here] / total[here]))
            rows.append(row)
    return DataFrame(rows)


def records(out: Path, alias_path: Path, min_month: int, min_lineage: int) -> None:
    """Writes the records of the analysis: (results folder) -> summary.json, lineages.tsv, resolution_by_month.tsv."""
    report = read_json(out / "tree.json")
    sites = report["details"]["input"]["sites"]
    sup = out / "supports"
    ours = read_tree(sup / "tree.boot.support.nwk")
    maple = read_tree(sup / "maple.boot.support.nwk")
    o = branches(
        ours,
        sites,
        {
            "boot": ours,
            "repeats": read_tree(sup / "tree.repeats.support.nwk"),
            "maple": read_tree(sup / "tree.maple.support.nwk"),
        },
    )
    m = branches(
        maple,
        sites,
        {
            "boot": maple,
            "ours": read_tree(sup / "maple.ours.support.nwk"),
            "repeats": read_tree(sup / "maple.repeats.support.nwk"),
        },
    )
    tips = read_csv(out / "tips.tsv", sep="\t", dtype=str, keep_default_na=False)
    alias = loads(alias_path.read_text(encoding="utf-8"))

    runs = sorted((out / "boot").glob("rep*.json")), sorted(
        (out / "repeat").glob("seed*.json")
    )
    boot_s = [search_time(read_json(p)) for p in runs[0]]
    device = report["details"]["devices"]
    summary_runs = {
        "genomes": report["details"]["input"]["sequences"],
        "sites": sites,
        "devices": device,
        "n_gpus": (
            len(device) if report["details"]["settings"]["accelerator"] == "gpu" else 0
        ),
        "reference_s": search_time(report),
        "rounds_run": report["rounds_run"],
        "gpu_memory_gib": report["details"]["memory"]["gpu_memory_peak_mib"] / 1024,
        "host_memory_gib": report["details"]["memory"]["host_ram_mib"] / 1024,
        "bootstrap_trees": len(boot_s),
        "bootstrap_mean_s": float(sum(boot_s) / len(boot_s)) if boot_s else nan,
        "bootstrap_total_gpu_h": float(sum(boot_s) / 3600),
        "repeat_searches": len(runs[1]),
    }

    b = o.rename(
        columns={"boot_exact": "exact", "boot_tbe": "tbe", "repeats_exact": "repeat"}
    )
    b["maple_correct"], b["maple_transfer"] = b.maple_exact, b.maple_tbe
    b["high"] = (b.exact >= 0.95).astype(float)
    real = b[b["sub"] >= MIN_SUBSTITUTIONS]
    fields = {
        "in_maple_pct": "maple_correct",
        "transfer_to_maple_pct": "maple_transfer",
        "exact_pct": "exact",
        "tbe_pct": "tbe",
        "repeat_pct": "repeat",
    }
    one = real[(real["sub"] > 0.9) & (real["sub"] < 1.1)]
    labelled = real.exact.notna()
    summary_ours = {
        "branches": len(b),
        "real": len(real),
        "below_half_substitution_pct": pct(b["sub"] < MIN_SUBSTITUTIONS),
        "below_half_in_maple_pct": pct(b.maple_correct[b["sub"] < MIN_SUBSTITUTIONS]),
        "below_half_exact_pct": pct(b.exact[b["sub"] < MIN_SUBSTITUTIONS]),
        "real_means": {k: pct(real[c]) for k, c in fields.items()},
        "by_substitutions": bins(real, "sub", SUBSTITUTIONS, fields),
        "by_exact": bins(
            real,
            "exact",
            SUPPORT,
            {"in_maple_pct": "maple_correct", "repeat_pct": "repeat"},
        ),
        "by_size": bins(real, "p", tuple((lo, hi + 1) for lo, hi in SIZES), fields),
        "exact95": {
            "branches": int(real.high.sum()),
            "share_pct": pct(real.high),
            "in_maple_pct": pct(real.maple_correct[real.high == 1]),
        },
        "by_transfer_support": [
            {
                "at_least": t,
                "branches": int((real.tbe >= t).sum()),
                "share_pct": pct(real.tbe >= t),
                "in_maple_pct": pct(real.maple_correct[real.tbe >= t]),
                "repeat_pct": pct(real.repeat[real.tbe >= t]),
            }
            for t in TRANSFER
        ],
        "one_substitution": {
            "branches": len(one),
            "exact_pct": pct(one.exact),
            "exact_median_pct": float(100 * one.exact.median()),
            "in_maple_pct": pct(one.maple_correct),
            "repeat_pct": pct(one.repeat),
        },
        "auc_exact_real": auc(
            real.exact[labelled].to_numpy(),
            real.maple_correct[labelled].to_numpy() == 1,
        ),
        "tree_length_substitutions": float(b["sub"].sum()),
    }

    mm = m.rename(
        columns={"boot_exact": "exact", "boot_tbe": "tbe", "repeats_exact": "repeat"}
    )
    mm["in_ours"], mm["to_ours"] = mm.ours_exact, mm.ours_tbe
    mm["any"] = (mm.exact > 0).astype(float)
    for name, col, cut in (("tbe70", "tbe", 0.70),):
        mm[name] = (mm[col] >= cut).astype(float)
    mr = mm[mm["sub"] >= MIN_SUBSTITUTIONS]
    mfields = {
        "in_ours_pct": "in_ours",
        "exact_pct": "exact",
        "tbe_pct": "tbe",
        "tbe70_pct": "tbe70",
        "repeat_pct": "repeat",
    }
    summary_maple = {
        "branches": len(mm),
        "zero_length": int((mm["sub"] == 0).sum()),
        "real": len(mr),
        "below_half_substitution_pct": pct(mm["sub"] < MIN_SUBSTITUTIONS),
        "real_means": {k: pct(mr[c]) for k, c in mfields.items()}
        | {
            "transfer_to_ours_pct": pct(mr.to_ours),
            "in_any_bootstrap_tree_pct": pct(mr["any"]),
        },
        "by_substitutions": bins(mr, "sub", SUBSTITUTIONS, mfields),
        "tree_length_substitutions": float(mm["sub"].sum()),
    }

    lin = lineage_rows(ours, maple, o, m, tips, alias, min_lineage)
    lin.to_csv(out / "lineages.tsv", sep="\t", index=False, float_format="%.6g")
    keys = (
        "genomes",
        "ours_size",
        "ours_misplaced",
        "ours_sub",
        "ours_exact",
        "ours_tbe",
        "ours_moved_per_tree",
        "ours_repeat",
        "ours_to_maple",
        "maple_size",
        "maple_misplaced",
        "maple_sub",
        "maple_exact",
        "maple_tbe",
        "maple_to_ours",
    )
    by_full = lin.set_index("full")
    variants = [
        {"name": by_full.loc[v, "who"], "lineage": by_full.loc[v, "lineage"]}
        | {k: float(by_full.loc[v, k]) for k in keys}
        for v in VARIANTS
        if v in by_full.index
    ]
    share_o, share_m = (
        lin.ours_least_misplaced / lin.genomes,
        lin.maple_least_misplaced / lin.genomes,
    )
    lineages = {
        "lineages": len(lin),
        "min_genomes": int(lin.genomes.min()) if len(lin) else 0,
        "ours_within_1pct": int((share_o <= 0.01).sum()),
        "maple_within_1pct": int((share_m <= 0.01).sum()),
        "ours_median_misplaced_pct": float(100 * share_o.median()),
        "maple_median_misplaced_pct": float(100 * share_m.median()),
        "ours_tbe_pct": pct(lin.ours_tbe),
        "ours_exact_pct": pct(lin.ours_exact),
        "ours_tbe95": int((lin.ours_tbe >= 0.95).sum()),
        "ours_exact95": int((lin.ours_exact >= 0.95).sum()),
    }
    summary = {
        "data": {
            "alignment": "Zenodo 12733488 (Viridian, MAPLE format)",
            "maple_tree": "Zenodo 15546527 (MAPLE 0.6.11 tree, Taxonium JSONL)",
            "lineage_labels": "the lineage field of that file; aliases from cov-lineages/pango-designation",
        },
        "runs": summary_runs,
        "ours": summary_ours,
        "maple": summary_maple,
        "variants": variants,
        "lineages": lineages,
    }
    write_json(out / "summary.json", summary)
    resolution(ours, o, tips, alias, min_month).to_csv(
        out / "resolution_by_month.tsv", sep="\t", index=False, float_format="%.6g"
    )
    lines = [dumps({"run": "tree"} | report)]
    lines += [dumps({"run": p.stem} | read_json(p)) for p in (*runs[0], *runs[1])]
    (out / "runs.jsonl").write_text("\n".join(lines) + "\n")
    print(
        f"records: {len(b)} branches of our tree ({len(real)} real), {len(lin)} lineages"
    )


def main() -> None:
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=STEPS,
        default=list(STEPS),
        help="steps to run, in order",
    )
    parser.add_argument(
        "--alignment", type=Path, default=Path(ALIGNMENT), help="the genomes (.maple)"
    )
    parser.add_argument(
        "--maple-jsonl", type=Path, default=Path(f"{DATA}/15546527/{MAPLE_JSONL}.zip")
    )
    parser.add_argument(
        "--alias",
        type=Path,
        default=Path(f"{DATA}/alias_key.json"),
        help="Pango alias key",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(DEFAULT_RESULTS_DIR),
        help="results folder; this script writes into its viridian/",
    )
    parser.add_argument("--p2vbme", default=P2VBME)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=100,
        help="bootstrap trees B used by the supports",
    )
    parser.add_argument(
        "--replicates", default=None, help="replicates to run, e.g. 1-25 (default: 1-B)"
    )
    parser.add_argument("--repeats", default="2-9", help="seeds of the repeat searches")
    parser.add_argument(
        "--seed", type=int, default=1, help="seed of the tree and of the bootstrap"
    )
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument(
        "--min-month-genomes",
        type=int,
        default=5000,
        help="months with fewer dated genomes go",
    )
    parser.add_argument(
        "--min-lineage-genomes", type=int, default=1000, help="smallest lineage matched"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="run steps whose outputs exist again"
    )
    args = parser.parse_args()
    args.results_dir = args.results_dir / "viridian"
    out, exe, ow = args.results_dir, args.p2vbme, args.overwrite
    # Without dedup, as the paper's runs: every genome is searched, also the third and later identical copies
    run = [
        "-s",
        str(args.alignment),
        "-m",
        "JC69",
        "--no-dedup",
        "-a",
        args.accelerator,
        "-d",
        args.devices,
        "--report-full",
    ]
    boots = [out / f"boot/rep{r:03d}.boottrees" for r in range(1, args.bootstrap + 1)]
    repeats = [out / f"repeat/seed{s}.nwk" for s in span(args.repeats)]
    for step in args.steps:
        if step == "maple":
            maple_tree(args.maple_jsonl, args.alignment, out, ow)
        elif step == "tree":
            run_p2vbme(exe, [*run, "--seed", str(args.seed)], out / "tree", ow)
        elif step == "replicates":
            for r in span(args.replicates or f"1-{args.bootstrap}"):
                rep = ["--seed", str(args.seed), "-b", f"{r}-{r}"]
                run_p2vbme(exe, [*run, *rep], out / f"boot/rep{r:03d}", ow)
        elif step == "repeats":
            for s in span(args.repeats):
                run_p2vbme(exe, [*run, "--seed", str(s)], out / f"repeat/seed{s}", ow)
        elif step == "supports":
            boot = concatenate_trees(boots, out / "boot.nwk", ow)
            rep = concatenate_trees(repeats, out / "repeats.nwk", ow)
            ours, maple = out / "tree.nwk", out / "maple.nwk"
            for prefix, tree, replicates in (
                ("tree.boot", ours, boot),
                ("tree.repeats", ours, rep),
                ("tree.maple", ours, maple),
                ("maple.boot", maple, boot),
                ("maple.repeats", maple, rep),
                ("maple.ours", maple, ours),
            ):
                supports(exe, tree, replicates, out / f"supports/{prefix}", ow)
        elif step == "records":
            records(out, args.alias, args.min_month_genomes, args.min_lineage_genomes)


if __name__ == "__main__":
    main()
