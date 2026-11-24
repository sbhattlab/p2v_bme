"""Helpers several scripts share: running p2v-bme, reading its outputs, comparing trees."""

from json import dumps, loads
from os import environ, sched_getaffinity
from pathlib import Path
from re import findall, finditer, sub
from shutil import copyfileobj
from subprocess import run

from numpy import (
    arange,
    argsort,
    concatenate,
    cumsum,
    empty,
    flatnonzero,
    frombuffer,
    full,
    int64,
    lexsort,
    minimum,
    nan,
    ndarray,
    ones,
    searchsorted,
    uint8,
    unique,
    where,
    zeros,
)


def read_json(path: Path) -> dict:
    """Reads a JSON file: path -> its object."""
    return loads(Path(path).read_text())


def write_json(path: Path, data: dict) -> None:
    """Writes an object as indented JSON, creating the folder: (path, data) -> None."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(data, indent=1) + "\n")


def run_p2vbme(p2vbme: str, args: list[str], prefix: Path, overwrite: bool) -> dict:
    """Runs p2v-bme unless its report exists: (args, prefix) -> the report <prefix>.json.

    args are p2v-bme's options without -p;
    the run writes <prefix>.nwk, <prefix>.json and <prefix>.log,
    and for a search (-s) this function writes <prefix>.host.json: the CPU threads the run had
    (OMP_NUM_THREADS, else the cores it may use), which the tables state for GPU runs.
    A failed run leaves no report (only its log): it runs again.
    The report decides: p2v-bme gets --overwrite,
    so the other outputs of a run without a report do not stop it.
    """
    report = Path(f"{prefix}.json")
    if overwrite or not report.exists():
        report.parent.mkdir(parents=True, exist_ok=True)
        run([p2vbme, *args, "-p", str(prefix), "-q", "--overwrite"], check=True)
        if "-s" in args:
            threads = environ.get("OMP_NUM_THREADS")
            host = {"cpu_threads": int(threads) if threads else len(sched_getaffinity(0))}
            write_json(Path(f"{prefix}.host.json"), host)
    return read_json(report)


def supports(p2vbme: str, tree: Path, replicates: Path, prefix: Path, overwrite: bool) -> dict:
    """Computes the supports of a tree's branches over replicate trees:
    (tree, replicates, prefix) -> report.

    Writes <prefix>.support.nwk (labels "TBE/exact" in %) and <prefix>.json;
    skipped if the report exists.
    """
    return run_p2vbme(p2vbme, ["-t", str(tree), "--supports", str(replicates)], prefix, overwrite)


def compare_to_truth(p2vbme: str, tree: Path, true_tree: Path, overwrite: bool) -> dict:
    """Compares a tree with the true tree: (tree, true tree) -> {rf_pct, mean_tbe, mean_exact}.

    Writes <tree without .nwk>.vs_true.{support.nwk,json}.
    For binary trees on the same leaves, the normalised RF distance RF / (2 (n - 3)) is
    1 - mean exact support.
    """
    prefix = Path(str(tree).removesuffix(".nwk") + ".vs_true")
    s = supports(p2vbme, tree, true_tree, prefix, overwrite)["supports"]
    return {
        "rf_pct": 100.0 - s["mean_exact"],
        "mean_tbe": s["mean_tbe"],
        "mean_exact": s["mean_exact"],
    }


def support_labels(path: Path) -> list[tuple[float, float]]:
    """Reads the branch labels of a .support.nwk in the file's order: path -> [(TBE, exact)] in %.

    Two files written for the same tree list the same branches in the same order.
    """
    return [
        (float(t), float(e)) for t, e in findall(r"\)([0-9.]+)/([0-9.]+)", Path(path).read_text())
    ]


def branch_table(path: Path) -> tuple[int, dict[str, ndarray]]:
    """Reads the branches above the internal nodes of a Newick tree, in the file's order:
    path -> (n leaves, columns).

    Columns: node (n_leaves + k for the k-th closing bracket),
    size (leaves below the node, as the file roots the tree), p (leaves on the smaller side),
    length (nan if none), tbe and exact (a "TBE/exact" label in %, nan if none).
    The root (the outermost bracket) has no branch and is left out.
    Two files written for the same tree list the same branches in the same order.
    """
    text = Path(path).read_text().strip()
    chars = frombuffer(text.encode(), dtype=uint8)
    is_open, is_close = chars == ord("("), chars == ord(")")
    # A leaf name starts after "(" or "," at a character that is not "(".
    starts = zeros(chars.size, dtype=bool)
    starts[1:] = (is_open[:-1] | (chars[:-1] == ord(","))) & ~is_open[1:]
    leaves_before = cumsum(starts)
    n_leaves = int(leaves_before[-1])
    # Brackets pair up level by level: the k-th "(" of a depth closes at the k-th ")" of that depth.
    depth = cumsum(is_open.astype(int64) - is_close.astype(int64))
    opens, closes = flatnonzero(is_open), flatnonzero(is_close)
    open_order = lexsort((opens, depth[opens]))
    close_order = lexsort((closes, depth[closes] + 1))
    open_of = empty(closes.size, dtype=int64)
    open_of[close_order] = opens[open_order]
    size = leaves_before[closes] - leaves_before[open_of]
    tbe, exact, length = (full(closes.size, nan) for _ in range(3))
    for k, m in enumerate(finditer(r"\)([^:,();]*)(?::([^,();]*))?", text)):
        label, value = m.group(1), m.group(2)
        if "/" in label:
            tbe[k], exact[k] = (float(x) for x in label.split("/"))
        if value:
            length[k] = float(value)
    branch = depth[closes] > 0  # the root's ")" closes depth 1 -> 0
    columns = {
        "node": n_leaves + arange(closes.size),
        "size": size,
        "p": minimum(size, n_leaves - size),
        "length": length,
        "tbe": tbe,
        "exact": exact,
    }
    return n_leaves, {k: v[branch] for k, v in columns.items()}


def read_tree(path: Path) -> dict:
    """Reads a Newick tree into node arrays:
    path -> {names, parent, first, size, length, tbe, exact}.

    Leaves are nodes 0 .. n-1 in the file's order (a depth-first order),
    internal nodes n, n+1, ... in the order of their closing brackets,
    so children come before their parent and the root (outermost bracket) is last.
    The leaves below node u are first[u] .. first[u] + size[u] - 1; parent is -1 at the root.
    length: the branch above the node (nan if none);
    tbe, exact: its "TBE/exact" label in % (nan if none).
    Names must not be quoted.
    """
    text = Path(path).read_text().strip()
    if "'" in text or "[" in text:
        raise ValueError(f"{path}: quoted names and comments are not supported")
    chars = frombuffer(text.encode(), dtype=uint8)
    is_open, is_close = chars == ord("("), chars == ord(")")
    starts = zeros(chars.size, dtype=bool)
    starts[1:] = (is_open[:-1] | (chars[:-1] == ord(","))) & ~is_open[1:]
    leaves_before = cumsum(starts)
    leaf_at = flatnonzero(starts)
    n = leaf_at.size
    depth = cumsum(is_open.astype(int64) - is_close.astype(int64))
    opens, closes = flatnonzero(is_open), flatnonzero(is_close)
    open_of = empty(closes.size, dtype=int64)
    open_of[lexsort((closes, depth[closes] + 1))] = opens[lexsort((opens, depth[opens]))]
    # The parent of a node is the first ")" after it that closes the level the node sits in.
    level = depth[closes] + 1  # level of each internal node's own brackets
    key = level * (chars.size + 1) + closes
    order = argsort(key)
    sorted_keys = key[order]

    def enclosing(levels: ndarray, positions: ndarray) -> ndarray:
        """Finds the ")" of the bracket enclosing each position:
        (its level, position) -> internal node index."""
        at = searchsorted(sorted_keys, levels * (chars.size + 1) + positions)
        found = order[minimum(at, order.size - 1)]
        ok = (at < order.size) & (level[found] == levels)
        return where(ok, n + found, -1)

    parent = concatenate([enclosing(depth[leaf_at], leaf_at), enclosing(level - 1, closes)])
    leaves = [(name, value) for name, value in findall(r"[(,]([^(),:;]+)(?::([^,();]*))?", text)]
    if len(leaves) != n:
        raise ValueError(f"{path}: {len(leaves)} leaf names for {n} leaves")
    length = concatenate([full(n, nan), full(closes.size, nan)])
    tbe, exact = full(n + closes.size, nan), full(n + closes.size, nan)
    for i, (_, value) in enumerate(leaves):
        if value:
            length[i] = float(value)
    for k, m in enumerate(finditer(r"\)([^:,();]*)(?::([^,();]*))?", text)):
        label, value = m.group(1), m.group(2)
        if "/" in label:
            tbe[n + k], exact[n + k] = (float(x) for x in label.split("/"))
        if value:
            length[n + k] = float(value)
    first = concatenate([arange(n), leaves_before[open_of]])
    size = concatenate([ones(n, dtype=int64), leaves_before[closes] - leaves_before[open_of]])
    return {
        "names": [name.strip() for name, _ in leaves],
        "parent": parent,
        "first": first,
        "size": size,
        "length": length,
        "tbe": tbe,
        "exact": exact,
    }


def with_edge_lengths(tree: Path, edge: float, out: Path) -> Path:
    """Writes a tree's topology with every edge of the same length: (tree, length) -> out.

    Branch lengths and internal labels are dropped;
    the comparison programs read start trees with lengths.
    """
    text = Path(tree).read_text().strip()
    text = sub(r":[^,();]*", "", text)
    text = sub(r"\)[^,();]*", ")", text)
    Path(out).write_text(sub(r"([^(),;]+|\))(?=[,)])", rf"\g<1>:{edge:.12g}", text) + "\n")
    return Path(out)


def concatenate_trees(paths: list[Path], out: Path, overwrite: bool) -> Path:
    """Writes trees one per line into one file, for p2v-bme --supports: (tree files, out) -> out."""
    if overwrite or not out.exists():
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"{len(missing)} trees missing, e.g. {missing[0]}")
        with open(out, "wb") as fo:
            for p in paths:
                with open(p, "rb") as fi:
                    copyfileobj(fi, fo)
    return out


def span(text: str) -> list[int]:
    """Parses a range such as 1-100 or 7: text -> the integers."""
    a, _, b = text.partition("-")
    return list(range(int(a), int(b or a) + 1))


def auc(score: ndarray, positive: ndarray) -> float:
    """Computes the area under the ROC curve of a score for a true/false outcome:
    (score, outcome) -> AUC.

    By rank sum, tied scores sharing their mean rank.
    """
    order = argsort(score, kind="stable")
    _, first, counts = unique(score[order], return_index=True, return_counts=True)
    ranks = empty(score.size)
    ranks[order] = (first + (counts + 1) / 2.0).repeat(counts)
    n1 = int(positive.sum())
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * (positive.size - n1)))


def pct(x) -> float:
    """Averages a 0/1 or 0..1 series in per cent:
    series -> per cent (nan if empty; a pandas Series skips nan)."""
    return float(100.0 * x.mean()) if len(x) else nan
