"""Helpers several scripts share: running p2v-bme, reading its outputs, comparing trees."""

import json
import os
import re
import shutil
import subprocess

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

import numpy as np

from paper._config import DEFAULT_RESULTS_DIR, P2VBME


def read_json(path: Path) -> dict:
    """Reads a JSON file: path -> its object."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    """Writes an object as indented JSON, creating the folder: (path, data) -> None."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def parse_common_args(doc: str, devices: bool = True) -> ArgumentParser:
    """Builds the command line the p2v-bme scripts share:
    docstring -> parser with --results-dir, --p2vbme, --overwrite, and --accelerator, --devices."""
    parser = ArgumentParser(description=doc, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path(DEFAULT_RESULTS_DIR))
    parser.add_argument("--p2vbme", default=P2VBME)
    parser.add_argument("--overwrite", action="store_true", help="redo what exists")
    if devices:
        parser.add_argument("--accelerator", default="auto", help="p2v-bme -a (default: auto)")
        parser.add_argument("--devices", default="auto", help="p2v-bme -d (default: auto)")
    return parser


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
        subprocess.run([p2vbme, *args, "-p", str(prefix), "-q", "--overwrite"], check=True)
        if "-s" in args:
            threads = os.environ.get("OMP_NUM_THREADS")
            host = {"cpu_threads": int(threads) if threads else len(os.sched_getaffinity(0))}
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
        (float(t), float(e))
        for t, e in re.findall(r"\)([0-9.]+)/([0-9.]+)", Path(path).read_text(encoding="utf-8"))
    ]


def branch_table(path: Path) -> tuple[int, dict[str, np.ndarray]]:
    """Reads the branches above the internal nodes of a Newick tree, in the file's order:
    path -> (n leaves, columns).

    Columns: node (n_leaves + k for the k-th closing bracket),
    size (leaves below the node, as the file roots the tree), p (leaves on the smaller side),
    length (nan if none), tbe and exact (a "TBE/exact" label in %, nan if none).
    The root (the outermost bracket) has no branch and is left out.
    Two files written for the same tree list the same branches in the same order.
    """
    text = Path(path).read_text(encoding="utf-8").strip()
    chars = np.frombuffer(text.encode(), dtype=np.uint8)
    is_open, is_close = chars == ord("("), chars == ord(")")
    # A leaf name starts after "(" or "," at a character that is not "(".
    starts = np.zeros(chars.size, dtype=bool)
    starts[1:] = (is_open[:-1] | (chars[:-1] == ord(","))) & ~is_open[1:]
    leaves_before = np.cumsum(starts)
    n_leaves = int(leaves_before[-1])
    # Brackets pair up level by level: the k-th "(" of a depth closes at the k-th ")" of that depth.
    depth = np.cumsum(is_open.astype(np.int64) - is_close.astype(np.int64))
    opens, closes = np.flatnonzero(is_open), np.flatnonzero(is_close)
    open_order = np.lexsort((opens, depth[opens]))
    close_order = np.lexsort((closes, depth[closes] + 1))
    open_of = np.empty(closes.size, dtype=np.int64)
    open_of[close_order] = opens[open_order]
    size = leaves_before[closes] - leaves_before[open_of]
    tbe, exact, length = (np.full(closes.size, np.nan) for _ in range(3))
    for k, m in enumerate(re.finditer(r"\)([^:,();]*)(?::([^,();]*))?", text)):
        label, value = m.group(1), m.group(2)
        if "/" in label:
            tbe[k], exact[k] = (float(x) for x in label.split("/"))
        if value:
            length[k] = float(value)
    branch = depth[closes] > 0  # the root's ")" closes depth 1 -> 0
    columns = {
        "node": n_leaves + np.arange(closes.size),
        "size": size,
        "p": np.minimum(size, n_leaves - size),
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
    text = Path(path).read_text(encoding="utf-8").strip()
    if "'" in text or "[" in text:
        raise ValueError(f"{path}: quoted names and comments are not supported")
    chars = np.frombuffer(text.encode(), dtype=np.uint8)
    is_open, is_close = chars == ord("("), chars == ord(")")
    starts = np.zeros(chars.size, dtype=bool)
    starts[1:] = (is_open[:-1] | (chars[:-1] == ord(","))) & ~is_open[1:]
    leaves_before = np.cumsum(starts)
    leaf_at = np.flatnonzero(starts)
    n = leaf_at.size
    depth = np.cumsum(is_open.astype(np.int64) - is_close.astype(np.int64))
    opens, closes = np.flatnonzero(is_open), np.flatnonzero(is_close)
    open_of = np.empty(closes.size, dtype=np.int64)
    open_of[np.lexsort((closes, depth[closes] + 1))] = opens[np.lexsort((opens, depth[opens]))]
    # The parent of a node is the first ")" after it that closes the level the node sits in.
    level = depth[closes] + 1  # level of each internal node's own brackets
    key = level * (chars.size + 1) + closes
    order = np.argsort(key)
    sorted_keys = key[order]

    def enclosing(levels: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Finds the ")" of the bracket enclosing each position:
        (its level, position) -> internal node index."""
        at = np.searchsorted(sorted_keys, levels * (chars.size + 1) + positions)
        found = order[np.minimum(at, order.size - 1)]
        ok = (at < order.size) & (level[found] == levels)
        return np.where(ok, n + found, -1)

    parent = np.concatenate([enclosing(depth[leaf_at], leaf_at), enclosing(level - 1, closes)])
    leaves = re.findall(r"[(,]([^(),:;]+)(?::([^,();]*))?", text)
    if len(leaves) != n:
        raise ValueError(f"{path}: {len(leaves)} leaf names for {n} leaves")
    length = np.concatenate([np.full(n, np.nan), np.full(closes.size, np.nan)])
    tbe, exact = np.full(n + closes.size, np.nan), np.full(n + closes.size, np.nan)
    for i, (_, value) in enumerate(leaves):
        if value:
            length[i] = float(value)
    for k, m in enumerate(re.finditer(r"\)([^:,();]*)(?::([^,();]*))?", text)):
        label, value = m.group(1), m.group(2)
        if "/" in label:
            tbe[n + k], exact[n + k] = (float(x) for x in label.split("/"))
        if value:
            length[n + k] = float(value)
    first = np.concatenate([np.arange(n), leaves_before[open_of]])
    size = np.concatenate(
        [np.ones(n, dtype=np.int64), leaves_before[closes] - leaves_before[open_of]]
    )
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
    text = Path(tree).read_text(encoding="utf-8").strip()
    text = re.sub(r":[^,();]*", "", text)
    text = re.sub(r"\)[^,();]*", ")", text)
    Path(out).write_text(
        re.sub(r"([^(),;]+|\))(?=[,)])", rf"\g<1>:{edge:.12g}", text) + "\n", encoding="utf-8"
    )
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
                    shutil.copyfileobj(fi, fo)
    return out


def span(text: str) -> list[int]:
    """Parses a range such as 1-100 or 7: text -> the integers."""
    a, _, b = text.partition("-")
    return list(range(int(a), int(b or a) + 1))


def auc(score: np.ndarray, positive: np.ndarray) -> float:
    """Computes the area under the ROC curve of a score for a true/false outcome:
    (score, outcome) -> AUC.

    By rank sum, tied scores sharing their mean rank.
    """
    order = np.argsort(score, kind="stable")
    _, first, counts = np.unique(score[order], return_index=True, return_counts=True)
    ranks = np.empty(score.size)
    ranks[order] = (first + (counts + 1) / 2.0).repeat(counts)
    n1 = int(positive.sum())
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * (positive.size - n1)))


def pct(x) -> float:
    """Averages a 0/1 or 0..1 series in per cent:
    series -> per cent (nan if empty; a pandas Series skips nan)."""
    return float(100.0 * x.mean()) if len(x) else np.nan
