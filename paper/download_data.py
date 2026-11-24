"""Downloads the paper's empirical and source datasets from their original deposits, checked by size and MD5.

Sources (licence; what to cite, DOIs as in the dataset check of the paper):
  rnasim       RNASim's 200,000-sequence alignment and true tree, the source of every RNASim and JC set
               (simulate_data.py); apples2-data (Balaban et al. 2022, doi:10.1111/1755-0998.13527), pinned to commit
               5f8de20; no licence stated. Simulator: Guo et al. 2009, doi:10.48550/arXiv.0912.2326. The checksums
               are those of the copy the paper's datasets were made from.
  silva144     SILVA 144 SSU Ref aligned sequences with taxonomy (CC BY 4.0; cite Chuvochina et al. 2026,
               doi:10.1093/nar/gkaf1247; dataset doi:10.82364/144/SSU/Ref/FASTA/aligned).
  silva144-arb SILVA 144 SSU Ref ARB database, the source of the GTDB labels (field tax_gtdb; CC BY 4.0, GTDB itself
               CC BY-SA 4.0; cite Westram et al. 2011, doi:10.1002/9781118010518.ch46, and GTDB). Dumping the labels
               needs ARB, which is not in the pixi environments. Fetched only when named in --sources (7.2 GB, read
               only by run_silva's gtdb step).
  cmaple       CMAPLE's supplementary data, version 1.0.0 (Zenodo 10.5281/zenodo.11003148, CC BY 4.0; Ly-Trong et
               al. 2024, doi:10.1093/molbev/msae134); later versions lack the simulated sets. The nine simulated
               pandemic sets are extracted from its zip (Validation/cmaple_vs_maple/simulated): alignments made by
               CMAPLE's authors with AliSim on random Yule-Harding trees (30,000 sites, GTR+F+G+I, mean branch length
               5e-5, 1e-4 or 1.5e-4), their true trees, and CMAPLE's and MAPLE's trees and run logs.
  viridian     Viridian's 2M SARS-CoV-2 genomes processed with MAPLE (Zenodo 10.5281/zenodo.12733488 and
               10.5281/zenodo.15546527, CC BY 4.0; De Maio et al. 2025, doi:10.1038/s41586-025-09567-x); the
               files run_viridian reads: the alignment, extracted from its zip, and MAPLE's tree with each genome's
               lineage and date (the .jsonl zip, read as it is). With it, the Pango alias key that expands the
               lineage names (cov-lineages/pango-designation, pinned to commit dbfab0d, the version the paper's
               records used; CC BY-NC 4.0; Rambaut et al. 2020, doi:10.1038/s41564-020-0770-5).

Inputs:  the deposits' URLs (below)
Options: --sources, --datasets-dir, --dry-run, --overwrite
Flow:    each source
           |
           v
         each file: kept if its size and MD5 match, else fetched with curl to <file>.part, checked, renamed (a .part
         that fails the check is deleted)
           |
           v
         zips: the members the scripts read, extracted and checked against the zip's CRC
Outputs: data/datasets/<source>/...                     the files as published, and what was extracted from them
         data/datasets/<source>/download_<source>.json  URL, size and MD5 of every file, and when it was checked
         (silva144-arb writes to silva144/)

    pixi run -e paper python -m paper.download_data --sources rnasim --dry-run
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections.abc import Callable
from datetime import datetime, timezone
from hashlib import md5
from json import dumps, loads
from pathlib import Path
from shutil import copyfileobj
from subprocess import run
from urllib.request import urlopen
from zipfile import ZipFile
from zlib import crc32

from paper._config import DEFAULT_DATASETS_DIR

BLOCK = 8 * 1024 * 1024
APPLES2 = (
    "https://raw.githubusercontent.com/balabanmetin/apples2-data/5f8de20764899b9f95f5fda4fb5be867ed045211/"
    "singlegene/rnasim/variable-size/rnasim-source"
)
SILVA = "https://ftp.arb-silva.de/release_144"
ZENODO = "https://zenodo.org/api/records"
PANGO = (
    "https://raw.githubusercontent.com/cov-lineages/pango-designation/dbfab0db6b314825d290f6ac1cf1861686c97a6c/"
    "pango_designation"
)
# source -> files: (folder below the source's, name, URL, bytes, MD5); MD5 None: from the deposit's .md5 file
FILES = {
    "rnasim": [
        (
            "",
            "aln_dna.fa.xz",
            f"{APPLES2}/aln_dna.fa.xz",
            32177056,
            "290e9badd3d8af2258c4b7f1587b6224",
        ),
        (
            "",
            "true_topo.tree",
            f"{APPLES2}/true_topo.tree",
            5652026,
            "4c564b283f7a9088d033bd12d7dce2bf",
        ),
    ],
    "silva144": [
        (
            "",
            "SILVA_144_SSURef_tax_silva_full_align_trunc.fasta.gz",
            f"{SILVA}/Exports/SILVA_144_SSURef_tax_silva_full_align_trunc.fasta.gz",
            20393029126,
            "cd83af97382f1d5a5629d21ff6d857d8",
        ),
    ],
    "silva144-arb": [
        (
            "",
            "SILVA_144_SSURef_opt.arb.gz",
            f"{SILVA}/ARB_files/SILVA_144_SSURef_opt.arb.gz",
            7166345635,
            None,
        ),
    ],
    "viridian": [
        (str(record), name, f"{ZENODO}/{record}/files/{name}/content", size, digest)
        for record, name, size, digest in (
            (
                15546527,
                "Viridian_2M_noShortDel_deeper_SPRTA0611_tree_combined.jsonl.zip",
                152502098,
                "7d3f3c580f7b1a2c9db927319f37e5e7",
            ),
            (
                12733488,
                "Viridian_2M_SARS-COV-2_alignment.maple.zip",
                73120005,
                "2b7884b6a7f5e086eb80aaf96fcce859",
            ),
        )
    ]
    + [
        (
            "",
            "alias_key.json",
            f"{PANGO}/alias_key.json",
            20126,
            "5537e72164c5ee80c01e5bd3f2456aad",
        ),
    ],
}
CMAPLE_RECORD = 11003148  # version 1.0.0; later versions lack the simulated sets
CMAPLE_BYTES = 1660813065  # the record's one zip
# The nine simulated pandemic sets: alignment, true tree, CMAPLE's and MAPLE's trees and their run logs, by size and
# branch-length scale
CMAPLE_SIZES = ("100K", "200K", "500K")
CMAPLE_SCALES = ("0.00005", "0.0001", "0.00015")
CMAPLE_MEMBERS = [
    name
    for size in CMAPLE_SIZES
    for scale in CMAPLE_SCALES
    for name in (
        f"aln_{size}_{scale}.maple",
        f"tree_{size}_{scale}.nwk",
        f"aln_{size}_{scale}.maple.treefile",
        f"aln_{size}_{scale}_wo_ref_tree.tree",
        f"CMaple_{size}_{scale}.log",
        f"Maple_{size}_{scale}.log",
    )
]
VIRIDIAN_MEMBERS = (".maple",)  # members extracted from the Viridian zips, by suffix
OPT_IN = ("silva144-arb",)  # sources fetched only when named in --sources


def file_md5(path: Path) -> str:
    """Computes a file's MD5 without reading it into memory: path -> hex digest."""
    h = md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def file_crc32(path: Path) -> int:
    """Computes a file's CRC-32, as zip archives record it: path -> CRC."""
    value = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK), b""):
            value = crc32(block, value)
    return value


def fetch(url: str, path: Path, size: int, digest: str | None, overwrite: bool) -> str:
    """Downloads a file unless a checked copy exists: (URL, path, bytes, MD5) -> its MD5.

    Without a recorded MD5, the deposit's <URL>.md5 file gives it.
    """
    if digest is None:
        with urlopen(f"{url}.md5") as response:
            digest = response.read().decode().split()[0]
    if path.exists() and not overwrite:
        if path.stat().st_size == size and file_md5(path) == digest:
            print(f"download_data: {path} checked", flush=True)
            return digest
        raise SystemExit(
            f"{path} differs from the deposit (size or MD5); remove it or pass --overwrite"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    print(f"download_data: {url} ({size:,} bytes)", flush=True)
    run(
        [
            "curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            "3",
            "--connect-timeout",
            "30",
        ]
        + ["--continue-at", "-", "--output", str(part), url],
        check=True,
    )
    if part.stat().st_size != size or file_md5(part) != digest:
        part.unlink()  # else curl would resume onto it next time
        raise SystemExit(f"{part}: size or MD5 differs from the deposit; deleted it")
    part.replace(path)
    return digest


def extract(archive: Path, wanted: Callable[[str], bool], folder: Path) -> list[str]:
    """Extracts members of a zip by base name: (zip, test on the base name, folder) -> the base names extracted.

    A member already extracted is kept if its size and CRC match the zip's record.
    """
    done = []
    with ZipFile(archive) as z:
        for member in z.infolist():
            name = Path(member.filename).name
            if (
                member.is_dir()
                or name.startswith("._")
                or "__MACOSX" in member.filename
                or not wanted(name)
            ):
                continue
            target = folder / name
            if not (
                target.exists()
                and target.stat().st_size == member.file_size
                and file_crc32(target) == member.CRC
            ):
                folder.mkdir(parents=True, exist_ok=True)
                part = target.with_name(name + ".part")
                with z.open(member) as src, part.open(
                    "wb"
                ) as dst:  # reading to the end checks the CRC
                    copyfileobj(src, dst, BLOCK)
                part.replace(target)
            done.append(name)
    return done


def cmaple_file() -> tuple[str, str, str, int, str]:
    """Looks up the CMAPLE record's zip on Zenodo: () -> (folder, name, URL, bytes, MD5)."""
    with urlopen(f"{ZENODO}/{CMAPLE_RECORD}") as response:
        files = loads(response.read())["files"]
    if len(files) != 1 or files[0]["size"] != CMAPLE_BYTES:
        raise SystemExit(
            f"Zenodo record {CMAPLE_RECORD} no longer holds one zip of {CMAPLE_BYTES:,} bytes"
        )
    f = files[0]
    return (
        "",
        f["key"],
        f["links"]["self"],
        f["size"],
        f["checksum"].removeprefix("md5:"),
    )


def download(source: str, folder: Path, dry_run: bool, overwrite: bool) -> None:
    """Fetches one source and extracts what the scripts read: (source, its folder) -> files in it."""
    if dry_run:
        if source == "cmaple":
            print(
                f"{source}: the one zip of {ZENODO}/{CMAPLE_RECORD} ({CMAPLE_BYTES:,} bytes) -> {folder}"
            )
            print(
                f"{source}: extract {len(CMAPLE_MEMBERS)} files of the simulated sets -> {folder / 'simulated'}"
            )
        for sub, name, url, size, _ in FILES.get(source, []):
            print(f"{source}: {url} ({size:,} bytes) -> {folder / sub / name}")
        return
    files = [cmaple_file()] if source == "cmaple" else FILES[source]
    record = []
    for sub, name, url, size, digest in files:
        path = folder / sub / name
        entry = {
            "path": str(Path(sub) / name),
            "url": url,
            "bytes": size,
            "md5": fetch(url, path, size, digest, overwrite),
        }
        if source == "cmaple":
            entry["extracted"] = extract(
                path, lambda member: member in CMAPLE_MEMBERS, folder / "simulated"
            )
            missing = sorted(set(CMAPLE_MEMBERS) - set(entry["extracted"]))
            if missing:
                raise SystemExit(
                    f"{path} lacks {len(missing)} files of the simulated sets, e.g. {missing[0]}"
                )
        elif path.suffix == ".zip":
            entry["extracted"] = extract(
                path, lambda n: n.endswith(VIRIDIAN_MEMBERS), folder / sub / "extracted"
            )
        record.append(entry)
    checked = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (folder / f"download_{source}.json").write_text(
        dumps({"source": source, "checked_utc": checked, "files": record}, indent=1)
    )


def main() -> None:
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    sources = [*FILES, "cmaple"]
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=sources,
        default=[s for s in sources if s not in OPT_IN],
        help=f"what to download (default: all but {', '.join(OPT_IN)})",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path(DEFAULT_DATASETS_DIR),
        help="output folder",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be fetched, fetch nothing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="fetch files again even if a checked copy exists",
    )
    args = parser.parse_args()
    for source in args.sources:
        folder = args.datasets_dir / source.removesuffix("-arb")
        download(source, folder, args.dry_run, args.overwrite)


if __name__ == "__main__":
    main()
