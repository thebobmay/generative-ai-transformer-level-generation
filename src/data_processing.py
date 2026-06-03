"""Data loading and chunking utilities for VGLC level files."""
import json
from pathlib import Path


def load_level_corpus(
    data_dir: Path,
    glob_pattern: str = "mario-*.txt",
) -> tuple[list[dict], dict]:
    """Load VGLC level text files and tile specification from a directory.

    Parameters
    ----------
    data_dir : Path
        Directory containing level .txt files and smb.json tile spec.
    glob_pattern : str
        Glob pattern used to select level files. Defaults to 'mario-*.txt'
        to load only the SMB1 overworld levels. Pass '*.txt' to include all
        level files in the directory (e.g. for the expanded corpus experiment).

    Returns
    -------
    levels : list[dict]
        Each dict has keys 'name' (str) and 'rows' (list of str).
    tile_spec : dict
        Tile character to property list mapping from smb.json.
    """
    data_dir = Path(data_dir)

    with open(data_dir / "smb.json") as f:
        tile_spec = json.load(f)["tiles"]

    levels = []
    for path in sorted(data_dir.glob(glob_pattern)):
        with open(path) as f:
            rows = [line.rstrip("\n") for line in f.readlines()]
        levels.append({"name": path.stem, "rows": rows})

    return levels, tile_spec


def normalize_corpus_rows(
    levels: list[dict],
    target_rows: int = 14,
) -> tuple[list[dict], list[dict]]:
    """Normalize level heights by trimming leading all-sky rows.

    Levels taller than target_rows have leading rows consisting entirely of
    the empty-sky tile ('-') stripped until the target height is reached or
    no more sky rows remain to trim. Levels that still do not equal
    target_rows after trimming are returned as skipped.

    Parameters
    ----------
    levels : list[dict]
        Level corpus as returned by load_level_corpus.
    target_rows : int
        Required number of rows. Default 14.

    Returns
    -------
    normalized : list[dict]
        Levels with exactly target_rows rows.
    skipped : list[dict]
        Levels that could not be normalized to target_rows.
    """
    normalized, skipped = [], []
    for lvl in levels:
        rows = list(lvl["rows"])
        while len(rows) > target_rows and set(rows[0]) <= {"-"}:
            rows = rows[1:]
        if len(rows) == target_rows:
            normalized.append({"name": lvl["name"], "rows": rows})
        else:
            skipped.append(lvl)
    return normalized, skipped


def chunk_level(level: dict, chunk_width: int = 32, stride: int = 16) -> list[dict]:
    """Split a level into fixed-width overlapping chunks.

    Parameters
    ----------
    level : dict
        Level dict with keys 'name' and 'rows' (list of str).
    chunk_width : int
        Number of tile columns per chunk. Default 32.
    stride : int
        Column step between chunk start positions. Default 16.

    Returns
    -------
    list[dict]
        Each dict has keys 'level_name', 'col_start', and 'rows'.
    """
    rows = level["rows"]
    n_cols = len(rows[0])
    chunks = []
    for start in range(0, n_cols - chunk_width + 1, stride):
        chunk_rows = [row[start:start + chunk_width] for row in rows]
        chunks.append({"level_name": level["name"], "col_start": start, "rows": chunk_rows})
    return chunks
