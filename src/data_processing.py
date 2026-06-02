"""Data loading and chunking utilities for VGLC level files."""
import json
from pathlib import Path


def load_level_corpus(data_dir: Path) -> tuple[list[dict], dict]:
    """Load all VGLC level text files and tile specification from a directory.

    Parameters
    ----------
    data_dir : Path
        Directory containing level .txt files and smb.json tile spec.

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
    for path in sorted(data_dir.glob("mario-*.txt")):
        with open(path) as f:
            rows = [line.rstrip("\n") for line in f.readlines()]
        levels.append({"name": path.stem, "rows": rows})

    return levels, tile_spec


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
