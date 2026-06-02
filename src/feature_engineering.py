"""Structural feature computation for VGLC Super Mario Bros level chunks."""

# Tile category sets derived from VGLC smb.json tile specification
SOLID_TILES       = {'X', 'S', '?', 'Q', '<', '>', '[', ']', 'B', 'b'}
HAZARD_TILES      = {'E', 'B'}
ENEMY_TILES       = {'E'}
COLLECTIBLE_TILES = {'o'}


def compute_enemy_density(chunk_rows: list[str]) -> float:
    """Fraction of tiles that are enemies (E).

    Parameters
    ----------
    chunk_rows : list[str]
        Tile row strings for a single chunk.

    Returns
    -------
    float
    """
    total = len(chunk_rows) * len(chunk_rows[0])
    return sum(row.count("E") for row in chunk_rows) / total


def compute_hazard_density(chunk_rows: list[str]) -> float:
    """Fraction of tiles that are hazards (enemies and cannons).

    Parameters
    ----------
    chunk_rows : list[str]
        Tile row strings for a single chunk.

    Returns
    -------
    float
    """
    total = len(chunk_rows) * len(chunk_rows[0])
    return sum(sum(1 for c in row if c in HAZARD_TILES) for row in chunk_rows) / total


def compute_longest_gap(chunk_rows: list[str]) -> int:
    """Max consecutive empty columns in the ground row (bottom row).

    A gap is a run of '-' tiles in the bottom row, representing a pit
    the player must jump over.

    Parameters
    ----------
    chunk_rows : list[str]
        Tile row strings for a single chunk.

    Returns
    -------
    int
    """
    max_gap = current_gap = 0
    for tile in chunk_rows[-1]:
        if tile == "-":
            current_gap += 1
            max_gap = max(max_gap, current_gap)
        else:
            current_gap = 0
    return max_gap


def compute_solid_ratio(chunk_rows: list[str]) -> float:
    """Fraction of tiles that are solid (ground, blocks, pipes).

    Parameters
    ----------
    chunk_rows : list[str]
        Tile row strings for a single chunk.

    Returns
    -------
    float
    """
    total = len(chunk_rows) * len(chunk_rows[0])
    return sum(sum(1 for c in row if c in SOLID_TILES) for row in chunk_rows) / total


def compute_collectible_density(chunk_rows: list[str]) -> float:
    """Fraction of tiles that are collectibles (coins).

    Parameters
    ----------
    chunk_rows : list[str]
        Tile row strings for a single chunk.

    Returns
    -------
    float
    """
    total = len(chunk_rows) * len(chunk_rows[0])
    return sum(row.count("o") for row in chunk_rows) / total


def engineer_features(chunk: dict) -> dict:
    """Compute all structural features for a single level chunk.

    Parameters
    ----------
    chunk : dict
        Chunk dict with keys 'level_name', 'col_start', and 'rows'.

    Returns
    -------
    dict
        Keys: level_name, col_start, enemy_density, hazard_density,
        longest_gap, solid_ratio, collectible_density.
    """
    rows = chunk["rows"]
    return {
        "level_name":          chunk["level_name"],
        "col_start":           chunk["col_start"],
        "enemy_density":       compute_enemy_density(rows),
        "hazard_density":      compute_hazard_density(rows),
        "longest_gap":         compute_longest_gap(rows),
        "solid_ratio":         compute_solid_ratio(rows),
        "collectible_density": compute_collectible_density(rows),
    }
