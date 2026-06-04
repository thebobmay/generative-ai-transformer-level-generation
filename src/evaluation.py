"""Evaluation metrics for generated level segments."""


def tile_similarity(rows_a: list, rows_b: list) -> float:
    """Compute tile-level similarity between two level grids.

    Returns the fraction of tile positions that are identical between
    two same-size grids, read row by row left to right top to bottom.

    Parameters
    ----------
    rows_a, rows_b : list[str]
        Tile row strings for each grid.

    Returns
    -------
    float
        Similarity in [0, 1]. Returns 0.0 if grids are empty or mismatched.
    """
    a = "".join(rows_a)
    b = "".join(rows_b)
    if len(a) != len(b) or not a:
        return 0.0
    return sum(ca == cb for ca, cb in zip(a, b)) / len(a)


def evaluate_structural_validity(generated_levels: list, valid_vocab: set) -> dict:
    """Check generated levels for valid tile vocabulary, correct dimensions, and non-empty output.

    Parameters
    ----------
    generated_levels : list[dict]
        Level dicts with key 'rows', as returned by generate_level.
    valid_vocab : set
        Set of valid tile characters.

    Returns
    -------
    dict
        Keys: n_total, valid_vocab_rate, valid_dims_rate, nonempty_rate, all_valid_rate.
    """
    n = len(generated_levels)
    vocab_ok_count = dims_ok_count = nonempty_count = all_ok_count = 0
    for lvl in generated_levels:
        rows      = lvl["rows"]
        vocab_ok  = set("".join(rows)).issubset(valid_vocab)
        dims_ok   = len(rows) == 14 and all(len(r) == 32 for r in rows)
        nonempty  = bool(rows) and any(c != "-" for row in rows for c in row)
        vocab_ok_count += vocab_ok
        dims_ok_count  += dims_ok
        nonempty_count += nonempty
        all_ok_count   += vocab_ok and dims_ok and nonempty
    return {
        "n_total":          n,
        "valid_vocab_rate": vocab_ok_count / n,
        "valid_dims_rate":  dims_ok_count  / n,
        "nonempty_rate":    nonempty_count / n,
        "all_valid_rate":   all_ok_count   / n,
    }


def compute_diversity(generated_levels: list) -> float:
    """Return the fraction of generated levels that are unique tile grids.

    Parameters
    ----------
    generated_levels : list[dict]
        Level dicts with key 'rows'.

    Returns
    -------
    float
    """
    keys = [tuple(lvl["rows"]) for lvl in generated_levels]
    return len(set(keys)) / len(keys) if keys else 0.0


def compute_novelty(
    generated_levels: list,
    training_chunks: list,
    threshold: float = 0.90,
) -> dict:
    """Fraction of generated levels whose nearest training chunk is below a similarity threshold.

    For each generated level the maximum tile-level similarity against all
    training chunks is computed. A level is considered novel if that
    nearest-neighbour similarity is below the threshold.

    Parameters
    ----------
    generated_levels : list[dict]
        Level dicts with key 'rows'.
    training_chunks : list[dict]
        Corpus chunks, each with key 'rows'.
    threshold : float
        Similarity above which a level is considered a near-copy. Default 0.90.

    Returns
    -------
    dict
        Keys: rate (float), mean_max_similarity (float), threshold (float).
    """
    train_rows = [c["rows"] for c in training_chunks]
    sims = [
        max(tile_similarity(lvl["rows"], tr) for tr in train_rows)
        for lvl in generated_levels
    ]
    rate = sum(s < threshold for s in sims) / len(sims) if sims else 0.0
    return {
        "rate":                rate,
        "mean_max_similarity": sum(sims) / len(sims) if sims else 0.0,
        "threshold":           threshold,
    }
