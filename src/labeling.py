"""Heuristic difficulty scoring and labeling for level chunks."""


def compute_difficulty_score(row: dict) -> float:
    """Compute a heuristic difficulty score for a level chunk.

    Weights enemy density, longest ground gap, and hazard density as
    the primary difficulty signals derived from structural features.
    Weights are exploratory and not drawn from published research.

    Parameters
    ----------
    row : dict
        Feature dict as produced by engineer_features, or a pd.Series row.

    Returns
    -------
    float
        Raw difficulty score. Higher values indicate harder chunks.
    """
    return (
        row["enemy_density"] * 40
        + row["longest_gap"] * 5
        + row["hazard_density"] * 30
    )


def assign_difficulty_label(
    score: float,
    easy_threshold: float,
    hard_threshold: float,
) -> str:
    """Map a raw difficulty score to a categorical label.

    Thresholds should be set at the 33rd and 67th percentiles of the
    corpus score distribution to ensure roughly equal class sizes.

    Parameters
    ----------
    score : float
        Raw difficulty score from compute_difficulty_score.
    easy_threshold : float
        Scores below this value are labeled Easy.
    hard_threshold : float
        Scores at or above this value are labeled Hard.

    Returns
    -------
    str
        One of 'Easy', 'Medium', or 'Hard'.
    """
    if score < easy_threshold:
        return "Easy"
    elif score < hard_threshold:
        return "Medium"
    return "Hard"
