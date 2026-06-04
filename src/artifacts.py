"""Serializable artifact classes for the level generation pipeline."""
from src.tokenizer import (
    CHAR_TO_ID,
    CONTROL_TOKEN_NAMES,
    CONTROL_TOKENS,
    ID_TO_CHAR,
    SEQ_LEN,
    VOCAB_SIZE,
)


class LevelTokenizer:
    """Vocabulary wrapper for encoding and decoding level token sequences.

    Saved to outputs/model_artifacts/level_tokenizer.pkl for downstream use.
    All vocabulary data is stored as instance attributes so the object is
    fully self-contained after unpickling without requiring notebook context.
    """

    def __init__(self) -> None:
        self.char_to_id          = CHAR_TO_ID
        self.id_to_char          = ID_TO_CHAR
        self.control_tokens      = CONTROL_TOKENS
        self.control_token_names = CONTROL_TOKEN_NAMES
        self.vocab_size          = VOCAB_SIZE
        self.seq_len             = SEQ_LEN
        self.n_rows              = 14
        self.n_cols              = 32

    def encode(self, chunk: dict, difficulty: str) -> list:
        """Encode a chunk dict and difficulty label into a token sequence."""
        control     = self.control_tokens[difficulty]
        tile_tokens = [self.char_to_id[c] for row in chunk["rows"] for c in row]
        return [control] + tile_tokens

    def decode(self, token_ids: list) -> dict:
        """Decode a token sequence to a dict with 'difficulty' and 'rows'."""
        difficulty = self.control_token_names[token_ids[0]]
        flat       = [self.id_to_char[t] for t in token_ids[1:]]
        rows       = [
            "".join(flat[i * self.n_cols:(i + 1) * self.n_cols])
            for i in range(self.n_rows)
        ]
        return {"difficulty": difficulty, "rows": rows}


class DifficultyScorer:
    """Heuristic difficulty scorer for generated level segments.

    Saved to outputs/model_artifacts/difficulty_scorer.pkl for downstream use.
    Thresholds are fitted to the training corpus score distribution at the
    33rd and 67th percentiles.
    """

    _HAZARD_TILES = {"E", "B"}

    def __init__(self, easy_threshold: float, hard_threshold: float) -> None:
        self.easy_threshold = easy_threshold
        self.hard_threshold = hard_threshold

    def score(self, rows: list) -> float:
        """Compute heuristic difficulty score from a list of tile row strings."""
        total          = len(rows) * len(rows[0]) if rows else 1
        enemy_density  = sum(row.count("E") for row in rows) / total
        hazard_density = sum(
            sum(1 for c in row if c in self._HAZARD_TILES) for row in rows
        ) / total
        max_gap = current_gap = 0
        for tile in rows[-1]:
            if tile == "-":
                current_gap += 1
                max_gap = max(max_gap, current_gap)
            else:
                current_gap = 0
        return enemy_density * 40 + max_gap * 5 + hazard_density * 30

    def label(self, score: float) -> str:
        """Map a difficulty score to 'Easy', 'Medium', or 'Hard'."""
        if score < self.easy_threshold:
            return "Easy"
        if score < self.hard_threshold:
            return "Medium"
        return "Hard"

    def score_rows(self, rows: list) -> tuple:
        """Return (score, label) for a list of tile row strings."""
        s = self.score(rows)
        return s, self.label(s)
