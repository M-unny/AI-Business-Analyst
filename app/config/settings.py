"""Central configuration. Thresholds live here so they are tunable and testable."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # --- upload safety ---
    max_file_size_mb: int = 50
    max_uncompressed_mb: int = 500      # zip-bomb guard for .xlsx
    max_rows: int = 1_000_000
    max_columns: int = 500
    allowed_extensions: tuple[str, ...] = (".csv", ".xlsx")

    # --- type inference ---
    date_sample_size: int = 1_000
    date_parse_threshold: float = 0.90      # share of sample that must parse as dates
    numeric_text_threshold: float = 0.95    # share of strings that must parse as numbers
    id_unique_ratio: float = 0.95
    id_min_rows: int = 20
    id_max_avg_len: int = 40
    categorical_max_unique_ratio: float = 0.5
    categorical_max_unique: int = 50
    text_min_avg_len: int = 50

    # --- reporting ---
    high_missing_pct: float = 50.0
    semantic_min_confidence: float = 0.5

    project_root: Path = Path(__file__).resolve().parents[2]

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def max_uncompressed_bytes(self) -> int:
        return self.max_uncompressed_mb * 1024 * 1024


settings = Settings()
