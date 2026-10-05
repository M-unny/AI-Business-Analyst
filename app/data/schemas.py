"""Pydantic schemas. These double as the API response models in Milestone 10."""
from typing import Literal

from pydantic import BaseModel, Field

ColumnType = Literal["numeric", "categorical", "datetime", "boolean", "id", "text", "empty"]
DateFormat = Literal["iso", "dayfirst", "monthfirst", "ambiguous", "mixed", "other"]


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    inferred_type: ColumnType
    missing_count: int
    missing_pct: float
    unique_count: int
    cardinality_ratio: float = Field(description="unique values / non-null values")
    min_value: float | str | None = None
    max_value: float | str | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    integer_like: bool | None = None
    outlier_count: int | None = Field(default=None, description="IQR rule (1.5x), numeric only")
    top_values: dict[str, int] | None = None
    stored_as_text: bool = False
    date_format: DateFormat | None = None
    notes: list[str] = Field(default_factory=list)


class SemanticMatch(BaseModel):
    column: str
    role: str
    confidence: float
    evidence: list[str] = Field(default_factory=list)


class SemanticResult(BaseModel):
    primary: dict[str, SemanticMatch] = Field(description="column -> best role")
    candidates: dict[str, list[SemanticMatch]] = Field(description="column -> all plausible roles")
    roles: dict[str, list[str]] = Field(description="role -> columns, best first")
    unmatched_columns: list[str]

    def best_column(self, role: str) -> str | None:
        cols = self.roles.get(role)
        return cols[0] if cols else None


class DatasetMetadata(BaseModel):
    rows: int
    columns: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    date_columns: list[str]
    boolean_columns: list[str]
    id_columns: list[str]
    text_columns: list[str]
    missing_columns: list[str]
    duplicate_rows: int
    file_name: str | None = None
    encoding: str | None = None


class DatasetProfile(BaseModel):
    metadata: DatasetMetadata
    memory_usage_mb: float
    missing_pct: float
    duplicate_pct: float
    quality_score: float = Field(description="0-100 heuristic: 0.5*completeness + 0.3*uniqueness + 0.2*validity")
    columns: list[ColumnProfile]
    semantic: SemanticResult
    warnings: list[str] = Field(default_factory=list)
