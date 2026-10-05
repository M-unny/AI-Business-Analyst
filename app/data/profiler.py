"""Dataset profiler: dataset-level stats, per-column stats, type inference,
date-format detection and semantic roles. Pure function of a DataFrame; never
mutates its input (cleaning is Milestone 2)."""
import re
import warnings as _warnings

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from app.config.settings import settings
from app.data.schemas import ColumnProfile, ColumnType, DatasetMetadata, DatasetProfile, DateFormat
from app.data.semantic import detect_semantic_roles
from app.utils.logging import get_logger
from app.utils.text import tokenize_name

logger = get_logger(__name__)

ID_TOKENS = {"id", "uuid", "guid", "key", "index", "idx", "code", "sku"}
ID_SUFFIXES = ("id", "no", "num", "code", "key")
ENTITY_WORDS = {
    "customer", "cust", "order", "product", "prod", "user", "client", "transaction", "txn",
    "invoice", "employee", "emp", "item", "account", "campaign", "store", "session", "stock",
}
BOOLEAN_STRINGS = [{"true", "false"}, {"yes", "no"}, {"y", "n"}, {"t", "f"}, {"0", "1"}]

_NUM_DATE_RE = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})$")
_ISO_DATE_RE = re.compile(r"^\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}")


# ------------------------------------------------------------------ helpers
def _f(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(v) or np.isinf(v)) else round(v, 6)


def _pct(part: int | float, whole: int | float) -> float:
    return round(100.0 * part / whole, 2) if whole else 0.0


def _looks_like_id_name(name: str) -> bool:
    tokens = tokenize_name(name)
    if not tokens:
        return False
    if tokens[-1] in ID_TOKENS or any(t in {"id", "uuid", "guid"} for t in tokens):
        return True
    if tokens[-1] in {"no", "num", "number"} and any(t in ENTITY_WORDS for t in tokens[:-1]):
        return True
    return any(
        t.endswith(suf) and t[: -len(suf)] in ENTITY_WORDS for t in tokens for suf in ID_SUFFIXES
    )


def _is_boolean_like(nn: pd.Series) -> bool:
    if pdt.is_bool_dtype(nn):
        return True
    if pdt.is_numeric_dtype(nn):
        u = nn.unique()
        return len(u) == 2 and set(u.tolist()) <= {0, 1}
    uniq = set(nn.astype(str).str.strip().str.lower().unique().tolist())
    return len(uniq) == 2 and any(uniq == pair for pair in BOOLEAN_STRINGS)


def _clean_numeric(nn: pd.Series) -> pd.Series:
    num = pd.to_numeric(nn, errors="coerce").astype("float64")
    return num.replace([np.inf, -np.inf], np.nan).dropna()


def _is_integer_like(num: pd.Series) -> bool:
    return bool(len(num) and (num % 1 == 0).all())


def _coerce_numeric_text(strs: pd.Series) -> pd.Series:
    cleaned = (
        strs.str.replace(r"[,$€£₹%\s]", "", regex=True)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _try_numeric_text(strs: pd.Series) -> pd.Series | None:
    """Numbers stored as text ('$1,200.50'). Skips leading-zero identifiers like '00123'."""
    sample = strs.sample(min(len(strs), 5000), random_state=0)
    if sample.str.match(r"^0\d").any():
        return None
    if _coerce_numeric_text(sample).notna().mean() < settings.numeric_text_threshold:
        return None
    return _coerce_numeric_text(strs).replace([np.inf, -np.inf], np.nan).dropna()


def _detect_date_format(sample: pd.Series) -> DateFormat:
    """Classify numeric day/month ordering. 01/02/2024 with no value > 12 in
    either slot is genuinely ambiguous and is reported rather than guessed."""
    parts = sample.str.extract(_NUM_DATE_RE).astype(float)
    if parts[0].notna().mean() >= 0.8:
        day_first = bool((parts[0] > 12).any())
        month_first = bool((parts[1] > 12).any())
        if day_first and month_first:
            return "mixed"
        if day_first:
            return "dayfirst"
        if month_first:
            return "monthfirst"
        return "ambiguous"
    if sample.str.match(_ISO_DATE_RE).mean() >= 0.8:
        return "iso"
    return "other"


def _try_dates(strs: pd.Series) -> tuple[pd.Series, DateFormat] | None:
    """Detect date strings by parse success on a sample, not by column name."""
    sample = strs.sample(min(len(strs), settings.date_sample_size), random_state=0)
    if sample.str.fullmatch(r"\d+").mean() > 0.5:          # '2019', '12345' -> not dates
        return None
    if not sample.str.contains(r"[-/.: ]|[A-Za-z]", regex=True).all():
        return None
    fmt = _detect_date_format(sample)
    dayfirst = fmt == "dayfirst"
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        parsed_sample = pd.to_datetime(sample, errors="coerce", dayfirst=dayfirst, format="mixed")
    if parsed_sample.notna().mean() < settings.date_parse_threshold:
        return None
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        full = pd.to_datetime(strs, errors="coerce", dayfirst=dayfirst)
        if full.notna().mean() < settings.date_parse_threshold:
            full = pd.to_datetime(strs, errors="coerce", dayfirst=dayfirst, format="mixed")
    return full.dropna(), fmt


def _classify_strings(strs: pd.Series, unique: int) -> ColumnType:
    n = len(strs)
    ratio = unique / n
    avg_len = float(strs.str.len().mean())
    space_ratio = float(strs.str.contains(" ").mean())
    if (ratio >= settings.id_unique_ratio and n >= settings.id_min_rows
            and avg_len <= settings.id_max_avg_len and space_ratio < 0.2):
        return "id"
    if avg_len > settings.text_min_avg_len:
        return "text"
    if ratio <= settings.categorical_max_unique_ratio or unique <= settings.categorical_max_unique:
        return "categorical"
    return "text"


# ------------------------------------------------------------ column profile
def profile_column(s: pd.Series, n_rows: int) -> ColumnProfile:
    name = str(s.name)
    nn = s.dropna()
    missing = int(len(s) - len(nn))
    unique = int(nn.nunique())
    base = dict(
        name=name, dtype=str(s.dtype), missing_count=missing,
        missing_pct=_pct(missing, n_rows), unique_count=unique,
        cardinality_ratio=round(unique / len(nn), 4) if len(nn) else 0.0,
    )
    if len(nn) == 0:
        return ColumnProfile(**base, inferred_type="empty", notes=["column is entirely empty"])

    notes: list[str] = []
    numeric: pd.Series | None = None
    dates: pd.Series | None = None
    stored_as_text = False
    date_format: DateFormat | None = None
    inferred: ColumnType

    if _is_boolean_like(nn):
        inferred = "boolean"
    elif pdt.is_datetime64_any_dtype(s):
        inferred, dates = "datetime", nn
    elif pdt.is_numeric_dtype(s):
        numeric = _clean_numeric(nn)
        inferred = "id" if (_looks_like_id_name(name) and unique > 1 and _is_integer_like(numeric)) \
            else "numeric"
    else:
        strs = nn.astype(str).str.strip()
        if _looks_like_id_name(name) and unique > 1:
            inferred = "id"
        elif (num := _try_numeric_text(strs)) is not None:
            inferred, numeric, stored_as_text = "numeric", num, True
            notes.append("numbers stored as text; convert before modelling")
        elif (res := _try_dates(strs)) is not None:
            inferred, (dates, date_format) = "datetime", res
            if date_format == "ambiguous":
                notes.append("day/month order is ambiguous (all values <= 12); assumed month-first")
            elif date_format == "mixed":
                notes.append("both day-first and month-first values present; parsing is unreliable")
        else:
            inferred = _classify_strings(strs, unique)

    extra: dict = {}
    if numeric is not None and len(numeric):
        q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
        iqr = q3 - q1
        outliers = int(((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
        extra.update(
            min_value=_f(numeric.min()), max_value=_f(numeric.max()), mean=_f(numeric.mean()),
            median=_f(numeric.median()), std=_f(numeric.std()),
            integer_like=_is_integer_like(numeric), outlier_count=outliers,
        )
    if dates is not None and len(dates):
        extra.update(min_value=dates.min().isoformat(), max_value=dates.max().isoformat())
    if inferred in ("categorical", "boolean"):
        vc = nn.astype(str).value_counts().head(5)
        extra["top_values"] = {str(k): int(v) for k, v in vc.items()}
    if unique == 1:
        notes.append("constant column (single unique value)")

    return ColumnProfile(
        **base, inferred_type=inferred, stored_as_text=stored_as_text,
        date_format=date_format, notes=notes, **extra,
    )


# ------------------------------------------------------------ dataset profile
def profile_dataset(
    df: pd.DataFrame, *, file_name: str | None = None, encoding: str | None = None,
    extra_warnings: list[str] | None = None,
) -> DatasetProfile:
    n_rows, n_cols = df.shape
    columns = [profile_column(df.iloc[:, i], n_rows) for i in range(n_cols)]

    def names(*types: str) -> list[str]:
        return [c.name for c in columns if c.inferred_type in types]

    dup_rows = int(df.duplicated().sum())
    total_cells = n_rows * n_cols
    missing_cells = sum(c.missing_count for c in columns)
    missing_pct = _pct(missing_cells, total_cells)
    dup_pct = _pct(dup_rows, n_rows)

    metadata = DatasetMetadata(
        rows=n_rows, columns=n_cols,
        numeric_columns=names("numeric"), categorical_columns=names("categorical"),
        date_columns=names("datetime"), boolean_columns=names("boolean"),
        id_columns=names("id"), text_columns=names("text"),
        missing_columns=[c.name for c in columns if c.missing_count > 0],
        duplicate_rows=dup_rows, file_name=file_name, encoding=encoding,
    )

    warns = list(extra_warnings or [])
    if dup_rows:
        warns.append(f"{dup_rows} duplicate rows ({dup_pct}%).")
    for c in columns:
        if c.missing_pct >= settings.high_missing_pct and c.inferred_type != "empty":
            warns.append(f"Column '{c.name}' is {c.missing_pct}% missing.")
        if c.inferred_type == "empty":
            warns.append(f"Column '{c.name}' is entirely empty.")
        if c.date_format in ("ambiguous", "mixed"):
            warns.append(f"Column '{c.name}': date format is {c.date_format}; verify day/month order.")
        if c.stored_as_text:
            warns.append(f"Column '{c.name}' holds numbers stored as text.")

    degenerate = sum(1 for c in columns if c.inferred_type == "empty" or c.unique_count <= 1)
    completeness, uniqueness = 100 - missing_pct, 100 - dup_pct
    validity = 100 * (1 - degenerate / n_cols) if n_cols else 0.0
    quality = round(0.5 * completeness + 0.3 * uniqueness + 0.2 * validity, 1)

    return DatasetProfile(
        metadata=metadata,
        memory_usage_mb=round(float(df.memory_usage(deep=True).sum()) / 1024**2, 3),
        missing_pct=missing_pct, duplicate_pct=dup_pct, quality_score=quality,
        columns=columns, semantic=detect_semantic_roles(columns), warnings=warns,
    )
