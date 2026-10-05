"""File validation, encoding detection and CSV/Excel -> DataFrame.

Uploaded files are never trusted: extension, size, magic bytes, zip expansion
size, row/column limits and encoding are all checked before/while parsing.
"""
import codecs
import csv
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import charset_normalizer
import pandas as pd

from app.config.settings import Settings, settings as default_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

XLSX_MAGIC = b"PK\x03\x04"


class DatasetLoadError(ValueError):
    """Raised for any user-facing problem with an uploaded file."""


@dataclass
class LoadedDataset:
    df: pd.DataFrame
    file_name: str
    file_type: str
    size_bytes: int
    encoding: str | None = None
    delimiter: str | None = None
    warnings: list[str] = field(default_factory=list)


def detect_encoding(content: bytes) -> str:
    if content.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if content.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    try:
        content.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass
    best = charset_normalizer.from_bytes(content[:200_000]).best()
    if best is not None and best.encoding:
        return best.encoding
    return "latin-1"


def _detect_delimiter(content: bytes, encoding: str) -> str:
    sample = content[:65_536].decode(encoding, errors="ignore")
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def validate_upload(file_name: str, content: bytes, cfg: Settings = default_settings) -> str:
    """Return the sanitised extension, or raise DatasetLoadError."""
    name = Path(file_name or "").name
    if not name:
        raise DatasetLoadError("File name is missing.")
    ext = Path(name).suffix.lower()
    if ext not in cfg.allowed_extensions:
        raise DatasetLoadError(
            f"Unsupported file type '{ext or 'none'}'. Allowed: {', '.join(cfg.allowed_extensions)}."
        )
    if not content:
        raise DatasetLoadError("The file is empty.")
    if len(content) > cfg.max_file_size_bytes:
        raise DatasetLoadError(f"File exceeds the {cfg.max_file_size_mb} MB limit.")

    if ext == ".xlsx":
        if not content.startswith(XLSX_MAGIC):
            raise DatasetLoadError("File content does not match the .xlsx format.")
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                total = sum(i.file_size for i in zf.infolist())
        except zipfile.BadZipFile as exc:
            raise DatasetLoadError("Corrupt .xlsx file.") from exc
        if total > cfg.max_uncompressed_bytes:
            raise DatasetLoadError("Excel file expands to an unsafe size.")
    else:  # .csv
        head = content[:8192]
        is_utf16 = head.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE))
        if b"\x00" in head and not is_utf16:
            raise DatasetLoadError("File looks binary, not a text CSV.")
    return ext


def _clean_columns(df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    """Strip whitespace, name blank headers, make names unique."""
    seen: dict[str, int] = {}
    new_cols: list[str] = []
    for i, col in enumerate(df.columns):
        name = str(col).strip()
        if not name or name.lower() == "nan":
            name = f"unnamed_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        new_cols.append(name)
    if new_cols != [str(c) for c in df.columns]:
        warnings.append("Column names were trimmed/renamed to be non-blank and unique.")
    df = df.copy()
    df.columns = new_cols
    return df


def load_dataset(content: bytes, file_name: str, cfg: Settings = default_settings) -> LoadedDataset:
    """Validate and parse an uploaded file held in memory (API-friendly)."""
    ext = validate_upload(file_name, content, cfg)
    warnings: list[str] = []
    encoding = delimiter = None

    try:
        if ext == ".csv":
            encoding = detect_encoding(content)
            delimiter = _detect_delimiter(content, encoding)
            read = lambda enc: pd.read_csv(  # noqa: E731
                io.BytesIO(content), encoding=enc, sep=delimiter,
                low_memory=False, nrows=cfg.max_rows + 1,
            )
            try:
                df = read(encoding)
            except UnicodeDecodeError:
                warnings.append(f"Decoding with '{encoding}' failed; fell back to latin-1.")
                encoding = "latin-1"
                df = read(encoding)
        else:
            xl = pd.ExcelFile(io.BytesIO(content), engine="openpyxl")
            if len(xl.sheet_names) > 1:
                warnings.append(
                    f"Workbook has {len(xl.sheet_names)} sheets; only '{xl.sheet_names[0]}' was loaded."
                )
            df = xl.parse(sheet_name=0)
    except pd.errors.EmptyDataError as exc:
        raise DatasetLoadError("The file contains no data.") from exc
    except (pd.errors.ParserError, ValueError, zipfile.BadZipFile, KeyError) as exc:
        if isinstance(exc, DatasetLoadError):
            raise
        raise DatasetLoadError(f"Could not parse file: {exc}") from exc

    if df.shape[0] == 0:
        raise DatasetLoadError("The file has headers but no data rows.")
    if df.shape[0] > cfg.max_rows:
        raise DatasetLoadError(f"Dataset exceeds the {cfg.max_rows:,} row limit.")
    if df.shape[1] > cfg.max_columns:
        raise DatasetLoadError(f"Dataset exceeds the {cfg.max_columns} column limit.")

    df = _clean_columns(df, warnings)
    logger.info("Loaded %s: %d rows x %d cols", file_name, *df.shape)
    return LoadedDataset(
        df=df, file_name=Path(file_name).name, file_type=ext.lstrip("."),
        size_bytes=len(content), encoding=encoding, delimiter=delimiter, warnings=warnings,
    )


def load_dataset_from_path(path: str | Path, cfg: Settings = default_settings) -> LoadedDataset:
    p = Path(path)
    if not p.is_file():
        raise DatasetLoadError(f"File not found: {p}")
    if p.stat().st_size > cfg.max_file_size_bytes:
        raise DatasetLoadError(f"File exceeds the {cfg.max_file_size_mb} MB limit.")
    return load_dataset(p.read_bytes(), p.name, cfg)
