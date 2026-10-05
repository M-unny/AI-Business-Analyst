import dataclasses
import io

import pandas as pd
import pytest

from app.config.settings import settings
from app.data.loader import DatasetLoadError, detect_encoding, load_dataset


def test_load_basic_csv():
    r = load_dataset(b"a,b\n1,2\n3,4\n", "x.csv")
    assert r.df.shape == (2, 2)
    assert r.file_type == "csv" and r.encoding == "utf-8"


def test_semicolon_delimiter():
    r = load_dataset(b"a;b;c\n1;2;3\n4;5;6\n", "x.csv")
    assert list(r.df.columns) == ["a", "b", "c"]
    assert r.delimiter == ";"


def test_utf8_bom_is_stripped_from_header():
    r = load_dataset(b"\xef\xbb\xbfname,v\nA,1\n", "x.csv")
    assert list(r.df.columns) == ["name", "v"]


def test_latin1_file_is_decoded():
    text = "name,city\n" + "\n".join(f"José {i},Zürich" for i in range(60))
    r = load_dataset(text.encode("latin-1"), "x.csv")
    assert r.df["name"].iloc[0] == "José 0"
    assert r.df["city"].iloc[0] == "Zürich"


def test_detect_encoding_utf8():
    assert detect_encoding("héllo".encode("utf-8")) == "utf-8"


@pytest.mark.parametrize("name", ["x.txt", "x.exe", "noext", "x.csv.exe"])
def test_rejects_bad_extension(name):
    with pytest.raises(DatasetLoadError, match="Unsupported"):
        load_dataset(b"a,b\n1,2\n", name)


def test_rejects_empty_file():
    with pytest.raises(DatasetLoadError, match="empty"):
        load_dataset(b"", "x.csv")


def test_rejects_oversized_file():
    cfg = dataclasses.replace(settings, max_file_size_mb=0)
    with pytest.raises(DatasetLoadError, match="limit"):
        load_dataset(b"a,b\n1,2\n", "x.csv", cfg)


def test_rejects_binary_masquerading_as_csv():
    with pytest.raises(DatasetLoadError, match="binary"):
        load_dataset(b"\x00\x01\x02\x03" * 100, "x.csv")


def test_rejects_header_only_csv():
    with pytest.raises(DatasetLoadError, match="no data rows"):
        load_dataset(b"a,b,c\n", "x.csv")


def test_row_limit_enforced():
    cfg = dataclasses.replace(settings, max_rows=3)
    with pytest.raises(DatasetLoadError, match="row limit"):
        load_dataset(b"a\n1\n2\n3\n4\n5\n", "x.csv", cfg)


def test_column_limit_enforced():
    cfg = dataclasses.replace(settings, max_columns=2)
    with pytest.raises(DatasetLoadError, match="column limit"):
        load_dataset(b"a,b,c\n1,2,3\n", "x.csv", cfg)


def test_xlsx_roundtrip():
    buf = io.BytesIO()
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_excel(buf, index=False)
    r = load_dataset(buf.getvalue(), "book.xlsx")
    assert r.df.shape == (2, 2) and r.file_type == "xlsx"


def test_fake_xlsx_rejected():
    with pytest.raises(DatasetLoadError, match="does not match"):
        load_dataset(b"a,b\n1,2\n", "x.xlsx")


def test_blank_and_padded_headers_are_fixed():
    r = load_dataset(b" a ,,a\n1,2,3\n", "x.csv")
    assert len(set(r.df.columns)) == 3
    assert "a" in r.df.columns
    assert any("renamed" in w for w in r.warnings)


def test_path_is_stripped_from_file_name():
    r = load_dataset(b"a\n1\n", "../../etc/data.csv")
    assert r.file_name == "data.csv"
