import pandas as pd
import pytest
from pathlib import Path

from src.data  import make_dataset


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def sample_chunk():
    return pd.DataFrame({
        "an": [10, 2012, 99],
        "mois": [1, 5, 12],
        "jour": [10, 20, 15],
        "hrmn": ["1230", "0815", "2359"],
        "an_nais": [1990, 1980, 1970],
        "grav": [1, 2, 3],
        "lat": [45.0, 46.0, 47.0],
        "long": [5.0, 6.0, 7.0]
    })


# -----------------------------
# Unit tests
# -----------------------------
def test_fix_year():
    assert make_dataset.fix_year(10) == 2010
    assert make_dataset.fix_year(99) == 2099
    assert make_dataset.fix_year(2015) == 2015


def test_process_chunk_basic(sample_chunk):
    df = make_dataset.process_chunk(sample_chunk)

    # Check filtering years (2010–2016)
    assert df["an"].min() >= 2010
    assert df["an"].max() <= 2016

    # Check hour extraction
    assert "hour" in df.columns
    assert df["hour"].iloc[0] == 12

    # Check victim age
    assert "victim_age" in df.columns
    assert all(df["victim_age"].between(0, 100))


def test_process_chunk_columns(sample_chunk):
    df = make_dataset.process_chunk(sample_chunk)

    # Ensure only expected columns are returned
    for col in df.columns:
        assert col in [
            "an","mois","jour","hour","lum","int","atm","col","catr","circ",
            "nbv","vosp","surf","infra","situ","lat","long","place","catu",
            "sexe","locp","actp","etatp","catv","victim_age","grav"
        ]


# -----------------------------
# Integration test (mock I/O)
# -----------------------------
def test_main(tmp_path, monkeypatch):
    # Create fake CSV
    fake_data = pd.DataFrame({
        "an": [2012, 2015, 2016],
        "mois": [1, 2, 3],
        "jour": [10, 11, 12],
        "hrmn": ["1200", "1300", "1400"],
        "an_nais": [1990, 1985, 1980],
        "grav": [1, 2, 3],
        "lat": [45, 46, 47],
        "long": [5, 6, 7]
    })

    fake_csv = tmp_path / "accidents_full.csv"
    fake_data.to_csv(fake_csv, index=False)

    # Redirect paths
    monkeypatch.setattr(make_dataset, "DATA_PATH", str(fake_csv))
    monkeypatch.setattr(make_dataset, "OUTPUT_DIR", tmp_path)

    # Run main
    make_dataset.main()

    # Check outputs exist
    assert (tmp_path / "X_train.csv").exists()
    assert (tmp_path / "X_test.csv").exists()
    assert (tmp_path / "y_train.csv").exists()
    assert (tmp_path / "y_test.csv").exists()

    # Optional: check content
    X_train = pd.read_csv(tmp_path / "X_train.csv")
    assert not X_train.empty