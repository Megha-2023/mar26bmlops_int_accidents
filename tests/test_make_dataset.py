import pandas as pd
import pytest

from src.data.make_dataset import (
    add_victim_age,
    clean_time_column,
    filter_years,
    fix_year_column,
    select_features,
    split_train_test,
)


# -----------------------------
# Year handling
# -----------------------------
def test_fix_year_column_converts_two_digit_years():
    df = pd.DataFrame({"an": [16, 2015, 99]})

    result = fix_year_column(df)

    assert result["an"].tolist() == [2016, 2015, 2099]


def test_filter_years_keeps_only_2010_to_2016():
    df = pd.DataFrame({"an": [2009, 2010, 2013, 2016, 2017]})

    result = filter_years(df)

    assert result["an"].tolist() == [2010, 2013, 2016]


# -----------------------------
# Time cleaning
# -----------------------------
def test_clean_time_column_creates_hour_from_hrmn():
    df = pd.DataFrame({"hrmn": [1330, "0815", 45]})

    result = clean_time_column(df)

    assert result["hour"].tolist() == [13, 8, 0]


def test_clean_time_column_sets_invalid_values_to_zero():
    df = pd.DataFrame({"hrmn": ["bad_value", None]})

    result = clean_time_column(df)

    assert result["hour"].tolist() == [0, 0]


# -----------------------------
# Victim age engineering
# -----------------------------
def test_add_victim_age_creates_age_column():
    df = pd.DataFrame({"an": [2016, 2014], "an_nais": [1980, 2000]})

    result = add_victim_age(df)

    assert result["victim_age"].tolist() == [36, 14]


def test_add_victim_age_filters_unrealistic_ages():
    df = pd.DataFrame(
        {
            "an": [2016, 2016, 2016],
            "an_nais": [1980, 1890, 2020],  # ages: 36, 126, -4
        }
    )

    result = add_victim_age(df)

    assert result["victim_age"].tolist() == [36]


# -----------------------------
# Feature selection
# -----------------------------
def test_select_features_keeps_only_expected_columns():
    df = pd.DataFrame(
        {
            "an": [2016],
            "mois": [5],
            "jour": [12],
            "hour": [14],
            "grav": [3],
            "extra_col": ["drop me"],
        }
    )

    result = select_features(df)

    assert "extra_col" not in result.columns
    assert result.columns.tolist() == ["an", "mois", "jour", "hour", "grav"]


# -----------------------------
# Train/test split
# -----------------------------
def test_split_train_test_splits_by_year_and_transforms_target():
    df = pd.DataFrame(
        {
            "an": [2014, 2015, 2016],
            "mois": [1, 2, 3],
            "jour": [10, 11, 12],
            "hour": [8, 9, 10],
            "grav": [1, 2, 4],
        }
    )

    X_train, X_test, y_train, y_test = split_train_test(df)

    assert len(X_train) == 2
    assert len(X_test) == 1
    assert "an" not in X_train.columns
    assert "an" not in X_test.columns
    assert y_train.tolist() == [0, 1]
    assert y_test.tolist() == [3]


def test_split_train_test_raises_if_train_set_is_empty():
    df = pd.DataFrame(
        {
            "an": [2016],
            "mois": [1],
            "jour": [10],
            "hour": [8],
            "grav": [2],
        }
    )

    with pytest.raises(ValueError, match="X_train is empty"):
        split_train_test(df)