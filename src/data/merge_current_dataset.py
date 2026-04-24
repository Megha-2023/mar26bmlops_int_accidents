from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "accidents_2016_2018.csv"
YEARS = [2016, 2017, 2018]


def load_year_tables(year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the four BAAC raw tables for a given year.
    """
    usagers_file = DATA_DIR / "usagers" / f"usagers-{year}.csv"
    vehicules_file = DATA_DIR / "vehicules" / f"vehicules-{year}.csv"
    lieux_file = DATA_DIR / "lieux" / f"lieux-{year}.csv"
    caracteristiques_file = DATA_DIR / "caracteristiques" / f"caracteristiques-{year}.csv"

    usagers = pd.read_csv(
        usagers_file,
        sep=",",
        encoding="latin1",
        engine="python",
        on_bad_lines="skip",
    )
    vehicules = pd.read_csv(
        vehicules_file,
        sep=",",
        encoding="latin1",
        engine="python",
        on_bad_lines="skip",
    )
    lieux = pd.read_csv(
        lieux_file,
        sep=",",
        encoding="latin1",
        engine="python",
        on_bad_lines="skip",
    )
    caracteristiques = pd.read_csv(
        caracteristiques_file,
        sep=",",
        encoding="latin1",
        engine="python",
        on_bad_lines="skip",
    )

    return usagers, vehicules, lieux, caracteristiques


def merge_one_year(year: int) -> pd.DataFrame:
    """
    Merge BAAC raw tables for one year into a single DataFrame.
    """
    usagers, vehicules, lieux, caracteristiques = load_year_tables(year)

    # Merge usagers with vehicules using accident + vehicle keys
    merged = usagers.merge(
        vehicules,
        on=["Num_Acc", "num_veh"],
        how="left",
    )

    # Merge with lieux using accident key
    merged = merged.merge(
        lieux,
        on="Num_Acc",
        how="left",
    )

    # Merge with caracteristiques using accident key
    merged = merged.merge(
        caracteristiques,
        on="Num_Acc",
        how="left",
    )

    # Keep one row per accident to mimic the previous project approach
    merged = merged.drop_duplicates(subset=["Num_Acc"]).copy()

    # Standardize accident id column name
    merged = merged.rename(columns={"Num_Acc": "num_acc"})

    # Add explicit source year column
    merged["source_year"] = year

    return merged


def main() -> None:
    all_years = []

    for year in YEARS:
        print(f"\nProcessing year {year}...")
        year_df = merge_one_year(year)
        print(f"Year {year} merged shape: {year_df.shape}")
        all_years.append(year_df)

    final_df = pd.concat(all_years, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\nDone.")
    print(f"Saved merged current dataset to: {OUTPUT_FILE}")
    print(f"Final shape: {final_df.shape}")
    print(f"Columns: {final_df.columns.tolist()}")


if __name__ == "__main__":
    main()