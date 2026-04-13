import pandas as pd

from src.monitoring.evidently_report import generate_data_drift_report


def main() -> None:
    reference_data = pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4, 5],
            "feature_2": [10, 11, 12, 13, 14],
        }
    )

    current_data = pd.DataFrame(
        {
            "feature_1": [2, 3, 4, 5, 100],
            "feature_2": [10, 10, 11, 12, 13],
        }
    )

    output_path = generate_data_drift_report(
        reference_data=reference_data,
        current_data=current_data,
        report_name="test_data_drift_report.html",
    )

    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()