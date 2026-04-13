from src.monitoring.data_loader import load_current_data, load_reference_data
from src.monitoring.evidently_report import generate_data_drift_report


def main() -> None:
    reference_data = load_reference_data()
    current_data = load_current_data()

    output_path = generate_data_drift_report(
        reference_data=reference_data,
        current_data=current_data,
        report_name="xtrain_vs_xtest_drift_report.html",
    )

    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()