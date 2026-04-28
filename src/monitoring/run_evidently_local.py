import pandas as pd

from data_loader import load_current_data, load_reference_data
from model_loader import generate_predictions
from evidently_report import (
    DEFAULT_REPORTS_DIR,
    generate_data_drift_report,
    generate_prediction_drift_report,
)
from src.model_registry import load_local_model


def attach_prediction_column(model, data: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the input data with a prediction column added.
    """
    data_with_predictions = data.copy()
    data_with_predictions["prediction"] = generate_predictions(model, data)
    return data_with_predictions


def main() -> None:
    reference_data = load_reference_data()
    current_data = load_current_data()
    model = load_local_model()

    data_drift_report_path = generate_data_drift_report(
        reference_data=reference_data,
        current_data=current_data,
        report_name="xtrain_vs_xtest_drift_report.html",
        reports_dir=DEFAULT_REPORTS_DIR,
    )

    reference_with_predictions = attach_prediction_column(model, reference_data)
    current_with_predictions = attach_prediction_column(model, current_data)

    prediction_drift_report_path = generate_prediction_drift_report(
        reference_data=reference_with_predictions,
        current_data=current_with_predictions,
        prediction_column="prediction",
        report_name="xtrain_vs_xtest_prediction_drift_report.html",
        reports_dir=DEFAULT_REPORTS_DIR,
    )

    print(f"Data drift report saved to: {data_drift_report_path}")
    print(f"Prediction drift report saved to: {prediction_drift_report_path}")


if __name__ == "__main__":
    main()
