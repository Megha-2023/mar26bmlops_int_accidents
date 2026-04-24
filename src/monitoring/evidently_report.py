from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


def get_reports_dir() -> Path:
    """Return the directory where monitoring reports will be stored."""
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def generate_data_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    report_name: str = "data_drift_report.html",
) -> Path:
    """
    Generate an Evidently data drift HTML report for all columns.

    Args:
        reference_data: Baseline dataset used as the reference.
        current_data: Dataset to compare against the reference.
        report_name: Output HTML report filename.

    Returns:
        Path to the generated HTML report.
    """
    report = Report([DataDriftPreset()])
    evaluation = report.run(reference_data, current_data)

    output_path = get_reports_dir() / report_name
    evaluation.save_html(str(output_path))

    return output_path


def generate_prediction_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    prediction_column: str = "prediction",
    report_name: str = "prediction_drift_report.html",
) -> Path:
    """
    Generate an Evidently prediction drift HTML report.

    This compares only the prediction column between the reference and current
    datasets.

    Args:
        reference_data: Reference dataset containing the prediction column.
        current_data: Current dataset containing the prediction column.
        prediction_column: Name of the prediction column to compare.
        report_name: Output HTML report filename.

    Returns:
        Path to the generated HTML report.
    """
    if prediction_column not in reference_data.columns:
        raise ValueError(
            f"Reference data must contain '{prediction_column}' column."
        )

    if prediction_column not in current_data.columns:
        raise ValueError(
            f"Current data must contain '{prediction_column}' column."
        )

    report = Report([DataDriftPreset(columns=[prediction_column])])
    evaluation = report.run(reference_data, current_data)

    output_path = get_reports_dir() / report_name
    evaluation.save_html(str(output_path))

    return output_path