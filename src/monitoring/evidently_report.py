import os
from pathlib import Path

import pandas as pd

try:
    # Evidently >=0.7
    from evidently import Report
except ImportError:
    # Evidently 0.4.x fallback
    from evidently.report import Report

try:
    # Evidently >=0.7
    from evidently.presets import DataDriftPreset
except ImportError:
    # Evidently 0.4.x fallback
    from evidently.metric_preset import DataDriftPreset


DEFAULT_REPORTS_DIR = os.getenv("MONITORING_REPORTS_DIR", "metrics/reports")


def get_reports_dir(reports_dir: str = DEFAULT_REPORTS_DIR) -> Path:
    """Return the directory where monitoring reports will be stored."""
    output_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def generate_data_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    report_name: str = "data_drift_report.html",
    reports_dir: str = DEFAULT_REPORTS_DIR,
) -> Path:
    """
    Generate an Evidently data drift HTML report for all columns.
    """
    report = Report([DataDriftPreset()])
    report.run(
        reference_data=reference_data,
        current_data=current_data,
    )

    output_path = get_reports_dir(reports_dir) / report_name
    report.save_html(str(output_path))

    return output_path


def generate_prediction_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    prediction_column: str = "prediction",
    report_name: str = "prediction_drift_report.html",
    reports_dir: str = DEFAULT_REPORTS_DIR,
) -> Path:
    """
    Generate an Evidently prediction drift HTML report.
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
    report.run(
        reference_data=reference_data,
        current_data=current_data,
    )

    output_path = get_reports_dir(reports_dir) / report_name
    report.save_html(str(output_path))

    return output_path
