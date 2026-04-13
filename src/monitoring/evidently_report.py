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
    Generate an Evidently data drift HTML report.

    Args:
        reference_data: Baseline dataset used as the reference.
        current_data: Dataset to compare against the reference.
        report_name: Output HTML report filename.

    Returns:
        Path to the generated HTML report.
    """
    report = Report([DataDriftPreset()])
    my_eval = report.run(reference_data, current_data)

    output_path = get_reports_dir() / report_name
    my_eval.save_html(str(output_path))

    return output_path