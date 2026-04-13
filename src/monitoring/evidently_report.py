from pathlib import Path


def get_reports_dir() -> Path:
    """Return the directory where Evidently reports will be stored."""
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir