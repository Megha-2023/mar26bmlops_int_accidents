import pandas as pd
from unittest.mock import Mock

from src.models import evaluate_model


def test_main_prints_evaluation_results_and_uses_expected_paths(monkeypatch, capsys):
    """
    Test that evaluate_model.main():
    - reads the expected test datasets
    - loads the expected trained model file
    - calls model.predict() once with X_test
    - prints the expected evaluation output

    This is a unit test, so we replace all file/model dependencies with fakes.
    """

    # -------------------------------------------------------------------------
    # Arrange: create fake input data that will stand in for the CSV files
    # -------------------------------------------------------------------------
    x_test_df = pd.DataFrame({"feature1": [1, 2, 3]})
    y_test_series = pd.Series([0, 1, 1], name="target")

    # We will record every path passed to pd.read_csv so we can assert on it later.
    read_csv_calls = []

    def fake_read_csv(path):
        """
        Fake replacement for pandas.read_csv.

        Depending on the path requested by the code under test, return the
        corresponding fake dataset. Also store the path so we can verify that
        the correct files were requested.
        """
        read_csv_calls.append(path)

        if path == "data/preprocessed/X_test.csv":
            return x_test_df

        if path == "data/preprocessed/y_test.csv":
            # Return a one-column DataFrame because the production code
            # later uses .squeeze() to convert it into a Series.
            return y_test_series.to_frame()

        raise AssertionError(f"Unexpected path passed to read_csv: {path}")

    # -------------------------------------------------------------------------
    # Arrange: create a fake model object
    # -------------------------------------------------------------------------
    fake_model = Mock()
    fake_model.predict.return_value = [0, 1, 0]

    # We also want to verify which path was used when loading the model,
    # so we store it in a small list.
    load_calls = []

    def fake_joblib_load(path):
        """
        Fake replacement for joblib.load.

        Records the requested model path and returns a fake model object.
        """
        load_calls.append(path)
        return fake_model

    # -------------------------------------------------------------------------
    # Arrange: patch external dependencies used inside evaluate_model.py
    # -------------------------------------------------------------------------
    monkeypatch.setattr(evaluate_model.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(evaluate_model.joblib, "load", fake_joblib_load)

    # Patch metric functions so the test focuses only on this module's behavior,
    # not on sklearn internals.
    monkeypatch.setattr(evaluate_model, "accuracy_score", lambda y_true, y_pred: 0.6667)
    monkeypatch.setattr(evaluate_model, "f1_score", lambda y_true, y_pred, average: 0.6250)
    monkeypatch.setattr(
        evaluate_model,
        "classification_report",
        lambda y_true, y_pred: "fake classification report",
    )

    # -------------------------------------------------------------------------
    # Act: run the function under test
    # -------------------------------------------------------------------------
    evaluate_model.main()

    # Capture everything printed to the console
    captured = capsys.readouterr()

    # -------------------------------------------------------------------------
    # Assert: verify the expected files were read
    # -------------------------------------------------------------------------
    assert read_csv_calls == [
        "data/preprocessed/X_test.csv",
        "data/preprocessed/y_test.csv",
    ]

    # -------------------------------------------------------------------------
    # Assert: verify the expected model path was used
    # -------------------------------------------------------------------------
    assert load_calls == ["models/xgb_model.pkl"]

    # -------------------------------------------------------------------------
    # Assert: verify the model was asked to predict exactly once on X_test
    # -------------------------------------------------------------------------
    fake_model.predict.assert_called_once_with(x_test_df)

    # -------------------------------------------------------------------------
    # Assert: verify the printed output contains the expected information
    # -------------------------------------------------------------------------
    assert "Evaluation Results" in captured.out
    assert "Accuracy: 0.6667" in captured.out
    assert "F1 Score:" in captured.out
    assert "0.6250" in captured.out
    assert "fake classification report" in captured.out
    
def test_main_calls_metric_functions_with_expected_arguments(monkeypatch):
    """
    Test that evaluate_model.main() passes the correct values to:
    - accuracy_score
    - f1_score with average='weighted'
    - classification_report

    This test focuses on the arguments passed into the metric functions,
    not on printed output.
    """

    # -------------------------------------------------------------------------
    # Arrange: create fake input data
    # -------------------------------------------------------------------------
    x_test_df = pd.DataFrame({"feature1": [10, 20, 30]})

    # This is the true target data that the script should load from y_test.csv
    y_test_df = pd.DataFrame({"target": [0, 1, 1]})

    # This is the prediction output that our fake model will return
    y_pred = [0, 1, 0]

    def fake_read_csv(path):
        """
        Fake replacement for pandas.read_csv.

        Returns the fake feature matrix or target dataframe depending
        on which file path is requested.
        """
        if path == "data/preprocessed/X_test.csv":
            return x_test_df

        if path == "data/preprocessed/y_test.csv":
            return y_test_df

        raise AssertionError(f"Unexpected path passed to read_csv: {path}")

    # Create a fake model whose predict() method returns our fake predictions.
    fake_model = Mock()
    fake_model.predict.return_value = y_pred

    def fake_joblib_load(path):
        """
        Fake replacement for joblib.load.

        Returns the fake model and checks that the model path is correct.
        """
        assert path == "models/xgb_model.pkl"
        return fake_model

    # -------------------------------------------------------------------------
    # Arrange: containers to record the arguments passed to metric functions
    # -------------------------------------------------------------------------
    metric_calls = {}

    def fake_accuracy_score(y_true, predicted):
        """
        Fake accuracy_score that records the received arguments.
        """
        metric_calls["accuracy_score"] = {
            "y_true": y_true,
            "y_pred": predicted,
        }
        return 0.6667

    def fake_f1_score(y_true, predicted, average):
        """
        Fake f1_score that records the received arguments, including
        the averaging strategy used by the production code.
        """
        metric_calls["f1_score"] = {
            "y_true": y_true,
            "y_pred": predicted,
            "average": average,
        }
        return 0.6250

    def fake_classification_report(y_true, predicted):
        """
        Fake classification_report that records the received arguments.
        """
        metric_calls["classification_report"] = {
            "y_true": y_true,
            "y_pred": predicted,
        }
        return "fake classification report"

    # -------------------------------------------------------------------------
    # Arrange: patch dependencies inside the module under test
    # -------------------------------------------------------------------------
    monkeypatch.setattr(evaluate_model.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(evaluate_model.joblib, "load", fake_joblib_load)
    monkeypatch.setattr(evaluate_model, "accuracy_score", fake_accuracy_score)
    monkeypatch.setattr(evaluate_model, "f1_score", fake_f1_score)
    monkeypatch.setattr(evaluate_model, "classification_report", fake_classification_report)

    # -------------------------------------------------------------------------
    # Act: execute the function under test
    # -------------------------------------------------------------------------
    evaluate_model.main()

    # -------------------------------------------------------------------------
    # Assert: build the expected y_true after the production code applies
    # .squeeze() to the one-column target dataframe
    # -------------------------------------------------------------------------
    expected_y_true = y_test_df.squeeze()

    # Check accuracy_score arguments
    assert "accuracy_score" in metric_calls
    pd.testing.assert_series_equal(
        metric_calls["accuracy_score"]["y_true"],
        expected_y_true,
    )
    assert metric_calls["accuracy_score"]["y_pred"] == y_pred

    # Check f1_score arguments, especially average='weighted'
    assert "f1_score" in metric_calls
    pd.testing.assert_series_equal(
        metric_calls["f1_score"]["y_true"],
        expected_y_true,
    )
    assert metric_calls["f1_score"]["y_pred"] == y_pred
    assert metric_calls["f1_score"]["average"] == "weighted"

    # Check classification_report arguments
    assert "classification_report" in metric_calls
    pd.testing.assert_series_equal(
        metric_calls["classification_report"]["y_true"],
        expected_y_true,
    )
    assert metric_calls["classification_report"]["y_pred"] == y_pred

def test_main_accepts_single_column_target_dataframe_and_squeezes_it(monkeypatch):
    """
    Test that evaluate_model.main() correctly handles y_test loaded as a
    single-column DataFrame and converts it into a Series via .squeeze().

    This test is valuable because pandas.read_csv() usually returns a DataFrame,
    while sklearn metric functions typically expect a 1D target structure.
    The production code uses .squeeze() to bridge that difference.
    """

    # -------------------------------------------------------------------------
    # Arrange: fake feature matrix and one-column target DataFrame
    # -------------------------------------------------------------------------
    x_test_df = pd.DataFrame({"feature1": [5, 6, 7]})
    y_test_df = pd.DataFrame({"target": [1, 0, 1]})

    # Fake predictions from the mocked model
    y_pred = [1, 0, 0]

    def fake_read_csv(path):
        """
        Fake replacement for pandas.read_csv.

        Returns a DataFrame for both files, including y_test.csv, so that
        we can verify the production code squeezes the target correctly.
        """
        if path == "data/preprocessed/X_test.csv":
            return x_test_df

        if path == "data/preprocessed/y_test.csv":
            return y_test_df

        raise AssertionError(f"Unexpected path passed to read_csv: {path}")

    fake_model = Mock()
    fake_model.predict.return_value = y_pred

    # Store the exact y_true object received by one of the metric functions
    captured_y_true = {}

    def fake_accuracy_score(y_true, predicted):
        """
        Record the y_true object passed into accuracy_score so we can verify
        that it is a Series rather than a DataFrame after squeezing.
        """
        captured_y_true["value"] = y_true
        return 0.6667

    # -------------------------------------------------------------------------
    # Arrange: patch dependencies
    # -------------------------------------------------------------------------
    monkeypatch.setattr(evaluate_model.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(evaluate_model.joblib, "load", lambda path: fake_model)
    monkeypatch.setattr(evaluate_model, "accuracy_score", fake_accuracy_score)
    monkeypatch.setattr(evaluate_model, "f1_score", lambda y_true, y_pred, average: 0.6250)
    monkeypatch.setattr(
        evaluate_model,
        "classification_report",
        lambda y_true, y_pred: "fake classification report",
    )

    # -------------------------------------------------------------------------
    # Act
    # -------------------------------------------------------------------------
    evaluate_model.main()

    # -------------------------------------------------------------------------
    # Assert: verify that y_true was squeezed to a Series
    # -------------------------------------------------------------------------
    assert "value" in captured_y_true
    assert isinstance(captured_y_true["value"], pd.Series)

    # Compare the actual squeezed Series with what we expect from pandas
    pd.testing.assert_series_equal(
        captured_y_true["value"],
        y_test_df.squeeze(),
    )