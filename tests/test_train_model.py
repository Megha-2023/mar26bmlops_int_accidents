import pandas as pd

from src.models import train_model


# ---------------------------------
# Training pipeline orchestration
# ---------------------------------
# This test verifies that the training script:
# - reads the expected input files
# - builds and trains a model
# - evaluates it
# - saves it to the expected output path
def test_train_model_main_runs_expected_workflow(monkeypatch):
    # Small fake datasets returned by pd.read_csv
    X_train = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
    X_test = pd.DataFrame({"feature1": [5, 6], "feature2": [6, 7]})
    y_train = pd.DataFrame({"grav": [0, 1]})
    y_test = pd.DataFrame({"grav": [1, 0]})

    read_calls = []

    def fake_read_csv(path, *args, **kwargs):
        read_calls.append(path)

        if path == "data/preprocessed/X_train.csv":
            return X_train
        if path == "data/preprocessed/X_test.csv":
            return X_test
        if path == "data/preprocessed/y_train.csv":
            return y_train
        if path == "data/preprocessed/y_test.csv":
            return y_test

        raise AssertionError(f"Unexpected read_csv path: {path}")

    monkeypatch.setattr(train_model.pd, "read_csv", fake_read_csv)

    # Fake model object to capture fit/predict calls
    class FakeModel:
        def __init__(self):
            self.fit_called_with = None
            self.predict_called_with = None

        def fit(self, X, y):
            self.fit_called_with = (X, y)

        def predict(self, X):
            self.predict_called_with = X
            return [1, 0]

    fake_model = FakeModel()
    model_init_kwargs = {}

    def fake_xgb_classifier(**kwargs):
        model_init_kwargs.update(kwargs)
        return fake_model

    monkeypatch.setattr(train_model, "XGBClassifier", fake_xgb_classifier)

    # Mock evaluation helpers
    accuracy_calls = []
    classification_calls = []

    def fake_accuracy_score(y_true, y_pred):
        accuracy_calls.append((list(y_true), list(y_pred)))
        return 1.0

    def fake_classification_report(y_true, y_pred):
        classification_calls.append((list(y_true), list(y_pred)))
        return "fake classification report"

    monkeypatch.setattr(train_model, "accuracy_score", fake_accuracy_score)
    monkeypatch.setattr(train_model, "classification_report", fake_classification_report)

    # Mock model saving
    dump_calls = []

    def fake_dump(model, path):
        dump_calls.append((model, path))

    monkeypatch.setattr(train_model.joblib, "dump", fake_dump)

    # Mock directory creation
    mkdir_calls = []

    def fake_mkdir(self, parents=False, exist_ok=False):
        mkdir_calls.append(
            {"path": self, "parents": parents, "exist_ok": exist_ok}
        )

    monkeypatch.setattr(train_model.Path, "mkdir", fake_mkdir)

    # Run the training script
    train_model.main()

    # Assert expected input files were read
    assert read_calls == [
        "data/preprocessed/X_train.csv",
        "data/preprocessed/X_test.csv",
        "data/preprocessed/y_train.csv",
        "data/preprocessed/y_test.csv",
    ]

    # Assert model was created with the expected configuration
    assert model_init_kwargs["n_estimators"] == 200
    assert model_init_kwargs["learning_rate"] == 0.05
    assert model_init_kwargs["max_depth"] == 6
    assert model_init_kwargs["subsample"] == 0.8
    assert model_init_kwargs["colsample_bytree"] == 0.8
    assert model_init_kwargs["eval_metric"] == "logloss"
    assert model_init_kwargs["random_state"] == 42

    # Assert fit and predict were called with the right data
    fit_X, fit_y = fake_model.fit_called_with
    assert fit_X.equals(X_train)
    assert fit_y.tolist() == [0, 1]

    assert fake_model.predict_called_with.equals(X_test)

    # Assert evaluation functions were called
    assert accuracy_calls == [([1, 0], [1, 0])]
    assert classification_calls == [([1, 0], [1, 0])]

    # Assert output directory creation and model saving happened
    assert len(mkdir_calls) == 1
    assert mkdir_calls[0]["parents"] is True
    assert mkdir_calls[0]["exist_ok"] is True

    assert len(dump_calls) == 1
    assert dump_calls[0][0] is fake_model
    assert str(dump_calls[0][1]).replace("\\", "/") == "models/xgb_model.pkl"