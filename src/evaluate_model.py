import pandas as pd
import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

print("Loading data...")

usagers = pd.read_csv("data/usagers.csv")

y = usagers["grav"]
X = usagers[["sexe", "catu"]]

X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Loading trained model...")

model = joblib.load("models/accident_model.pkl")

predictions = model.predict(X_test)

print("Evaluation results:")

print(classification_report(y_test, predictions))
