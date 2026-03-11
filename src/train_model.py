import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

print("Loading datasets...")

usagers = pd.read_csv("data/usagers.csv")

# Target variable: severity
y = usagers["grav"]

# Simple features for first model
X = usagers[["sexe", "catu"]]

X = X.fillna(0)

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

model = RandomForestClassifier()
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/accident_model.pkl")

print("Model trained and saved.")
