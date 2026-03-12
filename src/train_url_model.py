import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from src.feature_engineering import extract_url_features


def main():
    df = pd.read_csv("data/urls.csv")

    feature_rows = df["url"].apply(extract_url_features)
    X = pd.DataFrame(feature_rows.tolist())
    y = df["label"].map({"legitimate": 0, "phishing": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("URL Model Results")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/url_model.pkl")
    print("Saved URL model to models/url_model.pkl")


if __name__ == "__main__":
    main()