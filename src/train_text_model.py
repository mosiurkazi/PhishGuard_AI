import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from src.preprocess import clean_text


def main():
    df = pd.read_csv("data/emails.csv")
    df["clean_text"] = df["text"].astype(str).apply(clean_text)

    X = df["clean_text"]
    y = df["label"].map({"legitimate": 0, "phishing": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    print("Text Model Results")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/text_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    print("Saved text model and vectorizer in models/")


if __name__ == "__main__":
    main()