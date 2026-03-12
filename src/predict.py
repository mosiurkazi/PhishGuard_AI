import joblib
import pandas as pd
from src.feature_engineering import extract_url_features
from src.preprocess import clean_text

URL_MODEL_PATH = "models/url_model.pkl"
TEXT_MODEL_PATH = "models/text_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

url_model = joblib.load(URL_MODEL_PATH)
text_model = joblib.load(TEXT_MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def explain_url(url: str) -> list:
    features = extract_url_features(url)
    reasons = []

    if features["has_ip"]:
        reasons.append("URL contains an IP address")
    if features["https"] == 0:
        reasons.append("URL does not use HTTPS")
    if features["suspicious_word_count"] > 0:
        reasons.append("URL contains suspicious keywords")
    if features["subdomain_count"] > 2:
        reasons.append("URL has many subdomains")
    if features["url_length"] > 75:
        reasons.append("URL is unusually long")
    if not reasons:
        reasons.append("No strong suspicious indicators found")

    return reasons


def predict_url(url: str):
    features = extract_url_features(url)
    X = pd.DataFrame([features])
    pred = url_model.predict(X)[0]
    prob = url_model.predict_proba(X)[0][pred]

    return {
        "label": "Phishing" if pred == 1 else "Legitimate",
        "confidence": round(float(prob) * 100, 2),
        "reasons": explain_url(url),
    }


def explain_text(text: str) -> list:
    text_lower = text.lower()
    triggers = []
    keywords = ["verify", "urgent", "click", "password", "bank", "account", "suspended"]

    for word in keywords:
        if word in text_lower:
            triggers.append(f"Contains suspicious term: {word}")

    if not triggers:
        triggers.append("Prediction based mainly on learned text patterns")

    return triggers


def predict_text(text: str):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    pred = text_model.predict(X)[0]
    prob = text_model.predict_proba(X)[0][pred]

    return {
        "label": "Phishing" if pred == 1 else "Legitimate",
        "confidence": round(float(prob) * 100, 2),
        "reasons": explain_text(text),
    }