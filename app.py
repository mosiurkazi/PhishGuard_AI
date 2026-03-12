import streamlit as st
from src.predict import predict_url, predict_text

st.set_page_config(page_title="PhishGuard AI", page_icon="🛡️", layout="centered")

st.title("🛡️ PhishGuard AI")
st.write("Detect phishing URLs and suspicious email or message text using machine learning.")

option = st.radio("Choose input type", ["URL", "Email/Text"])

if option == "URL":
    url_input = st.text_input("Enter a URL")
    if st.button("Analyze URL"):
        if url_input.strip():
            result = predict_url(url_input.strip())
            st.subheader("Prediction")
            st.write(f"**Label:** {result['label']}")
            st.write(f"**Confidence:** {result['confidence']}%")
            st.subheader("Why?")
            for reason in result["reasons"]:
                st.write(f"- {reason}")
        else:
            st.warning("Please enter a URL.")

if option == "Email/Text":
    text_input = st.text_area("Paste email or message text")
    if st.button("Analyze Text"):
        if text_input.strip():
            result = predict_text(text_input.strip())
            st.subheader("Prediction")
            st.write(f"**Label:** {result['label']}")
            st.write(f"**Confidence:** {result['confidence']}%")
            st.subheader("Why?")
            for reason in result["reasons"]:
                st.write(f"- {reason}")
        else:
            st.warning("Please enter some text.")