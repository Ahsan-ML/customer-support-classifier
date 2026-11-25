import os
import re
import joblib
import streamlit as st

st.set_page_config(page_title="Customer Support Ticket Classifier", page_icon="🎫")


def find(name):
    for p in (name, os.path.join("models", name)):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} not found. Upload it next to streamlit_app.py")


@st.cache_resource
def load():
    return joblib.load(find("best_model.pkl")), joblib.load(find("tfidf_vectorizer.pkl"))


def clean_text(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


model, vectorizer = load()

st.title("🎫 Customer Support Ticket Classifier")
st.write("Traditional ML (TF-IDF + the best of 4 classifiers) predicts the support "
         "category of a customer message.")

examples = ["I want to cancel my order", "my refund has not arrived",
            "I cannot log into my account", "where is my parcel?"]
choice = st.selectbox("Try an example (optional)", [""] + examples)
message = st.text_area("Enter customer message", value=choice, height=100,
                       placeholder="e.g. I paid for my order, but it is still showing as pending.")

if st.button("Predict Category"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        pred = model.predict(vectorizer.transform([clean_text(message)]))[0]
        st.success(f"Predicted category: **{pred}**")
