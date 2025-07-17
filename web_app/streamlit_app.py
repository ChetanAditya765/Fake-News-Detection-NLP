import streamlit as st
import joblib
import os
import numpy as np
from lime.lime_text import LimeTextExplainer

# Model file mapping
MODEL_FILES = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "Support Vector Machine": "svm_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl"
}

VECTORIZER_PATH = "models/vectorizer.pkl"

@st.cache_resource
def load_vectorizer():
    if os.path.exists(VECTORIZER_PATH):
        return joblib.load(VECTORIZER_PATH)
    return None

@st.cache_resource
def load_model(model_filename):
    full_path = os.path.join("models", model_filename)
    if os.path.exists(full_path):
        return joblib.load(full_path)
    return None

st.title("📰 Fake News Detection Demo")
st.write("Paste a news article text to check if it’s Fake or Real.")

# Load vectorizer
vectorizer = load_vectorizer()
if not vectorizer:
    st.error("❌ Vectorizer not found in `models/vectorizer.pkl`. Train your models first.")
    st.stop()

# Model selection
model_choice = st.selectbox("🔍 Select Model", list(MODEL_FILES.keys()))
model = load_model(MODEL_FILES[model_choice])
if not model:
    st.error(f"❌ Model file not found: {MODEL_FILES[model_choice]}")
    st.stop()

# Input box
text_input = st.text_area("📝 Enter News Article Text", height=200)

# Predict
if st.button("Predict") and text_input:
    X_input = vectorizer.transform([text_input])
    
    if hasattr(model, "predict_proba"):
        pred_prob = model.predict_proba(X_input)[0]
        prediction = "Fake" if pred_prob[1] > 0.5 else "Real"
        confidence = pred_prob[1] if prediction == "Fake" else pred_prob[0]
        st.write(f"### 🧠 Prediction: {prediction}")
        st.write(f"✅ Confidence: {confidence:.2f}")
    else:
        pred_label = model.predict(X_input)[0]
        prediction = "Fake" if pred_label == 1 else "Real"
        st.write(f"### 🧠 Prediction: {prediction}")
        st.write("⚠️ This model doesn't support probability output.")

    # LIME explanation
    try:
        explainer = LimeTextExplainer(class_names=["Real", "Fake"])

        def predict_fn(texts):
            transformed = vectorizer.transform(texts)
            return model.predict_proba(transformed)

        explanation = explainer.explain_instance(
            text_input, predict_fn, num_features=10
        )
        st.write("#### 🔍 Top Words Influencing the Prediction:")
        fig = explanation.as_pyplot_figure()
        st.pyplot(fig)
    except Exception as e:
        st.warning("⚠️ LIME explanation could not be generated.")
        st.exception(e)

# Sidebar sample
st.sidebar.header("📰 Sample Articles")
sample_articles = {
    "Sample 1": "Breaking news! NASA announces discovery of aliens on Mars.",
    "Sample 2": "Economic growth hits new high this quarter, says finance minister.",
    "Sample 3": "Celebrity X caught in scandal, authorities investigate."
}

if st.sidebar.button("Load Sample 1"):
    st.session_state['text_input'] = sample_articles["Sample 1"]
if st.sidebar.button("Load Sample 2"):
    st.session_state['text_input'] = sample_articles["Sample 2"]
if st.sidebar.button("Load Sample 3"):
    st.session_state['text_input'] = sample_articles["Sample 3"]

if 'text_input' in st.session_state:
    text_input = st.session_state['text_input']
