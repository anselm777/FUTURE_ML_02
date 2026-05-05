import streamlit as st
import pickle
import os
import pandas as pd
import plotly.express as px
import nltk

# -------------------------------
# NLTK (safe for deployment)
# -------------------------------
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Support Ticket Classifier",
    layout="wide"
)

# -------------------------------
# DARK + CARD STYLE
# -------------------------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}

.block-container {
    padding-top: 2rem;
}

.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
model_path = "backend/app/model/model.pkl"

if not os.path.exists(model_path):
    st.error("Model not found")
    st.stop()

model = pickle.load(open(model_path, "rb"))

# -------------------------------
# SESSION STATE
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# PRIORITY LOGIC
# -------------------------------
def get_priority(text):
    text = text.lower()
    if any(word in text for word in ["failed", "error", "crash", "urgent"]):
        return "🔴 High"
    elif any(word in text for word in ["delay", "slow", "forgot"]):
        return "🟡 Medium"
    else:
        return "🟢 Low"

# -------------------------------
# HEADER
# -------------------------------
st.markdown("<h1 style='text-align:center;'>🚀 Support Ticket Classifier</h1>", unsafe_allow_html=True)

st.markdown("""
<div class="card">
🤖 Classify customer issues and assign priority using Machine Learning
</div>
""", unsafe_allow_html=True)

# -------------------------------
# LAYOUT
# -------------------------------
col1, col2 = st.columns([2, 1])

# ===============================
# INPUT + RESULTS
# ===============================
with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    user_input = st.text_area("Enter customer issue:", height=150)

    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        predict = st.button("Predict")

    with col_btn2:
        if st.button("Clear"):
            st.session_state.history = []

    if predict:
        if user_input.strip() == "":
            st.warning("Please enter text")
        else:
            prediction = model.predict([user_input])[0]

            try:
                confidence = model.predict_proba([user_input]).max()
            except:
                confidence = None

            priority = get_priority(user_input)

            st.session_state.history.append(prediction)

            st.markdown("### 📊 Results")
            st.success(f"Category: {prediction}")

            if confidence:
                st.info(f"Confidence: {confidence:.2f}")

            st.warning(f"Priority: {priority}")

            # Plotly probability chart
            try:
                probs = model.predict_proba([user_input])[0]
                labels = model.classes_

                df_prob = pd.DataFrame({
                    "Category": labels,
                    "Probability": probs
                })

                fig = px.bar(
                    df_prob,
                    x="Category",
                    y="Probability",
                    title="Prediction Confidence",
                    color="Probability"
                )

                st.plotly_chart(fig, use_container_width=True)

            except:
                st.info("No probability available")

    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# ANALYTICS DASHBOARD
# ===============================
with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📊 Analytics")

    history = st.session_state.history

    if len(history) > 0:

        df_hist = pd.DataFrame(history, columns=["category"])

        # Pie chart
        pie_fig = px.pie(
            df_hist,
            names="category",
            title="Category Distribution"
        )

        st.plotly_chart(pie_fig, use_container_width=True)

        # Bar chart
        bar_fig = px.bar(
            df_hist["category"].value_counts().reset_index(),
            x="index",
            y="category",
            labels={"index": "Category", "category": "Count"},
            title="Category Count"
        )

        st.plotly_chart(bar_fig, use_container_width=True)

        st.metric("Total Predictions", len(history))

    else:
        st.info("No predictions yet")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("""
---
<div style='text-align:center; color:gray;'>
Built by Ansel Monteiro 🚀 | MSc Big Data Analytics
</div>
""", unsafe_allow_html=True)