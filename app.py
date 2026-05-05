import streamlit as st
import pickle
import os
import pandas as pd
import plotly.express as px

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Support Ticket Classifier",
    layout="wide"
)

# -------------------------------
# LOAD MODEL
# -------------------------------
model_path = "backend/app/model/model.pkl"

if not os.path.exists(model_path):
    st.error("❌ Model file not found")
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
st.title("🚀 Support Ticket Classifier")
st.markdown("### 🤖 Classify customer issues and assign priority using Machine Learning")

# -------------------------------
# LAYOUT
# -------------------------------
col1, col2 = st.columns([2, 1])

# ===============================
# LEFT SIDE (INPUT + RESULTS)
# ===============================
with col1:

    user_input = st.text_area("Enter customer issue:", height=150)

    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        predict = st.button("Predict")

    with col_btn2:
        if st.button("Clear"):
            st.session_state.history = []

    if predict:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text")
        else:
            # SINGLE CALL (optimized)
            probs = None
            try:
                probs = model.predict_proba([user_input])[0]
                confidence = max(probs)
            except:
                confidence = None

            prediction = model.predict([user_input])[0]
            priority = get_priority(user_input)

            st.session_state.history.append(prediction)

            # RESULTS
            st.subheader("📊 Results")
            st.success(f"Category: {prediction}")

            if confidence is not None:
                st.info(f"Confidence: {confidence:.2f}")

            st.warning(f"Priority: {priority}")

            # PROBABILITY CHART
            if probs is not None:
                labels = model.classes_

                df_prob = pd.DataFrame({
                    "Category": labels,
                    "Probability": probs
                })

                fig = px.bar(
                    df_prob,
                    x="Category",
                    y="Probability",
                    color="Probability",
                    title="Prediction Confidence",
                    text="Probability"
                )

                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Probability not available")

# ===============================
# RIGHT SIDE (ANALYTICS)
# ===============================
with col2:

    st.subheader("📊 Analytics")

    history = st.session_state.history

    if len(history) > 0:
        df_hist = pd.DataFrame(history, columns=["category"])

        # PIE CHART
        fig1 = px.pie(
            df_hist,
            names="category",
            title="Category Distribution"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # BAR CHART (FIXED VERSION)
        df_count = df_hist["category"].value_counts().reset_index()
        df_count.columns = ["category", "count"]
        df_count = df_count.sort_values(by="count", ascending=False)

        fig2 = px.bar(
            df_count,
            x="category",
            y="count",
            title="Category Count",
            text="count"
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.metric("Total Predictions", len(history))

    else:
        st.info("No predictions yet")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("Built by Ansel Monteiro 🚀 | MSc Big Data Analytics")