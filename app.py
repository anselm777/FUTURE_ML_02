import streamlit as st
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Support Ticket Classifier", layout="wide")

# -----------------------------
# Load model
# -----------------------------
model_path = "backend/app/model/model.pkl"

if not os.path.exists(model_path):
    st.error("Model file not found. Train and save your model first.")
    st.stop()

model = pickle.load(open(model_path, "rb"))

# -----------------------------
# Session state (history)
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# Priority logic
# -----------------------------
def get_priority(text):
    text = text.lower()
    if any(word in text for word in ["failed", "error", "crash", "urgent"]):
        return "🔴 High"
    elif any(word in text for word in ["delay", "slow", "forgot"]):
        return "🟡 Medium"
    else:
        return "🟢 Low"

# -----------------------------
# Title
# -----------------------------
st.markdown("## 🚀 Support Ticket Classifier")

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns([2, 1])

# =============================
# LEFT SIDE (INPUT + RESULT)
# =============================
with col1:

    # Sample buttons
    colA, colB = st.columns(2)

    with colA:
        if st.button("Sample: Payment Issue"):
            st.session_state.sample_text = "My payment failed and I need help"

    with colB:
        if st.button("Sample: Account Issue"):
            st.session_state.sample_text = "I forgot my password"

    # Text input
    user_input = st.text_area(
        "Enter customer issue:",
        value=st.session_state.get("sample_text", ""),
        height=150
    )

    # Predict button
    if st.button("Predict"):

        if user_input.strip() == "":
            st.warning("Please enter some text")

        else:
            prediction = model.predict([user_input])[0]

            try:
                probs = model.predict_proba([user_input])[0]
                confidence = float(max(probs))
            except:
                probs = None
                confidence = None

            priority = get_priority(user_input)

            # Save history
            st.session_state.history.append(prediction)

            # Results
            st.subheader("📊 Results")
            st.success(f"Category: {prediction}")

            if confidence is not None:
                st.info(f"Confidence: {confidence:.2f}")

            st.warning(f"Priority: {priority}")

            # -----------------------------
            # Probability Chart
            # -----------------------------
            if probs is not None:
                labels = model.classes_

                prob_df = pd.DataFrame({
                    "Category": labels,
                    "Probability": probs
                })

                st.subheader("📊 Prediction Confidence")
                st.bar_chart(prob_df.set_index("Category"))
            else:
                st.info("Probability not available")

# =============================
# RIGHT SIDE (DASHBOARD)
# =============================
with col2:

    st.subheader("📈 Analytics Dashboard")

    history = st.session_state.history

    if len(history) > 0:
        df_hist = pd.DataFrame(history, columns=["category"])

        # Pie Chart
        st.subheader("📊 Category Distribution")
        fig1, ax1 = plt.subplots()
        df_hist["category"].value_counts().plot.pie(
            autopct="%1.1f%%",
            startangle=90,
            ax=ax1
        )
        ax1.set_ylabel("")
        ax1.set_title("Distribution")
        st.pyplot(fig1)

        # Bar Chart
        st.subheader("📊 Category Count")
        fig2, ax2 = plt.subplots()
        df_hist["category"].value_counts().plot.bar(ax=ax2)
        st.pyplot(fig2)

        # Metrics
        st.metric("Total Predictions", len(history))

        # Reset button
        if st.button("Reset Analytics"):
            st.session_state.history = []
            st.success("Analytics reset")

    else:
        st.info("No predictions yet. Try entering some text!")

# =============================
# BATCH PROCESSING
# =============================
st.markdown("---")
st.subheader("📂 Batch Processing")

uploaded_file = st.file_uploader("Upload CSV file with 'text' column")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "text" not in df.columns:
        st.error("CSV must contain a 'text' column")
    else:
        df["prediction"] = model.predict(df["text"])
        st.dataframe(df)