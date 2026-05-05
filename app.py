import streamlit as st
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import nltk

# Download NLTK data (safe for deployment)
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Support Ticket Classifier",
    layout="wide"
)

# Reduce extra spacing (clean UI)
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
model_path = "backend/app/model/model.pkl"

if not os.path.exists(model_path):
    st.error("❌ Model file not found. Train and save your model first.")
    st.stop()

model = pickle.load(open(model_path, "rb"))

# -------------------------------
# SESSION STATE (for charts)
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
# TITLE & DESCRIPTION
# -------------------------------
st.title("🚀 Support Ticket Classifier")

st.markdown("""
### 🤖 AI-powered system to classify customer issues  
Predicts category + assigns priority using Machine Learning
""")

st.markdown("**Try examples:**")
st.code("""
Payment failed again
App keeps crashing
Forgot my password
Order not delivered
""")

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
        predict_clicked = st.button("Predict")

    with col_btn2:
        if st.button("Clear"):
            st.session_state.history = []

    if predict_clicked:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text")
        else:
            prediction = model.predict([user_input])[0]

            try:
                confidence = model.predict_proba([user_input]).max()
            except:
                confidence = None

            priority = get_priority(user_input)

            # Store history
            st.session_state.history.append(prediction)

            # -------------------
            # RESULTS
            # -------------------
            st.subheader("📊 Results")

            st.success(f"Category: {prediction}")

            if confidence:
                st.info(f"Confidence: {confidence:.2f}")

            st.warning(f"Priority: {priority}")

            # -------------------
            # PROBABILITY BARS
            # -------------------
            try:
                probs = model.predict_proba([user_input])[0]
                labels = model.classes_

                st.subheader("📊 Category Probabilities")

                for label, prob in zip(labels, probs):
                    st.write(f"{label}: {prob:.2f}")
                    st.progress(float(prob))
            except:
                st.info("Probability not available")

# ===============================
# RIGHT SIDE (CHARTS)
# ===============================
with col2:
    st.subheader("📊 Analytics")

    history = st.session_state.history

    if len(history) > 0:
        df_hist = pd.DataFrame(history, columns=["category"])

        c1, c2 = st.columns(2)

        # PIE CHART (SMALL)
        with c1:
            st.markdown("**Distribution**")
            fig1, ax1 = plt.subplots(figsize=(3, 3))
            df_hist["category"].value_counts().plot.pie(
                autopct="%1.0f%%", ax=ax1
            )
            ax1.set_ylabel("")
            st.pyplot(fig1)

        # BAR CHART (SMALL)
        with c2:
            st.markdown("**Count**")
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            df_hist["category"].value_counts().plot.bar(ax=ax2)
            st.pyplot(fig2)

        st.metric("Total Predictions", len(history))

    else:
        st.info("No predictions yet")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("Built by Ansel Monteiro | MSc Big Data Analytics 🚀")