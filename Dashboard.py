"""
Dashboard.py
Full Fraud Detection Dashboard
Run with: streamlit run Dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide", page_icon="🛡️")

# ==========================================================
# Load data & artifacts
# ==========================================================
df = pd.read_csv("featured_data.csv")

with open("model_results.json") as f:
    model_results = json.load(f)

model = joblib.load("best_model.pkl")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("Y_test.csv").squeeze()

st.title("🛡️ Fraud Detection Dashboard")
st.caption("Machine learning-based fraud detection across customer, card, merchant, and transaction data")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "🤖 Model Performance", "🔍 Feature Insights", "🧪 Try a Prediction"]
)

# ==========================================================
# TAB 1: Overview
# ==========================================================
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    total_txn = len(df)
    fraud_txn = df["Fraud_Flag"].sum()
    fraud_rate = fraud_txn / total_txn
    fraud_amount = df.loc[df["Fraud_Flag"] == 1, "Transaction_Amount"].sum()

    col1.metric("Total Transactions", f"{total_txn:,}")
    col2.metric("Fraudulent Transactions", f"{fraud_txn:,}")
    col3.metric("Fraud Rate", f"{fraud_rate:.2%}")
    col4.metric("Total Fraud Amount", f"₹{fraud_amount:,.0f}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        cat_fraud = df.groupby("Merchant_Category")["Fraud_Flag"].mean().sort_values(ascending=False).head(10)
        fig = px.bar(cat_fraud, orientation="h", title="Fraud Rate by Merchant Category (Top 10)",
                     labels={"value": "Fraud Rate", "Merchant_Category": ""})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        hour_fraud = df.groupby("Transaction_Hour")["Fraud_Flag"].mean()
        fig = px.line(hour_fraud, title="Fraud Rate by Hour of Day", markers=True,
                      labels={"value": "Fraud Rate", "Transaction_Hour": "Hour"})
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        reason_counts = df[df["Fraud_Flag"] == 1]["Fraud_Reason"].value_counts()
        fig = px.pie(reason_counts, values=reason_counts.values, names=reason_counts.index,
                     title="Fraud Reasons Breakdown")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        channel_fraud = df.groupby("Transaction_Channel")["Fraud_Flag"].mean().sort_values(ascending=False)
        fig = px.bar(channel_fraud, title="Fraud Rate by Transaction Channel",
                     labels={"value": "Fraud Rate", "Transaction_Channel": ""})
        st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# TAB 2: Model Performance
# ==========================================================
with tab2:
    st.subheader("Model Comparison")
    results_df = pd.DataFrame(model_results["results"])[["name", "roc_auc", "pr_auc"]]
    results_df.columns = ["Model", "ROC-AUC", "PR-AUC"]
    st.dataframe(results_df, use_container_width=True)
    st.success(f"🏆 Best model: **{model_results['best_model']}** (selected by PR-AUC)")

    # Compute live ROC/PR curves using the saved best model + test set
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random"))
        fig.update_layout(title="ROC Curve (Best Model)", xaxis_title="False Positive Rate",
                           yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR Curve"))
        fig.update_layout(title="Precision-Recall Curve (Best Model)", xaxis_title="Recall",
                           yaxis_title="Precision")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Confusion Matrix (Best Model)")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, text_auto=True, x=["Predicted: Not Fraud", "Predicted: Fraud"],
                     y=["Actual: Not Fraud", "Actual: Fraud"], color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# TAB 3: Feature Insights
# ==========================================================
with tab3:
    if hasattr(model, "feature_importances_"):
        fi_df = pd.DataFrame({
            "feature": X_test.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False).head(20)
        fig = px.bar(fi_df.sort_values("importance"), x="importance", y="feature", orientation="h",
                     title=f"Top 20 Most Important Features — {model_results['best_model']}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance is not available for this model type (e.g. Logistic Regression).")

    st.subheader("Transaction Amount Distribution: Fraud vs Legit")
    fig = px.histogram(df, x="Transaction_Amount", color="Fraud_Flag", nbins=60,
                        barmode="overlay", opacity=0.6, labels={"Fraud_Flag": "Fraud Flag"})
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# TAB 4: Try a Prediction
# ==========================================================
with tab4:
    st.subheader("Score a Sample Transaction")
    st.caption("Pulls a random real transaction from the test set and shows the model's prediction.")

    if st.button("🎲 Pick a random transaction"):
        idx = np.random.randint(0, len(X_test))
        sample = X_test.iloc[[idx]]
        actual = y_test.iloc[idx]
        proba = model.predict_proba(sample)[0, 1]
        pred = model.predict(sample)[0]

        st.markdown("---")
        colA, colB, colC = st.columns(3)
        colA.metric("Predicted Fraud Probability", f"{proba:.1%}")
        colB.metric("Model Prediction", "Fraud" if pred == 1 else "Not Fraud")
        colC.metric("Actual Label", "Fraud" if actual == 1 else "Not Fraud")

        if pred == 1:
            st.error("🚨 Flagged as high fraud risk")
        else:
            st.success("✅ Low fraud risk")