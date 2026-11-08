import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    ConfusionMatrixDisplay,
    classification_report,
)

st.set_page_config(page_title="Fetal Health Classifier", layout="wide")
st.title("Fetal Health Risk Classification")
st.write("Upload a test CSV file and choose a model from the dropdown to run predictions and check evaluation metrics.")

file = st.file_uploader("Upload test CSV", type="csv")
if file is None:
    st.info("Upload test_data.csv to begin")
    st.stop()

df = pd.read_csv(file)
st.subheader("Uploaded Data Preview")
st.dataframe(df.head())

models = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/support_vector_machine.pkl",
}

choice = st.selectbox("Choose a model", list(models))
model = joblib.load(models[choice])

X = df.drop(columns=["fetal_risk"])
y = df["fetal_risk"]

pred = model.predict(X)
prob = model.predict_proba(X)[:, 1]

acc = accuracy_score(y, pred)
auc = roc_auc_score(y, prob)
prec = precision_score(y, pred, zero_division=0)
rec = recall_score(y, pred, zero_division=0)
f1 = f1_score(y, pred, zero_division=0)
mcc = matthews_corrcoef(y, pred)

st.subheader("Evaluation Metrics")
c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{acc:.3f}")
c2.metric("AUC", f"{auc:.3f}")
c3.metric("Precision", f"{prec:.3f}")

c4, c5, c6 = st.columns(3)
c4.metric("Recall", f"{rec:.3f}")
c5.metric("F1 Score", f"{f1:.3f}")
c6.metric("MCC", f"{mcc:.3f}")

st.subheader("Confusion Matrix and Classification Report")
col_left, col_right = st.columns(2)

with col_left:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, display_labels=["Normal", "At risk"], ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(choice)
    st.pyplot(fig)

with col_right:
    report = classification_report(y, pred, target_names=["Normal", "At risk"], digits=3)
    st.text(report)