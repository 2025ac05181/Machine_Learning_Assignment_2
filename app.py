import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
)

# Page Setup
st.set_page_config(
    page_title="Credit Risk Audit Dashboard",
    page_icon="🔎",
    layout="wide"
)

#CSS
st.markdown("""
    <style>
        div[data-testid="stTabs"] > div:first-child {
            position: sticky !important;
            top: 0px !important;
            z-index: 999 !important;
            background-color: #0e1117 !important; 
            padding-top: 10px !important;
            padding-bottom: 10px !important;
            border-bottom: 1px solid #333 !important;
        }
        
        div[data-testid="metric-container"], div[data-testid="stMetric"] {
            background: rgba(30, 41, 59, 0.7) !important; 
            backdrop-filter: blur(10px) !important; 
            padding: 15px !important;
            border-radius: 8px !important;
            border: 1px solid rgba(56, 189, 248, 0.3) !important; 
            box-shadow: 0 4px 15px rgba(56, 189, 248, 0.15) !important; 
        }
        
        div[data-testid="metric-container"] > div, div[data-testid="stMetric"] > div {
            color: #e0f2fe !important; 
            text-shadow: 0 0 5px rgba(224, 242, 254, 0.3) !important;
        }
        
        div[data-testid="metric-container"] label, div[data-testid="stMetric"] label {
            color: #7dd3fc !important;
        }
    </style>
""", unsafe_allow_html=True)

#Models
model_names = [
    "Logistic Regression",
    "Decision Tree Classifier",
    "K-Nearest Neighbor Classifier"
    "Naive Byes Classifier - Gaussian",
    "Ensemble Model - Random Forest",
]

#Header
st.title("🔎 Credit Risk Audit System")
st.caption(
    "BITS Pilani WILP · M.Tech (AIML/DSE) · Machine Learning Assignment 2  |  "
    "Financial Intelligence & Credit Risk Audit Dashboard"
)

with st.expander("📌 Audit System Protocol & Overview", expanded=False):
    st.markdown(
        """
        ### System Objectives
        This application evaluates 5 distinct machine learning models designed to classify loan applicants 
        into **Good** or **Bad** credit risks. 

        **Evaluation Protocol:**
        1. Ingest test data using the control panel (`test_data.csv`).
        2. Select an audit model to inspect classification accuracy, confusion matrix, and precision/recall balance.
        3. Navigate to the comparative matrix tab to view cross-model performance rankings.
        """
    )

st.markdown("---")

st.subheader("🕹️ Audit Control Panel")
ctrl_col1, ctrl_col2 = st.columns(2)
# File Upload Button
with ctrl_col1:
    uploaded_file = st.file_uploader(
        "1. Ingest Test Evidence (CSV)", 
        type=["csv"],
        help="Upload the test_data.csv file to proceed."
    )
# Model Selection Button
with ctrl_col2:
    selected_model_name = st.selectbox(
        "2. Select Classifier for Audit", 
        model_names
    )
st.markdown("---")

# ---------------------------------------------------------------------------
# UI Hard Stop
# ---------------------------------------------------------------------------
if uploaded_file is None:
    st.info("👆 Please ingest `test_data.csv` using the control panel above.")
    st.stop()
else:
    st.success(f"File '{uploaded_file.name}' successfully loaded into memory!.")
    # Removed st.stop() here so the script can proceed to evaluation

 
# Load config from the newly defined data folder
with open("data/Feature_Config.json") as f:
    config = json.load(f)
        
TARGET_COL = config["target_col"]
POSITIVE_LABEL = config["positive_label"]
    
# Load model artifact
model_path = f"model/{config['model_files'][selected_model_name]}"
lr_model = joblib.load(model_path)
# Ingest data
data = pd.read_csv(uploaded_file)

if TARGET_COL not in data.columns:
    st.error(f"Missing target column: '{TARGET_COL}' in the uploaded file.")
    st.stop()

X = data.drop(columns=[TARGET_COL])
y_true = (data[TARGET_COL] == POSITIVE_LABEL).astype(int)

# Generate predictions
y_pred = lr_model.predict(X)
y_proba = lr_model.predict_proba(X)[:, 1]

# Compile metrics
metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "AUC Score": roc_auc_score(y_true, y_proba),
    "Precision": precision_score(y_true, y_pred, zero_division=0),
    "Recall": recall_score(y_true, y_pred, zero_division=0),
    "F1 Score": f1_score(y_true, y_pred, zero_division=0),
    "MCC Score": matthews_corrcoef(y_true, y_pred),
}
    
st.subheader(f"Classifier Audit Results: {selected_model_name}")
    
# Render metric headers
col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
cols = [col_m1, col_m2, col_m3, col_m4, col_m5, col_m6]
for col, (m_title, val) in zip(cols, metrics.items()):
    col.metric(m_title, f"{val:.4f}")

st.markdown("---")

# Render visualizations
col_viz1, col_viz2 = st.columns([1, 1.2])

with col_viz1:
    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Predicted Good", "Predicted Bad"],
        yticklabels=["Actual Good", "Actual Bad"], ax=ax
    )
    plt.tight_layout()
    st.pyplot(fig)

with col_viz2:
    st.markdown("#### Full Classification Report")
    report_dict = classification_report(
        y_true, y_pred, target_names=["Good Risk", "Bad Risk"], output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose().round(3)
    st.dataframe(
        report_df.style.background_gradient(cmap="Blues", axis=0),
        use_container_width=True
    )