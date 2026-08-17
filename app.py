import streamlit as st

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

st.caption(
    "🔒 **Environment Note:** Currently running in UI-Only Deployment Mode. All backend logic is temporarily disabled."
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
    st.stop()