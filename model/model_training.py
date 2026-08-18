import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
DATA_PATH = "data/german_credit.csv"
TARGET_COL = "credit_risk"
POSITIVE_LABEL = "Bad" 

#loading data
df = pd.read_csv(DATA_PATH)

numeric_features = [
    "duration_months", "credit_amount", "installment_rate_pct",
    "present_residence_since", "age_years", "existing_credits_count",
    "num_dependents",
]
categorical_features = [c for c in df.columns if c not in numeric_features + [TARGET_COL]]

X = df.drop(columns=[TARGET_COL])
y = (df[TARGET_COL] == POSITIVE_LABEL).astype(int)  # Bad=1, Good=0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
 
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Decision Tree Classifier": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    "K-Nearest Neighbor Classifier": KNeighborsClassifier(n_neighbors=15),
    "Naive Byes Classifier - Gaussian": GaussianNB(),
    "Ensemble Model - Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, random_state=RANDOM_STATE),
}

#Model Training and evaluation
results = []
filenames = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree Classifier": "decision_tree.joblib",
    "K-Nearest Neighbor Classifier": "knn.joblib",
    "Naive Byes Classifier - Gaussian": "naive_bayes.joblib",
    "Ensemble Model - Random Forest": "random_forest.joblib",
}

for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    results.append(metrics)

    joblib.dump(pipe, f"model/{filenames[name]}")
    print(f"Trained {name:32s} | " + " | ".join(f"{k}={v:.3f}" for k, v in metrics.items() if k != "ML Model Name"))

# Model Comparison
results_df = pd.DataFrame(results).round(4)
results_df.to_csv("data/comparison_table.csv", index=False)
print("\nComparison table:\n", results_df.to_string(index=False))

test_export = X_test.copy()
test_export[TARGET_COL] = df.loc[X_test.index, TARGET_COL].values
test_export.to_csv("test_data.csv", index=False)
print(f"\nSaved held-out test_data.csv with {len(test_export)} rows.")

# Data Features
with open("data/Feature_Config.json", "w") as f:
    json.dump({
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target_col": TARGET_COL,
        "positive_label": POSITIVE_LABEL,
        "model_files": filenames,
    }, f, indent=2)