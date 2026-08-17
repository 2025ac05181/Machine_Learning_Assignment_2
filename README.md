# Credit Risk Classification - ML Assignment 2

## a. Problem Statement

Evaluate and compare five supervised machine learning models to classify loan applications into `Good` and `Bad` credit risk categories based on their financial, demographic, and employment profiles.

### i. Business Impact & Error Asymmetry:

Accurate credit scoring is critical for financial institutions to minimize default risk while maximizing loan approvals. Classification errors in this domain carry asymmetric costs:

- **False Positive (Predicting `Good` for a `Bad` risk):** Exposes the institution to a loan default and direct capital loss.
- **False Negative (Predicting `Bad` for a `Good` risk):** Causes foregone interest income and lost customer acquisition.

### ii. Evaluation Focus:

Because misclassifying default risk has severe financial implications, this project benchmarks models beyond raw Accuracy, utilizing **Precision, Recall, F1-Score, AUC, and Matthews Correlation Coefficient (MCC)** to determine the optimal balance between risk mitigation and revenue capture.

## b. Dataset Description

This project utilizes the **Statlog (German Credit Data)** dataset to serve as the foundational evidence for predicting applicant creditworthiness. By analyzing a historical ledger of financial and demographic footprints, our models act as automated investigators, piecing together distinct patterns to uncover underlying default risks before a loan is approved.

| Attribute          | Details                                                                                                                                                                                                                                                                                                     |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Source**         | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)                                                                                                                                                                                                       |
| **Instances**      | 1,000 loan applicants                                                                                                                                                                                                                                                                                       |
| **Features**       | 20 attributes (7 numeric + 13 categorical), e.g. checking account status, credit history, purpose, credit amount, savings account, employment duration, installment rate, personal status, present residence duration, property, age, housing, job, number of dependents, telephone, foreign worker status. |
| **Target Class**   | `credit_risk` — `Good` (700 instances) or `Bad` (300 instances), i.e. a moderately imbalanced binary classification problem.                                                                                                                                                                                |
| **Preprocessing**  | Numeric features are standardized (`StandardScaler`); categorical features are one-hot encoded (`OneHotEncoder`). Both steps are wrapped into a single `sklearn.Pipeline` per model so the exact training-time preprocessing is automatically re-applied at inference time.                                 |
| **Data Splitting** | Stratified 80/20 train/test split (800 train / 200 test, `random_state=42`). The 200-row test split is exported as `test_data.csv` and is what the Streamlit app expects to be uploaded.\*                                                                                                                  |

## c. GitHub Repository and deployment Link

**Repository Link:** [Machine Learning Assignment 2](https://github.com/2025ac05181/Machine_Learning_Assignment_2)

**Deployment Link:** [Streamlit App](http://2025ac05181-ml-assignment-2.streamlit.app/)

## d. Models Used & Evaluation

### Performance Metrics Comparison

The following table compares the evaluation metrics for the five implemented classification models tested on the withheld `test_data.csv`.

| ML Model Name                | Accuracy | AUC | Precision | Recall | F1  | MCC |
| ---------------------------- | -------- | --- | --------- | ------ | --- | --- |
| **Logistic Regression**      |          |     |           |        |     |     |
| **Decision Tree**            |          |     |           |        |     |     |
| **kNN**                      |          |     |           |        |     |     |
| **Naive Bayes**              |          |     |           |        |     |     |
| **Random Forest (Ensemble)** |          |     |           |        |     |     |

### Model Observations

The table below details specific observations regarding the performance, trade-offs, and behavioral characteristics of each model when classifying credit risk.

| ML Model Name                        | Observation about model performance |
| ------------------------------------ | ----------------------------------- |
| **Logistic Regression**              |                                     |
| **Decision Tree**                    |                                     |
| **kNN**                              |                                     |
| **Naive Bayes**                      |                                     |
| **Random Forest (Ensemble)**         |                                     |
| **Overall Winner for your dataset?** |                                     |
