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

| ML Model Name                | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
| ---------------------------- | -------- | ----- | --------- | ------ | ----- | ----- |
| **Logistic Regression**      | 0.795    | 0.804 | 0.721     | 0.517  | 0.602 | 0.481 |
| **Decision Tree**            | 0.725    | 0.686 | 0.549     | 0.467  | 0.505 | 0.318 |
| **kNN**                      | 0.760    | 0.765 | 0.875     | 0.233  | 0.368 | 0.370 |
| **Naive Bayes**              | 0.745    | 0.778 | 0.582     | 0.533  | 0.557 | 0.379 |
| **Random Forest (Ensemble)** | 0.800    | 0.804 | 0.763     | 0.483  | 0.592 | 0.489 |

### Model Observations

The table below details specific observations regarding the performance, trade-offs, and behavioral characteristics of each model when classifying credit risk.

| ML Model Name                        | Observation about model performance                                                                                                                                                                                                                                                                                                                                                                             |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression**              | While it narrowly lost the top spot in Accuracy, it holds the highest **F1 Score (0.602)**. It absorbed the new engineered features (like `credit_per_person`) flawlessly into its linear equation. By applying a `C=0.1` regularization penalty, I found it successfully balances catching defaults (Recall: 0.517) without making too many false accusations.                                                 |
| **Decision Tree**                    | Previously the weakest link, its MCC doubled from 0.163 to **0.318**. By handing the tree pre-calculated mathematical ratios instead of forcing it to split on scattered raw numbers, I was able to provide it with clear, logical branching paths. However, even with fine-tuning, a single tree remains my weakest model overall due to its inherent instability.                                             |
| **kNN**                              | Tuning this model (specifically switching to Manhattan distance with `p=1` and `weights='distance'`) turned it into a hyper-conservative profiler. It boasts a staggering 0.875 Precision—meaning when it flags a loan as "Bad", it is almost never wrong. However, this comes at the cost of the lowest Recall (0.233). It simply refuses to classify a bad risk unless the mathematical distance is absolute. |
| **Naive Bayes**                      | Adding the `var_smoothing` threshold stabilized this model beautifully. It fixed its previously terrible precision while allowing it to retain the **highest Recall (0.533)** of any model. It successfully identifies the most actual defaulters, making it my ideal model for a risk-averse scenario where missing a bad loan is considered the worst possible outcome.                                       |
| **Random Forest (Ensemble)**         | With an **Accuracy of 0.800** and the highest **MCC (0.489)**, the forest has taken the lead. By allowing the trees to grow deeper (`max_depth=None`) and heavily penalizing missed defaults (`class_weight='balanced'`), the ensemble was able to fully exploit the complex, non-linear patterns hidden in my newly engineered features without overfitting.                                                   |
| **Overall Winner for your dataset?** | **Random Forest** wins the pure metrics race (Highest MCC and Accuracy). I consider it the most robust model for predicting the absolute truth of this dataset. However, **Logistic Regression** remains a highly compelling alternative if the business prioritizes a smoother balance between Precision and Recall (F1 Score).                                                                                |
