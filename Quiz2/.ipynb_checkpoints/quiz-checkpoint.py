# Quiz 2: explainability project
# Datasets: COMPAS, credit card fraud
# Models: logistic regression, random forest
# Explainability: SHAP, LIME

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import shap
from lime.lime_tabular import LimeTabularExplainer

plt.rcParams["figure.figsize"] = (10, 6)


# Helper functions
def evaluate_binary(model, X_test, y_test, name="model"):
    """
    Print classification metrics for a binary classifier.
    """
    y_pred = model.predict(X_test)
    print(f"\n=== {name} ===")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
        print("ROC AUC:", round(roc_auc_score(y_test, y_score), 4))
        print("PR AUC :", round(average_precision_score(y_test, y_score), 4))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


def fit_logreg(X_train, y_train):
    """
    Logistic regression with class balancing.
    """
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train, y_train)
    return model


def fit_rf(X_train, y_train):
    """
    Random forest with class balancing.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_test_scale(X, y, test_size=0.2, random_state=42):
    """
    Split data and also create a standardized version for logistic regression.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


# 1) COMPAS
print("Loading COMPAS...")

compas = pd.read_csv("compas-scores-two-years.csv")

# Target used in many COMPAS versions
target_col_compas = "two_year_recid"

# Keep only rows with target available
compas = compas.dropna(subset=[target_col_compas]).copy()

# Drop columns that are not useful for modeling if present
drop_cols_compas = [
    "two_year_recid",
    "id",
    "name",
    "first",
    "last",
    "dob",
    "c_case_number",
    "c_offense_date",
    "c_arrest_date",
    "decile_score_text",
    "score_text"
]
drop_cols_compas = [c for c in drop_cols_compas if c in compas.columns]

y_compas = compas[target_col_compas].astype(int)
X_compas = compas.drop(columns=drop_cols_compas)

# Convert categorical columns to numeric using one-hot encoding
X_compas = pd.get_dummies(X_compas, drop_first=True)

# Clean any missing/infinite values if they appear
X_compas = X_compas.replace([np.inf, -np.inf], np.nan).fillna(0)

# Split and scale
Xc_train, Xc_test, Xc_train_lr, Xc_test_lr, yc_train, yc_test, scaler_c = train_test_scale(
    X_compas, y_compas
)

# Train models
lr_compas = fit_logreg(Xc_train_lr, yc_train)
rf_compas = fit_rf(Xc_train, yc_train)

# Evaluate
evaluate_binary(lr_compas, Xc_test_lr, yc_test, name="COMPAS - Logistic Regression")
evaluate_binary(rf_compas, Xc_test, yc_test, name="COMPAS - Random Forest")


# SHAP for COMPAS
print("\nSHAP explanations for COMPAS...")

# Use a sample for faster plots
Xc_test_sample = Xc_test.sample(min(500, len(Xc_test)), random_state=42)
Xc_test_lr_sample = Xc_test_lr[:len(Xc_test_sample)]

# Logistic Regression SHAP
explainer_lr_compas = shap.LinearExplainer(lr_compas, Xc_train_lr)
shap_values_lr_compas = explainer_lr_compas(Xc_test_lr_sample)

shap.summary_plot(
    shap_values_lr_compas.values,
    features=Xc_test_sample,
    feature_names=Xc_test_sample.columns,
    show=True
)

# Random Forest SHAP
explainer_rf_compas = shap.TreeExplainer(rf_compas)
shap_values_rf_compas = explainer_rf_compas(Xc_test_sample)

shap.summary_plot(
    shap_values_rf_compas.values,
    features=Xc_test_sample,
    feature_names=Xc_test_sample.columns,
    show=True
)

# Local SHAP explanation for one COMPAS example
idx = 0
print("\nLocal SHAP example for COMPAS - Random Forest")
shap.plots.waterfall(shap_values_rf_compas[idx], max_display=10)


# LIME for COMPAS
print("\nLIME explanations for COMPAS...")

lime_compas = LimeTabularExplainer(
    training_data=Xc_train.values,
    feature_names=Xc_train.columns.tolist(),
    class_names=["No Recidivism", "Recidivism"],
    mode="classification",
    discretize_continuous=True
)

i = 0
exp_compas_rf = lime_compas.explain_instance(
    Xc_test.iloc[i].values,
    rf_compas.predict_proba,
    num_features=10
)

print("\nLIME - COMPAS - Random Forest")
print(exp_compas_rf.as_list())
exp_compas_rf.show_in_notebook(show_table=True)


# 2) CREDIT CARD FRAUD
print("\nLoading credit fraud dataset...")

fraud = pd.read_csv("creditcard.csv")

target_col_fraud = "Class"
y_fraud = fraud[target_col_fraud].astype(int)
X_fraud = fraud.drop(columns=[target_col_fraud])

# Split and scale
Xf_train, Xf_test, Xf_train_lr, Xf_test_lr, yf_train, yf_test, scaler_f = train_test_scale(
    X_fraud, y_fraud
)

# Train models
lr_fraud = fit_logreg(Xf_train_lr, yf_train)
rf_fraud = fit_rf(Xf_train, yf_train)

# Evaluate
evaluate_binary(lr_fraud, Xf_test_lr, yf_test, name="Fraud - Logistic Regression")
evaluate_binary(rf_fraud, Xf_test, yf_test, name="Fraud - Random Forest")


# SHAP for fraud
print("\nSHAP explanations for fraud...")

Xf_test_sample = Xf_test.sample(min(500, len(Xf_test)), random_state=42)
Xf_test_lr_sample = Xf_test_lr[:len(Xf_test_sample)]

# Logistic Regression SHAP
explainer_lr_fraud = shap.LinearExplainer(lr_fraud, Xf_train_lr)
shap_values_lr_fraud = explainer_lr_fraud(Xf_test_lr_sample)

shap.summary_plot(
    shap_values_lr_fraud.values,
    features=Xf_test_sample,
    feature_names=Xf_test_sample.columns,
    show=True
)

# Random Forest SHAP
explainer_rf_fraud = shap.TreeExplainer(rf_fraud)
shap_values_rf_fraud = explainer_rf_fraud(Xf_test_sample)

shap.summary_plot(
    shap_values_rf_fraud.values,
    features=Xf_test_sample,
    feature_names=Xf_test_sample.columns,
    show=True
)

# Local SHAP explanation for one fraud example
j = 0
print("\nLocal SHAP example for fraud - Random Forest")
shap.plots.waterfall(shap_values_rf_fraud[j], max_display=10)


# LIME for fraud
print("\nLIME explanations for fraud...")

lime_fraud = LimeTabularExplainer(
    training_data=Xf_train.values,
    feature_names=Xf_train.columns.tolist(),
    class_names=["Legit", "Fraud"],
    mode="classification",
    discretize_continuous=True
)

k = 0
exp_fraud_rf = lime_fraud.explain_instance(
    Xf_test.iloc[k].values,
    rf_fraud.predict_proba,
    num_features=10
)

print("\nLIME - Fraud - Random Forest")
print(exp_fraud_rf.as_list())
exp_fraud_rf.show_in_notebook(show_table=True)


# Summary table for the report
results = pd.DataFrame([
    ["COMPAS", "Logistic Regression",
     accuracy_score(yc_test, lr_compas.predict(Xc_test_lr)),
     roc_auc_score(yc_test, lr_compas.predict_proba(Xc_test_lr)[:, 1]),
     average_precision_score(yc_test, lr_compas.predict_proba(Xc_test_lr)[:, 1])],

    ["COMPAS", "Random Forest",
     accuracy_score(yc_test, rf_compas.predict(Xc_test)),
     roc_auc_score(yc_test, rf_compas.predict_proba(Xc_test)[:, 1]),
     average_precision_score(yc_test, rf_compas.predict_proba(Xc_test)[:, 1])],

    ["Fraud", "Logistic Regression",
     accuracy_score(yf_test, lr_fraud.predict(Xf_test_lr)),
     roc_auc_score(yf_test, lr_fraud.predict_proba(Xf_test_lr)[:, 1]),
     average_precision_score(yf_test, lr_fraud.predict_proba(Xf_test_lr)[:, 1])],

    ["Fraud", "Random Forest",
     accuracy_score(yf_test, rf_fraud.predict(Xf_test)),
     roc_auc_score(yf_test, rf_fraud.predict_proba(Xf_test)[:, 1]),
     average_precision_score(yf_test, rf_fraud.predict_proba(Xf_test)[:, 1])]
], columns=["Dataset", "Model", "Accuracy", "ROC_AUC", "PR_AUC"])

print("\nSummary table:")
print(results)

results.to_csv("q1_explainability_results.csv", index=False)
