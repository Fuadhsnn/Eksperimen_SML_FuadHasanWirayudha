# ===============================
# Modelling & Hyperparameter Tuning
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import optuna

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import mlflow
import dagshub
import os
# ===============================
# Setup DagsHub + MLflow
# ===============================
dagshub.init(
    repo_owner="Fuadhsnn",
    repo_name="Eksperimen_SML_FuadHasanWirayudha",
    mlflow=True
)

mlflow.set_experiment("Diabetes_Prediction_Fuad")

# ===============================
# Load Dataset (PATH AMAN)
# ===============================
DATA_PATH = "cleaned_pima_diabetes_fuad.csv"


df = pd.read_csv(DATA_PATH)

X = df.drop("Outcome", axis=1)
y = df["Outcome"].astype(int) 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# Hyperparameter Tuning with Optuna
# ===============================
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 2, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return accuracy_score(y_test, preds)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

best_params = study.best_params

# ===============================
# Train Final Model
# ===============================
model = RandomForestClassifier(
    **best_params,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ===============================
# Metrics
# ===============================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# ===============================
# MLflow Manual Logging
# ===============================

for param, value in best_params.items():
    mlflow.log_param(param, value)

mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("precision", precision)
mlflow.log_metric("recall", recall)
mlflow.log_metric("f1_score", f1)

mlflow.log_artifact(cm_path)
mlflow.log_artifact(fi_path)
mlflow.log_artifact(model_path)


    # ===============================
    # Confusion Matrix Artifact
    # ===============================
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path)

    # ===============================
    # Feature Importance Artifact
    # ===============================
    importances = model.feature_importances_
    feature_names = X.columns

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x="importance", y="feature", data=fi_df)
    plt.title("Feature Importance")
    plt.tight_layout()

    fi_path = "feature_importance.png"
    plt.savefig(fi_path)
    plt.close()

    mlflow.log_artifact(fi_path)

    # ===============================
    # Save Model
    # ===============================
    model_path = "diabetes_model.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

print("Training selesai dan berhasil dicatat ke MLflow (DagsHub)")
