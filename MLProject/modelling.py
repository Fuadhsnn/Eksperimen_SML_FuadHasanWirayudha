# ===============================
# MLflow Manual Logging
# ===============================

# Log parameters
for param, value in best_params.items():
    mlflow.log_param(param, value)

# Log metrics
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("precision", precision)
mlflow.log_metric("recall", recall)
mlflow.log_metric("f1_score", f1)

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
