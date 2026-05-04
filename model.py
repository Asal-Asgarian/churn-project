import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# 1. Load data
# =========================
df = pd.read_csv("churn_data.csv")

# =========================
# 2. Features & Target
# =========================
X = df[["tenure", "monthly_charges", "support_calls"]]
y = df["churn"]

# =========================
# 3. Train/Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 4. Model
# =========================
model = RandomForestClassifier(n_estimators=100, random_state=42)

# =========================
# 5. Train
# =========================
model.fit(X_train, y_train)

# =========================
# 6. Predict
# =========================
y_pred = model.predict(X_test)

# =========================
# 7. Evaluation
# =========================
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("=== Model Performance ===")
print(f"Accuracy: {accuracy:.2f}")

print("\nConfusion Matrix:")
print(cm)

# =========================
# 8. New Customer Prediction
# =========================
new_customer = pd.DataFrame([{
    "tenure": 10,
    "monthly_charges": 80,
    "support_calls": 6
}])

prediction = model.predict(new_customer)
print("\nPrediction for new customer:", prediction)

# =========================
# 9. Feature Importance
# =========================
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(importance)

# =========================
# 10. Visualization
# =========================

# Confusion Matrix Plot
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix.png")
plt.close()

# Feature Importance Plot
plt.figure(figsize=(6,4))
sns.barplot(x="Importance", y="Feature", data=importance)
plt.title("Feature Importance")

plt.savefig("outputs/feature_importance.png")
plt.close()