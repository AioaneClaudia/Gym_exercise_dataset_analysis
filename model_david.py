import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

#  =============================
#  RANDOM FOREST REGRESSOR
#  =============================
#
# ==============================
# 1. LOAD DATA
# ==============================

X_train = pd.read_csv("data/X_train.csv")
X_test  = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test  = pd.read_csv("data/y_test.csv").squeeze()

print("Date incarcate cu succes!")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# # ==============================
# # 2. TRAIN MODEL
# # ==============================

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

print("Modelul a fost antrenat cu succes!")

# # ==============================
# # 3. PREDICT
# # ==============================
#
y_pred = model.predict(X_test)

print("First 5 predictions:", [f"{x:.2f}" for x in y_pred[:5]])
print("First 5 actual values:", [f"{x:.2f}" for x in y_test.values[:5]])

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

mae  = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# # ==============================
# # 5. FEATURE IMPORTANCE
# # ==============================
#
# # proprietate RandomForest, modelul alege o variabila care reduce cel mai mult eroarea (split DecisionTree)
importances = model.feature_importances_
feature_names = X_train.columns

feat_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nFeature Importances:")
print(feat_df)

plt.figure(figsize=(10, 6))
plt.barh(feat_df['Feature'], feat_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance - Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
#
y_pred_rounded = np.round(y_pred).astype(int)

plt.figure(figsize=(8, 6))
jitter = np.random.uniform(-0.1, 0.1, size=len(y_test))
plt.scatter(y_test + jitter, y_pred_rounded + jitter, alpha=0.5, color='steelblue')
plt.plot([2, 5], [2, 5], color='red', linewidth=2, linestyle='--')
plt.xlabel('Valori Reale')
plt.ylabel('Valori Prezise')
plt.title('Predictii vs Valori Reale - Random Forest')
plt.tight_layout()
plt.show()
# #
# ==============================
#  7. OVERFITTING CHECK
#  ==============================
#
y_pred_train = model.predict(X_train)

r2_train = r2_score(y_train, y_pred_train)
r2_test  = r2_score(y_test, y_pred)

print(f"R² Train: {r2_train:.4f}")
print(f"R² Test:  {r2_test:.4f}")
print(f"Diferenta: {r2_train - r2_test:.4f}")

# ==============================
# KNN CLASSIFIER
# ==============================

# ==============================
# 1. TRAIN MODEL
# ==============================
knn = KNeighborsClassifier(n_neighbors=40)
knn.fit(X_train, y_train)

print("KNN succsessfully trained!")

y_pred_knn = knn.predict(X_test)

print("First 5 predictions:", y_pred_knn[:5])
print("First 5 real values:", y_test.values[:5])

# ==============================
# 3. EVALUATE
# ==============================

accuracy = accuracy_score(y_test, y_pred_knn)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))

# ==============================
# 4. GRAFIC
# ==============================

jitter = np.random.uniform(-0.1, 0.1, size=len(y_test))
plt.figure(figsize=(8, 6))
plt.scatter(y_test + jitter, y_pred_knn + jitter, alpha=0.5, color='steelblue')
plt.plot([2, 5], [2, 5], color='red', linewidth=2, linestyle='--')
plt.xlabel('Valori Reale')
plt.ylabel('Valori Prezise')
plt.title('Predictii vs Valori Reale - KNN Classifier')
plt.tight_layout()
plt.show()

# ==============================
# 5. OVERFITTING CHECK
# ==============================

y_pred_knn_train = knn.predict(X_train)

acc_train = accuracy_score(y_train, y_pred_knn_train)
acc_test  = accuracy_score(y_test, y_pred_knn)

print(f"Accuracy Train: {acc_train:.4f}")
print(f"Accuracy Test:  {acc_test:.4f}")
print(f"Diferenta: {acc_train - acc_test:.4f}")