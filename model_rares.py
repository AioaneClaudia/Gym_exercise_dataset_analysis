import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import preprocessing
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report
)
import matplotlib.pyplot as plt

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()

# ==============================
# 1. LINEAR REGRESSION
# ==============================

# separate features and target variable
feature_name = 'Calories_Burned'
features = X_train[[feature_name]]
target = y_train

# create and fit a linear regression
workout_estimator = LinearRegression()
workout_estimator.fit(features, target)

# plot the original values
plt.scatter(features, target, c='green', label='train')
plt.scatter(X_test[[feature_name]], y_test, c='blue', label='test')

# plot the predicted values
plt.plot(features, workout_estimator.predict(features), c='red', label='prediction')

plt.xlabel(feature_name)
plt.ylabel('Workout Frequency (days/week)')
plt.legend()
plt.title(f"Linear Regression: {feature_name} vs Workout Frequency")
plt.show()

# print the model that was fitted (the regression formula)
print(f"Workout_Frequency = {workout_estimator.coef_[0]:.2f} * {feature_name} + {workout_estimator.intercept_:.2f}")

# ==============================
# 2. DECISION TREE CLASSIFIER
# ==============================

# Convert numeric to categories
def categorize(x):
    if x <= 3:
        return "low"      # 1-3 days
    elif x == 4:
        return "medium"   # 4 days
    else:
        return "high"     # 5 days

# Apply categorization
y_train_cat = pd.Series(y_train).apply(categorize)
y_test_cat = pd.Series(y_test).apply(categorize)

# Encode categories to numbers (low=0, medium=1, high=2)
label_encoder = preprocessing.LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_cat)
y_test_encoded = label_encoder.transform(y_test_cat)

# Create decision tree with max depth 4
dtc = DecisionTreeClassifier(max_depth=4, random_state=42)

# Train the model
dtc.fit(X_train, y_train_encoded)

# Make predictions
y_pred_cls = dtc.predict(X_test)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test_encoded, y_pred_cls))
print("\nClassification Report:\n")
print(classification_report(y_test_encoded, y_pred_cls))

# Visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(
    dtc,
    feature_names=X_train.columns,
    class_names=label_encoder.classes_,
    filled=True
)
plt.show()