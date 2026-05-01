# House Price Prediction Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib
import os

# -----------------------------
# STEP 1: CREATE SYNTHETIC DATA
# -----------------------------
np.random.seed(42)

data_size = 500

data = pd.DataFrame({
    "area": np.random.randint(500, 4000, data_size),
    "bedrooms": np.random.randint(1, 6, data_size),
    "bathrooms": np.random.randint(1, 4, data_size),
    "floors": np.random.randint(1, 3, data_size),
    "age": np.random.randint(0, 30, data_size),
    "parking": np.random.randint(0, 2, data_size),
    "price": 50000 
             + np.random.randint(500, 4000, data_size) * 50
             + np.random.randint(1, 6, data_size) * 10000
             - np.random.randint(0, 30, data_size) * 1000
})

# Save dataset
os.makedirs("data", exist_ok=True)
data.to_csv("data/housing_data.csv", index=False)

print("Dataset Created!")
print(data.head())

# -----------------------------
# STEP 2: DATA VISUALIZATION
# -----------------------------
plt.figure()
sns.histplot(data["price"], kde=True)
plt.title("Price Distribution")
plt.savefig("outputs/price_distribution.png")

plt.figure()
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("outputs/correlation.png")

# -----------------------------
# STEP 3: PREPROCESSING
# -----------------------------
X = data.drop("price", axis=1)
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 4: TRAIN MODELS
# -----------------------------
lr_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100)

lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# -----------------------------
# STEP 5: PREDICTIONS
# -----------------------------
lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# -----------------------------
# STEP 6: EVALUATION FUNCTION
# -----------------------------
def evaluate(y_true, y_pred, model_name):
    print(f"\n{model_name} Performance:")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R2 Score:", r2_score(y_true, y_pred))

evaluate(y_test, lr_pred, "Linear Regression")
evaluate(y_test, rf_pred, "Random Forest")

# -----------------------------
# STEP 7: SAVE MODEL
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(rf_model, "models/house_model.pkl")

print("\nModel Saved!")

# -----------------------------
# STEP 8: SAMPLE PREDICTION
# -----------------------------
sample_house = np.array([[2000, 3, 2, 2, 5, 1]])

prediction = rf_model.predict(sample_house)

print("\nSample House Prediction:")
print("Predicted Price:", prediction[0])

# -----------------------------
# STEP 9: ACTUAL VS PREDICTED GRAPH
# -----------------------------
plt.figure()
plt.scatter(y_test, rf_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.savefig("outputs/prediction.png")

print("\nProject Completed Successfully!")
# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
importances = rf_model.feature_importances_
features = X.columns

plt.figure()
plt.barh(features, importances)
plt.title("Feature Importance")
plt.savefig("outputs/feature_importance.png")

# -----------------------------
# MODEL COMPARISON
# -----------------------------
models = ["Linear Regression", "Random Forest"]
scores = [
    r2_score(y_test, lr_pred),
    r2_score(y_test, rf_pred)
]

plt.figure()
plt.bar(models, scores)
plt.title("Model Comparison (R2 Score)")
plt.savefig("outputs/model_comparison.png")


# -----------------------------
# SAVE PREDICTIONS
# -----------------------------
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": rf_pred
})

results.to_csv("outputs/predictions.csv", index=False)
plt.figure()
sns.histplot(data["price"], kde=True)
plt.title("Price Distribution")
plt.savefig("outputs/price_distribution.png")
plt.close()

plt.figure()
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("outputs/correlation.png")
plt.close()

plt.figure()
plt.scatter(y_test, rf_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.savefig("outputs/prediction.png")
plt.close()

importances = rf_model.feature_importances_
features = X.columns

plt.figure()
plt.barh(features, importances)
plt.title("Feature Importance")
plt.savefig("outputs/feature_importance.png")
plt.close()

models = ["Linear Regression", "Random Forest"]
scores = [
    r2_score(y_test, lr_pred),
    r2_score(y_test, rf_pred)
]

plt.figure()
plt.bar(models, scores)
plt.title("Model Comparison (R2 Score)")
plt.savefig("outputs/model_comparison.png")
plt.close()