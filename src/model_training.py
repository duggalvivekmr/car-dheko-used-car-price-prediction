import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Paths
data_path = "data/processed/preprocessed_data.csv"
feature_path = "model/features.pkl"
model_path = "model/best_random_forest_model.pkl"

# Load data and features
df = pd.read_csv(data_path)
features = joblib.load(feature_path)
target = "new_car_detail_12_price"

# Ensure all features are present
missing = [f for f in features if f not in df.columns]
if missing: 
    raise ValueError(f"Missing features in DataFrame: {missing}")   

# Prepare data
X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Grid search for hyperparameter tuning
print("🔍 Starting hyperparameter tuning...with GridSearchCV")
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f"✅ Best model found with parameters: {grid_search.best_params_}")

# Evaluate model
y_pred = best_model.predict(X_test) 
print("\n📊 Model Evaluation Metrics:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Feature importance
importances = best_model.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# Plot top 15 features
plt.figure(figsize=(10, 6))
feat_imp[:15].plot(kind='barh')
plt.xlabel('Feature Importance')
plt.title('Top 15 Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('model/feature_importance.png')
plt.show()

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(best_model, model_path)
print(f"\n✅ Best model trained and saved to '{model_path}'")
