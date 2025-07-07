# model_training.py
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Load preprocessed data
df = pd.read_csv("data/processed/preprocessed_data.csv")

# Drop unwanted columns
drop_candidates = [
    'new_car_overview_1_top_Fuel Type', 'new_car_detail_17_trendingText.heading', 'new_car_detail_18_trendingText.desc', 
    'new_car_specs_1_top_Max Power', 'new_car_overview_1_top_Registration Year', 'new_car_feature_1_top_item_4', 
    'new_car_overview_1_top_Kms Driven', 'new_car_feature_1_top_item_3', 'new_car_specs_1_top_Engine', 
    'new_car_overview_1_top_RTO', 'new_car_overview_1_top_Engine Displacement', 'new_car_specs_2_data_Miscellaneous', 
    'new_car_feature_2_data_Interior', 'new_car_feature_0_heading', 'new_car_feature_1_top_item_8', 
    'new_car_overview_1_top_Transmission', 'new_car_feature_1_top_item_7', 'new_car_feature_2_data_Entertainment & Communication', 
    'new_car_feature_2_data_Comfort & Convenience', 'new_car_feature_2_data_Exterior', 'new_car_detail_16_trendingText.imgUrl', 
    'new_car_specs_1_top_Mileage', 'new_car_specs_2_data_Dimensions & Capacity', 'new_car_specs_2_data_Engine and Transmission', 
    'new_car_feature_3_commonIcon', 'new_car_overview_1_top_Ownership', 'new_car_feature_1_top_item_6', 
    'new_car_feature_1_top_item_2', 'new_car_overview_1_top_Seats', 'new_car_overview_1_top_Insurance Validity', 
    'new_car_specs_3_commonIcon', 'new_car_specs_1_top_Wheel Size', 'new_car_overview_2_bottomData', 
    'new_car_feature_1_top_item_0', 'new_car_specs_0_heading', 'new_car_specs_1_top_Torque', 
    'new_car_feature_1_top_item_1', 'new_car_feature_1_top_item_5', 'new_car_overview_0_heading', 
    'new_car_feature_2_data_Safety'
    ] # Add the same drop list from modeling.ipynb
df = df.drop(columns=drop_candidates)

# Split features and target
target = 'new_car_detail_12_price'
X_raw = df.drop(columns=[target])
y = df[target]

# Drop columns with all missing values before imputation
x_non_allnan = X_raw.dropna(axis=1, how='all')

# Impute missing values
X_raw = X_raw.dropna(axis=1, how='all')
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and test set
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/best_random_forest_model.pkl")
X_test.to_csv("data/processed/X_test.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)
print("✅ Model trained and saved to 'model/best_random_forest_model.pkl'")
