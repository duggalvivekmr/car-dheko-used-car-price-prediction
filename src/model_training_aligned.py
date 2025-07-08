import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load aligned data
df = pd.read_csv("data/processed/model_data.csv")

# Define aligned features (same as scaler)
features = [
    "car_links",
    "city",
    "new_car_detail_0_it",
    "new_car_detail_1_ft",
    "new_car_detail_2_bt",
    "new_car_detail_3_km",
    "new_car_detail_4_transmission",
    "new_car_detail_5_ownerNo",
    "new_car_detail_6_owner",
    "new_car_detail_7_oem",
    "new_car_detail_8_model",
    "new_car_detail_9_modelYear",
    "new_car_detail_10_centralVariantId",
    "new_car_detail_11_variantName",
    "new_car_detail_13_priceActual",
    "new_car_detail_14_priceSaving",
    "new_car_detail_15_priceFixedText",
    "car_age",
    "new_car_overview_1_top_Year of Manufacture",
    "new_car_specs_1_top_Seats"
]

# Set target
target = "new_car_detail_12_price"

# Clean and split
X = df[features].fillna(0)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train and save model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/best_random_forest_model.pkl")

print("✅ Model trained with correct feature set and saved.")


