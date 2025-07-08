import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load only model-used columns
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

df = pd.read_csv("data/processed/model_data.csv")  # Use your correct path
scaler = StandardScaler()
scaler.fit(df[features])
joblib.dump(scaler, "model/scaler.pkl")
print("✅ scaler.pkl saved with correct feature set.")