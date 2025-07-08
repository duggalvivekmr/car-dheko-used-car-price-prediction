import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import joblib
import os

# Load your cleaned dataset
df = pd.read_csv("data/processed/model_data.csv")

# Categorical columns to encode
categorical_cols = [
    "city",
    "new_car_detail_1_ft",
    "new_car_detail_2_bt",
    "new_car_detail_4_transmission",
    "new_car_detail_6_owner",
    "new_car_detail_7_oem",
    "new_car_detail_8_model",
    "new_car_detail_11_variantName"
]
# Generate ordinal encoders
ordinal_encoders = {}

for col in categorical_cols:
    if col in df.columns:
        categories = df[col].astype(str).unique().tolist()
        if '__unknown__' not in categories:
            categories.append('__unknown__')
        encoder = OrdinalEncoder(categories=[categories], handle_unknown='use_encoded_value', unknown_value=len(categories))
        df[col] = df[col].astype(str).apply(lambda x: x if x in categories else '__unknown__')
        encoder.fit(df[[col]])
        ordinal_encoders[col] = encoder

# Save to model folder
os.makedirs("model", exist_ok=True)
joblib.dump(ordinal_encoders, "model/ordinal_encoders.pkl")

print("✅ ordinal_encoders.pkl saved successfully.")
