# predictor.py

import joblib
import pandas as pd


# Load model artifacts
model = joblib.load("model/best_random_forest_model.pkl")
scaler = joblib.load("model/scaler.pkl")
ordinal_encoders = joblib.load("model/ordinal_encoders.pkl")

# Categorical columns that need ordinal encoding (only non-numeric, string-based)
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

def prepare_input(raw_input: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw_input])

    # Apply ordinal encoding to categorical columns
    for col in categorical_cols:
        if col in df.columns and col in ordinal_encoders:
            encoder = ordinal_encoders[col]
            df[col] = df[col].astype(str).apply(lambda x: x if x in encoder.categories_[0] else '__unknown__')
            if '__unknown__' not in encoder.categories_[0]:
                encoder.categories_ = [encoder.categories_[0].tolist() + ['__unknown__']]
            df[col] = encoder.transform(df[[col]])
        else:
            print(f"⚠️ Warning: Column '{col}' missing from input or encoders.")

    # Fill missing numeric columns expected by scaler 
    for col in scaler.feature_names_in_:
        if col not in df.columns:
            df[col] = 0.0   # Defalt fallback
    
    print("🧪 scaler.feature_names_in_:", scaler.feature_names_in_.tolist())
    print("📦 input.columns:", df.columns.tolist())

    # Transform numerical features
    df = df[scaler.feature_names_in_]
    df[df.columns] = scaler.transform(df)
    return df

def predict_price(raw_input: dict) -> float:
    X = prepare_input(raw_input)
    return model.predict(X)[0]
