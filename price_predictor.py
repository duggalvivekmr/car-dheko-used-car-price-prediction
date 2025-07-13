# predictor.py

import joblib
import pandas as pd
import numpy as np


# Load model artifacts
model = joblib.load("model/best_random_forest_model.pkl")
features = joblib.load("model/features.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

# Define categorical columns based on available encoders
categorical_cols = list(label_encoders.keys())


def prepare_input(raw_input: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw_input])

    # Apply label encoding to categorical columns
    for col in categorical_cols:
        if col in df.columns:
            encoder = label_encoders[col]
            if df[col].iloc[0] not in encoder.classes_:
                df[col] = '__unknown__'
                if '__unknown__' not in encoder.classes_:
                    encoder.classes_ = np.append(encoder.classes_, '__unknown__')
            df[col] = encoder.transform(df[col].astype(str))
        else:    
            print(f"⚠️ Warning: Missing column '{col}' in input. Filling with 0.")
            df[col] = 0

    # Fill missing numeric columns expected by scaler 
    for col in features:
        if col not in df.columns:
            df[col] = 0.0   # Defalt fallback
    
    # Ensure all features are present
    df = df[features]
    
     # Debug: print features going into model
    print("\n🔍 Features sent to model:")
    for col in df.columns:
        print(f"{col}: {df[col].iloc[0]}")

    return df

def predict_price(raw_input: dict) -> float:
    X = prepare_input(raw_input)
    return model.predict(X)[0]
