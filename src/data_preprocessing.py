# data preprocessing.py
import pandas as pd # import pandas for data manipulation
import numpy as np  # import necessary libraries
import json # import JSON for handling nested dictionaries  
import ast  # import ast for safely evaluating string representations of Python objects
import os # import os for file handling
from pandas import json_normalize # import json_normalize for flattening nested JSON structures
import joblib  # for saving model artifacts like scalers and encoders
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler # import necessary preprocessing tools
from sklearn.impute import SimpleImputer # import SimpleImputer for handling missing values
import re # import regular expressions for text processing

# -------------------------------------------------------------------- 
# Utility function to flatten nested dictionaries in DataFrame columns  
# --------------------------------------------------------------------
from preprocessing_utils import (
    parse_stringified_dicts,
    flatten_dict_columns,
    extract_feature_flags, 
    extract_top_features,
    extract_spec_fields,
    auto_flatten_nested_columns
)

# ------------- 1. Load and concatenate the city datasets-------------
def load_and_concate_datas(data_dir):
    all_dfs = [] # Initialize an empty list to store DataFrames 
    for file in os.listdir(data_dir):
        if file.endswith('.xlsx'):
            city = file.split('_')[0].capitalize()
            file_path = os.path.join(data_dir, file)
            df = pd.read_excel(file_path) # Read each Excel file into a DataFrame
            df['city'] = city # Add a new column for the city name
            all_dfs.append(df) # Append the DataFrame to the list
    return pd.concat(all_dfs, ignore_index=True) # Concatenate all DataFrames into a single DataFrame

# -------------- 2. Clean and standardize columns --------------
def clean_and_standardize_columns(df):
    # Remove units and standatdize km field
    if 'km' in df.columns:
        df['km'] = df['km'].astype(str).str.replace(r'[^0-9]', '',regex=True)
        df['km'] = pd.to_numeric    (df['km'], errors='coerce') # Convert km to numeric, coercing errors to NaN
    # Remove units and standardize price field
    if 'price' in df.columns:
        df['price'] = df['price'].astype(str).str.replace(r'[^\d]', '', regex=True)
        df['price'] = pd.to_numeric(df['price'], errors='coerce') # Convert price to numeric, coercing errors to NaN
    return df # Return the cleaned DataFrame

# -------------- 3. Handle missing values -----------------------
def impute_missing_values(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns # Select numeric columns  
    cat_cols = df.select_dtypes(include=['object']).columns # Select categorical columns    

    for col in num_cols:
        imputer = SimpleImputer(strategy='median')
        imputed = imputer.fit_transform(df[[col]])
        if imputed.shape[0] == len(df):
            df[col] = imputed.ravel()  # Impute missing values with median  
        else:
            raise ValueError(f"Imputation size mismatch in numeric column: {col}")
    
    for col in cat_cols: 
        if df[col].dropna().apply(lambda x: isinstance(x, (list, dict))).any():
            print(f"⚠️ Skipping '{col}' during imputation — unhashable types.")
            continue
        imputer = SimpleImputer(strategy='most_frequent')
        imputed = imputer.fit_transform(df[[col]])
        if imputed.shape[0] == len(df):  # Ensure the imputed values match the original DataFrame length
            df[col] = imputed.ravel()  # Impute missing values with the most frequent value
        else:
            raise ValueError(f"Imputation size mismatch in categorical column: {col}")  
    return df  # Return the DataFrame with imputed values

#----------------- 4. Encode the categorical variables --------------
def encode_features(df):  
    label_encoders = {}  # Dictionary to store label encoders  
    for col in df.select_dtypes(include=['object']).columns:  # Iterate over categorical columns  
        # skip columns with list/dict types
        if df[col].dropna().apply(lambda x: isinstance(x, (list, dict))).any():
            print(f"⚠️ Skipping '{col}' during encoding — unhashable types.")
            continue

        if df[col].nunique() > 1: # skip constant fields or non-only  
            le = LabelEncoder()  
            df[col] = le.fit_transform(df[col].astype(str)) # Fit and transform the column with label encoding  
            label_encoders[col] = le    # Store the label encoder for each column
    return df, label_encoders  # Return the DataFrame and the label encoders

#----------------- 5. Normalize the numeric features ----------------
def normalize_features(df, feature_columns):    
    scaler = MinMaxScaler()  # Create a MinMaxScaler instance
    if feature_columns:
        if not feature_columns:
            raise ValueError("No feature columns provided for normalization.")
    # Optional: Check if all columns exist in dataframe
    missing = [col for col in feature_columns if col not in df.columns]
    if missing: 
        raise ValueError(f"Feature columns {missing} not found in DataFrame.")
    df[feature_columns] = scaler.fit_transform(df[feature_columns])  # Normalize the specified feature columns
    return df, scaler  # Return the DataFrame and the scaler    

#------------------ 6. Remove outliers -------------------
def remove_outliers(df, column):
    q1 = df[column].quantile(0.25)  # Calculate the first quartile
    q3 = df[column].quantile(0.75)  # Calculate the third quartile
    iqr = q3 - q1  # Calculate the interquartile range
    return df[~((df[column] < (q1 - 1.5 *iqr)) | (df[column] > (q3 + 1.5 * iqr)))] # Return the DataFrame without outliers

#------------------ 7. Main pipeline function -------------------  
def run_preprocessing_pipeline(data_dir, feature_columns):
    df = load_and_concate_datas(data_dir)  # Load and concatenate the datasets

    # Parse and flatten nested dictionary lists/columns
    nested_columns = ['new_car_detail', 'new_car_overview', 'new_car_feature', 'new_car_specs']
    df = parse_stringified_dicts(df, nested_columns)  # Parse string fields to dictionaries
    df.dropna(axis=1, how='all', inplace=True)

    # Drop columns where more than 95% of the data is missing
    threshold = 0.95
    df = df.loc[:, df.isnull().mean() < threshold]
    null_stats = df.isnull().mean().sort_values(ascending=False)
    print("📉 Columns with high missing percentage:")
    print(null_stats[null_stats > 0.80])


    # ✅ Flatten those parsed dictionaries 
    for col in nested_columns:
        if col in df.columns:
            flattened = flatten_dict_columns(df, col)
            df = pd.concat([df, flattened], axis=1).reset_index(drop=True  ) # Concatenate the flattened DataFrame with the original DataFrame

    df.drop(columns=nested_columns, inplace=True, errors='ignore')  # Drop the original nested columns        
    print("✅ Columns after flattening:", df.columns.tolist())
    print("✅ Sample data:")
    print(df.head(3))
    df.to_csv("intermediate_debug_flattened.csv", index=False)


    # Handle top-level nested structures specifically  
    if 'new_car_specs_top' in df.columns:
        df_specs_top = df['new_car_specs_top'].apply(extract_top_features).apply(pd.Series)
        df = pd.concat([df, df_specs_top], axis=1)
        df.reset_index(drop=True, inplace=True)
        df.drop(columns=['new_car_specs_top'], inplace=True)
    
    if 'new_car_overview_top' in df.columns:
        df_overview_top = df['new_car_overview_top'].apply(extract_top_features).apply(pd.Series)
        df = pd.concat([df, df_overview_top], axis=1)
        df.reset_index(drop=True, inplace=True)
        df.drop(columns=['new_car_overview_top'], inplace=True)

    if 'new_car_feature_top' in df.columns:
        df_feature_top = df['new_car_feature_top'].apply(extract_feature_flags).apply(pd.Series)
        df = pd.concat([df, df_feature_top], axis=1)  # Concatenate the new feature flags to the DataFrame
        df.reset_index(drop=True, inplace=True)

    if 'new_car_specs_data' in df.columns:
        df_specs = df['new_car_specs_data'].apply(extract_spec_fields).apply(pd.Series)
        df = pd.concat([df, df_specs], axis=1)
        df.reset_index(drop=True, inplace=True)

    if 'new_car_feature_data' in df.columns:
        df_feature = df['new_car_feature_data'].apply(extract_spec_fields).apply(pd.Series)
        df = pd.concat([df, df_feature], axis=1)
        df.reset_index(drop=True, inplace=True)
    
    # Drop redundant nested columns if present
    for col in ['new_car_detail', 'new_car_overview', 'new_car_feature', 'new_car_specs', 
                'new_car_specs_data', 'new_car_feature_data', 'new_car_specs_top', 
                'new_car_overview_top', 'new_car_feature_top']:
        if col in df.columns:
            df.drop(columns=col, inplace=True, errors='ignore')

    # --- Add car_age feature ---
    if 'new_car_detail_9_modelYear' in df.columns:
        df['car_age'] = 2025 - df['new_car_detail_9_modelYear']  # Assuming current year is 2025
        print("🛠️ Added 'car_age' feature from new_car_detail_9_modelYear")
    else:
        print("⚠️ 'new_car_detail_9_modelYear' column not found. 'car_age' not created.")

    df = clean_and_standardize_columns(df)  # Clean and standardize the columns
    print(f"🧪 Checking duplicate columns: {df.columns[df.columns.duplicated()].tolist()}")
    print(f"🧪 Shape before imputation: {df.shape}")
    df = df.loc[:, ~df.columns.duplicated()] # Remove duplicate columns if any
    df = impute_missing_values(df)  # Handle missing values
    df, label_encoders = encode_features(df)  # Encode categorical features    
    if 'price' in df.columns: # Remove outliers from the price column if it exists
        df = remove_outliers(df, 'price')
    # Auto-select feature columns if not provided
    if not feature_columns:
        feature_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        print("✅ Auto-selected feature columns for normalization:", feature_columns)

    if not feature_columns:
        raise ValueError("❌ No numeric feature columns found to normalize.")

    df = auto_flatten_nested_columns(df)
    df, scaler = normalize_features(df, feature_columns)  # Normalize numeric features
    # Save scaler and label encoders for later use (e.g., during inference)
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(label_encoders, 'model/label_encoders.pkl')

    return df # Return the processed DataFrame 

#------------------- 8. Example Entry Point -------------------
# This section is for running the preprocessing pipeline directly from this script

if __name__ == '__main__':
        # Load data to determine feature columns
    temp_df = load_and_concate_datas(os.path.join('.', 'data', 'raw'))
    # Example: select all numeric columns for normalization
    feature_columns = None
    final_df = run_preprocessing_pipeline(os.path.join('.', 'data', 'raw'), feature_columns=None)
    final_df.to_csv('./data/processed/preprocessed_data.csv', index=False)
    print("\u2705 Preprocessing complete. Output saved to preprocessed_data.csv")
    print("Available Columns:")
    print(final_df.columns.tolist())  # Print the available columns in the final DataFrame  