# file: preprocessing_utils.py

import pandas as pd
import ast
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# ----------------- NESTED STRUCTURE HANDLING -----------------
def parse_stringified_dicts(df, columns):  
    for col in columns:  
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)  # Safely evaluate string representations of dictionaries
    return df  # Return the DataFrame with parsed string fields

def flatten_dict_columns(df, column_name):  
    expanded = pd.json_normalize(df[column_name])  # Normalize the nested dictionary column into a flat DataFrame
    expanded.columns = [f"{column_name}_{i}_{subcol}" for i, subcol in enumerate(expanded.columns)] # Flatten the column names to include the original column name and index
    return expanded  # Return the flattened DataFrame

def extract_feature_flags(feature_data): 
    if isinstance(feature_data, list):
        return {item.get("key", f"missing_{i}"): True for i, item in enumerate(feature_data) if isinstance(item, dict)}
    return {}

def extract_top_features(top_data):
    result = {}
    if isinstance(top_data, list):
        for d in top_data:
            if isinstance(d, dict):
               result.update(d)
    return result

def extract_spec_fields(specs_data):  
    result = {}  
    if isinstance(specs_data, list):
        for d in specs_data:  
            if isinstance(d, dict):  
                result.update(d)  # Update the result dictionary with the contents of each dictionary in the list
    return result  # Return the flattened dictionary   

def auto_flatten_nested_columns(df):
    """
    Automatically detects and flattens all columns in the DataFrame
    that contain lists of dictionaries or nested dictionaries.
    """
    import pandas as pd

    def is_nested(vale):
        """
        Check if a value is a list of dictionaries or a nested dictionary.
        """
        return isinstance(vale, (list, dict))
    cols_to_flatten = [
        col for col in df.columns
        if df[col].dropna().apply(is_nested).any()
    ]

    print(f" 🔍 Auto detecting nested columns: {cols_to_flatten}")

    for col in cols_to_flatten:
        sample = df[col].dropna().iloc[0]  # Get a sample value from the column

        if isinstance(sample, list): 
            if all(isinstance(item, dict) for item in sample):
                expanded = df[col].apply(lambda items: {
                    d.get('key') or d.get('heading') or f"item_{i}": d.get('value', True)
                    for i, d in enumerate(items) if isinstance(d, dict)    
                }).apply(pd.Series)  # Flatten the list of dictionaries into separate columns
            else:
                print(f" ⚠️ skipping {col} - unsupported list structure")
                continue  # Skip unsupported list structures

        elif isinstance(sample, dict):
            expanded - df[col].apply(pd.series)
        else:
            continue

        expanded.columns = [f"{col}_{c}" for c in expanded.columns]  # Rename columns to include the original column name 
        df = pd.concat([df.drop(columns=[col]), expanded], axis=1)  # Concatenate the expanded DataFrame with the original DataFrame

    return df