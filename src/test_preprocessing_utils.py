# file: test_preprocessing_utils.py
import pandas as pd
import os
import pytest
from preprocessing_utils import (
    parse_stringified_dicts, 
    flatten_dict_columns, 
    extract_feature_flags, 
    extract_top_features, 
    extract_spec_fields, 
    auto_flatten_nested_columns
) 

def test_extract_top_features():
    sample = [{"key": "Mileage", "value": "23 kmpl"}, {"key": "Power", "value": "100 hp"}]
    result = extract_top_features(sample)
    assert result["Mileage"] == "23 kmpl"
    assert result["Power"] == "100 hp"
    

def test_extract_feature_flags():
    sample = [{"value": "ABS"}, {"value": "Power Steering"}]
    result = extract_feature_flags(sample)
    assert "ABS" in result and result["ABS"] is True
    assert "Power Steering" in result and result["Power Steering"] is True

def test_extract_spec_fields():
    sample = [
        {"heading": "Engine", "list": [{"key": "Type", "value": "Diesel"}, {"key": "Power", "value": "100hp"}]},
        {"heading": "Dimension", "list": [{"key": "Length", "value": "4000mm"}]}
    ]
    result = extract_spec_fields(sample)
    assert result["Engine_Type"] == "Diesel"
    assert result["Engine_Power"] == "100hp"
    assert result["Dimension_Length"] == "4000mm"

def test_parse_stringified_dicts():
    df = pd.DataFrame({
        'nested': ['{"a": 1, "b": 2}', '{"a": 3, "b": 4}']
    })
    df = parse_stringified_dicts(df, ['nested'])
    assert isinstance(df['nested'][0], dict)
    assert df['nested'][0]['a'] == 1

def test_flatten_dict_columns():
    df = pd.DataFrame({
        'nested': [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
    })
    flattened = flatten_dict_columns(df, 'nested')
    assert any(col.startswith("nested_") and "a" in col for col in flattened.columns)
    assert any(col.startswith("nested_") and "b" in col for col in flattened.columns)

def test_auto_flatten_nested_columns():
    df = pd.DataFrame({
        'info': [[{'key': 'Mileage', 'value': '20 kmpl'}, {'key': 'Power', 'value': '80 hp'}]]
    })
    flattened = auto_flatten_nested_columns(df)
    assert 'info_Mileage' in flattened.columns
    assert flattened['info_Mileage'][0] == '20 kmpl'


