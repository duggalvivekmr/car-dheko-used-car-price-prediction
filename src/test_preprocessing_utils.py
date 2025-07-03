# file: test_preprocessing_utils.py
import pytest
import pandas as pd
from preprocessing_utils import (
    parse_stringified_dicts,
    flatten_dict_columns,
    extract_feature_flags,
    extract_top_features,
    extract_spec_fields,
    auto_flatten_nested_columns
)

def test_parse_stringified_dicts():
    df = pd.DataFrame({'col': ['{"a": 1}', '{"b": 2}']})
    df = parse_stringified_dicts(df, ['col'])
    assert isinstance(df['col'][0], dict)
    assert df['col'][0]['a'] == 1

def test_flatten_dict_columns():
    df = pd.DataFrame({'info': [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]})
    expanded = flatten_dict_columns(df, 'info')
    assert list(expanded.columns) == ['info_0_a', 'info_1_b']
    assert expanded.iloc[0, 0] == 1

def test_extract_feature_flags():
    feature_data = [{"key": "Airbags"}, {"key": "ABS"}]
    result = extract_feature_flags(feature_data)
    assert result == {"Airbags": True, "ABS": True}

def test_extract_top_features():
    top_data = [{"year": 2015}, {"fuel": "Petrol"}]
    result = extract_top_features(top_data)
    assert result == {"year": 2015, "fuel": "Petrol"}

def test_extract_spec_fields():
    specs_data = [{"mileage": "20 kmpl"}, {"engine": "1197 cc"}]
    result = extract_spec_fields(specs_data)
    assert result == {"mileage": "20 kmpl", "engine": "1197 cc"}

def test_auto_flatten_nested_columns_dict(monkeypatch):
    df = pd.DataFrame({'col': [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]})
    result = auto_flatten_nested_columns(df)
    assert "col_a" in result.columns
    assert result.iloc[0]["col_a"] == 1

def test_auto_flatten_nested_columns_list_of_dicts():
    df = pd.DataFrame({
        'col': [[{'key': 'ABS', 'value': True}, {'key': 'Airbags', 'value': True}],
                [{'key': 'ABS', 'value': True}]]
    })
    result = auto_flatten_nested_columns(df)
    assert "col_ABS" in result.columns
    assert result.iloc[0]["col_Airbags"] == True


