import pandas as pd
from joblib import load


def test_model_prediction_non_negative() -> None:
    model = load("4-Deployment/first_basic_model.joblib")

    test_inputs = [
        {
            "BOROUGH": 2,
            "TAX CLASS AT TIME OF SALE": 1.0,
            "YEAR BUILT": 1935,
            "GROSS SQUARE FEET": 1000,
            "LAND SQUARE FEET": 500,
            "COMMERCIAL UNITS": 10,
            "RESIDENTIAL UNITS": 0,
        },
        {
            "BOROUGH": 2,
            "TAX CLASS AT TIME OF SALE": 1.0,
            "YEAR BUILT": 1935,
            "GROSS SQUARE FEET": 1000,
            "LAND SQUARE FEET": 500,
            "COMMERCIAL UNITS": 10,
            "RESIDENTIAL UNITS": 4,
        },
        {
            "BOROUGH": 2,
            "TAX CLASS AT TIME OF SALE": 1.0,
            "YEAR BUILT": 1935,
            "GROSS SQUARE FEET": 1000,
            "LAND SQUARE FEET": 500,
            "COMMERCIAL UNITS": 10,
            "RESIDENTIAL UNITS": 7,
        },
        {
            "BOROUGH": 2,
            "TAX CLASS AT TIME OF SALE": 1.0,
            "YEAR BUILT": 1935,
            "GROSS SQUARE FEET": 1000,
            "LAND SQUARE FEET": 500,
            "COMMERCIAL UNITS": 10,
            "RESIDENTIAL UNITS": 8,
        },
        {
            "BOROUGH": 2,
            "TAX CLASS AT TIME OF SALE": 1.0,
            "YEAR BUILT": 1935,
            "GROSS SQUARE FEET": 1000,
            "LAND SQUARE FEET": 500,
            "COMMERCIAL UNITS": 10,
            "RESIDENTIAL UNITS": 10,
        },
        {
            "BOROUGH": 2,
            "TAX CLASS AT TIME OF SALE": 1.0,
            "YEAR BUILT": 1938,
            "GROSS SQUARE FEET": 1000,
            "LAND SQUARE FEET": 500,
            "COMMERCIAL UNITS": 10,
            "RESIDENTIAL UNITS": 4,
        },
        {
            "BOROUGH": 2,
            "TAX CLASS AT TIME OF SALE": 1.0,
            "YEAR BUILT": 1946,
            "GROSS SQUARE FEET": 1000,
            "LAND SQUARE FEET": 500,
            "COMMERCIAL UNITS": 10,
            "RESIDENTIAL UNITS": 4,
        },
        {
            "BOROUGH": 2,
            "TAX CLASS AT TIME OF SALE": 1.0,
            "YEAR BUILT": 2005,
            "GROSS SQUARE FEET": 1000,
            "LAND SQUARE FEET": 500,
            "COMMERCIAL UNITS": 10,
            "RESIDENTIAL UNITS": 4,
        },
        {
            "BOROUGH": 3,
            "TAX CLASS AT TIME OF SALE": 1.0,
            "YEAR BUILT": 2005,
            "GROSS SQUARE FEET": 1000,
            "LAND SQUARE FEET": 500,
            "COMMERCIAL UNITS": 10,
            "RESIDENTIAL UNITS": 4,
        },
        {
            "BOROUGH": 5,
            "TAX CLASS AT TIME OF SALE": 1.0,
            "YEAR BUILT": 2005,
            "GROSS SQUARE FEET": 1000,
            "LAND SQUARE FEET": 500,
            "COMMERCIAL UNITS": 10,
            "RESIDENTIAL UNITS": 4,
        },
    ]

    for case in test_inputs:
        df = pd.DataFrame([case])
        prediction = model.predict(df)[0]
        assert prediction >= 0, f"Negative prediction {prediction} for input: {case}"
