import os

import pandas as pd
import streamlit as st
from joblib import load
from sklearn.pipeline import Pipeline


# 🚀 NYC Housing Price Predictor App
def load_model(model_path: str) -> Pipeline:
    """Load the trained XGBoost model from disk using joblib."""
    with st.spinner("Loading model..."):
        return load(model_path)


def get_user_input() -> pd.DataFrame:
    """Collect user input from the sidebar and return it as a DataFrame."""
    st.sidebar.header("🏡 Property Details")

    borough = st.sidebar.selectbox("Borough", options=[1, 2, 3, 4, 5])

    tax_class = st.sidebar.selectbox("Tax Class at Time of Sale", options=[1.0, 2.0, 4.0])

    year_built = st.sidebar.slider("Year Built", min_value=1800, max_value=2025, value=1935)

    gross_sqft = st.sidebar.number_input(
        "Gross Square Feet", min_value=100, max_value=700000, value=3000
    )

    land_sqft = st.sidebar.number_input(
        "Land Square Feet", min_value=100, max_value=3500000, value=1500
    )

    residential_units = st.sidebar.number_input(
        "Residential Units", min_value=0, max_value=800, value=2
    )

    commercial_units = st.sidebar.number_input(
        "Commercial Units", min_value=0, max_value=175, value=0
    )

    user_data = pd.DataFrame.from_dict(
        {
            "BOROUGH": [borough],
            "TAX CLASS AT TIME OF SALE": [tax_class],
            "YEAR BUILT": [year_built],
            "GROSS SQUARE FEET": [gross_sqft],
            "LAND SQUARE FEET": [land_sqft],
            "RESIDENTIAL UNITS": [residential_units],
            "COMMERCIAL UNITS": [commercial_units],
        }
    )
    return user_data


def main() -> None:
    st.set_page_config(page_title="NYC House Price Predictor 🏠", page_icon="📊")

    st.image("4-Deployment/ts-nyc.jpg")

    title1 = "Estimate the sale price of a property in New York City"
    title2 = " using our trained machine learning model."

    st.title("📊 NYC House Price Estimator")
    st.write(title1 + title2)

    # Load model
    model_path = os.path.join("4-Deployment", "first_basic_model.joblib")
    model = load_model(model_path)

    # User input
    input_df = get_user_input()

    # Prediction
    prediction = model.predict(input_df)[0]

    # Values
    low = 500000
    mid = 1500000

    # Result display
    st.markdown("---")
    st.subheader("💵 Predicted Sale Price")
    if prediction < low:
        st.error(f"Estimated Price: ${prediction:,.0f} 😬 That's quite affordable for NYC.")
    elif prediction < mid:
        st.warning(f"Estimated Price: ${prediction:,.0f} 🏙️ Mid-range NYC property.")
    else:
        st.success(f"Estimated Price: ${prediction:,.0f} 💎 You're in luxury territory!")

    st.markdown("---")
    st.caption("Model: XGBoost | Data source: NYC Rolling Sales Dataset")


if __name__ == "__main__":
    main()
