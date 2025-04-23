import os
import pandas as pd
import streamlit as st
from joblib import load
from sklearn.pipeline import Pipeline

# 🚀 Load Model
@st.cache_resource
def load_model(model_path: str) -> Pipeline:
    with st.spinner("Loading model..."):
        return load(model_path)

# 🧍 Individual Prediction
def get_user_input_inside_tab() -> pd.DataFrame:
    st.header("🏡 Property Details")

    BOROUGH_MAPPING = {
        "Manhattan": 1,
        "Bronx": 2,
        "Brooklyn": 3,
        "Queens": 4,
        "Staten Island": 5,
    }

    col1, col2 = st.columns(2)

    with col1:
        borough_name = st.selectbox(
            "Borough", 
            options=list(BOROUGH_MAPPING.keys()),
            help="Administrative division of NYC where the property is located"
        )
        borough = BOROUGH_MAPPING[borough_name]

        tax_class = st.selectbox(
            "Tax Class at Time of Sale", 
            options=[1.0, 2.0, 4.0],
            help="Classification used to determine applicable property tax rates"
        )
        year_built = st.slider(
            "Year Built", 
            min_value=1800, max_value=2025, value=1935,
            help="The year the building was originally constructed"
        )

    with col2:
        gross_sqft = st.number_input(
            "Gross Square Feet", 
            min_value=100, max_value=700000, value=3000,
            help="Total floor area of the building (including all floors)"
        )
        land_sqft = st.number_input(
            "Land Square Feet", 
            min_value=100, max_value=3500000, value=1500,
            help="Total land area that the property occupies"
        )
        commercial_units = st.number_input(
            "Commercial Units", 
            min_value=0, max_value=175, value=0,
            help="Number of commercial spaces in the building"
        )
        residential_units = st.number_input(
            "Residential Units", 
            min_value=0, max_value=800, value=2,
            help="Number of residential apartments or housing units"
        )

    return pd.DataFrame.from_dict({
        "BOROUGH": [borough],
        "TAX CLASS AT TIME OF SALE": [tax_class],
        "YEAR BUILT": [year_built],
        "GROSS SQUARE FEET": [gross_sqft],
        "LAND SQUARE FEET": [land_sqft],
        "RESIDENTIAL UNITS": [residential_units],
        "COMMERCIAL UNITS": [commercial_units],
    })


def individual_prediction_tab(model: Pipeline):
    zero = 0
    low = 500000
    mid = 1500000

    input_df = get_user_input_inside_tab()
    prediction = max(model.predict(input_df)[0], 0)

    st.subheader("💵 Predicted Sale Price")
    if prediction == 0:
        st.error("⚠️ The model predicted a price of $0. " 
                 + "This result is invalid or unreliable. Please review the input values.")
    elif prediction < low:
        st.error(f"Estimated Price: ${prediction:,.0f} 😬 That's quite affordable for NYC.")
    elif prediction < mid:
        st.warning(f"Estimated Price: ${prediction:,.0f} 🏙️ Mid-range NYC property.")
    else:
        st.success(f"Estimated Price: ${prediction:,.0f} 💎 You're in luxury territory!")

# 📦 Batch Prediction
def preprocess_batch_data(df: pd.DataFrame) -> pd.DataFrame:
    for col in [
        "BOROUGH", "TAX CLASS AT TIME OF SALE", "YEAR BUILT",
        "GROSS SQUARE FEET", "LAND SQUARE FEET", "RESIDENTIAL UNITS", "COMMERCIAL UNITS"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def batch_prediction_tab(model: Pipeline):
    st.subheader("📤 Upload your CSV file with property data")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("🔍 Preview of uploaded data:")
            st.dataframe(df.head())

            required_cols = [
                "BOROUGH", "TAX CLASS AT TIME OF SALE", "YEAR BUILT",
                "GROSS SQUARE FEET", "LAND SQUARE FEET", "RESIDENTIAL UNITS", "COMMERCIAL UNITS"
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.warning(f"⚠️ Missing columns: {', '.join(missing_cols)}")
            elif st.button("Predict Prices"):
                df_clean = preprocess_batch_data(df)
                predictions = model.predict(df_clean)
                cleaned_predictions = [max(p, 0) for p in predictions.astype(int)]
                df_result = df.copy()
                df_result["Predicted Price"] = cleaned_predictions

                # 🔍 show warning if there is any zero
                if any(p == 0 for p in cleaned_predictions):
                    st.warning("⚠️ One or more predictions resulted in $0. These values may be invalid. "
                            "Please check the input data for potential inconsistencies.")

                st.success("✅ Predictions completed!")
                st.dataframe(df_result)
                st.download_button(
                    label="💾 Download results as CSV",
                    data=df_result.to_csv(index=False),
                    file_name="nyc_price_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"❌ Error processing the file: {e}")
    else:
        st.info("Please upload a CSV file with property information.")
        st.caption("Sample format:")
        st.dataframe(pd.DataFrame({
            "BOROUGH": [1, 2, 5],
            "TAX CLASS AT TIME OF SALE": [1, 2, 1],
            "YEAR BUILT": [1930, 2001, 1989],
            "GROSS SQUARE FEET": [3000, 1500, 1000],
            "LAND SQUARE FEET": [1500, 800, 800],
            "RESIDENTIAL UNITS": [2, 1, 3],
            "COMMERCIAL UNITS": [0, 1, 0],
        }))

# 🧠 Main
def main():
    st.set_page_config(page_title="NYC Price Estimator", page_icon="📊")
    st.image("4-Deployment/ts-nyc.jpg")

    st.title("📊 NYC House Price Estimator")
    st.write("Estimate the sale price of a property in New York City using our trained machine learning model.")

    model_path = os.path.join("4-Deployment", "first_basic_model.joblib")
    model = load_model(model_path)

    tab1, tab2 = st.tabs(["Individual Prediction", "Batch Prediction"])

    with tab1:
        individual_prediction_tab(model)
    with tab2:
        batch_prediction_tab(model)

if __name__ == "__main__":
    main()