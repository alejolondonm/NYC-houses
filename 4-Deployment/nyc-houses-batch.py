import os
import pandas as pd
import streamlit as st
from joblib import load
from sklearn.pipeline import Pipeline

def load_model(model_file_path: str) -> Pipeline:
    with st.spinner("Loading model..."):
        return load(model_file_path)

def preprocess_batch_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["BOROUGH"] = pd.to_numeric(df["BOROUGH"], errors="coerce")
    df["TAX CLASS AT TIME OF SALE"] = pd.to_numeric(df["TAX CLASS AT TIME OF SALE"], errors="coerce")
    df["YEAR BUILT"] = pd.to_numeric(df["YEAR BUILT"], errors="coerce")
    df["GROSS SQUARE FEET"] = pd.to_numeric(df["GROSS SQUARE FEET"], errors="coerce")
    df["LAND SQUARE FEET"] = pd.to_numeric(df["LAND SQUARE FEET"], errors="coerce")
    df["RESIDENTIAL UNITS"] = pd.to_numeric(df["RESIDENTIAL UNITS"], errors="coerce")
    df["COMMERCIAL UNITS"] = pd.to_numeric(df["COMMERCIAL UNITS"], errors="coerce")
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
                "BOROUGH",
                "TAX CLASS AT TIME OF SALE",
                "YEAR BUILT",
                "GROSS SQUARE FEET",
                "LAND SQUARE FEET",
                "RESIDENTIAL UNITS",
                "COMMERCIAL UNITS",
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.warning(f"⚠️ Missing columns: {', '.join(missing_cols)}")
            elif st.button("Predict Prices"):
                df_clean = preprocess_batch_data(df)
                predictions = model.predict(df_clean)

                df_result = df.copy()
                df_result["Predicted Price"] = predictions.astype(int)

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
        sample_data = pd.DataFrame({
            "BOROUGH": [1, 2],
            "TAX CLASS AT TIME OF SALE": [1, 2],
            "YEAR BUILT": [1930, 2001],
            "GROSS SQUARE FEET": [3000, 1500],
            "LAND SQUARE FEET": [1500, 800],
            "RESIDENTIAL UNITS": [2, 1],
            "COMMERCIAL UNITS": [0, 1],
        })
        st.dataframe(sample_data)

def main():
    st.set_page_config(page_title="Batch Price Prediction", page_icon="📦")
    st.image("4-Deployment/ts-nyc.jpg")
    st.title("📦 Batch NYC Property Price Predictor")
    st.write("Upload a CSV file to estimate sale prices in batch.")

    model_path = os.path.join("4-Deployment", "first_basic_model.joblib")
    model = load_model(model_path)

    batch_prediction_tab(model)

if __name__ == "__main__":
    main()