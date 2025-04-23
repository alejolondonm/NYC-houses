# 📊 ***NYC House Price Estimator — Streamlit App***

This is a Streamlit-based web application that allows users to estimate the **sale price of residential properties in New York City** 🏙️ based on key property features. The app uses a regression model trained with **XGBoost** and wrapped in a `scikit-learn` pipeline.

## 🎯 ***Objective***

This app is designed to:

- 💬 **Collect property features** through an interactive interface.
- 🤖 **Predict housing prices** using a pre-trained model.
- 🧪 **Validate predictions**, ensuring no negative values.
- 🖥️ **Deploy seamlessly** both locally and on Streamlit Cloud.

## 🏗️ ***How It Works***

The app loads a trained model from the file:

```plaintext
4-Deployment/first_basic_model.joblib
```

It supports two modes via tab navigation:

1. 🧍 **Individual Prediction Tab**
   - Users input property details interactively.
   - The app validates and predicts the estimated sale price.

2. 📦 **Batch Prediction Tab**
   - Users upload a `.csv` file with property records.
   - The app returns a downloadable file with predicted prices.
   - Invalid or suspicious results (like 0) are flagged.

Each tab uses the same underlying model and preprocessing pipeline.

***This README will be updated once the PR is approved, as access routes change!!!***

Then, it collects user inputs like:

- Borough
- Tax Class at Time of Sale
- Year Built
- Number of Units

And uses the model to make a live price prediction 💵.

## 🛠️ ***Tech Stack***
- `Python 3.11`
- `Streamlit` for UI
- `scikit-learn` for pipeline structure
- `XGBoost` for regression modeling
- `Joblib` for model serialization
- `Poetry` for dependency management

## ⚙️ ***How to Run the App Locally***
### 1️⃣ ***Clone the Repository***
```
git clone https://github.com/alejolondonm/NYC-houses.git
cd NYC-houses
```

### 2️⃣ ***Set Up a Virtual Environment***
```
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
```
```
.venv\Scripts\Activate         # Windows
```

### 3️⃣ ***Install Dependencies***
```
poetry install
```

## 🚀 ***Run the App Locally***
With `poetry`:
```
poetry run streamlit run 4-Deployment/nyc-houses-streamlit.py
```
Or without poetry:
```
streamlit run 4-Deployment/nyc-houses-streamlit.py
```
But if this option is used, all the dependencies of the `pyproject.toml` file must be installed.

> ⚠️ **Note:** The `first_basic_model.joblib` file was trained with:
> - `xgboost==1.6.2`
> - `scikit-learn==1.3.2`
> Make sure to use the same versions to avoid compatibility issues when reloading the model.

### 🧪 ***Validation***
The app prevents ⚠️ Negative predictions (flagged with warnings)

## 📈 ***Expected Output***

Once the app is running, the app will:
- Let users choose between Individual and Batch predictions via tabs.
- Display a 💵 price estimate or alert for suspicious outputs.
- Allow download of results in batch mode.


### 🛑 To Deactivate the Virtual Environment  
```
deactivate
```

# 🚀✨
