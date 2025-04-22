# 📊 ***NYC House Price Estimator — Streamlit App***

This is a Streamlit-based web application that allows users to estimate the **sale price of residential properties in New York City** 🏙️ based on key property features. The app uses a regression model trained with **XGBoost** and wrapped in a `scikit-learn` pipeline.

## 🎯 ***Objective***

This app is designed to:

- 💬 **Collect property features** through an interactive sidebar.
- 🤖 **Predict housing prices** using a pre-trained model.
- 🖥️ **Deploy seamlessly** in both local environments and on Streamlit Cloud.

## 🏗️ ***How It Works***

The app loads a trained model from the file:

```plaintext
Deployment/first_basic_model.joblib
```

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
poetry run streamlit run Deployment/nyc-houses-streamlit.py
```
Or without poetry:
```
streamlit run Deployment/nyc-houses-streamlit.py
```
But if this option is used, all the dependencies of the `pyproject.toml` file must be installed.

> ⚠️ **Note:** The `first_basic_model.joblib` file was trained with:
> - `xgboost==1.6.2`
> - `scikit-learn==1.3.2`
> Make sure to use the same versions to avoid compatibility issues when reloading the model.

## 📈 ***Expected Output***

Once the app is running, you’ll see:

- A sidebar with interactive property inputs 🏡
- A live-predicted sale price displayed on screen 💵


### 🛑 To Deactivate the Virtual Environment  
```
deactivate
```

# 🚀✨
