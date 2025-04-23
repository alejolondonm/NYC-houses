# 🏠 ***NYC Houses - Basic Training Pipeline***

A machine learning pipeline designed to train an initial regression model for predicting housing prices in New York City 🏙️.  
This script handles the data cleaning, preprocessing, and model selection workflow using `XGBoost`.

## 🎯 ***Objective***

The main goals of this training script are:

- 🤖 **Model Development**  
  - Build a robust regression pipeline using **XGBoost** and **scikit-learn**.  
  - Search for the best hyperparameters via **GridSearchCV**.  

- 🔍 **Data Cleaning and Preprocessing**  
  - Handle missing values, outliers, and type conversions for selected variables.  
  - Focus on key predictive features only.

- 💾 **Model Saving & Reusability**  
  - Export the best model as a `.joblib` file to be reused in deployment or evaluation scripts.  

## 🧠 Features Used in Training

Only the most relevant features were selected:

```python
selected_features = [
    "COMMERCIAL UNITS",
    "TAX CLASS AT TIME OF SALE",
    "RESIDENTIAL UNITS",
    "GROSS SQUARE FEET",
    "BOROUGH",
    "SALE PRICE",
    "YEAR BUILT",
    "LAND SQUARE FEET",
]
```

## 🛠️ Tech Stack

- **Python 3.12** 🐍  
- **Pandas & NumPy** for data manipulation  
- **Scikit-Learn** for preprocessing & pipelines  
- **XGBoost** for regression modeling
- **Joblib** for saving trained models  

# ⚙️ Setting Up the Project  

To run this training pipeline on your local environment:

## 1️⃣ **Clone the Repository**  

```bash
git clone https://github.com/alejolondonm/NYC-houses.git
cd NYC-houses
```

## 2️⃣ Set Up the Virtual Environment

```
# Create a virtual environment
python -m venv .venv  

# Activate the virtual environment
# Windows (PowerShell)
.venv\Scripts\Activate  
```
```
# macOS/Linux
source .venv/bin/activate
```

## 3️⃣ Install `uv` for Dependency Management  
```bash
pip install uv
```

## 4️⃣ Install All Dependencies Using `uv`

```bash
python -m uv pip install -r uv.lock
```

# 🚀 Running the Pipeline

From the root of the repository, run the training pipeline:

```bash
python Pipeline/basic-training-pipeline.py
```

If successful, the best model will be saved to:

```
Pipeline/models/first_basic_model.joblib
```

## 📉 Output Metrics

At the end of training, you'll see:

- **MAE** (Mean Absolute Error)  
- **RMSE** (Root Mean Squared Error)  
- **R²** score on the test set  
- Best hyperparameters found by GridSearch  

### 🛑 Deactivate the Virtual Environment  
```bash
deactivate
```

# 🚀✨
