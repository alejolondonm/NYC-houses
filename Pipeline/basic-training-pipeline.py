# Train Model pipeline
#
# ## By:
# [Alejandro Londoño](https://github.com/alejolondonm)
#
# ## Date:
# 2025-04-07
#
# ## Description:
#
# Pipeline script for training the first selected regression model
# based on the NYC housing dataset.

# Import  libraries
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBRegressor

# Variables
COMMERCIAL_UNITS_THRESHOLD = 500
TOTAL_UNITS_THRESHOLD = 1000


# Cleaning Function
def cleaning(df: pd.DataFrame) -> pd.DataFrame:
    # Remove duplicates
    df = df.sort_values(by="SALE PRICE", ascending=False, na_position="last")
    df = df.drop_duplicates(keep="first")

    # Drop rows with missing SALE PRICE or zeros
    df = df[df["SALE PRICE"].notna()]
    df = df[df["SALE PRICE"] > 0]

    # Drop rows where key area columns are null
    df = df.dropna(subset=["GROSS SQUARE FEET", "LAND SQUARE FEET"])

    # Replace 0s with NaNs in important columns
    columns_to_nullify = ["GROSS SQUARE FEET", "LAND SQUARE FEET", "YEAR BUILT"]
    df[columns_to_nullify] = df[columns_to_nullify].replace(0, np.nan)

    # Nullify extreme outliers
    if "TOTAL UNITS" in df.columns:
        df.loc[df["TOTAL UNITS"] > TOTAL_UNITS_THRESHOLD, "TOTAL UNITS"] = np.nan
    df.loc[df["COMMERCIAL UNITS"] > COMMERCIAL_UNITS_THRESHOLD, "COMMERCIAL UNITS"] = (
        np.nan
    )

    # Remove outliers
    lower_bound = df["SALE PRICE"].quantile(0.028)
    upper_bound = df["SALE PRICE"].quantile(0.99)
    df = df[(df["SALE PRICE"] >= lower_bound) & (df["SALE PRICE"] <= upper_bound)]

    df = df.drop_duplicates()

    return df


# --------------------------
# 1. Load and Prepare Data
# --------------------------

# Load data
URL_DATA = "https://github.com/JoseRZapata/Data_analysis_notebooks/raw/refs/heads/main/data/datasets/nyc-rolling-sales_data.csv"
df_raw = pd.read_csv(URL_DATA, low_memory=False)

# Data preparation
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

df = df_raw[selected_features].copy()

# Replace known null-like strings
df.replace(["NULL", "None", "", "?", " ", "  ", " -  "], np.nan, inplace=True)

numeric_cols = [
    "SALE PRICE",
    "GROSS SQUARE FEET",
    "LAND SQUARE FEET",
    "COMMERCIAL UNITS",
    "RESIDENTIAL UNITS",
    "YEAR BUILT",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Convert data types

# numerical columns
numeric_features = [
    "GROSS SQUARE FEET",
    "LAND SQUARE FEET",
    "COMMERCIAL UNITS",
    "RESIDENTIAL UNITS",
]

# Numerical variables
df[numeric_features] = df[numeric_features].astype("float")

# categorical columns
categorical_features = ["TAX CLASS AT TIME OF SALE", "BOROUGH"]
ordinal_features = ["YEAR BUILT"]

# Categorical variables
df[categorical_features] = df[categorical_features].astype("category")

# Target variable
target = "SALE PRICE"
df[target] = df[target].astype("float")


# Data preprocessing
df = cleaning(df)

# -----------------------------
# 2. Split Features and Target
# -----------------------------

# Train / Test split

X = df.drop("SALE PRICE", axis=1)
Y = df["SALE PRICE"]

# 80% train, 20% test
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ----------------------
# 3. Define Pipelines
# ----------------------
numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

categorical_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

categorical_ord_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "ordinal",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipe, numeric_features),
        ("categoric", categorical_pipe, categorical_features),
        ("ordinal", categorical_ord_pipe, ordinal_features),
    ]
)


# --------------------------
# 4. Build Final Pipeline
# --------------------------
pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("model", XGBRegressor(random_state=42))]
)

# Hyperparameter tuning
score = "r2"

hyperparameters = {
    "model__max_depth": [3, 5, 7, 9],
    "model__n_estimators": [100, 200, 300, 400],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
}

grid_search = GridSearchCV(
    pipeline,
    hyperparameters,
    cv=5,
    scoring=score,
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(x_train, y_train)


# --------------------
# 5. Evaluate Model
# --------------------

# Predict on test set
y_pred = grid_search.predict(x_test)

# Calculate R² on original scale
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n--- Evaluation Metrics ---")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.4f}")

# Print best hyperparameters
print("\n--- Best Hyperparameters ---")
print(grid_search.best_params_)


# ---------------------------
# 6. Save if Model is Valid
# ---------------------------

# baseline score
BASELINE_SCORE = 0.54

# Model Validation
model_validation = r2 > BASELINE_SCORE

MODEL_OUTPUT_PATH = Path(__file__).resolve().parent / "models"
if model_validation:
    # Save model if validation passed
    MODEL_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    dump(
        grid_search.best_estimator_,
        MODEL_OUTPUT_PATH / "first_basic_model.joblib",
        protocol=5,
    )
    print("\nModel validation passed")
    print(f"\n✅ Model saved to {MODEL_OUTPUT_PATH}")
else:
    # Raise error if validation failed
    print("\n❌ Model did not pass the threshold. Not saved.")
    print(f"Model validation failed: score {r2} below baseline {BASELINE_SCORE}")
    raise ValueError()
