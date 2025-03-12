# 🏠 ***NYC Houses - Real Estate Data Analysis***

A data science project focused on analyzing real estate data from New York City 🏙️. This project explores property characteristics and sales trends using machine learning models.

## 🎯 ***Objective***

The main goals of this project are:

- 📌 **End-to-End Model Development**:  
  - Build a Proof of Concept (PoC) with an initial data science pipeline.  
  - Implement basic **MLOps Level 0** practices.  

- 🤝 **Team Collaboration & Agile Methodologies**:  
  - Collaborative development using **Pull Requests (PRs)** and code reviews.  
  - Follow Agile methodologies to structure tasks and iterations effectively.  

- 💻 **Software Development Best Practices**:  
  - **Environment & Dependency Management**: Ensuring a reproducible and organized workflow.  
  - **Linting, Formatting & Static Typing**: Maintain clean, readable, and error-free code.  
  - **Version Control**: Proper code versioning and tracking using Git.  

## 🏗️ Project Overview

This project covers the complete data science workflow:

1. **Data Collection & Preprocessing** 📊  
   - Process raw NYC housing data.  
   - Handle missing values, outliers, and categorical variables.  

2. **Exploratory Data Analysis (EDA)** 🔍  
   - Visualize trends in property sales.  
   - Identify key variables affecting housing prices.  

3. **Machine Learning Modeling** 🤖  
   - Train and evaluate regression models to predict house prices.  
   - Use algorithms like **XGBoost** and **Gradient Boosting** for optimal performance.  

4. **Model Interpretation & Error Analysis** 📉  
   - Analyze feature importance to understand key predictors.  
   - Investigate model errors to improve accuracy in future iterations.

## 🛠️ Tech Stack

- **Python** 🐍  
- **Pandas & NumPy** for data manipulation  
- **Matplotlib & Seaborn** for visualization  
- **Scikit-Learn & XGBoost** for machine learning  
- **Jupyter Notebooks** for experimentation  
- **PyArrow & Parquet** for optimized data storage  
- **Joblib** for model serialization  

# ⚙️ Setting Up the Project  

To run this project on another computer, follow these steps:

## 1️⃣ **Clone the Repository**  
First, clone the GitHub repository to your local machine:  

```bash
git clone https://github.com/alejolondonm/NYC-houses.git
cd NYC-houses
```

## 2️⃣ Set Up the Virtual Environment
This project uses a virtual environment to manage dependencies. To create and activate it, run:

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
## 3️⃣ Install uv for Dependency Management
```
pip install uv
```
## 4️⃣ Install Dependencies with uv
```
python -m uv pip install -r uv.lock
```

# 🚀 Running the Project
After installing dependencies, you can now execute the notebooks in the project as needed.

### 🛑 Deactivate the Virtual Environment
```
deactivate
```

# 🚀✨
