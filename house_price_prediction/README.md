# 🏡 House Price Prediction using California Housing Dataset

This project builds a regression model that predicts **median house values** in California districts based on census features such as:

- Median income  
- Average rooms  
- Population  
- House age  
- Latitude & longitude  

The dataset is the official replacement for the Boston Housing dataset.

---

## 📊 Dataset
- **California Housing Dataset** (1990 US Census)
- Included in scikit-learn
- 20,640 rows × 8 features

---

## 📝 Project Workflow

### 1. Data Loading  
Dataset loaded using `fetch_california_housing()`.

### 2. Exploratory Data Analysis (EDA)  
- Summary statistics  
- Distribution plots  
- Correlation heatmap  

### 3. Model Training  
Used **Linear Regression**.

### 4. Model Evaluation  
Metrics used:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

### 5. Visualization  
- Actual vs Predicted scatter plot  

---

## 🚀 Run the Model
-py model.py
