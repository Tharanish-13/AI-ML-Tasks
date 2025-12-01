# 🌸 Iris Flower Classification

This project builds a machine learning model that predicts the **species of an Iris flower** using four botanical measurements:

- Sepal Length  
- Sepal Width  
- Petal Length  
- Petal Width  

The goal is to classify flowers into one of the three species:

- **Setosa**
- **Versicolor**
- **Virginica**

---

## 📊 Dataset
- **Iris Dataset** (Fisher’s Iris Dataset)
- Included in scikit-learn
- 150 samples × 4 features × 3 species

---

## 📝 Project Workflow

### 1. Data Loading  
Dataset loaded using `load_iris()` from scikit-learn.

### 2. Exploratory Data Analysis (EDA)
- Summary statistics  
- Distribution plots  
- Class visualization  
- Basic feature exploration  

### 3. Model Training  
Used **Logistic Regression** for multi-class flower classification.

### 4. Model Evaluation  
Metrics used:
- Accuracy Score  
- Confusion Matrix  
- Classification Report (Precision, Recall, F1-Score)

### 5. Visualization  
- Confusion Matrix heatmap using Seaborn

### 6. User Input Prediction  
The model accepts manual input values for all four measurements and predicts the flower species.

Example:

Enter Sepal Length (cm): 5.1
Enter Sepal Width (cm): 3.5
Enter Petal Length (cm): 1.4
Enter Petal Width (cm): 0.2

Predicted Species: setosa