# Credit Card Fraud Detection Using Machine Learning 💳🔍

This project builds a fraud detection model using Logistic Regression, Random Forest, and XGBoost.  
The dataset is highly imbalanced, so **SMOTE oversampling** is used to balance the classes and improve the model’s recall on fraudulent transactions.

---

## 📂 Dataset  
The dataset (`creditcard.csv`) is NOT included in this repository because GitHub does not allow files larger than 100 MB.

Download it from Kaggle:  
🔗 https://www.kaggle.com/mlg-ulb/creditcardfraud

Place it in the project folder before running the notebook.

---

## 🚀 Workflow Summary

### **1. Import Libraries**
The notebook imports:
- Pandas, NumPy
- Scikit-learn (train_test_split, StandardScaler, Logistic Regression, Random Forest)
- XGBoost
- SMOTE from imblearn
- accuracy_score for evaluation

---

### **2. Load Dataset**
```python
df = pd.read_csv("creditcard.csv")
