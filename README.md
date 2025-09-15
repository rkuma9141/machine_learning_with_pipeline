# Machine Learning Project (With Pipeline)

This project demonstrates how to build a **Machine Learning model using Scikit-Learn Pipelines**.  
Pipelines make the workflow **clean, less error-prone, and reusable** by combining preprocessing and modeling steps together.

---

## 📌 Why Use Pipelines?
Without pipelines, you need to separately apply:
- Encoding
- Scaling
- Model training
- Transformation for new inputs  

👉 With pipelines:
- All steps are chained together.
- No need to manually repeat preprocessing for new data.
- Prevents data leakage (scaler/encoder fitted only on training set).
- Code becomes clean and professional.

---

## 🚀 Steps

### 1. Import Required Libraries
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
