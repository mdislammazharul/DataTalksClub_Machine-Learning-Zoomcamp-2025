#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc
)

import pickle


# In[2]:


# read csv file
df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
df.info()


# In[3]:


df.head(10)


# In[4]:


# Set some style preferences
sns.set(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams['figure.figsize'] = (10, 6)

# --- Basic Information ---
print("Shape of dataset:", df.shape)
print("\nData types and non-null counts:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# --- Univariate Analysis ---
# Histograms for numerical columns
df.hist(bins=20, figsize=(15, 10), color='skyblue', edgecolor='black')
plt.suptitle('Distribution of Numerical Features')
plt.show()

# Count plots for binary/categorical variables
categorical_cols = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'DEATH_EVENT']
for col in categorical_cols:
    sns.countplot(x=col, data=df)
    plt.title(f'Count of {col}')
    plt.show()

# --- Bivariate Analysis ---
# Correlation matrix
corr = df.select_dtypes(include=['number']).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Relationship of features with DEATH_EVENT
numeric_features = ['age', 'creatinine_phosphokinase', 'ejection_fraction',
                    'platelets', 'serum_creatinine', 'serum_sodium', 'time']

for col in numeric_features:
    sns.boxplot(x='DEATH_EVENT', y=col, data=df)
    plt.title(f'{col} vs DEATH_EVENT')
    plt.show()

# --- Multivariate Analysis ---
# Pairplot of selected features
sns.pairplot(df[['age', 'ejection_fraction', 'serum_creatinine', 'time', 'DEATH_EVENT']], hue='DEATH_EVENT', diag_kind='kde')
plt.show()

# --- Feature Relationships ---
# Example: Check survival by sex
sns.barplot(x='sex', y='DEATH_EVENT', data=df, errorbar=None)
plt.title('Death Event Rate by Sex')
plt.show()

# Example: Check survival by age group
df['age_group'] = pd.cut(df['age'], bins=[0, 40, 60, 80, 100], labels=['<40', '40-60', '60-80', '80+'])
sns.barplot(x='age_group', y='DEATH_EVENT', data=df, errorbar=None)
plt.title('Death Event Rate by Age Group')
plt.show()
df = df.drop(columns='age_group')

# --- Outlier Detection ---
for col in numeric_features:
    sns.boxplot(x=df[col], color='lightblue')
    plt.title(f'Outliers in {col}')
    plt.show()

# --- Summary Insights (print basic observations) ---
print("\nCorrelation with DEATH_EVENT:")
print(df.select_dtypes(include=['number']).corr()['DEATH_EVENT'].sort_values(ascending=False))


# In[5]:


"""
Split the data into 3 parts: train/validation/test with 60%/20%/20% distribution. Use train_test_split function for that with random_state=1
"""

seed = 1
d_test = 0.2
d_val = 0.2

# Separate features and target first
X = df.drop(columns='DEATH_EVENT')
y = df['DEATH_EVENT']

# Step 1: Split into train_val and test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=d_test, random_state=seed, stratify=y
)

# Step 2: Split train_val into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=d_val, random_state=seed, stratify=y_train_val
)

print(f'Shape of full training set: {X_train_val.shape}')
print(f'Shape of validation set: {X_val.shape}')
print(f'Shape of test set: {X_test.shape}')


# In[6]:


"""
ROC AUC could also be used to evaluate feature importance of numerical variables. Let's do that-
For each numerical variable, use it as score (aka prediction) and compute the AUC with the y variable as ground truth.
Use the training dataset for that.
If your AUC is < 0.5, invert this variable by putting "-" in front

(e.g. -df_train['balance'])
"""
# Select only numeric columns (no objects or categories)

df_train_numerical_col = X_train.select_dtypes(include="number")

max_auc = {'column': None, 'auc': -1}

for col in df_train_numerical_col.columns:
    fpr, tpr, thresholds = roc_curve(y_train, df_train_numerical_col[col])
    auc_value = auc(fpr, tpr)
    print(f'{col}: {auc_value:.4f}')

    # If AUC < 0.5, invert the column
    if auc_value < 0.5:
        print(f"    ↳ Inverting {col} (AUC < 0.5)")
        df_train_numerical_col[col] = -df_train_numerical_col[col]
        auc_value = 1 - auc_value  # inverted AUC improves to >0.5

    # Track the best feature by AUC
    if auc_value > max_auc['auc']:
        max_auc['column'] = col
        max_auc['auc'] = auc_value

print("\nBest column by AUC:")
print(max_auc)


# In[7]:


models = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__solver": ["lbfgs", "liblinear"]
        }
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "model__max_depth": [3, 5, 7, None],
            "model__min_samples_split": [2, 5, 10],
            "model__criterion": ["gini", "entropy"]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, 10, None],
            "model__min_samples_split": [2, 5, 10]
        }
    }
}

# ==========================
# Train, Tune & Evaluate
# ==========================
results = []

for name, mp in models.items():
    print(f"\n Training {name}...")
    # Create pipeline (scaler for LR, others unaffected)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", mp["model"])
    ])
    
    grid = GridSearchCV(pipe, mp["params"], cv=5, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        "Model": name,
        "Best Params": grid.best_params_,
        "Accuracy": acc,
        "ROC_AUC": auc,
        "F1_Score": f1
    })
    
    print(f"Best Params: {grid.best_params_}")
    print(f"Accuracy: {acc:.3f}, ROC-AUC: {auc:.3f}, F1: {f1:.3f}")

# ==========================
# 5. Compare Models
# ==========================
results_df = pd.DataFrame(results).sort_values(by="ROC_AUC", ascending=False)
print("\n Model Comparison:")
print(results_df)

best_model_name = results_df.iloc[0]["Model"]
print(f"\n Best Model: {best_model_name}")


# In[8]:


# Example patient data
sample = pd.DataFrame([{
    'age': 65,
    'anaemia': 0,
    'creatinine_phosphokinase': 250,
    'diabetes': 1,
    'ejection_fraction': 35,
    'high_blood_pressure': 1,
    'platelets': 250000,
    'serum_creatinine': 1.9,
    'serum_sodium': 130,
    'sex': 1,
    'smoking': 0,
    'time': 120
}])

# Predict class (0 = survived, 1 = death)
pred_class = best_model.predict(sample)[0]

# Predict probability of death
pred_proba = best_model.predict_proba(sample)[0][1]

print(f"Predicted Class: {pred_class}")
print(f"Probability of Death: {pred_proba:.2f}")


# In[9]:


# Save model to file
with open("heart_failure_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Model saved as heart_failure_model.pkl")


# In[10]:


get_ipython().system('jupyter nbconvert --to python Mid_Term_Project.ipynb')


# In[ ]:




