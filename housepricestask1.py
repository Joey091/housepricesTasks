# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 02:43:42 2024

@author: Asus
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import joblib

# Load the dataset
df = pd.read_csv('boston.csv', encoding='latin')

# Display initial data
print(df.head())
print(df.describe())
print(df.dtypes)

# 1. Handling Missing Values
df.fillna(df.median(), inplace=True)

# 2. Handling Duplicates
df = df.drop_duplicates()

# 3. Handling Categorical Data
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 4. Handling Outliers using IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Splitting features and target variable
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# 5. Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Dictionary to store model performance
model_performance = {}

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
model_performance['Linear Regression'] = {
    'R2 Score': r2_score(y_test, y_pred_lr),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr))
}

# Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
model_performance['Decision Tree'] = {
    'R2 Score': r2_score(y_test, y_pred_dt),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_dt))
}

# Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
model_performance['Random Forest'] = {
    'R2 Score': r2_score(y_test, y_pred_rf),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf))
}

# Support Vector Regressor
svr = SVR()
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)
model_performance['Support Vector Regressor'] = {
    'R2 Score': r2_score(y_test, y_pred_svr),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_svr))
}

# K-Nearest Neighbors Regressor
knn = KNeighborsRegressor()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
model_performance['K-Nearest Neighbors'] = {
    'R2 Score': r2_score(y_test, y_pred_knn),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_knn))
}

# Hyperparameter Tuning with Grid Search for the Random Forest Model
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
}
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search_rf.fit(X_train, y_train)

# Best parameters and performance for the tuned Random Forest model
best_rf = grid_search_rf.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
model_performance['Tuned Random Forest'] = {
    'R2 Score': r2_score(y_test, y_pred_best_rf),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_best_rf))
}

# Model Comparison
performance_df = pd.DataFrame(model_performance).T
print("Model Performance Comparison:\n", performance_df)

# Saving the Best Model
best_model = best_rf
joblib.dump(best_model, 'best_model.pkl')
print("Best model saved as 'best_model.pkl'")
