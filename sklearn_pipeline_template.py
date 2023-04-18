"""
This code preprocesses the data, splits it into training and testing sets, creates a pipeline for each model, and trains and evaluates the models using cross-validation. The negative mean squared error is used as the scoring metric, and it is converted to root mean squared error (RMSE) for easier interpretation.

Feel free to replace the models in the models dictionary with other models you'd like to try. Don't forget to import the corresponding classes from scikit-learn if you add or change models.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# 1. Preprocess the data
# Assuming you have a dataframe called 'df' and a series called 'series'
X = df
y = series

# Define the column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Col1', 'Col3']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Col2'])
    ])

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create a function that builds a pipeline with the specified model
def create_pipeline(model):
    return Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# 4. Train and evaluate multiple models using cross-validation
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Support Vector Machine': SVR()
}

for name, model in models.items():
    pipeline = create_pipeline(model)
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print(f'{name} - Mean RMSE: {rmse_scores.mean():.2f}, Std: {rmse_scores.std():.2f}')


