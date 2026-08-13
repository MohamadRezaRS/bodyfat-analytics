import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

"""
I am training three distinct models (Linear Regression, Random Forest, and SVR) 
to combat the inherent risks of our extremely small dataset size. When dealing 
with a limited number of instances, complex models tend to memorize the training 
data (overfitting) rather than learning the underlying signal, which destroys 
out-of-sample performance. Linear Regression provides a strict, interpretable baseline. 
"""

datasets = {
    "Male": pd.read_csv("data/processed/processed_male.csv"),
    "Female": pd.read_csv("data/processed/processed_female.csv")
}

def get_pipelines():
    return {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(max_depth=4, random_state=42))
        ]),
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=1.0, epsilon=0.1))
        ])
    }

scoring_metrics = {
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error"
}

for gender, df in datasets.items():
    X = df.drop(columns=["BodyFat"])
    y = df["BodyFat"]
    feature_names = list(X.columns)

    print(gender)
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    models = get_pipelines()
    
    best_score = -np.inf
    best_model_name = None
    best_pipeline = None

    for name, pipeline in models.items():
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring_metrics)
        
        mean_r2 = scores["test_r2"].mean()
        mean_mae = -scores["test_mae"].mean()
        mean_rmse = -scores["test_rmse"].mean()
        
        print(f"{name} -> R2: {mean_r2:.4f} | MAE: {mean_mae:.4f} | RMSE: {mean_rmse:.4f}")
        
        if mean_r2 > best_score:
            best_score = mean_r2
            best_model_name = name
            best_pipeline = pipeline

    print(f"best model: {best_model_name} (R2: {best_score:.4f})\n")

    best_pipeline.fit(X, y)

    file_path = f"models/model_{gender.lower()}.pkl"
    with open(file_path, "wb") as f:
        pickle.dump({"pipeline": best_pipeline, "features": feature_names}, f)