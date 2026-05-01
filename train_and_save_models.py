"""
Pre-train and save models to disk for instant loading in Streamlit app.
Run this script once locally: python train_and_save_models.py
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

print("Loading California Housing dataset...")
data = fetch_california_housing(as_frame=True)
df = data.frame.copy()
df.columns = [
    "MedIncome", "HouseAge", "AvgRooms", "AvgBedrooms",
    "Population", "AvgOccupants", "Latitude", "Longitude", "Price"
]
df["Price"] = df["Price"] * 100_000

# Feature engineering
df["RoomsPerPerson"]   = df["AvgRooms"] / df["AvgOccupants"].clip(0.5)
df["BedroomRatio"]     = df["AvgBedrooms"] / df["AvgRooms"].clip(1)
df["IncomePerPerson"]  = df["MedIncome"] / df["AvgOccupants"].clip(0.5)
df["PopDensity"]       = df["Population"] / df["AvgOccupants"].clip(0.5)

FEATURES = ["MedIncome", "HouseAge", "AvgRooms", "AvgBedrooms",
            "Population", "AvgOccupants", "Latitude", "Longitude",
            "RoomsPerPerson", "BedroomRatio", "IncomePerPerson", "PopDensity"]

X = df[FEATURES]
y = df["Price"]

# Train-test split
test_size = 0.20
random_seed = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_seed)

# Scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("Training models...")
models_def = {
    "Random Forest":        RandomForestRegressor(n_estimators=50, max_depth=8,
                                                  random_state=random_seed, n_jobs=-1),
    "Extra Trees":          ExtraTreesRegressor(n_estimators=50, max_depth=8,
                                                random_state=random_seed, n_jobs=-1),
    "Gradient Boosting":    GradientBoostingRegressor(n_estimators=50, max_depth=5,
                                                      random_state=random_seed, subsample=0.8),
    "Decision Tree":        DecisionTreeRegressor(max_depth=8, random_state=random_seed),
    "Ridge Regression":     Ridge(alpha=1.0),
    "Lasso Regression":     Lasso(alpha=50.0, max_iter=5000),
}

results = {}
trained = {}
for name, mdl in models_def.items():
    print(f"  Training {name}...")
    if name in ("Ridge Regression", "Lasso Regression"):
        mdl.fit(X_train_sc, y_train)
        preds = mdl.predict(X_test_sc)
    else:
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
    
    results[name] = {
        "mae":  mean_absolute_error(y_test, preds),
        "rmse": np.sqrt(mean_squared_error(y_test, preds)),
        "r2":   r2_score(y_test, preds),
        "preds": preds,
    }
    trained[name] = mdl

best_name = max(results, key=lambda k: results[k]["r2"])
best_model = trained[best_name]
best_preds = results[best_name]["preds"]

# Save everything to disk
print("\nSaving models to disk...")
with open("models_trained.pkl", "wb") as f:
    pickle.dump({
        "trained_models": trained,
        "model_results": results,
        "best_name": best_name,
        "best_model": best_model,
        "best_preds": best_preds,
        "scaler": scaler,
        "df": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_sc": X_train_sc,
        "X_test_sc": X_test_sc,
        "FEATURES": FEATURES,
    }, f)

print("✓ Models saved to models_trained.pkl")
print(f"\nBest model: {best_name} (R² = {results[best_name]['r2']:.4f})")
print("You can now commit this file to GitHub and deploy on Streamlit Cloud!")
