# House Price Predictor Pro

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview
**House Price Predictor Pro** is an interactive end-to-end Machine Learning dashboard built with **Streamlit**. It allows users to explore the California Housing dataset through advanced Exploratory Data Analysis (EDA) and compare multiple regression models (Random Forest, Gradient Boosting, etc.) to predict housing prices in real-time.

This project was completed as part of the **Practical Machine Learning Training (Phase 2)** by Chhay Lyhour.

## Key Features
* **Live Predictions:** Adjust housing features (income, house age, rooms) to see instant price estimates.
* **Interactive EDA:** Dynamic heatmaps, distribution plots, and scatter maps.
* **Model Comparison:** Compare $R^2$ and MAE (Mean Absolute Error) across 6 different regression algorithms.
* **Feature Importance:** Visualizes which factors (like Median Income) drive housing prices most.

## Tech Stack
* **Frontend:** Streamlit
* **Data Handling:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn, Plotly
* **Machine Learning:** Scikit-Learn (Random Forest, XGBoost, Ridge/Lasso)

---

## Setup & Installation

Follow these steps to run the project locally on your machine.

### 1. Clone the Repository
```bash
git clone https://github.com/Chhay-Lyhour/House-Price-Predictor-Pro.git
cd House-Price-Predictor-Pro
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
> **Note:** If you don't have a `requirements.txt`, create one with: `streamlit`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.

### 4. Run the Application
```bash
streamlit run app.py
```

---

## Data Insights
The model is trained on the **California Housing Dataset**. Key findings from the EDA include:
* **Median Income:** The strongest predictor of house prices ($r \approx 0.68$).
* **Geography:** Proximity to the coast significantly inflates property values regardless of house age.


**Author:** [Chhay Lyhour](https://github.com/Chhay-Lyhour)  
**Date:** April 29, 2026