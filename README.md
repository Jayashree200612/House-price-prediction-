# House-price-prediction-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("housing.csv")  # Use your dataset here

df = load_data()
st.title("House Price Predictor")

# Display dataset
if st.checkbox("Show Raw Data"):
    st.write(df.head())

# Preprocessing
df = df.select_dtypes(include=[np.number])
df = df.dropna()

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection
model_choice = st.selectbox("Choose a Model", ["Linear Regression", "Random Forest", "XGBoost"])

if model_choice == "Linear Regression":
    model = LinearRegression()
elif model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
else:
    model = XGBRegressor(n_estimators=100, random_state=42)

# Train Model
model.fit(X_train, y_train)
preds = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

st.subheader("Model Performance")
st.write(f"RMSE: {rmse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Prediction
st.subheader("Try Predicting with Custom Input")
input_data = {}
for col in X.columns[:5]:  # Only first 5 features for simplicity
    input_data[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

if st.button("Predict Price"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted House Price: ${prediction:,.2f}")
