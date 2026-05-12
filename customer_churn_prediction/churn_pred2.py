from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import os

# Set Streamlit layout to wide
st.set_page_config(layout="wide")

# Get current directory
BASE_DIR = os.path.dirname(__file__)

# Load the trained model
model_path = os.path.join(BASE_DIR, 'best_model.pkl')

with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load the MinMaxScaler
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

# Define the input features for the model
feature_names = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "EstimatedSalary", "Geography_France", "Geography_Germany", "Geography_Spain",
    "Gender_Female", "Gender_Male", "HasCrCard_0", "HasCrCard_1",
    "IsActiveMember_0", "IsActiveMember_1"
]

# Columns requiring scaling
scale_vars = ["CreditScore", "EstimatedSalary", "Tenure", "Balance", "Age", "NumOfProducts"]

# Updated default values
default_values = [
    600, 30, 2, 8000, 2, 60000,
    True, False, False, True, False, False, True, False, True
]

# Sidebar setup
pic1_path = os.path.join(BASE_DIR, "Pic 1.PNG")
st.sidebar.image(pic1_path, use_container_width=True)

st.sidebar.header("User Inputs")

# Collect user inputs
user_inputs = {}

for i, feature in enumerate(feature_names):
    if feature in scale_vars:
        user_inputs[feature] = st.sidebar.number_input(
            feature,
            value=default_values[i],
            step=1 if isinstance(default_values[i], int) else 0.01
        )

    elif isinstance(default_values[i], bool):
        user_inputs[feature] = st.sidebar.checkbox(
            feature,
            value=default_values[i]
        )

    else:
        user_inputs[feature] = st.sidebar.number_input(
            feature,
            value=default_values[i],
            step=1
        )

# Convert inputs to DataFrame
input_data = pd.DataFrame([user_inputs])

# Apply MinMaxScaler
input_data_scaled = input_data.copy()
input_data_scaled[scale_vars] = scaler.transform(input_data[scale_vars])

# App Header
pic2_path = os.path.join(BASE_DIR, "Pic 2.PNG")
st.image(pic2_path, use_container_width=True)

st.title("Customer Churn Prediction")

# Page Layout
left_col, right_col = st.columns(2)

# Left Side - Feature Importance
with left_col:
    st.header("Feature Importance")

    feature_importance_path = os.path.join(
        BASE_DIR,
        "feature_importance.xlsx"
    )

    feature_importance_df = pd.read_excel(
        feature_importance_path,
        usecols=["Feature", "Feature Importance Score"]
    )

    fig = px.bar(
        feature_importance_df.sort_values(
            by="Feature Importance Score",
            ascending=True
        ),
        x="Feature Importance Score",
        y="Feature",
        orientation="h",
        title="Feature Importance",
        labels={
            "Feature Importance Score": "Importance",
            "Feature": "Features"
        },
        width=400,
        height=500
    )

    st.plotly_chart(fig)

# Right Side - Prediction
with right_col:
    st.header("Prediction")

    if st.button("Predict"):

        probabilities = model.predict_proba(input_data_scaled)[0]

        prediction = model.predict(input_data_scaled)[0]

        prediction_label = (
            "Churned"
            if prediction == 1
            else "Retain"
        )

        st.subheader(f"Predicted Value: {prediction_label}")

        st.write(
            f"Predicted Probability: {probabilities[1]:.2%} (Churn)"
        )

        st.write(
            f"Predicted Probability: {probabilities[0]:.2%} (Retain)"
        )

        st.markdown(
            f"### Output: **{prediction_label}**"
        )
