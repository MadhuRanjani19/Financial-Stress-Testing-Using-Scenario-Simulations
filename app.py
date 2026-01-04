import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

st.title("Financial Stress Testing Using Scenario Simulation")
st.write("Predict Stress Level (0 = Normal, 1 = Moderate, 2 = Severe).")

uploaded = st.file_uploader("Upload cleaned_financial_stress_data.csv", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)-
    st.write("Dataset Preview")
    st.dataframe(df.head())

    # Features & Target
    X = df.drop("Stress_Level", axis=1)
    y = df["Stress_Level"]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric="mlogloss"),
        "SVM": SVC()
    }

    accuracies = {}

    # Train each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracies[name] = accuracy_score(y_test, preds)

    # Best model
    best_model_name = max(accuracies, key=accuracies.get)
    best_model = models[best_model_name]

    st.success(f"Best Model: {best_model_name} (Accuracy: {accuracies[best_model_name]:.4f})")

    # Accuracy chart
    st.write("Model Accuracy Comparison")
    fig, ax = plt.subplots()
    ax.bar(accuracies.keys(), accuracies.values())
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.write("---")
    st.write("## Scenario Simulation – Predict Stress Level")

    # User input for simulation
    user_inputs = []
    for col in X.columns:
        value = st.number_input(f"Enter {col}", value=float(df[col].mean()))
        user_inputs.append(value)

    user_array = np.array(user_inputs).reshape(1, -1)
    user_scaled = scaler.transform(user_array)

    predicted_stress = best_model.predict(user_scaled)[0]

    st.write("### Predicted Stress Level:")

    if predicted_stress == 0:
        st.success("0 → Normal Condition")
    elif predicted_stress == 1:
        st.warning("1 → Moderate Stress")
    else:
        st.error("2 → Severe Stress")
