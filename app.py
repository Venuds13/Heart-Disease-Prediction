import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# App title
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("❤️ Heart Disease Prediction App")

# Load and display data
@st.cache_data
def load_data():
    df = pd.read_csv('heart_disease_data.csv')
    return df

heart_data = load_data()

# Prepare the data
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Sidebar - About
st.sidebar.title("About")
st.sidebar.info("This app uses a Logistic Regression model to predict the likelihood of heart disease.")

# Input form
st.subheader("Enter Patient Details")

with st.form("heart_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
        cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", value=120)
    
    with col2:
        chol = st.number_input("Cholesterol", value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [1, 0])
        restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate Achieved", value=150)
    
    with col3:
        exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
        oldpeak = st.number_input("ST Depression", value=1.0, format="%.1f")
        slope = st.selectbox("Slope of ST Segment (0-2)", [0, 1, 2])
        ca = st.selectbox("Major Vessels Colored by Fluoroscopy (0-3)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", [1, 2, 3])

    submitted = st.form_submit_button("Predict")

# Prediction
if submitted:
    input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg,
                           thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)

    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.success("✅ The person does **not** have heart disease.")
    else:
        st.error("⚠️ The person **has** heart disease.")

# Model accuracy (optional)
with st.expander("See Model Accuracy"):
    train_acc = accuracy_score(Y_train, model.predict(X_train))
    test_acc = accuracy_score(Y_test, model.predict(X_test))
    st.write(f"Training Accuracy: `{train_acc:.2f}`")
    st.write(f"Test Accuracy: `{test_acc:.2f}`")
