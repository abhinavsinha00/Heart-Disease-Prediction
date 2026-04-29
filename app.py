import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and expected columns
model = joblib.load("heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

st.title("Heart Stroke Prediction by abhinav sinha ❤️")
st.markdown("Provide the following details to check your heart stroke risk:")

cp_map = {"ATA": 0,"NAP": 1,"TA": 2,"ASY": 3}

restecg_map = {"Normal": 0,"ST": 1,"LVH": 2}

# Collect user input
cp = st.selectbox("Chest Pain Type", list(cp_map.keys()))
restcg= st.selectbox("Resting ECG", list(restecg_map.keys()))
age = st.slider("Age", 18, 100, 50)
sex = st.selectbox("Sex", ["0", "1"],format_func=lambda x: "Female" if x==0 else "Male")
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
thalach = st.slider("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise-Induced Angina", [0, 1])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
ca=st.selectbox("Number of majaor vessels(ca)",[0,1,2,3,4])
thal=st.selectbox("Thalassemia (thal)",[0,1,2,3])

# When Predict is clicked
if st.button("Predict"):

    # Create a raw input dictionary
    raw_input = {
        'Age': age,
        'RestingBP': trestbps,
        'Cholesterol': chol,
        'FastingBS': fbs,
        'MaxHR': thalach,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + cp: 1,
        'RestingECG_' + restcg: 1,
        'ExerciseAngina_' + str(exang): 1,
        'ST_Slope_' + slope: 1
    }

    # Create input dataframe
    input_df = pd.DataFrame([raw_input])

    # Fill in missing columns with 0s
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_columns]

    # Scale the input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]

    # Show result
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")