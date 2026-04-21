import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ----------------------------
# Load Model
# ----------------------------
try:
    model = pickle.load(open("random_forest_model.pkl", "rb"))
except Exception as e:
    st.error("❌ Model could not be loaded. Check file path or corruption.")
    st.exception(e)
    st.stop()

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="Salary Predictor", page_icon="💰")
st.title("📊 Salary Prediction App")
st.write("Enter employee details to predict salary using a Random Forest model.")

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("Employee Input")

age = st.sidebar.slider("Age", 18, 65, 30)

gender_text = st.sidebar.selectbox("Gender", ["Female", "Male"])
gender = 1 if gender_text == "Male" else 0

education_text = st.sidebar.selectbox(
    "Education Level",
    ["Bachelor's Degree", "Master's Degree", "PhD"]
)

education_map = {
    "Bachelor's Degree": 0,
    "Master's Degree": 1,
    "PhD": 2
}
education = education_map[education_text]

years_exp = st.sidebar.slider("Years of Experience", 0.0, 40.0, 5.0, 0.5)

job_title = st.sidebar.number_input(
    "Job Title (Encoded ID 0–198)",
    min_value=0,
    max_value=198,
    value=0
)

# ----------------------------
# Input Display
# ----------------------------
input_data = np.array([[age, gender, education, job_title, years_exp]])

df = pd.DataFrame(
    input_data,
    columns=["Age", "Gender", "Education Level", "Job Title", "Years of Experience"]
)

st.subheader("📌 Input Data")
st.dataframe(df)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Salary"):
    try:
        prediction = model.predict(input_data)[0]

        st.success(f"💰 Predicted Salary: ${prediction:,.2f}")
        st.metric("Estimated Salary", f"${prediction:,.2f}")

    except Exception as e:
        st.error("❌ Prediction failed. See details below:")
        st.exception(e)
