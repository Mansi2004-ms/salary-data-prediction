import streamlit as st
import pickle
import numpy as np

# ----------------------------
# Load Model
# ----------------------------
try:
    model = pickle.load(open("random_forest_model.pkl", "rb"))
except Exception as e:
    st.error("❌ Failed to load model file")
    st.exception(e)
    st.stop()

# ----------------------------
# UI Setup
# ----------------------------
st.set_page_config(page_title="Salary Predictor", page_icon="💰")
st.title("📊 Salary Prediction App")

st.sidebar.header("Enter Employee Details")

# ----------------------------
# Inputs
# ----------------------------
age = st.sidebar.number_input("Age", 18, 65, 30)

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
gender = 1 if gender == "Male" else 0

education = st.sidebar.selectbox(
    "Education Level",
    ["Bachelor's", "Master's", "PhD"]
)

education = {
    "Bachelor's": 0,
    "Master's": 1,
    "PhD": 2
}[education]

job_title = st.sidebar.number_input(
    "Job Title (Encoded ID)",
    0, 198, 0
)

years_exp = st.sidebar.number_input(
    "Years of Experience",
    0.0, 40.0, 5.0
)

# ----------------------------
# Input Array (IMPORTANT FIX)
# ----------------------------
input_data = np.array([[age, gender, education, job_title, years_exp]])

st.subheader("Input Data Preview")
st.write(input_data)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Salary"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Predicted Salary: ${prediction:,.2f}")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.exception(e)
