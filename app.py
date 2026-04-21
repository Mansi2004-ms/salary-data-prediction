import streamlit as st
import pickle
import pandas as pd

# ----------------------------
# Load Model
# ----------------------------
try:
    model = pickle.load(open('random_forest_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file not found. Please place 'random_forest_model.pkl' in the same folder.")
    st.stop()

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title='Salary Predictor', page_icon='💰')
st.title('📊 Salary Prediction App')
st.write("Predict employee salary using a trained Random Forest model.")

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
gender = 1 if gender == "Male" else 0

education = st.sidebar.selectbox(
    "Education Level",
    ["Bachelor's Degree", "Master's Degree", "PhD"]
)

education_map = {
    "Bachelor's Degree": 0,
    "Master's Degree": 1,
    "PhD": 2
}
education_level = education_map[education]

years_of_experience = st.sidebar.slider(
    "Years of Experience", 0.0, 40.0, 5.0, 0.5
)

# ----------------------------
# Job Title Handling (Improved UX)
# ----------------------------
st.sidebar.info(
    "⚠ Job Title should match training encoding. "
    "For now, select a simplified index (0–198)."
)

job_title = st.sidebar.selectbox(
    "Job Title (Encoded ID)",
    list(range(199))
)

# ----------------------------
# Create Input DataFrame
# ----------------------------
input_data = pd.DataFrame([[
    age,
    gender,
    education_level,
    job_title,
    years_of_experience
]],
columns=[
    "Age",
    "Gender",
    "Education Level",
    "Job Title",
    "Years of Experience"
])

# ----------------------------
# Show Input
# ----------------------------
st.subheader("Entered Data")
st.dataframe(input_data)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Salary"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Predicted Salary: ${prediction:,.2f}")
        st.metric("Estimated Salary", f"${prediction:,.2f}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
