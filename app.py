import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💰",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
try:
    model = joblib.load("salary_predictor_model.joblib")
except FileNotFoundError:
    st.error("❌ Model file 'salary_predictor_model.joblib' not found.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# -----------------------------
# Title
# -----------------------------
st.title("💰 Salary Prediction App")
st.write("Predict salary based on years of experience.")

# -----------------------------
# User Input
# -----------------------------
years_experience = st.slider(
    "Select Years of Experience",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    step=0.1
)

# -----------------------------
# Create Input Data
# -----------------------------
input_data = pd.DataFrame({
    "YearsExperience": [years_experience],
    "YearsExperience_squared": [years_experience ** 2]
})

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Salary"):

    try:
        prediction = model.predict(input_data)

        st.success("Prediction Successful!")

        st.metric(
            label="Predicted Salary",
            value=f"${prediction[0]:,.2f}"
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# -----------------------------
# Display Input Data
# -----------------------------
with st.expander("View Input Data"):
    st.dataframe(input_data)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built using Python, Scikit-Learn and Streamlit")
