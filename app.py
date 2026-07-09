
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('salary_predictor_model.joblib')

st.title('Salary Prediction App')
st.write('Enter the years of experience to predict the salary.')

# Get user input for YearsExperience
years_experience = st.slider('Years of Experience', min_value=0.0, max_value=20.0, value=5.0, step=0.1)

# Create the feature DataFrame for prediction
# The model expects two features: YearsExperience and YearsExperience_squared
input_data = pd.DataFrame({
    'YearsExperience': [years_experience],
    'YearsExperience_squared': [years_experience**2]
})

# Make prediction
predicted_salary = model.predict(input_data)[0]

st.subheader('Predicted Salary')
st.write(f'${predicted_salary:,.2f}')
