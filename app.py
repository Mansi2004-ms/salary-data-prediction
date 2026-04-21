
import streamlit as st
import pickle
import pandas as pd

# Load the trained model
try:
    model = pickle.load(open('random_forest_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: 'random_forest_model.pkl' not found. Please ensure the model is saved in the same directory as this app.")
    st.stop()

st.set_page_config(page_title='Salary Predictor', page_icon=':moneybag:')
st.title('📊 Salary Prediction App')
st.write('Enter the employee details below to get a predicted salary. This model uses a Random Forest Regressor trained on the provided dataset.')

st.sidebar.header('Input Employee Data')

# --- Input Features --- (using values based on the dataset's range and encoding)

# Age (float)
age = st.sidebar.slider('Age', min_value=18.0, max_value=65.0, value=30.0, step=1.0)

# Gender (int: 0 for Female, 1 for Male - based on LabelEncoder)
gender = st.sidebar.selectbox('Gender', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')

# Education Level (int: 0 for Bachelor's, 1 for Master's, 2 for PhD - based on LabelEncoder)
education_map = {0: "Bachelor's Degree", 1: "Master's Degree", 2: "PhD"}
education_level_input = st.sidebar.selectbox('Education Level', options=sorted(education_map.keys()), format_func=lambda x: education_map[x])

# Job Title (int: numerical ID - since LabelEncoder wasn't saved, we assume direct input of ID)
# Max value 198 observed from the original Job Title encoding in the notebook.
st.sidebar.info("For 'Job Title', please enter the numerical ID it was encoded to during training (e.g., from 0 to 198). This requires knowing the original encoding.")
job_title = st.sidebar.number_input('Job Title (Numerical ID)', min_value=0, max_value=198, value=0, step=1)

# Years of Experience (float)
years_of_experience = st.sidebar.slider('Years of Experience', min_value=0.0, max_value=40.0, value=5.0, step=0.5)


# Create a DataFrame for prediction
input_data = pd.DataFrame([[
age,
    gender,
    education_level_input,
    job_title,
    years_of_experience
]],
columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])

# Display input data
st.subheader('Employee Data Inputted:')
st.dataframe(input_data)

# Make prediction
if st.button('Predict Salary'):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f'### Predicted Salary: ${prediction:,.2f}')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}. Please check your inputs.")
