import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
with open('alzhemiers.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to make a prediction
def predict_diagnosis(input_data):
    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_data], columns=['PhysicalActivity', 'MMSE', 'FunctionalAssessment', 
                                                   'MemoryComplaints', 'BehavioralProblems', 'ADL'])
    
    # Use the model to make a prediction
    prediction = model.predict(input_df)
    
    # Return the prediction result
    if prediction[0] == 1:
        return "Diagnosis: Alzheimer's Disease"
    else:
        return "Diagnosis: Healthy"

# Streamlit app UI
st.title("Alzheimer's Disease Diagnosis Prediction")

# Input fields for the user to enter values for each feature
physical_activity = st.sidebar.number_input("Physical Activity (0-10)", min_value=0, max_value=10)

mmse = st.sidebar.number_input("MMSE", min_value=0, max_value=30)

functional_assessment = st.sidebar.number_input("Functional Assessment (0-10)", min_value=0, max_value=10)

memory_complaints = st.selectbox("Memory Complaints (0-1)", [0, 1])

behavioral_problems = st.selectbox("Behavioral Problems (0-1)", [0, 1])

adl = st.sidebar.number_input("Activities of Daily Living (0-10)", min_value=0, max_value=10)

# Prepare input data
input_data = {
    'PhysicalActivity': physical_activity,
    'MMSE': mmse,
    'FunctionalAssessment': functional_assessment,
    'MemoryComplaints': memory_complaints,
    'BehavioralProblems': behavioral_problems,
    'ADL': adl
}

# Make prediction when the user clicks the button
if st.button('Predict'):
    result = predict_diagnosis(input_data)
    st.write(result)
