import streamlit as st
import pickle
import numpy as np

with open('rf_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

# Streamlit interface
st.title('Numerical Input Form')
st.header('Enter Your Numbers')

# Define default values or initializations
number1 = number2 = number3 = number4 = number5 = 0.0
number6 = number7 = number8 = number9 = number10 = 0.0

# Create a form
with st.form(key='numerical_form'):
    col1, col2 = st.columns(2)
    
    # Define input fields in two columns
    with col1:
        number1 = st.number_input('Potential', value=0.0)
        number2 = st.number_input('Value (Euros)', value=0.0)
        number3 = st.number_input('Wage (Euros)', value=0.0)
        number4 = st.number_input('Passing Ability', value=0.0)
        number5 = st.number_input('Dribbling Ability', value=0.0)
        
    with col2:
        number6 = st.number_input('Physique', value=0.0)
        number7 = st.number_input('Movement Reaction Rating', value=0.0)
        number8 = st.number_input('Mental Composure', value=0.0)
        number9 = st.number_input('Right Foot', value=0.0)
        number10 = st.number_input('Left Foot', value=0.0)
        
    # Add a submit button to the form
    submit_button = st.form_submit_button(label='Submit')

# Handle form submission
if submit_button:
    # Display submitted numbers
    st.subheader('Submitted Numbers')
    st.write(f'Potential: {number1}')
    st.write(f'Value (Euros): {number2}')
    st.write(f'Wage (Euros): {number3}')
    st.write(f'Passing Ability: {number4}')
    st.write(f'Dribbling Ability: {number5}')
    st.write(f'Physique: {number6}')
    st.write(f'Movement Reaction Rating: {number7}')
    st.write(f'Mental Composure: {number8}')
    st.write(f'Right Foot: {number9}')
    st.write(f'Left Foot: {number10}')
    
    # Example of using the loaded model (replace with your model logic)
    input_data = [[number1, number2, number3, number4, number5, 
                   number6, number7, number8, number9, number10]]
    prediction = rf_model.predict(input_data)
    st.write(f'Prediction: {prediction}')
    
    st.markdown('---')