import streamlit as st
import numpy as np
import pandas as pd
import pickle
import logging

# Configure logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

# Function to fix dtype of tree nodes
def fix_tree(tree):
    if tree is not None:
        tree.left_child = fix_tree(tree.left_child)
        tree.right_child = fix_tree(tree.right_child)
        tree.feature = int(tree.feature)
        tree.threshold = float(tree.threshold)
        tree.impurity = float(tree.impurity)
        tree.n_node_samples = int(tree.n_node_samples)
        tree.weighted_n_node_samples = float(tree.weighted_n_node_samples)
    return tree

# Function to load the model
def load_model():
    try:
        # Load the model from the pickle file
        with open('diabetes-prediction-rfc-model.pkl', 'rb') as file:
            model = pickle.load(file)
        
        # Apply dtype fix to each estimator
        for estimator in model.estimators_:
            fix_tree(estimator.tree_)
        
        return model
    except Exception as e:
        logging.error(f"Error loading the model: {e}")
        st.error("An error occurred while loading the model. Please try again later.")
        st.stop()

# Load the model
model = load_model()

# Function to predict diabetes
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age):
    try:
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]])
        prediction = model.predict(input_data)
        return prediction
    except Exception as e:
        logging.error(f"Error predicting diabetes: {e}")
        return None

# Streamlit app
def main():
    # Page title
    st.title("Diabetes Prediction")

    # Sidebar
    st.sidebar.title("User Input")

    # Input features
    Pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 3)
    Glucose = st.sidebar.slider("Glucose", 0, 199, 117)
    BloodPressure = st.sidebar.slider("BloodPressure", 0, 122, 72)
    SkinThickness = st.sidebar.slider("SkinThickness", 0, 99, 23)
    Insulin = st.sidebar.slider("Insulin", 0, 846, 30)
    BMI = st.sidebar.slider("BMI", 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider("DPF", 0.078, 2.42, 0.3725)
    Age = st.sidebar.slider("Age", 21, 81, 29)

    # Prediction
    if st.sidebar.button("Predict"):
        result = predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age)
        if result is not None:
            if result[0] == 0:
                st.success('The person is not diabetic')
            else:
                st.success('The person is diabetic')
        else:
            st.error("An error occurred while predicting. Please try again later.")

if __name__ == '__main__':
    main()
