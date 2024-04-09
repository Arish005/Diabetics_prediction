import streamlit as st
import pickle
import numpy as np

# Custom function to load model with dtype conversion
def load_model(file_path):
    with open(file_path, 'rb') as f:
        return fix_pickle(pickle.load(f))

# Function to fix dtype mismatch
def fix_pickle(obj):
    if isinstance(obj, tuple):
        return tuple(fix_pickle(item) for item in obj)
    elif isinstance(obj, list):
        return [fix_pickle(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.astype([
            ('left_child', '<i8'), 
            ('right_child', '<i8'), 
            ('feature', '<i8'), 
            ('threshold', '<f8'), 
            ('impurity', '<f8'), 
            ('n_node_samples', '<i8'), 
            ('weighted_n_node_samples', '<f8')
        ])
    else:
        return obj

# Load the trained model
try:
    model = load_model('diabetes-prediction-rfc-model.pkl')
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Function to predict diabetes
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age):
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]])
    prediction = model.predict(input_data)
    return prediction

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
        if result[0] == 0:
            st.success('The person is not diabetic')
        else:
            st.success('The person is diabetic')

if __name__ == '__main__':
    main()
