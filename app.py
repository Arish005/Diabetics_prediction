import streamlit as st
import pickle
import numpy as np

# Load the model
with open('diabetes-prediction-rfc-model.pkl', 'rb') as file:
    classifier = pickle.load(file)

# Function to predict diabetes
def predict_diabetes(preg, glucose, bp, st, insulin, bmi, dpf, age):
    data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
    prediction = classifier.predict(data)
    return prediction

# Streamlit app
def main():
    # Page title
    st.title("Diabetes Prediction")

    # Input features
    preg = st.slider("Pregnancies", 0, 17, 3)
    glucose = st.slider("Glucose", 0, 199, 117)
    bp = st.slider("Blood Pressure", 0, 122, 72)
    sthickness = st.slider("Skin Thickness", 0, 99, 23)
    insulin = st.slider("Insulin", 0, 846, 30)
    bmi = st.slider("BMI", 0.0, 67.1, 32.0)
    dpf = st.slider("DPF", 0.078, 2.42, 0.3725)
    age = st.slider("Age", 21, 81, 29)

    # Prediction
    if st.button("Predict"):
        result = predict_diabetes(preg, glucose, bp, sthickness, insulin, bmi, dpf, age)
        if result[0] == 0:
            st.success('The person is not diabetic')
        else:
            st.success('The person is diabetic')

if __name__ == '__main__':
    main()
