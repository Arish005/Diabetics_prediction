import pickle
import numpy as np
import streamlit as st

# Load the model
model_file = 'diabetes-prediction-rfc-model.pkl'

try:
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
        if isinstance(model, DecisionTreeClassifier):
            model = RandomForestClassifier()  # Convert Decision Tree to Random Forest
            model.estimators_ = [model]
            st.warning("Loaded model is a Decision Tree. Converted to Random Forest for prediction.")
except FileNotFoundError:
    st.error(f"Error: Model file '{model_file}' not found. Please make sure the file exists.")
    st.stop()
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
    st.title("Diabetes Predictor")

    # Sidebar for user input
    st.sidebar.title("User Input")
    Pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 3)
    Glucose = st.sidebar.slider("Glucose", 0, 199, 117)
    BloodPressure = st.sidebar.slider("Blood Pressure", 0, 122, 72)
    SkinThickness = st.sidebar.slider("Skin Thickness", 0, 99, 23)
    Insulin = st.sidebar.slider("Insulin", 0, 846, 30)
    BMI = st.sidebar.slider("BMI", 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider("DPF", 0.078, 2.42, 0.3725)
    Age = st.sidebar.slider("Age", 21, 81, 29)

    # Prediction
    if st.sidebar.button("Predict"):
        try:
            result = predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age)
            if result[0] == 0:
                st.success('The person is not diabetic')
            else:
                st.success('The person is diabetic')
        except Exception as e:
            st.error(f"Error predicting: {e}")

    st.write("Made by Arish")

if __name__ == '__main__':
    main()
