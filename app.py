import streamlit as st
import joblib
import numpy as np

# Load the Random Forest Classifier model
def load_model():
    try:
        # Load the model
        filename = 'diabetes-prediction-rfc-model.pkl'
        classifier = joblib.load(filename)
        return classifier
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

def main():
    st.title('Diabetes Prediction')

    # Load the model
    classifier = load_model()
    if classifier is None:
        st.write("Please check the model file and try again later.")
        return

    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=17, value=0)
    glucose = st.number_input('Glucose', min_value=0, max_value=200, value=0)
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=122, value=0)
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=99, value=0)
    insulin = st.number_input('Insulin', min_value=0, max_value=846, value=0)
    bmi = st.number_input('BMI', min_value=0.0, max_value=67.1, value=0.0)
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.078, max_value=2.42, value=0.078)
    age = st.number_input('Age', min_value=21, max_value=81, value=21)

    if st.button('Predict'):
        data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        prediction = classifier.predict(data)
        if prediction[0] == 1:
            st.error('You might have diabetes.')
        else:
            st.success('You may not have diabetes.')

if __name__ == '__main__':
    main()
