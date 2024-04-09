import pickle
from sklearn.ensemble import RandomForestClassifier

# Assume you have trained a RandomForestClassifier model
model = RandomForestClassifier()
# Train your model...

# Save the model to a pickle file
filename = 'diabetes-prediction-rfc-model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)
