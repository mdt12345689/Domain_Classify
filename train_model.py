import json
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


def train_and_save(X, y, model_file, label_encoder_file, vectorizer_file, num_epochs):
    # Label encode the target variable y
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Create a mapping dictionary from app names to their corresponding labels
    app_mapping = {label: app_name for label, app_name in enumerate(label_encoder.classes_)}

    text_clf = MultinomialNB()
    vect = CountVectorizer()
    for epoch in range(num_epochs):
        X_transformed = vect.fit_transform(X)
        text_clf.partial_fit(X_transformed, y_encoded, classes=np.unique(y_encoded))

    # Save the model
    joblib.dump(text_clf, model_file)

    # Save the label encoder
    joblib.dump(label_encoder, label_encoder_file)

    # Save the vectorizer
    joblib.dump(vect, vectorizer_file)

    # Print classification report
    y_pred_encoded = text_clf.predict(X_transformed)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    report = classification_report(y, y_pred)
    print(f"Epoch {num_epochs}/{num_epochs}:\n{report}")


# Read data from JSON file                     SUA LAI FILE JSON
with open('Project_Domain.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Prepare training data
X = [app['Domain'] for app in data]
y = [app['App'] for app in data]

# File paths
model_file = 'model.sav'
label_encoder_file = 'label_encoder.sav'
vectorizer_file = 'vectorizer.sav'

# Number of training epochs
num_epochs = int(input("Enter the number of training epochs: "))

# Train and save the model
train_and_save(X, y, model_file, label_encoder_file, vectorizer_file, num_epochs)
