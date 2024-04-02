import json
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

def predict_apps(domain_file, model_file, vectorizer_file, data_file):
    # Load model and vectorizer
    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)

    # Load domains from file
    with open(domain_file, 'r') as file:
        domains = file.readlines()

    # Create a LabelEncoder instance and fit it with app names
    with open(data_file, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        app_names = [item['App'] for item in data]
    label_encoder = LabelEncoder()
    label_encoder.fit(app_names)

    # Predict apps for each domain
    predictions = []
    for domain in domains:
        domain = domain.strip()  # Remove leading/trailing whitespaces and newlines

        # Transform domain using the loaded vectorizer
        X_transformed = vectorizer.transform([domain])

        # Predict the app for the domain
        prediction_encoded = model.predict(X_transformed)
        app_name = label_encoder.inverse_transform(prediction_encoded)[0]

        predictions.append((domain, app_name))

    return predictions

def add_data(data_file, json_file, domain, app_name):
    # Check if domain already exists in data
    domain_exists = False
    with open(data_file, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        for item in data:
            if item['Domain'] == domain:
                domain_exists = True
                break

    # If domain does not exist, ask whether to add it to data
    if not domain_exists:
        add_to_data = input(f"Do you want to add {domain} => {app_name} to data? (1/0): ")
        if add_to_data == '1':
            with open(data_file, 'w', encoding='utf-8') as json_file:
                data.append({'App': app_name, 'Domain': domain})
                json.dump(data, json_file, ensure_ascii=False, indent=4)

# File paths                SUA LAI TEN FILE
domain_file = 'New_Domain.txt'
model_file = 'model.sav'
vectorizer_file = 'vectorizer.sav'
data_file = 'Project_Domain.json'

# Perform prediction
predictions = predict_apps(domain_file, model_file, vectorizer_file, data_file)

# Print predictions
for domain, app_name in predictions:
    print(f"Domain: {domain} => Predicted App: {app_name}")
    add_data(data_file, data_file,domain,app_name)  # Add data to db if it doesn't exist
